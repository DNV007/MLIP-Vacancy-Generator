"""Symmetry-canonical hashing for exact vacancy-pattern deduplication."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .structure_utils import get_indices_by_species, wrap_fractional_coords


def build_parent_symmetry_maps(structure: Structure,
                               symprec: float,
                               match_tol: float) -> List[Dict[int, int]]:
    """Build how each symmetry operation permutes parent-site indices.

    Returns
    -------
    list of dict
        One dictionary per symmetry operation. Each dictionary maps
        original_site_index -> transformed_equivalent_site_index.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    symm_ops = analyzer.get_symmetry_operations(cartesian=False)
    indices_by_species = get_indices_by_species(structure)

    frac_coords = np.array([wrap_fractional_coords(site.frac_coords) for site in structure])

    maps: List[Dict[int, int]] = []
    for op_index, op in enumerate(symm_ops):
        op_map: Dict[int, int] = {}
        for i, site in enumerate(structure):
            transformed = wrap_fractional_coords(op.operate(frac_coords[i]))
            symbol = site.specie.symbol
            candidates = indices_by_species[symbol]

            candidate_coords = frac_coords[candidates]
            diff = candidate_coords - transformed
            diff -= np.round(diff)
            dists = np.linalg.norm(diff, axis=1)

            best_local = int(np.argmin(dists))
            best_dist = float(dists[best_local])

            if best_dist >= match_tol:
                raise RuntimeError(
                    f"Could not map site {i} under symmetry operation {op_index}. "
                    f"Closest same-species site is {best_dist:.6f} fractional units away "
                    f"(tolerance is {match_tol}). "
                    f"Try a slightly larger site-matching tolerance."
                )
            op_map[i] = candidates[best_local]
        maps.append(op_map)
    return maps


def canonicalize_combo(combo: Tuple[int, ...],
                       symmetry_maps: List[Dict[int, int]]) -> Tuple[int, ...]:
    """Return the lexicographically smallest symmetry-related image of a combo."""
    images = []
    for op_map in symmetry_maps:
        transformed = tuple(sorted(op_map[i] for i in combo))
        images.append(transformed)
    return min(images)


def make_combo_hash(structure: Structure,
                    recipe: Dict[str, int],
                    canonical_combo: Tuple[int, ...],
                    scaling: Tuple[int, int, int]) -> str:
    """Create a short stable hash for an exact vacancy pattern."""
    species_part = ";".join(f"{symbol}:{recipe[symbol]}" for symbol in sorted(recipe))
    combo_part = ",".join(str(i) for i in canonical_combo)
    payload = f"scale={scaling};recipe={species_part};combo={combo_part}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def keep_only_unique_combinations_by_hash(
    raw_combos: List[Tuple[int, ...]],
    structure: Structure,
    recipe: Dict[str, int],
    symmetry_maps: List[Dict[int, int]],
    scaling: Tuple[int, int, int],
) -> Tuple[
    List[Tuple[int, ...]],
    Dict[Tuple[int, ...], str],
    Dict[Tuple[int, ...], Tuple[int, ...]],
]:
    """Keep one representative from each exact symmetry-unique vacancy pattern.

    Returns
    -------
    unique_combos
        One representative raw combo from each unique exact defect pattern.
    combo_to_hash
        Mapping from representative combo -> short canonical hash.
    combo_to_canonical
        Mapping from representative combo -> canonical tuple used for the hash.
    """
    unique_by_hash: Dict[str, Tuple[int, ...]] = {}
    combo_to_hash: Dict[Tuple[int, ...], str] = {}
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    for combo in raw_combos:
        canonical = canonicalize_combo(combo, symmetry_maps)
        combo_hash = make_combo_hash(structure, recipe, canonical, scaling)
        if combo_hash not in unique_by_hash:
            unique_by_hash[combo_hash] = combo
            combo_to_hash[combo] = combo_hash
            combo_to_canonical[combo] = canonical

    unique_combos = list(unique_by_hash.values())
    return unique_combos, combo_to_hash, combo_to_canonical
