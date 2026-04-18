#!/usr/bin/env python3
"""
Generate vacancy structures for defect studies and MLIP dataset building.

This version focuses on
*generation-time* uniqueness and diversity. It does **not** compare relaxed
structures. Instead, it offers two separate ideas that are useful before any
relaxation is run:

1. Exact uniqueness for vacancy patterns:
   Use the symmetry of the parent crystal to build a canonical occupation key,
   then hash that key. Two vacancy combinations that are related by a parent
   symmetry operation get the same hash.

2. Diversity selection for MLIP starting structures:
   Optionally compute **defect-local** SOAP descriptors at the removed-site
   positions and keep a diverse subset using farthest-point sampling.

Why both ideas exist
--------------------
- The symmetry-canonical hash answers: "Are these the same vacancy pattern in
  the ideal parent crystal?"
- SOAP answers: "Among many candidate starting structures, which ones look
  most different in the *defect-local* atomic environments around the removed
  sites?"

SOAP is therefore used here for *generation-time diversity pruning*, using
only the neighborhoods around the vacancy centers, not for post-relaxation
equivalence testing.

Important scientific note
-------------------------
This script generates candidate structures. It does not prove that two defects
will remain equivalent after relaxation. For relaxed defect studies or DFT
labelling, run calculations without symmetry constraints when appropriate.
For VASP this commonly means ISYM = 0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple  # 'Iterable' removed: was unused

import numpy as np
from numpy.random import SeedSequence
from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    _ASE_ADAPTOR_AVAILABLE = True
except Exception:
    _ASE_ADAPTOR_AVAILABLE = False

try:
    import ase.io
    _ASE_IO_AVAILABLE = True
except Exception:
    _ASE_IO_AVAILABLE = False

try:
    from dscribe.descriptors import SOAP
    _DSCRIBE_AVAILABLE = True
except Exception:
    _DSCRIBE_AVAILABLE = False


# Canonical field order for the metadata CSV so columns read logically in a
# spreadsheet rather than alphabetically scattered.
_METADATA_FIELD_ORDER = [
    "record_type",
    "defect_class",
    "combo_index",
    "seed_index",
    "canonical_hash",
    "poscar_file",
    "extxyz_file",
    "distance_filter_passed",
    "distance_filter_reason",
    "removed_indices",
    "canonical_removed_indices",
    "removed_species",
    "removed_frac_coords",
    "n_removed_total",
    "supercell_scaling",
    "perturb_radius_A",
    "max_random_displacement_A",
    "perturbed_atom_indices",
]


# -----------------------------------------------------------------------------
# Small input helpers
# -----------------------------------------------------------------------------

def ask_string(prompt: str, default: Optional[str] = None) -> str:
    """Ask the user for text input.

    Parameters
    ----------
    prompt
        Message shown to the user.
    default
        Value used when the user presses Enter without typing anything.

    Returns
    -------
    str
        The user response, or the default if the response is empty.
    """
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Please enter a value.")


def ask_choice(prompt: str, choices: Sequence[str], default: str) -> str:
    """Ask the user to pick one value from a fixed set of allowed choices.

    Loops until the user enters a valid choice, giving a friendly error
    message instead of a traceback on invalid input.
    """
    choices_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} [{choices_str}, default: {default}]: ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"Please choose one of: {choices_str}")


def ask_float(prompt: str, default: float) -> float:
    """Ask the user for a floating-point number and validate the input."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default  # already a float; no redundant cast needed
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def ask_int(prompt: str, default: Optional[int] = None,
            min_value: Optional[int] = None,
            max_value: Optional[int] = None) -> int:
    """Ask the user for an integer and validate the input."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue

        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask the user a yes/no question and return True or False."""
    default_text = "y" if default else "n"
    while True:
        raw = input(f"{prompt} [y/n, default: {default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with y or n.")


# -----------------------------------------------------------------------------
# Basic structure inspection
# -----------------------------------------------------------------------------

def get_species_counts(structure: Structure) -> Dict[str, int]:
    """Return how many atoms of each species are present in the structure."""
    counts: Dict[str, int] = defaultdict(int)
    for site in structure:
        counts[site.specie.symbol] += 1
    return dict(counts)


def print_species_counts(structure: Structure) -> None:
    """Print a simple species summary for the structure."""
    print("\nSpecies present in the structure:")
    for symbol, count in get_species_counts(structure).items():
        print(f"  {symbol}: {count}")


def print_space_group_info(structure: Structure, symprec: float) -> None:
    """Print the parent space group of the structure."""
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    symbol = analyzer.get_space_group_symbol()
    number = analyzer.get_space_group_number()
    print(f"\nParent space group: {symbol} (No. {number})")


def print_equivalent_site_groups(structure: Structure, symprec: float) -> None:
    """Print symmetry-equivalent site groups in the parent structure.

    This helps the user see which atoms are already equivalent before any defect
    is made. If all Mg sites appear in one group, then a single Mg vacancy is
    unique in the ideal parent crystal.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    symmetrized = analyzer.get_symmetrized_structure()

    print("\nEquivalent site groups in the parent structure:")
    for group in symmetrized.equivalent_indices:
        symbol = structure[group[0]].specie.symbol
        coords_text = ", ".join(
            [f"{idx}:{np.round(structure[idx].frac_coords, 4).tolist()}" for idx in group]
        )
        print(f"  {symbol}: [{coords_text}]")


# -----------------------------------------------------------------------------
# Supercell handling
# -----------------------------------------------------------------------------

def propose_supercell_scaling(structure: Structure,
                              min_lattice_length: float = 10.0,
                              min_atoms: Optional[int] = None,
                              max_atoms: int = 400) -> Tuple[int, int, int]:
    """Propose a diagonal supercell scaling (a, b, c).

    The rule is intentionally simple:
    - first make each lattice vector at least `min_lattice_length` long,
    - then, if requested, enlarge further until the atom count reaches
      `min_atoms`,
    - but do not exceed `max_atoms`.

    This is a practical starting point for defect generation, not a perfect
    guarantee of defect-image separation in every material.
    """
    lattice_lengths = structure.lattice.abc
    scale = [max(1, int(math.ceil(min_lattice_length / length)))
             for length in lattice_lengths]

    current_atoms = len(structure) * scale[0] * scale[1] * scale[2]

    if min_atoms is not None:
        while current_atoms < min_atoms:
            smallest_axis = min(range(3), key=lambda i: lattice_lengths[i] * scale[i])
            trial = scale.copy()
            trial[smallest_axis] += 1
            trial_atoms = len(structure) * trial[0] * trial[1] * trial[2]
            if trial_atoms > max_atoms:
                break
            scale = trial
            current_atoms = trial_atoms

    if len(structure) * scale[0] * scale[1] * scale[2] > max_atoms:
        raise ValueError(
            "Automatic supercell would exceed the allowed maximum number of atoms. "
            "Use a smaller minimum lattice length, a smaller minimum atom count, "
            "or increase max_atoms if you really intend a larger cell."
        )

    return int(scale[0]), int(scale[1]), int(scale[2])


def maybe_build_supercell(structure: Structure) -> Tuple[Structure, Tuple[int, int, int]]:
    """Optionally build a supercell from the parent structure."""
    use_supercell = ask_yes_no(
        "Do you want to automatically build a supercell before creating vacancies?",
        default=True,
    )
    if not use_supercell:
        return structure, (1, 1, 1)

    min_length = ask_float("Target minimum lattice-vector length in Å", 10.0)
    use_min_atoms = ask_yes_no("Also enforce a minimum total number of atoms?", default=False)
    min_atoms = None
    if use_min_atoms:
        min_atoms = ask_int("Minimum total number of atoms in the supercell", 96, min_value=1)
    max_atoms = ask_int("Maximum allowed total number of atoms", 400, min_value=1)

    scaling = propose_supercell_scaling(
        structure,
        min_lattice_length=min_length,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
    )

    supercell = structure.copy()
    supercell.make_supercell(scaling)

    print(
        f"\nUsing supercell scaling {scaling}. "
        f"Number of atoms: {len(structure)} -> {len(supercell)}"
    )
    return supercell, scaling


# -----------------------------------------------------------------------------
# Defect recipe parsing
# -----------------------------------------------------------------------------

def parse_removal_recipe(recipe_text: str,
                         available_species: Dict[str, int]) -> Dict[str, int]:
    """Parse text like 'Mg:1, Se:2' into a dictionary.

    Returns
    -------
    dict
        Example: {'Mg': 1, 'Se': 2}

    Raises
    ------
    ValueError
        If a species appears more than once, is unknown, or has an invalid count.
    """
    recipe: Dict[str, int] = {}
    parts = [part.strip() for part in recipe_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("No defect recipe was provided.")

    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"Could not parse '{part}'. Please use the form Species:Count, "
                f"for example Mg:1"
            )
        symbol, count_text = [item.strip() for item in part.split(":", 1)]

        # Check for duplicate species *before* trying to parse count_text as an
        # integer. Attempting int(count_text) inside the error message here would
        # raise a secondary ValueError when count_text is non-numeric (e.g.
        # "Mg:abc, Mg:2"), masking the real "appears more than once" message.
        if symbol in recipe:
            raise ValueError(
                f"Species '{symbol}' appears more than once in the recipe. "
                f"Combine into a single entry, e.g. '{symbol}:<total>'."
            )

        if symbol not in available_species:
            raise ValueError(
                f"Species '{symbol}' is not present. Available species are: "
                f"{', '.join(available_species)}"
            )

        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"Removal count for {symbol} must be an integer, got '{count_text}'."
            ) from exc

        if count < 1:
            raise ValueError(f"Removal count for {symbol} must be at least 1.")
        if count > available_species[symbol]:
            raise ValueError(
                f"Cannot remove {count} atoms of {symbol}; "
                f"only {available_species[symbol]} are present."
            )

        recipe[symbol] = count
    return recipe


def ask_removal_recipe(structure: Structure) -> Dict[str, int]:
    """Ask the user which atoms to remove.

    Accepted examples
    -----------------
    Mg:1
    Se:2
    Mg:1, Se:2
    """
    available = get_species_counts(structure)
    print("\nEnter the defect recipe as Species:Count pairs separated by commas.")
    print("Examples: Mg:1   or   Mg:1, Se:2")
    while True:
        raw = input("Defect recipe: ").strip()
        try:
            return parse_removal_recipe(raw, available)
        except ValueError as err:
            print(err)


# -----------------------------------------------------------------------------
# Vacancy combination generation
# -----------------------------------------------------------------------------

def get_indices_by_species(structure: Structure) -> Dict[str, List[int]]:
    """Return the site indices belonging to each chemical species."""
    result: Dict[str, List[int]] = defaultdict(list)
    for index, site in enumerate(structure):
        result[site.specie.symbol].append(index)
    return dict(result)


def generate_raw_removal_combinations(structure: Structure,
                                      recipe: Dict[str, int]) -> List[Tuple[int, ...]]:
    """Generate every possible atom-index combination for the requested recipe.

    Example
    -------
    If the user asks for Mg:1 and Se:2, this function generates all choices of
    one Mg index and two Se indices, then combines them into full defect tuples.
    """
    indices_by_species = get_indices_by_species(structure)
    specieswise_choices: List[List[Tuple[int, ...]]] = []

    for symbol in sorted(recipe):
        count = recipe[symbol]
        species_indices = indices_by_species[symbol]
        specieswise_choices.append(list(combinations(species_indices, count)))

    all_combos = []
    for choice_group in product(*specieswise_choices):
        flat = tuple(sorted(idx for subgroup in choice_group for idx in subgroup))
        all_combos.append(flat)

    return all_combos


# -----------------------------------------------------------------------------
# Exact uniqueness by symmetry-canonical hash
# -----------------------------------------------------------------------------

def wrap_fractional_coords(frac_coords: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into the [0, 1) interval."""
    return np.mod(frac_coords, 1.0)


def periodic_fractional_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the shortest distance between two fractional points.

    This is measured in fractional-coordinate space and is used only for site
    matching when a symmetry operation maps one parent site onto another parent
    site.
    """
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    diff -= np.round(diff)
    return float(np.linalg.norm(diff))


def build_parent_symmetry_maps(structure: Structure,
                               symprec: float,
                               match_tol: float) -> List[Dict[int, int]]:
    """Build how each symmetry operation permutes parent-site indices.

    Returns
    -------
    list of dict
        One dictionary per symmetry operation. Each dictionary maps
        original_site_index -> transformed_equivalent_site_index.

    Notes
    -----
    The closest-candidate strategy (rather than first-match) is used so that
    high-symmetry supercells with tightly spaced atoms of the same species are
    handled correctly. Per-site candidate distances are vectorised with NumPy
    to reduce the constant factor for large supercells.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    symm_ops = analyzer.get_symmetry_operations(cartesian=False)
    indices_by_species = get_indices_by_species(structure)

    # Pre-compute wrapped fractional coordinates as a (N, 3) array.
    frac_coords = np.array([wrap_fractional_coords(site.frac_coords) for site in structure])

    maps: List[Dict[int, int]] = []
    for op_index, op in enumerate(symm_ops):
        op_map: Dict[int, int] = {}
        for i, site in enumerate(structure):
            transformed = wrap_fractional_coords(op.operate(frac_coords[i]))
            symbol = site.specie.symbol
            candidates = indices_by_species[symbol]

            # Vectorised minimum-image distance from transformed to all
            # same-species candidate sites.
            candidate_coords = frac_coords[candidates]        # (K, 3)
            diff = candidate_coords - transformed              # (K, 3)
            diff -= np.round(diff)
            dists = np.linalg.norm(diff, axis=1)              # (K,)

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
    """Return the canonical representative of a vacancy combination.

    The canonical representative is chosen as the lexicographically smallest
    symmetry-related image of the atom-index tuple.
    """
    images = []
    for op_map in symmetry_maps:
        transformed = tuple(sorted(op_map[i] for i in combo))
        images.append(transformed)
    return min(images)


def make_combo_hash(structure: Structure,
                    recipe: Dict[str, int],
                    canonical_combo: Tuple[int, ...],
                    scaling: Tuple[int, int, int]) -> str:
    """Create a short stable hash for an exact vacancy pattern.

    The hash is based on:
    - the supercell scaling,
    - the defect recipe,
    - the canonical symmetry-reduced site tuple.

    Two raw combinations that are symmetry-equivalent in the parent crystal will
    receive the same hash if their canonical tuples are the same.
    """
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


# -----------------------------------------------------------------------------
# Dataset balancing and diversity selection
# -----------------------------------------------------------------------------

def defect_class_label(recipe: Dict[str, int]) -> str:
    """Return a short human-readable label such as 'Mg1+Se2'."""
    return "+".join(f"{symbol}{recipe[symbol]}" for symbol in sorted(recipe))


def cap_items_randomly(items: List[Tuple[int, ...]],
                       max_count: Optional[int],
                       rng: random.Random) -> List[Tuple[int, ...]]:
    """Randomly down-select a list if it is larger than `max_count`."""
    if max_count is None or len(items) <= max_count:
        return items
    chosen = rng.sample(items, max_count)
    chosen.sort()
    return chosen


def build_soap_descriptor(species: Sequence[str],
                          r_cut: float,
                          n_max: int,
                          l_max: int,
                          sigma: float):
    """Build a DScribe SOAP descriptor object for defect-local environments.

    The descriptor is configured with ``average="off"`` because we want one SOAP
    vector **per vacancy center**. We then combine those per-center vectors in a
    simple, order-independent way inside :func:`defect_local_soap_vector`.
    """
    if not _DSCRIBE_AVAILABLE:
        raise RuntimeError("DScribe is not available in this Python environment.")
    if not _ASE_ADAPTOR_AVAILABLE:
        raise RuntimeError("pymatgen-to-ASE conversion is not available.")

    return SOAP(
        species=list(species),
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average="off",
        sparse=False,
    )


def defect_local_soap_vector(defect_structure: Structure,
                             parent_structure: Structure,
                             removed_indices: Tuple[int, ...],
                             soap) -> np.ndarray:
    """Return one defect-local SOAP vector for one vacancy structure.

    How this works in plain language
    --------------------------------
    1. Take the positions of the removed atoms from the **parent** structure.
       These are the vacancy centers.
    2. Evaluate SOAP at those Cartesian points in the *defect* structure.
       This describes only the neighborhood around the vacancy, not the whole
       supercell.
    3. Average the per-vacancy SOAP vectors so the final descriptor does not
       depend on the order of the removed sites.

    This is still a heuristic for diversity pruning, not an exact notion of
    equivalence. Exact equivalence is handled separately by the symmetry-based
    canonical hash.
    """
    atoms = AseAtomsAdaptor.get_atoms(defect_structure)
    vacancy_centers = [parent_structure[i].coords for i in removed_indices]

    raw = np.asarray(soap.create(atoms, centers=vacancy_centers), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    # Average over vacancy centers to get an order-independent structure-level
    # descriptor for this defect class.
    vector = np.mean(raw, axis=0)
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm
    return vector


def farthest_point_sample(vectors: List[np.ndarray], target_count: int) -> List[int]:
    """Select a diverse subset of vectors by farthest-point sampling.

    The first vector chosen is the one farthest from the average vector. Each
    next vector is chosen to be as far as possible from the already selected set.

    This is a simple, understandable diversity-selection rule. It is not the
    only possible one, but it works well enough for pruning many similar defect
    candidates.

    Notes
    -----
    ``remaining`` is sorted before ``max`` to make tie-breaking deterministic
    across all platforms and Python versions.
    """
    if target_count >= len(vectors):
        return list(range(len(vectors)))
    if target_count <= 0:
        return []

    matrix = np.vstack(vectors)
    center = np.mean(matrix, axis=0)
    distances_to_center = np.linalg.norm(matrix - center, axis=1)
    first_index = int(np.argmax(distances_to_center))

    selected = [first_index]
    remaining = set(range(len(vectors))) - {first_index}
    min_dist_to_selected = np.linalg.norm(matrix - matrix[first_index], axis=1)

    while len(selected) < target_count and remaining:
        next_index = max(sorted(remaining), key=lambda i: min_dist_to_selected[i])
        selected.append(int(next_index))
        remaining.remove(next_index)
        new_dist = np.linalg.norm(matrix - matrix[next_index], axis=1)
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_dist)

    selected.sort()
    return selected


def select_diverse_structures_by_soap(defect_structures: List[Structure],
                                      parent_structure: Structure,
                                      removed_combos: List[Tuple[int, ...]],
                                      max_count: int,
                                      species: Sequence[str],
                                      r_cut: float,
                                      n_max: int,
                                      l_max: int,
                                      sigma: float) -> List[int]:
    """Pick a diverse subset of generated structures using *defect-local* SOAP.

    SOAP is computed only at the vacancy-center positions taken from the parent
    structure. The result is a diversity pruning step that focuses on the local
    defect environment instead of letting the bulk part of a large supercell
    dominate the descriptor.
    """
    if len(defect_structures) != len(removed_combos):
        raise ValueError("defect_structures and removed_combos must have the same length.")

    soap = build_soap_descriptor(species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma)
    vectors = [
        defect_local_soap_vector(defect_structure, parent_structure, combo, soap)
        for defect_structure, combo in zip(defect_structures, removed_combos)
    ]
    return farthest_point_sample(vectors, max_count)


# -----------------------------------------------------------------------------
# Structure creation and local perturbation
# -----------------------------------------------------------------------------

def remove_sites_from_structure(structure: Structure,
                               combo: Tuple[int, ...]) -> Structure:
    """Return a copy of the structure with the chosen sites removed."""
    new_structure = structure.copy()
    new_structure.remove_sites(sorted(combo, reverse=True))
    return new_structure


def minimum_image_cartesian_distance(lattice_matrix: np.ndarray,
                                     frac_a: Sequence[float],
                                     frac_b: Sequence[float]) -> float:
    """Return the shortest Cartesian distance between two fractional points."""
    diff = np.array(frac_a, dtype=float) - np.array(frac_b, dtype=float)
    diff -= np.round(diff)
    cart = diff @ lattice_matrix
    return float(np.linalg.norm(cart))


def get_nearby_atom_indices(defect_structure: Structure,
                            vacancy_cart_coords: Sequence[np.ndarray],
                            radius: float) -> List[int]:
    """Find atoms lying within `radius` of any removed-site position.

    The vacancy positions come from the parent structure. We search around each
    removed site separately and take the union of all nearby atoms.
    """
    lattice_matrix = np.array(defect_structure.lattice.matrix)
    vacancy_frac_coords = [defect_structure.lattice.get_fractional_coords(coord)
                           for coord in vacancy_cart_coords]

    nearby = set()
    for atom_index, site in enumerate(defect_structure):
        for vacancy_frac in vacancy_frac_coords:
            distance = minimum_image_cartesian_distance(
                lattice_matrix,
                site.frac_coords,
                vacancy_frac,
            )
            if distance <= radius:
                nearby.add(atom_index)
                break
    return sorted(nearby)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3D unit vector.

    Uses a ``while`` loop rather than recursion so that the astronomically
    unlikely degenerate draw of a near-zero vector cannot cause RecursionError.
    """
    while True:
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm >= 1e-12:
            return vec / norm


def make_rattled_copy_near_vacancies(defect_structure: Structure,
                                     parent_structure: Structure,
                                     removed_indices: Tuple[int, ...],
                                     radius: float,
                                     max_displacement: float,
                                     rng: np.random.Generator) -> Tuple[Structure, List[int]]:
    """Create a perturbed copy of a defect structure.

    Perturbations are applied only to atoms near the removed parent sites. This
    keeps the random noise local to the defect region rather than shaking the
    entire crystal.
    """
    vacancy_cart_coords = [parent_structure[i].coords for i in removed_indices]
    target_indices = get_nearby_atom_indices(defect_structure, vacancy_cart_coords, radius)

    new_structure = defect_structure.copy()
    for idx in target_indices:
        displacement = random_unit_vector(rng) * rng.uniform(0.0, max_displacement)
        new_structure.translate_sites([idx], displacement, frac_coords=False, to_unit_cell=True)

    return new_structure, target_indices


# -----------------------------------------------------------------------------
# Minimum-distance filtering
# -----------------------------------------------------------------------------

def element_radius(symbol: str) -> float:
    """Return a simple radius estimate in Å for minimum-distance screening.

    The function first tries to use pymatgen's radius information. If a radius is
    missing, it falls back to 1.0 Å. This is a *sanity filter* only. It is not a
    full chemistry or bonding model.
    """
    element = Element(symbol)

    for attr in ("atomic_radius", "atomic_radius_calculated"):
        value = getattr(element, attr, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return 1.0


def structure_passes_min_distance_filter(
    structure: Structure,
    radius_scale: float = 0.55,
    absolute_floor: float = 0.80,
) -> Tuple[bool, Optional[dict]]:
    """Reject obviously unphysical short contacts.

    The minimum allowed distance for a pair is approximated as
    max(absolute_floor, radius_scale * (radius_A + radius_B)).

    Uses NumPy broadcasting over the pre-computed distance matrix to find
    violations in one vectorised pass rather than an O(N²) Python loop.
    """
    dist = np.array(structure.distance_matrix)

    species = [site.specie.symbol for site in structure]
    radii = np.array([element_radius(s) for s in species])

    # Build the pairwise minimum-allowed distance matrix via broadcasting.
    min_allowed = np.maximum(absolute_floor, radius_scale * (radii[:, None] + radii[None, :]))

    # Ignore self-distances by setting the diagonal to a safe large value.
    np.fill_diagonal(dist, np.inf)

    violations = np.argwhere(dist < min_allowed)
    if violations.size > 0:
        i, j = int(violations[0, 0]), int(violations[0, 1])
        return False, {
            "pair": (i, j),
            "species": (species[i], species[j]),
            "distance": float(dist[i, j]),
            "minimum_allowed": float(min_allowed[i, j]),
        }
    return True, None


# -----------------------------------------------------------------------------
# File writing and metadata
# -----------------------------------------------------------------------------

def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def structure_filename(base_name: str,
                       class_label: str,
                       combo_index: int,
                       seed_index: Optional[int],
                       extension: str = ".vasp") -> str:
    """Build a readable filename for one output structure."""
    if seed_index is None:
        return f"{base_name}_{class_label}_base{combo_index:04d}{extension}"
    return f"{base_name}_{class_label}_base{combo_index:04d}_seed{seed_index:03d}{extension}"


def write_poscar(structure: Structure, path: str) -> None:
    """Write a structure in VASP POSCAR format."""
    structure.to(fmt="poscar", filename=path)


def maybe_write_extxyz(structure: Structure, path: str) -> bool:
    """Write extxyz if ASE support is available.

    Returns True when the file was written successfully, otherwise False.
    """
    if not (_ASE_ADAPTOR_AVAILABLE and _ASE_IO_AVAILABLE):
        return False
    atoms = AseAtomsAdaptor.get_atoms(structure)
    ase.io.write(path, atoms, format="extxyz")
    return True


def save_metadata_csv(records: List[dict], path: str) -> None:
    """Save metadata to CSV with a canonical, human-friendly column order.

    Uses the module-level ``_METADATA_FIELD_ORDER`` list so that the CSV is
    immediately readable in a spreadsheet without reordering columns. Any
    fields not in the list are appended at the end in sorted order so nothing
    is silently dropped.
    """
    if not records:
        return

    all_keys = {key for record in records for key in record}
    extra_keys = sorted(all_keys - set(_METADATA_FIELD_ORDER))
    fieldnames = [f for f in _METADATA_FIELD_ORDER if f in all_keys] + extra_keys

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_json(data: object, path: str) -> None:
    """Save Python data to a JSON file with readable indentation."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


# -----------------------------------------------------------------------------
# Metadata record helpers
# -----------------------------------------------------------------------------

def _base_record(class_label: str,
                 combo_index: int,
                 combo: Tuple[int, ...],
                 canonical_combo: Tuple[int, ...],
                 combo_hash: str,
                 removed_species: List[str],
                 removed_frac_coords: List[list],
                 scaling: Tuple[int, int, int],
                 poscar_name: str,
                 extxyz_name: Optional[str]) -> dict:
    """Build the metadata dict for an unperturbed base defect structure."""
    return {
        "record_type": "base",
        "defect_class": class_label,
        "combo_index": combo_index,
        "seed_index": "",
        "removed_indices": list(combo),
        "canonical_removed_indices": list(canonical_combo),
        "canonical_hash": combo_hash,
        "removed_species": removed_species,
        "removed_frac_coords": removed_frac_coords,
        "n_removed_total": len(combo),
        "poscar_file": poscar_name,
        "extxyz_file": extxyz_name or "",
        "supercell_scaling": list(scaling),
        "perturb_radius_A": "",
        "max_random_displacement_A": "",
        "perturbed_atom_indices": [],
        "distance_filter_passed": True,
        "distance_filter_reason": "",
    }


def _seed_record(class_label: str,
                 combo_index: int,
                 seed_offset: int,
                 combo: Tuple[int, ...],
                 canonical_combo: Tuple[int, ...],
                 combo_hash: str,
                 removed_species: List[str],
                 removed_frac_coords: List[list],
                 scaling: Tuple[int, int, int],
                 perturb_radius: float,
                 amplitude: float,
                 perturbed_indices: List[int],
                 passed: bool,
                 reason: Optional[dict],
                 poscar_name: str,
                 extxyz_name: Optional[str]) -> dict:
    """Build the metadata dict for one rattled seed (accepted or rejected).

    ``reason`` is serialised as a JSON string so that the CSV and JSON
    representations are consistent and machine-readable (a raw Python dict
    repr in a CSV cell is hard to parse).
    """
    return {
        "record_type": "seed" if passed else "rejected_seed",
        "defect_class": class_label,
        "combo_index": combo_index,
        "seed_index": seed_offset,
        "removed_indices": list(combo),
        "canonical_removed_indices": list(canonical_combo),
        "canonical_hash": combo_hash,
        "removed_species": removed_species,
        "removed_frac_coords": removed_frac_coords,
        "n_removed_total": len(combo),
        "poscar_file": poscar_name,
        "extxyz_file": extxyz_name or "",
        "supercell_scaling": list(scaling),
        "perturb_radius_A": perturb_radius,
        "max_random_displacement_A": float(amplitude),
        "perturbed_atom_indices": perturbed_indices,
        "distance_filter_passed": passed,
        "distance_filter_reason": json.dumps(reason) if reason else "",
    }


# -----------------------------------------------------------------------------
# Main generation workflow — sub-functions
# -----------------------------------------------------------------------------

def _fill_raw_combo_maps(
    combos: List[Tuple[int, ...]],
    structure: Structure,
    recipe: Dict[str, int],
    scaling: Tuple[int, int, int],
    symprec: float,
    match_tol: float,
    symmetry_maps: Optional[List[Dict[int, int]]],
    combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
) -> List[Dict[int, int]]:
    """Populate hash and canonical-combo look-up tables for a list of combos.

    Builds symmetry maps on the fly using the caller-supplied tolerances if
    they have not already been computed. Returns the (possibly newly built)
    symmetry maps so the caller can reuse them.
    """
    if symmetry_maps is None:
        symmetry_maps = build_parent_symmetry_maps(structure, symprec, match_tol)
    for combo in combos:
        canonical = canonicalize_combo(combo, symmetry_maps)
        h = make_combo_hash(structure, recipe, canonical, scaling)
        raw_combo_to_hash[combo] = h
        combo_to_hash[combo] = h
        combo_to_canonical[combo] = canonical
    return symmetry_maps


def _build_base_combos(
    raw_combos: List[Tuple[int, ...]],
    unique_combos: List[Tuple[int, ...]],
    symmetry_maps: Optional[List[Dict[int, int]]],
    working_structure: Structure,
    recipe: Dict[str, int],
    scaling: Tuple[int, int, int],
    symprec: float,
    match_tol: float,
    mode: str,
    combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
) -> Tuple[List[Tuple[int, ...]], str]:
    """Select the base set of combinations depending on the chosen mode.

    ``symprec`` and ``match_tol`` are passed explicitly so that any symmetry
    maps built on demand (for raw mode) use the tolerances the user actually
    supplied rather than hardcoded fallback values.

    Returns
    -------
    base_combos
        The list of combinations to use as base defects.
    base_source
        ``"raw"`` or ``"unique"`` (recorded in the summary JSON).
    """
    if mode == "raw":
        base_combos = list(raw_combos)
        _fill_raw_combo_maps(
            base_combos, working_structure, recipe, scaling, symprec, match_tol,
            symmetry_maps, combo_to_hash, combo_to_canonical, raw_combo_to_hash,
        )
        return base_combos, "raw"

    if mode in {"unique", "seeded"}:
        return list(unique_combos), "unique"

    # mlip mode — ask the user which base to use.
    base_source = ask_choice(
        "For mlip mode, use raw or unique base defects",
        choices=["raw", "unique"],
        default="unique",
    )
    if base_source == "raw":
        base_combos = list(raw_combos)
        _fill_raw_combo_maps(
            base_combos, working_structure, recipe, scaling, symprec, match_tol,
            symmetry_maps, combo_to_hash, combo_to_canonical, raw_combo_to_hash,
        )
        return base_combos, "raw"

    return list(unique_combos), "unique"


def _apply_diversity_pruning(
    base_combos: List[Tuple[int, ...]],
    working_structure: Structure,
    mode: str,
    sampling_seed: int,
    python_rng: random.Random,
) -> Tuple[List[Tuple[int, ...]], Optional[int], str, bool, dict]:
    """Optionally cap and/or SOAP-prune the base combo list.

    Returns
    -------
    base_combos
        Possibly pruned list.
    max_per_class
        The cap used (None if no cap was applied).
    base_selection_method
        Short label for the summary JSON.
    use_soap_pruning
        Whether SOAP was used.
    soap_config
        The SOAP hyper-parameters actually used.
    """
    soap_config: dict = {
        "r_cut": 5.0,
        "n_max": 8,
        "l_max": 6,
        "sigma": 0.5,
    }

    apply_cap = ask_yes_no(
        "Apply a balancing cap to the number of base defect structures?", default=True
    )
    if not apply_cap:
        return base_combos, None, "none", False, soap_config

    max_per_class = ask_int(
        "Maximum number of base defect structures for this defect class",
        50,
        min_value=1,
    )

    if len(base_combos) <= max_per_class:
        return base_combos, max_per_class, "cap_not_needed", False, soap_config

    use_soap_pruning = False
    if _DSCRIBE_AVAILABLE and _ASE_ADAPTOR_AVAILABLE:
        use_soap_pruning = ask_yes_no(
            "Use defect-local SOAP to keep a diverse subset of generated unrelaxed "
            "base defects instead of random down-selection?",
            default=(mode == "mlip"),
        )
    else:
        print("SOAP pruning is unavailable because DScribe and/or ASE support is missing.")

    if use_soap_pruning:
        soap_config["r_cut"] = ask_float("Defect-local SOAP cutoff r_cut in Å", 5.0)
        soap_config["n_max"] = ask_int("SOAP n_max", 8, min_value=1)
        soap_config["l_max"] = ask_int("SOAP l_max", 6, min_value=0)
        soap_config["sigma"] = ask_float("SOAP Gaussian width sigma in Å", 0.5)

        # Build candidate structures once; they are also cached in the write
        # loop via defect_structure_cache so no work is duplicated.
        print("Computing defect-local SOAP descriptors for diversity pruning…")
        candidate_structures = [
            remove_sites_from_structure(working_structure, combo)
            for combo in base_combos
        ]
        all_species = sorted(get_species_counts(working_structure).keys())
        selected_indices = select_diverse_structures_by_soap(
            candidate_structures,
            parent_structure=working_structure,
            removed_combos=base_combos,
            max_count=max_per_class,
            species=all_species,
            r_cut=soap_config["r_cut"],
            n_max=soap_config["n_max"],
            l_max=soap_config["l_max"],
            sigma=soap_config["sigma"],
        )
        uncapped_count = len(base_combos)
        base_combos = [base_combos[i] for i in selected_indices]
        print(f"SOAP-pruned base defects: {uncapped_count} -> {len(base_combos)}")
        return base_combos, max_per_class, "defect_local_soap_diversity", True, soap_config

    # Random cap fallback.
    uncapped_count = len(base_combos)
    base_combos = cap_items_randomly(base_combos, max_per_class, python_rng)
    if len(base_combos) < uncapped_count:
        print(f"Randomly capped base defects: {uncapped_count} -> {len(base_combos)}")
    return base_combos, max_per_class, "random_cap", False, soap_config


def _write_structures(
    base_combos: List[Tuple[int, ...]],
    working_structure: Structure,
    combo_to_hash: Dict[Tuple[int, ...], str],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    class_label: str,
    base_name: str,
    output_dir: str,
    scaling: Tuple[int, int, int],
    mode: str,
    write_base_structures: bool,
    write_extxyz: bool,
    seeds_per_base: int,
    perturb_radius: float,
    min_rattle: float,
    max_rattle: float,
    distance_scale: float,
    distance_floor: float,
    sampling_seed: int,
    summary: dict,
) -> List[dict]:
    """Write all POSCAR (and optionally extxyz) files and collect metadata.

    Returns
    -------
    metadata_records
        One dict per structure (base, accepted seed, or rejected seed).

    Notes
    -----
    Per-structure RNG seeds are derived via ``SeedSequence.spawn`` to avoid
    integer overflow. All child sequences are spawned at once before the loop
    (O(n_total)) rather than re-spawning an ever-growing list on every
    iteration. Defect structures are cached so base output and seed generation
    share the same object without rebuilding.
    """
    metadata_records: List[dict] = []
    n_total = len(base_combos)

    # Spawn all per-combo child sequences up front.
    ss = SeedSequence(sampling_seed)
    combo_sequences = ss.spawn(n_total)

    # Cache defect structures to avoid rebuilding the same structure twice.
    defect_structure_cache: Dict[Tuple[int, ...], Structure] = {}

    for combo_index, combo in enumerate(base_combos, start=1):
        print(f"  Writing structure {combo_index}/{n_total} …", end="\r", flush=True)

        if combo not in defect_structure_cache:
            defect_structure_cache[combo] = remove_sites_from_structure(
                working_structure, combo
            )
        defect_structure = defect_structure_cache[combo]

        removed_species = [working_structure[i].specie.symbol for i in combo]
        removed_frac_coords = [
            np.round(working_structure[i].frac_coords, 8).tolist() for i in combo
        ]
        combo_hash = combo_to_hash.get(combo, raw_combo_to_hash.get(combo, ""))
        canonical_combo = combo_to_canonical.get(combo, combo)

        if write_base_structures:
            poscar_name = structure_filename(base_name, class_label, combo_index, None, ".vasp")
            poscar_path = os.path.join(output_dir, poscar_name)
            write_poscar(defect_structure, poscar_path)

            extxyz_name = None
            if write_extxyz:
                extxyz_name = structure_filename(
                    base_name, class_label, combo_index, None, ".extxyz"
                )
                extxyz_path = os.path.join(output_dir, extxyz_name)
                if not maybe_write_extxyz(defect_structure, extxyz_path):
                    extxyz_name = None

            metadata_records.append(
                _base_record(
                    class_label, combo_index, combo, canonical_combo, combo_hash,
                    removed_species, removed_frac_coords, scaling,
                    poscar_name, extxyz_name,
                )
            )
            summary["base_structures_written"] += 1

        if mode in {"seeded", "mlip"} and seeds_per_base > 0:
            amplitudes = np.linspace(min_rattle, max_rattle, seeds_per_base)
            # Spawn one grandchild per seed from this combo's child sequence.
            seed_sequences = combo_sequences[combo_index - 1].spawn(seeds_per_base)

            for seed_offset, (amplitude, seed_seq) in enumerate(
                zip(amplitudes, seed_sequences), start=1
            ):
                np_rng = np.random.default_rng(seed_seq)

                rattled_structure, perturbed_indices = make_rattled_copy_near_vacancies(
                    defect_structure,
                    working_structure,
                    combo,
                    perturb_radius,
                    float(amplitude),
                    np_rng,
                )

                passed, reason = structure_passes_min_distance_filter(
                    rattled_structure,
                    radius_scale=distance_scale,
                    absolute_floor=distance_floor,
                )

                if not passed:
                    summary["seed_structures_rejected_by_distance_filter"] += 1
                    metadata_records.append(
                        _seed_record(
                            class_label, combo_index, seed_offset, combo,
                            canonical_combo, combo_hash, removed_species,
                            removed_frac_coords, scaling, perturb_radius,
                            amplitude, perturbed_indices, False, reason, "", None,
                        )
                    )
                    continue

                poscar_name = structure_filename(
                    base_name, class_label, combo_index, seed_offset, ".vasp"
                )
                poscar_path = os.path.join(output_dir, poscar_name)
                write_poscar(rattled_structure, poscar_path)

                extxyz_name = None
                if write_extxyz:
                    extxyz_name = structure_filename(
                        base_name, class_label, combo_index, seed_offset, ".extxyz"
                    )
                    extxyz_path = os.path.join(output_dir, extxyz_name)
                    if not maybe_write_extxyz(rattled_structure, extxyz_path):
                        extxyz_name = None

                metadata_records.append(
                    _seed_record(
                        class_label, combo_index, seed_offset, combo,
                        canonical_combo, combo_hash, removed_species,
                        removed_frac_coords, scaling, perturb_radius,
                        amplitude, perturbed_indices, True, None,
                        poscar_name, extxyz_name,
                    )
                )
                summary["seed_structures_written"] += 1

    print(" " * 60, end="\r")  # clear progress line
    return metadata_records


def describe_output_summary(records: List[dict]) -> None:
    """Print a short summary of what was written to disk.

    Counts only records whose ``record_type`` is not ``"rejected_seed"`` for
    the "total written" line, because rejected seeds are never written to disk.
    """
    print("\nGeneration summary:")
    if not records:
        print("  No structures were written.")
        return

    by_type: Dict[str, int] = defaultdict(int)
    for record in records:
        by_type[record["record_type"]] += 1

    for key in sorted(by_type):
        print(f"  {key}: {by_type[key]}")

    written = sum(1 for r in records if r["record_type"] != "rejected_seed")
    print(f"  Total written structures: {written}")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the interactive vacancy-generation workflow."""
    print("Vacancy structure generator with exact hashes and defect-local SOAP pruning")
    print("-" * 78)

    input_filename = ask_string("Enter POSCAR filename", "POSCAR")
    parent_structure = Structure.from_file(input_filename)

    print_species_counts(parent_structure)
    working_structure, scaling = maybe_build_supercell(parent_structure)
    print_species_counts(working_structure)

    symprec = ask_float("Symmetry tolerance symprec", 1e-3)
    match_tol = ask_float("Site-matching tolerance in fractional coordinates", 1e-4)

    print_space_group_info(working_structure, symprec)
    print_equivalent_site_groups(working_structure, symprec)

    recipe = ask_removal_recipe(working_structure)
    class_label = defect_class_label(recipe)

    mode = ask_choice(
        "Choose mode",
        choices=["raw", "unique", "seeded", "mlip"],
        default="mlip",
    )

    raw_combos = generate_raw_removal_combinations(working_structure, recipe)
    print(f"\nDefect class: {class_label}")
    print(f"Total raw combinations: {len(raw_combos)}")

    symmetry_maps: Optional[List[Dict[int, int]]] = None
    unique_combos: List[Tuple[int, ...]] = []
    combo_to_hash: Dict[Tuple[int, ...], str] = {}
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    raw_combo_to_hash: Dict[Tuple[int, ...], str] = {}

    if mode in {"unique", "seeded", "mlip"}:
        print("Building symmetry maps … (this may take a moment for large supercells)")
        symmetry_maps = build_parent_symmetry_maps(working_structure, symprec, match_tol)
        unique_combos, combo_to_hash, combo_to_canonical = (
            keep_only_unique_combinations_by_hash(
                raw_combos,
                structure=working_structure,
                recipe=recipe,
                symmetry_maps=symmetry_maps,
                scaling=scaling,
            )
        )
        for raw_combo in raw_combos:
            canonical = canonicalize_combo(raw_combo, symmetry_maps)
            raw_combo_to_hash[raw_combo] = make_combo_hash(
                working_structure, recipe, canonical, scaling
            )
        print(
            f"Exact parent-symmetry-unique combinations (by canonical hash): "
            f"{len(unique_combos)}"
        )

    base_combos, base_source = _build_base_combos(
        raw_combos, unique_combos, symmetry_maps,
        working_structure, recipe, scaling,
        symprec, match_tol,  # user-supplied tolerances forwarded explicitly
        mode,
        combo_to_hash, combo_to_canonical, raw_combo_to_hash,
    )

    sampling_seed = ask_int("Random seed for down-selection and perturbations", 12345, min_value=0)
    python_rng = random.Random(sampling_seed)

    base_combos, max_per_class, base_selection_method, use_soap_pruning, soap_config = (
        _apply_diversity_pruning(
            base_combos, working_structure, mode, sampling_seed, python_rng
        )
    )

    write_base_structures = True
    if mode in {"seeded", "mlip"}:
        write_base_structures = ask_yes_no(
            "Also write the unperturbed base defect structures?", default=True
        )

    seeds_per_base = 0
    perturb_radius = 0.0
    min_rattle = 0.0
    max_rattle = 0.0
    distance_scale = 0.55
    distance_floor = 0.80

    if mode in {"seeded", "mlip"}:
        seeds_per_base = ask_int("How many rattled seeds per base defect", 8, min_value=0)
        perturb_radius = ask_float(
            "Perturb atoms within this radius of each vacancy site (Å)", 4.0
        )
        min_rattle = ask_float(
            "Smallest maximum random displacement used for a seed (Å)", 0.02
        )
        max_rattle = ask_float(
            "Largest maximum random displacement used for a seed (Å)", 0.08
        )
        if max_rattle < min_rattle:
            raise ValueError("Largest displacement must be >= smallest displacement.")
        distance_scale = ask_float("Minimum-distance filter scale factor", 0.55)
        distance_floor = ask_float("Absolute minimum allowed interatomic distance (Å)", 0.80)

    write_extxyz = ask_yes_no(
        "Also write extxyz files when ASE is available?",
        default=False,
    )
    if write_extxyz and not (_ASE_ADAPTOR_AVAILABLE and _ASE_IO_AVAILABLE):
        print("ASE or pymatgen-to-ASE support is not available. extxyz export will be skipped.")

    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_dir = ask_string("Output directory", f"{base_name}_{class_label}_dataset")
    ensure_directory(output_dir)

    summary = {
        "input_file": input_filename,
        "supercell_scaling": scaling,
        "symprec": symprec,
        "site_match_tolerance": match_tol,
        "defect_recipe": recipe,
        "defect_class": class_label,
        "mode": mode,
        "raw_combinations": len(raw_combos),
        "base_structures_written": 0,
        "seed_structures_written": 0,
        "seed_structures_rejected_by_distance_filter": 0,
        "balancing_cap_per_class": max_per_class,
        "base_selection_method": base_selection_method,
        "defect_local_soap_diversity_used": use_soap_pruning,
        "soap_settings": soap_config if use_soap_pruning else None,
    }

    metadata_records = _write_structures(
        base_combos=base_combos,
        working_structure=working_structure,
        combo_to_hash=combo_to_hash,
        raw_combo_to_hash=raw_combo_to_hash,
        combo_to_canonical=combo_to_canonical,
        class_label=class_label,
        base_name=base_name,
        output_dir=output_dir,
        scaling=scaling,
        mode=mode,
        write_base_structures=write_base_structures,
        write_extxyz=write_extxyz,
        seeds_per_base=seeds_per_base,
        perturb_radius=perturb_radius,
        min_rattle=min_rattle,
        max_rattle=max_rattle,
        distance_scale=distance_scale,
        distance_floor=distance_floor,
        sampling_seed=sampling_seed,
        summary=summary,
    )

    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_json_path = os.path.join(output_dir, "metadata.json")
    summary_json_path = os.path.join(output_dir, "summary.json")

    save_metadata_csv(metadata_records, metadata_csv_path)
    save_json(metadata_records, metadata_json_path)
    save_json(summary, summary_json_path)

    describe_output_summary(metadata_records)
    print(f"\nMetadata written to: {metadata_csv_path}")
    print(f"Summary written to:  {summary_json_path}")


if __name__ == "__main__":
    main()
