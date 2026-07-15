"""Basic structure inspection, supercell building, and low-level geometry helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .input_helpers import ask_float, ask_int, ask_yes_no


# ---------------------------------------------------------------------------
# Species bookkeeping
# ---------------------------------------------------------------------------

def get_species_counts(structure: Structure) -> Dict[str, int]:
    """Return how many atoms of each species are present in the structure."""
    counts: Dict[str, int] = defaultdict(int)
    for site in structure:
        counts[site.specie.symbol] += 1
    return dict(counts)


def get_indices_by_species(structure: Structure) -> Dict[str, List[int]]:
    """Return the site indices belonging to each chemical species."""
    result: Dict[str, List[int]] = defaultdict(list)
    for index, site in enumerate(structure):
        result[site.specie.symbol].append(index)
    return dict(result)


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
    """Print symmetry-equivalent site groups in the parent structure."""
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    symmetrized = analyzer.get_symmetrized_structure()

    print("\nEquivalent site groups in the parent structure:")
    for group in symmetrized.equivalent_indices:
        symbol = structure[group[0]].specie.symbol
        coords_text = ", ".join(
            [f"{idx}:{np.round(structure[idx].frac_coords, 4).tolist()}" for idx in group]
        )
        print(f"  {symbol}: [{coords_text}]")


# ---------------------------------------------------------------------------
# Supercell
# ---------------------------------------------------------------------------

def propose_supercell_scaling(structure: Structure,
                              min_lattice_length: float = 10.0,
                              min_atoms: Optional[int] = None,
                              max_atoms: int = 400) -> Tuple[int, int, int]:
    """Propose a diagonal supercell scaling (a, b, c)."""
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


# ---------------------------------------------------------------------------
# Low-level geometry
# ---------------------------------------------------------------------------

def wrap_fractional_coords(frac_coords: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into the [0, 1) interval."""
    return np.mod(frac_coords, 1.0)


def periodic_fractional_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the shortest distance between two fractional points."""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    diff -= np.round(diff)
    return float(np.linalg.norm(diff))


def minimum_image_cartesian_vector(lattice_matrix: np.ndarray,
                                   frac_a: Sequence[float],
                                   frac_b: Sequence[float]) -> np.ndarray:
    """Return the shortest Cartesian vector from ``a`` to ``b`` under PBC."""
    diff = np.array(frac_b, dtype=float) - np.array(frac_a, dtype=float)
    diff -= np.round(diff)
    return diff @ lattice_matrix


def minimum_image_cartesian_distance(lattice_matrix: np.ndarray,
                                     frac_a: Sequence[float],
                                     frac_b: Sequence[float]) -> float:
    """Return the shortest Cartesian distance between two fractional points."""
    diff = np.array(frac_a, dtype=float) - np.array(frac_b, dtype=float)
    diff -= np.round(diff)
    cart = diff @ lattice_matrix
    return float(np.linalg.norm(cart))
