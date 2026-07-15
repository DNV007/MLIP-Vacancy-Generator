"""Structure creation, rattling, topology classification, and distance filtering."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Element, Structure

from .structure_utils import (
    minimum_image_cartesian_distance,
    minimum_image_cartesian_vector,
)


def remove_sites_from_structure(structure: Structure,
                                combo: Tuple[int, ...]) -> Structure:
    """Return a copy of the structure with the chosen sites removed."""
    new_structure = structure.copy()
    new_structure.remove_sites(sorted(combo, reverse=True))
    return new_structure


def parent_indices_to_defect_indices(total_sites: int,
                                     removed_indices: Tuple[int, ...]) -> Dict[int, int]:
    """Map parent-site indices to defect-structure indices after removals."""
    removed = set(removed_indices)
    mapping: Dict[int, int] = {}
    defect_index = 0
    for parent_index in range(total_sites):
        if parent_index in removed:
            continue
        mapping[parent_index] = defect_index
        defect_index += 1
    return mapping


def classify_vacancy_topology(
    structure: Structure,
    combo: Tuple[int, ...],
    adjacency_factor: float = 1.15,
) -> str:
    """Classify a vacancy combo by a simple connectivity/topology label."""
    if len(combo) == 1:
        return "single"

    lattice_matrix = np.array(structure.lattice.matrix)
    distances = []
    for i, idx_i in enumerate(combo):
        row = []
        for j, idx_j in enumerate(combo):
            if i == j:
                continue
            vec = minimum_image_cartesian_vector(
                lattice_matrix,
                structure[idx_i].frac_coords,
                structure[idx_j].frac_coords,
            )
            row.append(float(np.linalg.norm(vec)))
        distances.append(sorted(row))

    nearest = [row[0] for row in distances if row]
    if not nearest:
        return "isolated"
    threshold = adjacency_factor * float(min(nearest))

    n = len(combo)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            vec = minimum_image_cartesian_vector(
                lattice_matrix,
                structure[combo[i]].frac_coords,
                structure[combo[j]].frac_coords,
            )
            if float(np.linalg.norm(vec)) <= threshold:
                union(i, j)

    comps: Dict[int, int] = defaultdict(int)
    for i in range(n):
        comps[find(i)] += 1
    sizes = sorted(comps.values(), reverse=True)
    if len(sizes) == 1:
        return f"cluster_{sizes[0]}"
    return "cluster_" + "+".join(str(s) for s in sizes)


def element_radius(symbol: str) -> float:
    """Return a simple radius estimate in Å for minimum-distance screening."""
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
    """
    dist = np.array(structure.distance_matrix)

    species = [site.specie.symbol for site in structure]
    radii = np.array([element_radius(s) for s in species])

    min_allowed = np.maximum(absolute_floor, radius_scale * (radii[:, None] + radii[None, :]))

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


def get_nearby_atom_indices(defect_structure: Structure,
                            vacancy_cart_coords: Sequence[np.ndarray],
                            radius: float) -> List[int]:
    """Find atoms lying within ``radius`` of any removed-site position."""
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
    """Return a random 3D unit vector."""
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
    """Create a perturbed copy of a defect structure near the vacancy sites."""
    vacancy_cart_coords = [parent_structure[i].coords for i in removed_indices]
    target_indices = get_nearby_atom_indices(defect_structure, vacancy_cart_coords, radius)

    new_structure = defect_structure.copy()
    for idx in target_indices:
        displacement = random_unit_vector(rng) * rng.uniform(0.0, max_displacement)
        new_structure.translate_sites([idx], displacement, frac_coords=False, to_unit_cell=True)

    return new_structure, target_indices
