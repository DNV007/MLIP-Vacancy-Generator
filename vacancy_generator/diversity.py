"""Dataset balancing and defect-local SOAP diversity selection."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure

from ._compat import ASE_ADAPTOR_AVAILABLE, DSCRIBE_AVAILABLE


def cap_items_randomly(items: List[Tuple[int, ...]],
                       max_count: Optional[int],
                       rng: random.Random) -> List[Tuple[int, ...]]:
    """Randomly down-select a list if it is larger than ``max_count``."""
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
    """Build a DScribe SOAP descriptor object for defect-local environments."""
    if not DSCRIBE_AVAILABLE:
        raise RuntimeError("DScribe is not available in this Python environment.")
    if not ASE_ADAPTOR_AVAILABLE:
        raise RuntimeError("pymatgen-to-ASE conversion is not available.")

    from dscribe.descriptors import SOAP
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

    SOAP is evaluated at the removed-site positions from the parent structure,
    then averaged over vacancy centers to give an order-independent descriptor.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(defect_structure)
    vacancy_centers = [parent_structure[i].coords for i in removed_indices]

    raw = np.asarray(soap.create(atoms, centers=vacancy_centers), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    vector = np.mean(raw, axis=0)
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm
    return vector


def farthest_point_sample(vectors: List[np.ndarray], target_count: int) -> List[int]:
    """Select a diverse subset of vectors by farthest-point sampling.

    The first vector chosen is the one farthest from the mean. Each subsequent
    vector maximises the minimum distance to the already-selected set.
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
    """Pick a diverse subset of generated structures using defect-local SOAP."""
    if len(defect_structures) != len(removed_combos):
        raise ValueError("defect_structures and removed_combos must have the same length.")

    soap = build_soap_descriptor(species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma)
    vectors = [
        defect_local_soap_vector(defect_structure, parent_structure, combo, soap)
        for defect_structure, combo in zip(defect_structures, removed_combos)
    ]
    return farthest_point_sample(vectors, max_count)
