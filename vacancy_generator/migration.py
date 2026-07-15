"""Migration path construction: candidate enumeration and image building."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

from .structure_ops import parent_indices_to_defect_indices
from .structure_utils import minimum_image_cartesian_vector


# ---------------------------------------------------------------------------
# Local geometry helpers
# ---------------------------------------------------------------------------

def _safe_unit_vector(vector: np.ndarray,
                      fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Normalize a vector if possible, otherwise return a fallback."""
    vec = np.array(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm >= 1e-12:
        return vec / norm
    if fallback is None:
        return None
    fb = np.array(fallback, dtype=float)
    fb_norm = np.linalg.norm(fb)
    if fb_norm < 1e-12:
        return None
    return fb / fb_norm


def estimate_local_shell_normal(structure: Structure,
                                site_index: int,
                                neighbor_count: int = 6) -> Optional[np.ndarray]:
    """Estimate a local normal direction from the nearest neighbors of a site.

    Returns the least-variance principal component of the nearest-neighbor
    displacement cloud (i.e. the axis perpendicular to the local shell plane).
    """
    site = structure[site_index]
    lattice_matrix = np.array(structure.lattice.matrix)
    vecs: List[Tuple[float, np.ndarray]] = []
    for j, other in enumerate(structure):
        if j == site_index:
            continue
        vec = minimum_image_cartesian_vector(lattice_matrix, site.frac_coords, other.frac_coords)
        dist = float(np.linalg.norm(vec))
        if dist > 1e-8:
            vecs.append((dist, vec))
    if len(vecs) < 3:
        return None
    vecs.sort(key=lambda item: item[0])
    cloud = np.array([v for _, v in vecs[:neighbor_count]], dtype=float)
    cloud -= np.mean(cloud, axis=0)
    cov = cloud.T @ cloud
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    normal = eigvecs[:, int(np.argmin(eigvals))]
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    return normal / norm


def choose_perpendicular_unit(vector: np.ndarray,
                              preferred_normal: Optional[np.ndarray] = None) -> np.ndarray:
    """Choose a deterministic unit vector perpendicular to ``vector``."""
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    unit = vector / norm
    if preferred_normal is not None:
        normal = np.array(preferred_normal, dtype=float)
        n_norm = np.linalg.norm(normal)
        if n_norm >= 1e-12:
            normal = normal / n_norm
            perp = np.cross(unit, normal)
            perp_norm = np.linalg.norm(perp)
            if perp_norm >= 1e-12:
                return perp / perp_norm

    refs = np.eye(3)
    best = None
    best_norm = -1.0
    for ref in refs:
        perp = np.cross(unit, ref)
        perp_norm = np.linalg.norm(perp)
        if perp_norm > best_norm:
            best = perp
            best_norm = perp_norm
    if best is None or best_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return best / best_norm


def estimate_interstitial_target_point(
    structure: Structure,
    vacancy_index: int,
    source_index: int,
    target_family: str,
) -> Optional[np.ndarray]:
    """Estimate a plausible interstitial control point for a hop.

    Returns a Cartesian coordinate heuristic for the pocket center.
    Returns None for the 'direct' family.
    """
    if target_family == "direct":
        return None

    vacancy_site = structure[vacancy_index]
    source_site = structure[source_index]
    lattice_matrix = np.array(structure.lattice.matrix)

    vecs: List[Tuple[float, np.ndarray]] = []
    for j, other in enumerate(structure):
        if j == vacancy_index:
            continue
        vec = minimum_image_cartesian_vector(
            lattice_matrix,
            vacancy_site.frac_coords,
            other.frac_coords,
        )
        dist = float(np.linalg.norm(vec))
        if dist > 1e-8:
            vecs.append((dist, vec))
    if not vecs:
        return None

    vecs.sort(key=lambda item: item[0])
    nearest_distance = vecs[0][0]
    local_vectors = [vec for _, vec in vecs[: max(4, min(6, len(vecs)))]]

    if target_family == "tetrahedral":
        centroid = -np.sum([v / max(np.linalg.norm(v), 1e-12) for v in local_vectors[:4]], axis=0)
        direction = _safe_unit_vector(centroid)
        if direction is None:
            direction = estimate_local_shell_normal(structure, vacancy_index)
        if direction is None:
            direction = _safe_unit_vector(
                minimum_image_cartesian_vector(
                    lattice_matrix,
                    source_site.frac_coords,
                    vacancy_site.frac_coords,
                )
            )
        scale = 0.30 * nearest_distance
    elif target_family == "octahedral":
        direction = estimate_local_shell_normal(structure, vacancy_index, neighbor_count=6)
        if direction is None:
            direction = _safe_unit_vector(np.sum(local_vectors[:6], axis=0))
        if direction is None:
            direction = _safe_unit_vector(
                minimum_image_cartesian_vector(
                    lattice_matrix,
                    source_site.frac_coords,
                    vacancy_site.frac_coords,
                )
            )
        scale = 0.42 * nearest_distance
    else:
        return None

    if direction is None:
        return None
    return vacancy_site.coords + scale * direction


def expand_migration_target_families(mode: str) -> List[str]:
    """Expand the user-facing migration family mode into concrete families."""
    normalized = mode.strip().lower()
    if normalized == "all":
        return ["direct", "tetrahedral", "octahedral"]
    if normalized in {"direct", "tetrahedral", "octahedral"}:
        return [normalized]
    raise ValueError(f"Unknown migration target family mode: {mode}")


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def find_migration_candidates(
    parent_structure: Structure,
    vacancy_index: int,
    forbidden_indices: Tuple[int, ...],
    max_candidates: int,
    cutoff: float,
) -> List[int]:
    """Return same-species source-site candidates that can hop into a vacancy."""
    vacancy_site = parent_structure[vacancy_index]
    vacancy_symbol = vacancy_site.specie.symbol
    forbidden = set(forbidden_indices)
    lattice_matrix = np.array(parent_structure.lattice.matrix)

    candidates: List[Tuple[float, int]] = []
    for source_index, site in enumerate(parent_structure):
        if source_index == vacancy_index or source_index in forbidden:
            continue
        if site.specie.symbol != vacancy_symbol:
            continue
        distance = float(np.linalg.norm(
            minimum_image_cartesian_vector(
                lattice_matrix,
                site.frac_coords,
                vacancy_site.frac_coords,
            )
        ))
        if distance <= cutoff:
            candidates.append((distance, source_index))

    if not candidates:
        for source_index, site in enumerate(parent_structure):
            if source_index == vacancy_index or source_index in forbidden:
                continue
            if site.specie.symbol != vacancy_symbol:
                continue
            distance = float(np.linalg.norm(
                minimum_image_cartesian_vector(
                    lattice_matrix,
                    site.frac_coords,
                    vacancy_site.frac_coords,
                )
            ))
            candidates.append((distance, source_index))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx in candidates[:max_candidates]]


def enumerate_migration_assignments(
    parent_structure: Structure,
    vacancy_indices: Tuple[int, ...],
    max_candidates_per_vacancy: int,
    cutoff: float,
    max_assignments: int,
) -> List[Tuple[int, ...]]:
    """Enumerate injective source->vacancy assignments for a vacancy combo."""
    vacancy_indices = tuple(sorted(vacancy_indices))
    candidate_lists = [
        find_migration_candidates(
            parent_structure,
            vacancy_index=v_idx,
            forbidden_indices=vacancy_indices,
            max_candidates=max_candidates_per_vacancy,
            cutoff=cutoff,
        )
        for v_idx in vacancy_indices
    ]
    if any(len(cands) == 0 for cands in candidate_lists):
        return []

    vacancy_order = sorted(range(len(vacancy_indices)), key=lambda i: len(candidate_lists[i]))
    assignments: List[Tuple[int, ...]] = []

    def backtrack(depth: int, used_sources: set, current: List[int]) -> None:
        if len(assignments) >= max_assignments:
            return
        if depth == len(vacancy_order):
            full = [0] * len(vacancy_indices)
            for order_pos, source_idx in zip(vacancy_order, current):
                full[order_pos] = source_idx
            assignments.append(tuple(full))
            return

        vac_pos = vacancy_order[depth]
        for source_idx in candidate_lists[vac_pos]:
            if source_idx in used_sources:
                continue
            used_sources.add(source_idx)
            current.append(source_idx)
            backtrack(depth + 1, used_sources, current)
            current.pop()
            used_sources.remove(source_idx)

    backtrack(0, set(), [])
    return assignments


# ---------------------------------------------------------------------------
# Image building
# ---------------------------------------------------------------------------

def build_migration_image(
    defect_structure: Structure,
    parent_structure: Structure,
    vacancy_indices: Tuple[int, ...],
    source_indices: Tuple[int, ...],
    image_fraction: float,
    saddle_shift: float,
    saddle_sign: int,
    target_family: str = "direct",
) -> Structure:
    """Build one interpolated migration image for a source-vacancy assignment."""
    new_structure = defect_structure.copy()
    lattice_matrix = np.array(new_structure.lattice.matrix)
    parent_to_defect = parent_indices_to_defect_indices(len(parent_structure), vacancy_indices)

    for vacancy_index, source_index in zip(vacancy_indices, source_indices):
        defect_index = parent_to_defect[source_index]
        start_cart = np.array(new_structure[defect_index].coords, dtype=float)
        end_cart = np.array(parent_structure[vacancy_index].coords, dtype=float)
        delta = minimum_image_cartesian_vector(
            lattice_matrix,
            new_structure[defect_index].frac_coords,
            parent_structure[vacancy_index].frac_coords,
        )
        shell_normal = estimate_local_shell_normal(parent_structure, vacancy_index)
        side = choose_perpendicular_unit(delta, preferred_normal=shell_normal)

        if target_family == "direct":
            path = image_fraction * delta
            if saddle_shift > 0.0:
                path = path + (
                    4.0 * image_fraction * (1.0 - image_fraction) * saddle_sign * saddle_shift * side
                )
            new_cart = start_cart + path
        else:
            target_cart = estimate_interstitial_target_point(
                parent_structure,
                vacancy_index=vacancy_index,
                source_index=source_index,
                target_family=target_family,
            )
            if target_cart is None:
                path = image_fraction * delta
                if saddle_shift > 0.0:
                    path = path + (
                        4.0 * image_fraction * (1.0 - image_fraction) * saddle_sign * saddle_shift * side
                    )
                new_cart = start_cart + path
            else:
                target_cart = target_cart + saddle_sign * saddle_shift * side
                control_cart = 2.0 * target_cart - 0.5 * (start_cart + end_cart)
                t = float(image_fraction)
                new_cart = (
                    (1.0 - t) ** 2 * start_cart
                    + 2.0 * (1.0 - t) * t * control_cart
                    + t**2 * end_cart
                )
        new_structure.replace(
            defect_index,
            new_structure[defect_index].specie,
            new_cart,
            coords_are_cartesian=True,
        )

    return new_structure
