"""File writing, metadata record construction, and CSV/JSON serialisation."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

from ._compat import ASE_ADAPTOR_AVAILABLE, ASE_IO_AVAILABLE

# Canonical CSV column order for human-readable spreadsheet output.
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
    "vacancy_topology",
    "n_removed_total",
    "supercell_scaling",
    "perturb_radius_A",
    "max_random_displacement_A",
    "perturbed_atom_indices",
    "migration_target_family",
    "migration_target_points_A",
]


# ---------------------------------------------------------------------------
# Directory and filename helpers
# ---------------------------------------------------------------------------

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


def migration_filename(base_name: str,
                       class_label: str,
                       combo_index: int,
                       target_family: str,
                       path_index: int,
                       image_index: int,
                       saddle_sign: int,
                       extension: str = ".vasp") -> str:
    """Build a filename for one migration-path image."""
    sign_text = "p" if saddle_sign >= 0 else "m"
    family_text = target_family.replace(" ", "-")
    return (
        f"{base_name}_{class_label}_base{combo_index:04d}"
        f"_{family_text}_path{path_index:04d}_{sign_text}img{image_index:03d}{extension}"
    )


# ---------------------------------------------------------------------------
# Structure writers
# ---------------------------------------------------------------------------

def write_poscar(structure: Structure, path: str) -> None:
    """Write a structure in VASP POSCAR format."""
    structure.to(fmt="poscar", filename=path)


def maybe_write_extxyz(structure: Structure, path: str) -> bool:
    """Write extxyz if ASE support is available. Returns True on success."""
    if not (ASE_ADAPTOR_AVAILABLE and ASE_IO_AVAILABLE):
        return False
    import ase.io
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(structure)
    ase.io.write(path, atoms, format="extxyz")
    return True


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_metadata_csv(records: List[dict], path: str) -> None:
    """Save metadata to CSV with a canonical, human-friendly column order."""
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


# ---------------------------------------------------------------------------
# Metadata record builders
# ---------------------------------------------------------------------------

def build_base_record(class_label: str,
                      combo_index: int,
                      combo: Tuple[int, ...],
                      canonical_combo: Tuple[int, ...],
                      combo_hash: str,
                      removed_species: List[str],
                      removed_frac_coords: List[list],
                      vacancy_topology: str,
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
        "vacancy_topology": vacancy_topology,
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


def build_seed_record(class_label: str,
                      combo_index: int,
                      seed_offset: int,
                      combo: Tuple[int, ...],
                      canonical_combo: Tuple[int, ...],
                      combo_hash: str,
                      removed_species: List[str],
                      removed_frac_coords: List[list],
                      vacancy_topology: str,
                      scaling: Tuple[int, int, int],
                      perturb_radius: float,
                      amplitude: float,
                      perturbed_indices: List[int],
                      passed: bool,
                      reason: Optional[dict],
                      poscar_name: str,
                      extxyz_name: Optional[str]) -> dict:
    """Build the metadata dict for one rattled seed (accepted or rejected)."""
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
        "vacancy_topology": vacancy_topology,
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


def build_migration_record(
    class_label: str,
    combo_index: int,
    combo: Tuple[int, ...],
    canonical_combo: Tuple[int, ...],
    combo_hash: str,
    removed_species: List[str],
    removed_frac_coords: List[list],
    vacancy_topology: str,
    scaling: Tuple[int, int, int],
    poscar_name: str,
    extxyz_name: Optional[str],
    path_index: int,
    path_fraction: float,
    source_indices: Tuple[int, ...],
    vacancy_indices: Tuple[int, ...],
    target_family: str,
    target_points: Optional[List[np.ndarray]],
    saddle_sign: int,
    record_type: str,
) -> dict:
    """Build metadata for one migration-path image."""
    return {
        "record_type": record_type,
        "defect_class": class_label,
        "combo_index": combo_index,
        "seed_index": "",
        "removed_indices": list(combo),
        "canonical_removed_indices": list(canonical_combo),
        "canonical_hash": combo_hash,
        "removed_species": removed_species,
        "removed_frac_coords": removed_frac_coords,
        "vacancy_topology": vacancy_topology,
        "n_removed_total": len(combo),
        "poscar_file": poscar_name,
        "extxyz_file": extxyz_name or "",
        "supercell_scaling": list(scaling),
        "perturb_radius_A": "",
        "max_random_displacement_A": "",
        "perturbed_atom_indices": [],
        "migration_target_family": target_family,
        "migration_target_points_A": [] if not target_points else [np.round(tp, 8).tolist() for tp in target_points],
        "distance_filter_passed": True,
        "distance_filter_reason": "",
        "path_index": path_index,
        "path_fraction": float(path_fraction),
        "migration_source_indices": list(source_indices),
        "migration_vacancy_indices": list(vacancy_indices),
        "saddle_sign": saddle_sign,
    }
