"""File writing, metadata record construction, and CSV/JSON serialisation."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

from ._compat import ASE_ADAPTOR_AVAILABLE, ASE_IO_AVAILABLE
from .records import (
    BaseRecord,
    ComboContext,
    MetadataRecord,
    MigrationRecord,
    MigrationSeedRecord,
    SeedRecord,
)

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
    "path_index",
    "path_fraction",
    "image_index",
    "migration_source_indices",
    "migration_vacancy_indices",
    "saddle_sign",
    "dataset_type",
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
                       extension: str = ".vasp",
                       include_base_name: bool = True) -> str:
    """Build a readable filename for one output structure."""
    prefix = f"{base_name}_" if include_base_name else ""
    if seed_index is None:
        return f"{prefix}{class_label}_base{combo_index:04d}{extension}"
    return f"{prefix}{class_label}_base{combo_index:04d}_seed{seed_index:03d}{extension}"


def migration_filename(base_name: str,
                       class_label: str,
                       combo_index: int,
                       target_family: str,
                       path_index: int,
                       image_index: int,
                       saddle_sign: int,
                       extension: str = ".vasp",
                       seed_index: Optional[int] = None,
                       include_base_name: bool = True) -> str:
    """Build a filename for one migration-path image (or a rattled seed of one)."""
    sign_text = "p" if saddle_sign >= 0 else "m"
    family_text = target_family.replace(" ", "-")
    prefix = f"{base_name}_" if include_base_name else ""
    base = (
        f"{prefix}{class_label}_base{combo_index:04d}"
        f"_{family_text}_path{path_index:04d}_{sign_text}img{image_index:03d}"
    )
    if seed_index is not None:
        base += f"_seed{seed_index:03d}"
    return base + extension


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

def save_metadata_csv(records: List[object], path: str) -> None:
    """Save metadata to CSV with a canonical, human-friendly column order.

    Accepts either ``MetadataRecord`` instances or plain dicts (e.g. SOAP rows).
    ``None`` values are rendered by :mod:`csv` as empty strings, matching the
    previous ``""`` sentinel behaviour.
    """
    if not records:
        return

    rows = [r.to_dict() if isinstance(r, MetadataRecord) else r for r in records]
    all_keys = {key for row in rows for key in row}
    extra_keys = sorted(all_keys - set(_METADATA_FIELD_ORDER))
    fieldnames = [f for f in _METADATA_FIELD_ORDER if f in all_keys] + extra_keys

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    """Fallback serialiser for numpy types that ``json`` cannot handle natively."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _to_serialisable(data: object) -> object:
    """Recursively convert metadata records to plain dicts for serialisation."""
    if isinstance(data, MetadataRecord):
        return data.to_dict()
    if isinstance(data, (list, tuple)):
        return [_to_serialisable(item) for item in data]
    return data


def save_json(data: object, path: str) -> None:
    """Save Python data (records, lists of records, or plain data) to JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_serialisable(data), handle, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Metadata record builders
# ---------------------------------------------------------------------------

def _round_target_points(target_points: Optional[List[np.ndarray]]) -> List[list]:
    """Round migration target points and convert to plain nested lists."""
    if not target_points:
        return []
    return [np.round(tp, 8).tolist() for tp in target_points]


def build_base_record(ctx: ComboContext,
                      poscar_name: str,
                      extxyz_name: Optional[str]) -> BaseRecord:
    """Build the metadata record for an unperturbed base defect structure."""
    return BaseRecord(
        **ctx.common_fields(),
        record_type="base",
        poscar_file=poscar_name,
        extxyz_file=extxyz_name or "",
        distance_filter_passed=True,
        distance_filter_reason="",
    )


def build_seed_record(ctx: ComboContext,
                      seed_offset: int,
                      perturb_radius: float,
                      amplitude: float,
                      perturbed_indices: List[int],
                      passed: bool,
                      reason: Optional[dict],
                      poscar_name: str,
                      extxyz_name: Optional[str]) -> SeedRecord:
    """Build the metadata record for one rattled seed (accepted or rejected)."""
    return SeedRecord(
        **ctx.common_fields(),
        record_type="seed" if passed else "rejected_seed",
        seed_index=seed_offset,
        poscar_file=poscar_name,
        extxyz_file=extxyz_name or "",
        perturb_radius_A=perturb_radius,
        max_random_displacement_A=float(amplitude),
        perturbed_atom_indices=perturbed_indices,
        distance_filter_passed=passed,
        distance_filter_reason=json.dumps(reason) if reason else "",
    )


def build_migration_record(
    ctx: ComboContext,
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
) -> MigrationRecord:
    """Build metadata for one migration-path image."""
    return MigrationRecord(
        **ctx.common_fields(),
        record_type=record_type,
        poscar_file=poscar_name,
        extxyz_file=extxyz_name or "",
        migration_target_family=target_family,
        migration_target_points_A=_round_target_points(target_points),
        distance_filter_passed=True,
        distance_filter_reason="",
        path_index=path_index,
        path_fraction=float(path_fraction),
        migration_source_indices=list(source_indices),
        migration_vacancy_indices=list(vacancy_indices),
        saddle_sign=saddle_sign,
    )


def build_migration_seed_record(
    ctx: ComboContext,
    seed_offset: int,
    poscar_name: str,
    extxyz_name: Optional[str],
    path_index: int,
    image_index: int,
    path_fraction: float,
    source_indices: Tuple[int, ...],
    vacancy_indices: Tuple[int, ...],
    target_family: str,
    target_points: Optional[List[np.ndarray]],
    saddle_sign: int,
    base_record_type: str,
    perturb_radius: float,
    amplitude: float,
    perturbed_indices: List[int],
    passed: bool,
    reason: Optional[dict],
) -> MigrationSeedRecord:
    """Build metadata for one rattled seed of a migration-path image."""
    return MigrationSeedRecord(
        **ctx.common_fields(),
        record_type=f"{base_record_type}_seed" if passed else f"rejected_{base_record_type}_seed",
        seed_index=seed_offset,
        poscar_file=poscar_name,
        extxyz_file=extxyz_name or "",
        perturb_radius_A=perturb_radius,
        max_random_displacement_A=float(amplitude),
        perturbed_atom_indices=perturbed_indices,
        migration_target_family=target_family,
        migration_target_points_A=_round_target_points(target_points),
        distance_filter_passed=passed,
        distance_filter_reason=json.dumps(reason) if reason else "",
        path_index=path_index,
        path_fraction=float(path_fraction),
        image_index=image_index,
        migration_source_indices=list(source_indices),
        migration_vacancy_indices=list(vacancy_indices),
        saddle_sign=saddle_sign,
    )
