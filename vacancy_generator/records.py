"""Metadata record dataclasses for generated vacancy/migration structures."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sanitise(value: Any) -> Any:
    """Recursively convert numpy types to plain JSON/CSV-friendly Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_sanitise(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitise(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ComboContext:
    """Per-combination values shared by every record built in one loop iteration.

    ``main.py`` computes these once per vacancy combination and threads a single
    ``ComboContext`` into each record builder, instead of re-passing the same
    nine arguments to every builder call. :meth:`common_fields` returns the
    ~10 metadata fields that were previously hand-repeated in every builder body.

    Per-call fields (``poscar_file``/``extxyz_file`` and the distinguishing
    record fields) are NOT part of the context; builders supply those explicitly.
    """

    class_label: str
    combo_index: int
    combo: Tuple[int, ...]
    canonical_combo: Tuple[int, ...]
    combo_hash: str
    removed_species: List[str]
    removed_frac_coords: List[list]
    vacancy_topology: str
    scaling: Tuple[int, int, int]

    def common_fields(self) -> Dict[str, Any]:
        """Return the metadata fields common to every record for this combo."""
        return {
            "defect_class": self.class_label,
            "combo_index": self.combo_index,
            "removed_indices": list(self.combo),
            "canonical_removed_indices": list(self.canonical_combo),
            "canonical_hash": self.combo_hash,
            "removed_species": self.removed_species,
            "removed_frac_coords": self.removed_frac_coords,
            "vacancy_topology": self.vacancy_topology,
            "n_removed_total": len(self.combo),
            "supercell_scaling": list(self.scaling),
        }


@dataclass(kw_only=True)
class MetadataRecord:
    """Base metadata common to every generated structure.

    Keyword-only fields so subclasses can add their own required fields without
    tripping the "non-default follows default" ordering rule. Fields that are
    genuinely not applicable to a record type carry ``None`` defaults rather
    than fake ``""``/``0`` sentinels.
    """

    record_type: str
    defect_class: str
    combo_index: int
    canonical_hash: str
    poscar_file: str
    extxyz_file: str
    removed_indices: List[int]
    canonical_removed_indices: List[int]
    removed_species: List[str]
    removed_frac_coords: List[list]
    vacancy_topology: str
    n_removed_total: int
    supercell_scaling: List[int]
    distance_filter_passed: bool
    distance_filter_reason: str
    seed_index: Optional[int] = None
    dataset_type: str = "unassigned"

    def to_dict(self) -> dict:
        """Flatten to a numpy-free dict for CSV/JSON serialisation."""
        return {f.name: _sanitise(getattr(self, f.name)) for f in fields(self)}


@dataclass(kw_only=True)
class BaseRecord(MetadataRecord):
    """Unperturbed base defect structure. Perturbation fields are N/A here."""

    perturb_radius_A: Optional[float] = None
    max_random_displacement_A: Optional[float] = None
    perturbed_atom_indices: Optional[List[int]] = None


@dataclass(kw_only=True)
class SeedRecord(BaseRecord):
    """A rattled seed of a base defect (accepted or rejected)."""


@dataclass(kw_only=True)
class MigrationRecord(BaseRecord):
    """One migration-path image. Non-seeded, so perturbation fields are N/A."""

    migration_target_family: Optional[str] = None
    migration_target_points_A: List[list] = field(default_factory=list)
    path_index: Optional[int] = None
    path_fraction: Optional[float] = None
    image_index: Optional[int] = None
    migration_source_indices: Optional[List[int]] = None
    migration_vacancy_indices: Optional[List[int]] = None
    saddle_sign: Optional[int] = None


@dataclass(kw_only=True)
class MigrationSeedRecord(MigrationRecord):
    """A rattled seed of a migration-path image."""
