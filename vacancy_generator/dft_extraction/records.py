"""Result record dataclass for parsed DFT (VASP) single-point calculations.

Deliberately *not* a subclass of :class:`vacancy_generator.records.MetadataRecord`:
generation-time metadata and post-DFT results are separate tables, joined by the
``poscar_file`` key rather than merged into one row. Only the ``to_dict()``
convention (numpy-free flattening via ``_sanitise``) is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional

from ..records import _sanitise

# Valid values for ``DftResultRecord.status`` (job state, not physical convergence).
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_UNPARSABLE = "unparsable"
STATUS_UNCONVERGED = "unconverged"

VALID_STATUSES = (STATUS_OK, STATUS_MISSING, STATUS_UNPARSABLE, STATUS_UNCONVERGED)

# Preferred CSV column order for the DFT results table. Purely cosmetic: keys not
# listed here are appended alphabetically by ``save_metadata_csv``.
DFT_FIELD_ORDER = [
    "poscar_file",
    "status",
    "converged",
    "energy_eV",
    "n_atoms",
    "forces_file",
    "sigma_eV",
    "n_electronic_steps",
    "ediff",
    "stress_kB",
    "vasp_version",
    "extracted_at",
    "vasprun_path",
    "error",
]


@dataclass(kw_only=True)
class DftResultRecord:
    """One parsed VASP single-point result for one generated structure.

    ``status`` describes the *job* state (did we get a usable calculation at
    all), while ``converged`` is the strictly physical question of whether the
    SCF loop reached ``EDIFF``. A run can be ``status="unconverged"`` with
    ``converged=False`` yet still carry a (meaningless) energy — callers should
    filter on both.

    The forces array is intentionally not stored inline: it is written to a
    per-structure sidecar file and referenced by :attr:`forces_file`, so one CSV
    row stays one line.
    """

    poscar_file: str
    status: str
    converged: bool
    energy_eV: Optional[float] = None
    forces_file: Optional[str] = None
    n_atoms: Optional[int] = None
    sigma_eV: Optional[float] = None
    n_electronic_steps: Optional[int] = None
    ediff: Optional[float] = None
    stress_kB: Optional[List[list]] = None
    vasp_version: Optional[str] = None
    extracted_at: str
    vasprun_path: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Flatten to a numpy-free dict for CSV/JSON serialisation."""
        return {f.name: _sanitise(getattr(self, f.name)) for f in fields(self)}
