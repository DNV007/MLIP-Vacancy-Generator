"""Parse one finished VASP calculation into a :class:`DftResultRecord`.

Mirrors the parse/validate split used by :class:`vacancy_generator.config.DftSetup`:
this module turns a single ``(structure_dir, sidecar_json_path)`` pair into one
record, while :mod:`.runner` owns the filesystem walk that discovers those pairs.
"""

from __future__ import annotations

import json
import os
import posixpath
from datetime import datetime, timezone
from typing import Any, Optional

from ..io import save_json
from .records import (
    STATUS_MISSING,
    STATUS_OK,
    STATUS_UNCONVERGED,
    STATUS_UNPARSABLE,
    DftResultRecord,
)

VASPRUN_NAME = "vasprun.xml"
FORCES_NAME = "dft_forces.json"


def _now() -> str:
    """ISO-8601 UTC timestamp for the ``extracted_at`` field."""
    return datetime.now(timezone.utc).isoformat()


def read_join_key(sidecar_json_path: str) -> str:
    """Return the ``poscar_file`` join key stored in a generation-time sidecar JSON.

    The key is read verbatim (never reconstructed from the directory layout) so
    the results table joins exactly against the generation metadata CSV/JSON.

    Raises:
        ValueError: if the sidecar is unreadable, is not a JSON object, or has
            no usable ``poscar_file`` entry.
    """
    try:
        with open(sidecar_json_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read sidecar JSON {sidecar_json_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"sidecar JSON {sidecar_json_path} is not a metadata object")

    poscar_file = data.get("poscar_file")
    if not isinstance(poscar_file, str) or not poscar_file:
        raise ValueError(f"sidecar JSON {sidecar_json_path} has no 'poscar_file' key")
    return poscar_file


def _forces_relpath(structure_dir: str) -> str:
    """Sidecar forces path, relative to ``output_dir``, matching where it's actually written.

    Deliberately derived from ``structure_dir`` (where :func:`extract_one` writes the
    sidecar) rather than ``poscar_file``'s own directory component — those two can
    diverge, since the join key is read verbatim and never assumed to match the
    folder layout (see :func:`read_join_key`).
    """
    folder = os.path.basename(os.path.normpath(structure_dir))
    return posixpath.join(folder, FORCES_NAME)


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float conversion; ``None`` for missing/non-numeric values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_one(structure_dir: str, sidecar_json_path: str) -> DftResultRecord:
    """Parse the VASP result in ``structure_dir`` into a single result record.

    Always returns a record — a missing or broken ``vasprun.xml`` becomes a
    ``status="missing"``/``"unparsable"`` row rather than an exception, so one
    failed job never aborts a whole extraction run.

    Raises:
        ValueError: only if the sidecar JSON itself is unusable, in which case
            the join key is unknown and no meaningful row can be produced.
    """
    poscar_file = read_join_key(sidecar_json_path)
    vasprun_path = os.path.join(structure_dir, VASPRUN_NAME)

    if not os.path.isfile(vasprun_path):
        return DftResultRecord(
            poscar_file=poscar_file,
            status=STATUS_MISSING,
            converged=False,
            extracted_at=_now(),
            vasprun_path=vasprun_path,
            error=f"no {VASPRUN_NAME} in {structure_dir}",
        )

    # Imported lazily: pymatgen's vasp.outputs module is heavy, and the runner
    # should still start up quickly on a tree with no finished jobs.
    from pymatgen.io.vasp.outputs import Vasprun

    try:
        vasprun = Vasprun(
            vasprun_path,
            parse_dos=False,
            parse_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
        converged = bool(vasprun.converged_electronic)
        ionic_steps = vasprun.ionic_steps or []
        last_step = ionic_steps[-1] if ionic_steps else {}
        energy = _as_float(last_step.get("e_fr_energy"))
        forces = last_step.get("forces")
        stress = last_step.get("stress")
        electronic_steps = last_step.get("electronic_steps") or []
        parameters = vasprun.parameters or {}
        n_atoms = len(vasprun.final_structure)
        vasp_version = vasprun.vasp_version

        forces_relpath: Optional[str] = None
        if forces is not None:
            forces_relpath = _forces_relpath(structure_dir)
            save_json(
                {"poscar_file": poscar_file, "forces_eV_per_A": forces},
                os.path.join(structure_dir, FORCES_NAME),
            )
    except Exception as exc:  # noqa: BLE001 - any parser/write failure is a data problem
        return DftResultRecord(
            poscar_file=poscar_file,
            status=STATUS_UNPARSABLE,
            converged=False,
            extracted_at=_now(),
            vasprun_path=vasprun_path,
            error=f"{type(exc).__name__}: {exc}",
        )

    return DftResultRecord(
        poscar_file=poscar_file,
        status=STATUS_OK if converged else STATUS_UNCONVERGED,
        converged=converged,
        energy_eV=energy,
        forces_file=forces_relpath,
        n_atoms=n_atoms,
        sigma_eV=_as_float(parameters.get("SIGMA")),
        n_electronic_steps=len(electronic_steps) or None,
        ediff=_as_float(parameters.get("EDIFF")),
        stress_kB=stress,
        vasp_version=vasp_version,
        extracted_at=_now(),
        vasprun_path=vasprun_path,
        error=None if converged else "electronic convergence not reached",
    )
