"""Unit tests for the vacancy_generator.dft_extraction package."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from vacancy_generator.dft_extraction.extract import extract_one, read_join_key
from vacancy_generator.dft_extraction.records import DftResultRecord
from vacancy_generator.dft_extraction.runner import (
    extract_all,
    find_structure_dirs,
    run_extraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_structure_folder(output_dir, name: str, *, sidecar: bool = True,
                           poscar_file: str | None = None) -> str:
    """Create a generation-style structure subfolder; return its path."""
    structure_dir = os.path.join(str(output_dir), name)
    os.makedirs(structure_dir, exist_ok=True)
    with open(os.path.join(structure_dir, "POSCAR"), "w", encoding="utf-8") as handle:
        handle.write("dummy poscar\n")
    if sidecar:
        payload = {
            "record_type": "base",
            "poscar_file": poscar_file if poscar_file is not None else f"{name}/POSCAR",
        }
        with open(os.path.join(structure_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    return structure_dir


class _FakeVasprun:
    """Minimal stand-in for pymatgen's ``Vasprun`` with the attributes we read."""

    def __init__(self, *args, **kwargs):
        self.converged_electronic = True
        self.vasp_version = "6.4.2"
        self.parameters = {"EDIFF": 1e-6, "SIGMA": 0.05}
        self.final_structure = [object(), object(), object()]
        self.ionic_steps = [
            {
                "e_fr_energy": np.float64(-42.5),
                "forces": np.zeros((3, 3)),
                "stress": np.eye(3) * np.float64(2.0),
                "electronic_steps": [{"e_fr_energy": -42.0}, {"e_fr_energy": -42.5}],
            }
        ]


def _patch_vasprun(monkeypatch, cls) -> None:
    """Patch the lazily-imported ``Vasprun`` symbol used by ``extract_one``."""
    import pymatgen.io.vasp.outputs as outputs
    monkeypatch.setattr(outputs, "Vasprun", cls)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

class TestDftResultRecord:
    def test_to_dict_sanitises_numpy(self):
        record = DftResultRecord(
            poscar_file="foo/POSCAR",
            status="ok",
            converged=True,
            energy_eV=np.float64(-12.5),
            stress_kB=np.eye(3),
            n_atoms=np.int64(4),
            extracted_at="2026-01-01T00:00:00+00:00",
            vasprun_path="foo/vasprun.xml",
        )
        data = record.to_dict()
        assert isinstance(data["energy_eV"], float)
        assert not isinstance(data["energy_eV"], np.generic)
        assert isinstance(data["n_atoms"], int)
        assert data["stress_kB"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        json.dumps(data)  # must not raise

    def test_to_dict_keeps_none_fields(self):
        record = DftResultRecord(
            poscar_file="foo/POSCAR",
            status="missing",
            converged=False,
            extracted_at="2026-01-01T00:00:00+00:00",
            vasprun_path="foo/vasprun.xml",
            error="no vasprun.xml",
        )
        data = record.to_dict()
        assert data["energy_eV"] is None
        assert data["forces_file"] is None
        assert data["error"] == "no vasprun.xml"


# ---------------------------------------------------------------------------
# extract_one
# ---------------------------------------------------------------------------

class TestExtractOne:
    def test_missing_vasprun_returns_missing_status(self, tmp_path):
        structure_dir = _make_structure_folder(tmp_path, "Mg1_base0001")
        sidecar = os.path.join(structure_dir, "Mg1_base0001.json")

        record = extract_one(structure_dir, sidecar)

        assert record.status == "missing"
        assert record.converged is False
        assert record.poscar_file == "Mg1_base0001/POSCAR"
        assert record.energy_eV is None
        assert record.error

    def test_join_key_is_read_verbatim_from_sidecar(self, tmp_path):
        structure_dir = _make_structure_folder(
            tmp_path, "odd_folder", poscar_file="renamed/elsewhere/POSCAR"
        )
        sidecar = os.path.join(structure_dir, "odd_folder.json")
        record = extract_one(structure_dir, sidecar)
        assert record.poscar_file == "renamed/elsewhere/POSCAR"

    def test_sidecar_without_poscar_file_raises(self, tmp_path):
        structure_dir = os.path.join(str(tmp_path), "broken")
        os.makedirs(structure_dir)
        sidecar = os.path.join(structure_dir, "broken.json")
        with open(sidecar, "w", encoding="utf-8") as handle:
            json.dump({"record_type": "base"}, handle)
        with pytest.raises(ValueError):
            read_join_key(sidecar)

    def test_successful_parse_populates_fields(self, tmp_path, monkeypatch):
        structure_dir = _make_structure_folder(tmp_path, "Mg1_base0001")
        sidecar = os.path.join(structure_dir, "Mg1_base0001.json")
        open(os.path.join(structure_dir, "vasprun.xml"), "w").close()
        _patch_vasprun(monkeypatch, _FakeVasprun)

        record = extract_one(structure_dir, sidecar)

        assert record.status == "ok"
        assert record.converged is True
        assert record.energy_eV == pytest.approx(-42.5)
        assert record.n_atoms == 3
        assert record.n_electronic_steps == 2
        assert record.ediff == pytest.approx(1e-6)
        assert record.sigma_eV == pytest.approx(0.05)
        assert record.vasp_version == "6.4.2"
        assert record.error is None
        # Forces go to a sidecar file, never inline in the row.
        assert record.forces_file == "Mg1_base0001/dft_forces.json"
        assert os.path.isfile(os.path.join(structure_dir, "dft_forces.json"))
        json.dumps(record.to_dict())

    def test_unconverged_scf_is_flagged(self, tmp_path, monkeypatch):
        structure_dir = _make_structure_folder(tmp_path, "Mg1_base0002")
        sidecar = os.path.join(structure_dir, "Mg1_base0002.json")
        open(os.path.join(structure_dir, "vasprun.xml"), "w").close()

        class _Unconverged(_FakeVasprun):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.converged_electronic = False

        _patch_vasprun(monkeypatch, _Unconverged)
        record = extract_one(structure_dir, sidecar)

        assert record.status == "unconverged"
        assert record.converged is False
        assert record.energy_eV == pytest.approx(-42.5)

    def test_parser_exception_returns_unparsable(self, tmp_path, monkeypatch):
        structure_dir = _make_structure_folder(tmp_path, "Mg1_base0003")
        sidecar = os.path.join(structure_dir, "Mg1_base0003.json")
        open(os.path.join(structure_dir, "vasprun.xml"), "w").close()

        class _Broken:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("truncated xml")

        _patch_vasprun(monkeypatch, _Broken)
        record = extract_one(structure_dir, sidecar)

        assert record.status == "unparsable"
        assert record.converged is False
        assert "truncated xml" in record.error


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunner:
    def test_find_structure_dirs_skips_folders_without_sidecar(self, tmp_path):
        _make_structure_folder(tmp_path, "Mg1_base0001")
        _make_structure_folder(tmp_path, "Mg1_base0002")
        os.makedirs(os.path.join(str(tmp_path), "leftover_dir"))
        (tmp_path / "summary.json").write_text("{}")

        pairs = find_structure_dirs(str(tmp_path))

        assert [os.path.basename(d) for d, _ in pairs] == [
            "Mg1_base0001", "Mg1_base0002"
        ]

    def test_find_structure_dirs_is_one_level_deep(self, tmp_path):
        _make_structure_folder(tmp_path, "Mg1_base0001")
        nested = tmp_path / "Mg1_base0001" / "nested"
        nested.mkdir()
        (nested / "nested.json").write_text('{"poscar_file": "x/POSCAR"}')

        pairs = find_structure_dirs(str(tmp_path))
        assert len(pairs) == 1

    def test_missing_output_dir_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_structure_dirs(str(tmp_path / "nope"))

    def test_extract_all_skips_unusable_sidecar(self, tmp_path, capsys):
        _make_structure_folder(tmp_path, "good")
        bad_dir = os.path.join(str(tmp_path), "bad")
        os.makedirs(bad_dir)
        with open(os.path.join(bad_dir, "bad.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")

        records = extract_all(str(tmp_path))

        assert len(records) == 1
        assert records[0].poscar_file == "good/POSCAR"
        assert "warning" in capsys.readouterr().out

    def test_run_extraction_writes_results_table(self, tmp_path):
        _make_structure_folder(tmp_path, "Mg1_base0001")
        _make_structure_folder(tmp_path, "Mg1_base0002")

        records = run_extraction(str(tmp_path))

        assert len(records) == 2
        csv_path = tmp_path / "dft_results.csv"
        json_path = tmp_path / "dft_results.json"
        assert csv_path.is_file()
        assert json_path.is_file()

        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["status"] == "missing"
        # DFT_FIELD_ORDER drives the column order for this table.
        assert list(rows[0])[:3] == ["poscar_file", "status", "converged"]

        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        assert {r["poscar_file"] for r in data} == {
            "Mg1_base0001/POSCAR", "Mg1_base0002/POSCAR"
        }

    def test_run_extraction_on_empty_dir_writes_nothing(self, tmp_path):
        records = run_extraction(str(tmp_path))
        assert records == []
        assert not (tmp_path / "dft_results.csv").exists()
        assert not (tmp_path / "dft_results.json").exists()


# ---------------------------------------------------------------------------
# Regression: widening io.py's serialisers must not change existing behaviour
# ---------------------------------------------------------------------------

class TestSerialiserWideningRegression:
    @staticmethod
    def _combo_context():
        from vacancy_generator.records import ComboContext
        return ComboContext(
            class_label="Mg1", combo_index=1, combo=(0,), canonical_combo=(0,),
            combo_hash="abc123", removed_species=["Mg"],
            removed_frac_coords=[[0.25, 0.25, 0.25]], vacancy_topology="single",
            scaling=(1, 1, 1),
        )

    def test_metadata_records_still_serialise_to_csv(self, tmp_path):
        from vacancy_generator.io import build_base_record, save_metadata_csv
        records = [
            build_base_record(self._combo_context(), poscar_name="foo.vasp",
                              extxyz_name=None)
        ]
        path = str(tmp_path / "metadata.csv")
        save_metadata_csv(records, path)

        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["record_type"] == "base"
        # Canonical generation-metadata column order is unchanged.
        assert list(rows[0])[:3] == ["record_type", "defect_class", "combo_index"]
        # Optional fields render as empty strings, not "None".
        assert rows[0]["seed_index"] == ""

    def test_metadata_records_still_serialise_to_json(self, tmp_path):
        from vacancy_generator.io import build_base_record, save_json
        record = build_base_record(self._combo_context(), poscar_name="foo.vasp",
                                   extxyz_name=None)
        path = str(tmp_path / "record.json")
        save_json(record, path)

        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        assert data["record_type"] == "base"
        assert data["poscar_file"] == "foo.vasp"
        assert data["seed_index"] is None

    def test_plain_dicts_still_supported(self, tmp_path):
        from vacancy_generator.io import save_metadata_csv
        path = str(tmp_path / "rows.csv")
        save_metadata_csv([{"record_type": "base", "soap_distance": 0.5}], path)
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["soap_distance"] == "0.5"
