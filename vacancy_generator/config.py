"""Parsed, validated configuration for the .inp-file-driven migration runner.

``MigrationRunConfig.from_inp_file`` is the single place that turns a raw
``.inp`` file into typed, range-checked values; ``migration_runner.py`` then
works with one object instead of ~30 loose locals. Filesystem-touching
validation (``DftSetup.validate``) is kept separate from parsing so the
config can be constructed and unit-tested without a populated directory.
"""

from __future__ import annotations

import configparser
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

_VALID_FAMILY_MODES = {"all", "direct", "tetrahedral", "octahedral"}
_KSPACING_RE = re.compile(r"(?im)^\s*KSPACING\s*=")


def load_inp_file(path: str) -> configparser.ConfigParser:
    """Parse an INI-style .inp file and return the ConfigParser object."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(path)
    return cfg


# ---------------------------------------------------------------------------
# [dft_setup]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DftSetup:
    """Shared DFT reference files, copied to the output dir and symlinked
    into every per-structure subfolder.

    ``potcar``/``incar``/``dft_job`` are always required when the section is
    enabled. ``kpoints`` may be omitted if the referenced INCAR supplies a
    ``KSPACING`` fallback instead — that is only checked by :meth:`validate`,
    not at parse time, since it requires reading the INCAR file.
    """

    potcar: str
    incar: str
    dft_job: str
    kpoints: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: configparser.ConfigParser) -> Optional["DftSetup"]:
        """Return ``None`` if [dft_setup] is absent or ``enabled = false``."""
        if not cfg.has_section("dft_setup"):
            return None
        if not cfg.getboolean("dft_setup", "enabled", fallback=True):
            return None

        def _required(key: str) -> str:
            value = cfg.get("dft_setup", key, fallback="").strip()
            if not value:
                raise ValueError(
                    f"Missing required key '{key}' under [dft_setup]. "
                    "Set 'enabled = false' under [dft_setup] to skip DFT "
                    "reference-file staging entirely."
                )
            return value

        potcar = _required("path_to_potcar")
        incar = _required("path_to_incar")
        dft_job = _required("path_to_dft_job")
        kpoints = cfg.get("dft_setup", "path_to_kpoints", fallback="").strip() or None
        return cls(potcar=potcar, incar=incar, dft_job=dft_job, kpoints=kpoints)

    def validate(self) -> None:
        """Raise if a required file is missing, applying the KSPACING fallback."""
        for label, path in (
            ("path_to_potcar", self.potcar),
            ("path_to_incar", self.incar),
            ("path_to_dft_job", self.dft_job),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[dft_setup] {label} not found: {path}")

        if self.kpoints:
            if not os.path.isfile(self.kpoints):
                raise FileNotFoundError(
                    f"[dft_setup] path_to_kpoints not found: {self.kpoints}"
                )
            return

        with open(self.incar, "r", encoding="utf-8", errors="ignore") as fh:
            incar_text = fh.read()
        if not _KSPACING_RE.search(incar_text):
            raise ValueError(
                "[dft_setup] path_to_kpoints was not given, and INCAR "
                f"'{self.incar}' has no KSPACING entry — provide a KPOINTS "
                "file or add KSPACING to the INCAR."
            )

    def _file_map(self) -> dict:
        file_map = {
            "POTCAR": self.potcar,
            "INCAR": self.incar,
            os.path.basename(self.dft_job): self.dft_job,
        }
        if self.kpoints:
            file_map["KPOINTS"] = self.kpoints
        return file_map

    def stage(self, output_dir: str) -> None:
        """Copy the shared files into *output_dir* and relative-symlink them
        into every existing structure subfolder directly under it."""
        file_map = self._file_map()
        for dest_name, src_path in file_map.items():
            shutil.copy2(src_path, os.path.join(output_dir, dest_name))

        for entry in sorted(os.listdir(output_dir)):
            struct_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(struct_dir):
                continue
            for dest_name in file_map:
                link_path = os.path.join(struct_dir, dest_name)
                if os.path.lexists(link_path):
                    continue
                os.symlink(os.path.join("..", dest_name), link_path)


# ---------------------------------------------------------------------------
# Full run config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MigrationRunConfig:
    """Fully parsed and range-checked contents of a migration-runner .inp file."""

    poscar_file: str
    symprec: float
    match_tol: float

    use_supercell: bool
    min_length: float
    max_atoms: int

    recipe_text: str

    target_family_mode: str
    hop_cutoff: float
    max_candidates: int
    max_assignments: int
    path_images: int
    saddle_shift: float
    both_saddle_sides: bool
    seeds_per_image: int
    rattle_all: bool
    perturb_radius: float
    min_rattle: float
    max_rattle: float
    distance_scale: float
    distance_floor: float
    sampling_seed: int

    write_base_structures: bool
    write_extxyz: bool
    output_dir_raw: str

    dft_setup: Optional[DftSetup] = None

    @classmethod
    def from_inp_file(cls, path: str) -> "MigrationRunConfig":
        cfg = load_inp_file(path)

        try:
            recipe_text = cfg.get("defect", "recipe")
        except (configparser.NoSectionError, configparser.NoOptionError) as exc:
            raise ValueError(
                "Missing required key 'recipe' under [defect] in the input file. "
                "Expected format: 'Mg:1' or 'Mg:1, Se:2'."
            ) from exc

        config = cls(
            poscar_file=cfg.get("structure", "poscar_file", fallback="POSCAR"),
            symprec=cfg.getfloat("structure", "symprec", fallback=1e-3),
            match_tol=cfg.getfloat("structure", "match_tol", fallback=1e-4),
            use_supercell=cfg.getboolean("supercell", "enabled", fallback=False),
            min_length=cfg.getfloat("supercell", "min_length", fallback=10.0),
            max_atoms=cfg.getint("supercell", "max_atoms", fallback=400),
            recipe_text=recipe_text,
            target_family_mode=cfg.get("migration", "target_family", fallback="all"),
            hop_cutoff=cfg.getfloat("migration", "hop_cutoff", fallback=4.0),
            max_candidates=cfg.getint("migration", "max_candidates", fallback=6),
            max_assignments=cfg.getint("migration", "max_assignments", fallback=5),
            path_images=cfg.getint("migration", "path_images", fallback=5),
            saddle_shift=cfg.getfloat("migration", "saddle_shift", fallback=0.15),
            both_saddle_sides=cfg.getboolean("migration", "both_saddle_sides", fallback=False),
            seeds_per_image=cfg.getint("migration", "seeds_per_image", fallback=0),
            rattle_all=cfg.getboolean("migration", "rattle_all", fallback=True),
            perturb_radius=cfg.getfloat("migration", "perturb_radius", fallback=4.0),
            min_rattle=cfg.getfloat("migration", "min_rattle", fallback=0.02),
            max_rattle=cfg.getfloat("migration", "max_rattle", fallback=0.08),
            distance_scale=cfg.getfloat("migration", "distance_scale", fallback=0.55),
            distance_floor=cfg.getfloat("migration", "distance_floor", fallback=0.80),
            sampling_seed=cfg.getint("migration", "sampling_seed", fallback=12345),
            write_base_structures=cfg.getboolean("output", "write_base_structures", fallback=True),
            write_extxyz=cfg.getboolean("output", "write_extxyz", fallback=False),
            output_dir_raw=cfg.get("output", "output_dir", fallback="").strip(),
            dft_setup=DftSetup.from_config(cfg),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if self.symprec <= 0:
            raise ValueError("symprec must be > 0.")
        if self.match_tol <= 0:
            raise ValueError("match_tol must be > 0.")
        if self.use_supercell and self.min_length <= 0:
            raise ValueError("supercell min_length must be > 0.")
        if self.use_supercell and self.max_atoms < 1:
            raise ValueError("supercell max_atoms must be >= 1.")
        if self.target_family_mode not in _VALID_FAMILY_MODES:
            raise ValueError(
                f"Unknown target_family '{self.target_family_mode}'. "
                f"Valid values: {sorted(_VALID_FAMILY_MODES)}"
            )
        if self.hop_cutoff <= 0:
            raise ValueError("hop_cutoff must be > 0.")
        if self.path_images < 2:
            raise ValueError("path_images must be at least 2.")
        if self.seeds_per_image > 0:
            if self.min_rattle < 0:
                raise ValueError("min_rattle must be >= 0.")
            if self.max_rattle < self.min_rattle:
                raise ValueError("max_rattle must be >= min_rattle.")
            if not self.rattle_all and self.perturb_radius < 0:
                raise ValueError("perturb_radius must be >= 0.")
