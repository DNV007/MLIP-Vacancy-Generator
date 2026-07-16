"""Migration-path runner driven by a .inp configuration file.

Usage
-----
    python -m vacancy_generator.migration_runner generate_migration_images.inp

All parameters are read from the INI-style input file.  No interactive
prompts are issued.  Only the migration-path workflow is supported here;
use ``python -m vacancy_generator`` for the full interactive workflow.
"""

from __future__ import annotations

import configparser
import os
import sys
from typing import Dict, Tuple

from pymatgen.core import Structure

from .io import (
    ensure_directory,
    save_json,
    save_metadata_csv,
)
from .main import _write_migration_paths
from .recipe import (
    defect_class_label,
    generate_raw_removal_combinations,
    parse_removal_recipe,
)
from .reporting import build_dataset_report, describe_output_summary
from .structure_utils import (
    get_species_counts,
    print_equivalent_site_groups,
    print_space_group_info,
    print_species_counts,
    propose_supercell_scaling,
)
from .symmetry import (
    build_parent_symmetry_maps,
    keep_only_unique_combinations_by_hash,
)

_VALID_FAMILY_MODES = {"all", "direct", "tetrahedral", "octahedral"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> configparser.ConfigParser:
    """Parse an INI-style .inp file and return the ConfigParser object."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(path)
    return cfg


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_migration_from_config(config_path: str) -> None:
    """Run the migration-path pipeline using parameters from *config_path*."""
    cfg = load_config(config_path)

    # -- [structure] ---------------------------------------------------------
    poscar_file = cfg.get("structure", "poscar_file", fallback="POSCAR")
    symprec = cfg.getfloat("structure", "symprec", fallback=1e-3)
    match_tol = cfg.getfloat("structure", "match_tol", fallback=1e-4)

    # -- [supercell] ---------------------------------------------------------
    use_supercell = cfg.getboolean("supercell", "enabled", fallback=False)
    min_length = cfg.getfloat("supercell", "min_length", fallback=10.0)
    max_atoms = cfg.getint("supercell", "max_atoms", fallback=400)

    # -- [defect] ------------------------------------------------------------
    try:
        recipe_text = cfg.get("defect", "recipe")
    except (configparser.NoSectionError, configparser.NoOptionError) as exc:
        raise ValueError(
            "Missing required key 'recipe' under [defect] in the input file. "
            "Expected format: 'Mg:1' or 'Mg:1, Se:2'."
        ) from exc

    # -- [migration] ---------------------------------------------------------
    target_family_mode = cfg.get("migration", "target_family", fallback="all")
    hop_cutoff = cfg.getfloat("migration", "hop_cutoff", fallback=4.0)
    max_candidates = cfg.getint("migration", "max_candidates", fallback=6)
    max_assignments = cfg.getint("migration", "max_assignments", fallback=5)
    path_images = cfg.getint("migration", "path_images", fallback=5)
    saddle_shift = cfg.getfloat("migration", "saddle_shift", fallback=0.15)
    both_saddle_sides = cfg.getboolean("migration", "both_saddle_sides", fallback=False)
    seeds_per_image = cfg.getint("migration", "seeds_per_image", fallback=0)
    rattle_all = cfg.getboolean("migration", "rattle_all", fallback=True)
    perturb_radius = cfg.getfloat("migration", "perturb_radius", fallback=4.0)
    min_rattle = cfg.getfloat("migration", "min_rattle", fallback=0.02)
    max_rattle = cfg.getfloat("migration", "max_rattle", fallback=0.08)
    distance_scale = cfg.getfloat("migration", "distance_scale", fallback=0.55)
    distance_floor = cfg.getfloat("migration", "distance_floor", fallback=0.80)
    sampling_seed = cfg.getint("migration", "sampling_seed", fallback=12345)

    # -- [output] ------------------------------------------------------------
    write_base_structures = cfg.getboolean("output", "write_base_structures", fallback=True)
    write_extxyz = cfg.getboolean("output", "write_extxyz", fallback=False)
    output_dir_raw = cfg.get("output", "output_dir", fallback="").strip()

    # -- Validate all parameters before touching the filesystem --------------
    if symprec <= 0:
        raise ValueError("symprec must be > 0.")
    if match_tol <= 0:
        raise ValueError("match_tol must be > 0.")
    if use_supercell and min_length <= 0:
        raise ValueError("supercell min_length must be > 0.")
    if use_supercell and max_atoms < 1:
        raise ValueError("supercell max_atoms must be >= 1.")
    if target_family_mode not in _VALID_FAMILY_MODES:
        raise ValueError(
            f"Unknown target_family '{target_family_mode}'. "
            f"Valid values: {sorted(_VALID_FAMILY_MODES)}"
        )
    if hop_cutoff <= 0:
        raise ValueError("hop_cutoff must be > 0.")
    if path_images < 2:
        raise ValueError("path_images must be at least 2.")
    if seeds_per_image > 0:
        if min_rattle < 0:
            raise ValueError("min_rattle must be >= 0.")
        if max_rattle < min_rattle:
            raise ValueError("max_rattle must be >= min_rattle.")
        if not rattle_all and perturb_radius < 0:
            raise ValueError("perturb_radius must be >= 0.")

    # -- Load structure ------------------------------------------------------
    print(f"\nReading structure from: {poscar_file}")
    try:
        parent_structure = Structure.from_file(poscar_file)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read structure from '{poscar_file}': {exc}"
        ) from exc
    print_species_counts(parent_structure)

    # -- Supercell -----------------------------------------------------------
    scaling: Tuple[int, int, int]
    if use_supercell:
        scaling = propose_supercell_scaling(
            parent_structure,
            min_lattice_length=min_length,
            max_atoms=max_atoms,
        )
        working_structure = parent_structure.copy()
        working_structure.make_supercell(scaling)
        print(
            f"Supercell {scaling}: {len(parent_structure)} -> "
            f"{len(working_structure)} atoms"
        )
    else:
        working_structure = parent_structure
        scaling = (1, 1, 1)

    print_space_group_info(working_structure, symprec)
    print_equivalent_site_groups(working_structure, symprec)

    # -- Defect recipe -------------------------------------------------------
    available = get_species_counts(working_structure)
    recipe = parse_removal_recipe(recipe_text, available)
    class_label = defect_class_label(recipe)
    print(f"\nDefect recipe: {recipe}  |  Class label: {class_label}")

    # -- Symmetry and unique combos ------------------------------------------
    print("Building symmetry maps …")
    symmetry_maps = build_parent_symmetry_maps(working_structure, symprec, match_tol)

    raw_combos = generate_raw_removal_combinations(working_structure, recipe)
    unique_combos, combo_to_hash, combo_to_canonical = (
        keep_only_unique_combinations_by_hash(
            raw_combos,
            structure=working_structure,
            recipe=recipe,
            symmetry_maps=symmetry_maps,
            scaling=scaling,
        )
    )
    print(
        f"Raw combos: {len(raw_combos)}  |  "
        f"Symmetry-unique: {len(unique_combos)}"
    )

    # -- Output directory ----------------------------------------------------
    base_name = os.path.splitext(os.path.basename(poscar_file))[0]
    output_dir = output_dir_raw or f"{base_name}_{class_label}_migration_dataset"
    ensure_directory(output_dir)
    print(f"Output directory: {output_dir}")

    # -- Summary skeleton ----------------------------------------------------
    summary: dict = {
        "config_file": os.path.abspath(config_path),
        "input_file": poscar_file,
        "supercell_scaling": list(scaling),
        "symprec": symprec,
        "site_match_tolerance": match_tol,
        "defect_recipe": recipe,
        "defect_class": class_label,
        "mode": "migration",
        "raw_combinations": len(raw_combos),
        "unique_combinations": len(unique_combos),
        "base_structures_written": 0,
        "migration_hop_cutoff_A": hop_cutoff,
        "migration_max_candidates_per_vacancy": max_candidates,
        "migration_max_assignments_per_combo": max_assignments,
        "migration_path_images": path_images,
        "migration_saddle_shift_A": saddle_shift,
        "migration_write_both_sides": both_saddle_sides,
        "migration_target_family_mode": target_family_mode,
        "migration_seeds_per_image": seeds_per_image,
        "migration_rattle_all": rattle_all,
        "migration_assignments_enumerated": 0,
        "migration_path_images_written": 0,
        "migration_path_families_written": 0,
        "migration_seed_structures_written": 0,
        "migration_seed_structures_rejected": 0,
    }

    # -- Run migration pipeline ----------------------------------------------
    print("\nGenerating migration paths …")
    metadata_records = _write_migration_paths(
        base_combos=unique_combos,
        working_structure=working_structure,
        combo_to_hash=combo_to_hash,
        raw_combo_to_hash={},
        combo_to_canonical=combo_to_canonical,
        class_label=class_label,
        base_name=base_name,
        output_dir=output_dir,
        scaling=scaling,
        write_base_structures=write_base_structures,
        write_extxyz=write_extxyz,
        path_images=path_images,
        saddle_shift=saddle_shift,
        write_both_saddle_sides=both_saddle_sides,
        target_family_mode=target_family_mode,
        hop_cutoff=hop_cutoff,
        max_candidates_per_vacancy=max_candidates,
        max_assignments_per_combo=max_assignments,
        seeds_per_image=seeds_per_image,
        perturb_radius=perturb_radius,
        min_rattle=min_rattle,
        max_rattle=max_rattle,
        distance_scale=distance_scale,
        distance_floor=distance_floor,
        sampling_seed=sampling_seed,
        summary=summary,
        rattle_all=rattle_all,
    )

    # -- Save outputs --------------------------------------------------------
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_json_path = os.path.join(output_dir, "metadata.json")
    summary_json_path = os.path.join(output_dir, "summary.json")
    report_json_path = os.path.join(output_dir, "report.json")

    save_metadata_csv(metadata_records, metadata_csv_path)
    save_json(metadata_records, metadata_json_path)
    save_json(summary, summary_json_path)

    report = build_dataset_report(metadata_records, summary)
    save_json(report, report_json_path)

    describe_output_summary(metadata_records)
    print(f"\nMetadata  : {metadata_csv_path}")
    print(f"Summary   : {summary_json_path}")
    print(f"Report    : {report_json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m vacancy_generator.migration_runner <config.inp>")
        sys.exit(1)
    run_migration_from_config(sys.argv[1])


if __name__ == "__main__":
    main()
