"""Migration-path runner driven by a .inp configuration file.

Usage
-----
    python -m vacancy_generator.migration_runner generate_migration_images.inp

All parameters are read from the INI-style input file.  No interactive
prompts are issued.  Only the migration-path workflow is supported here;
use ``python -m vacancy_generator`` for the full interactive workflow.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

from pymatgen.core import Structure

from .config import MigrationRunConfig
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_migration_from_config(config_path: str) -> None:
    """Run the migration-path pipeline using parameters from *config_path*."""
    run_config = MigrationRunConfig.from_inp_file(config_path)
    if run_config.dft_setup is not None:
        run_config.dft_setup.validate()

    poscar_file = run_config.poscar_file
    symprec = run_config.symprec
    match_tol = run_config.match_tol
    use_supercell = run_config.use_supercell
    min_length = run_config.min_length
    max_atoms = run_config.max_atoms
    recipe_text = run_config.recipe_text
    target_family_mode = run_config.target_family_mode
    hop_cutoff = run_config.hop_cutoff
    max_candidates = run_config.max_candidates
    max_assignments = run_config.max_assignments
    path_images = run_config.path_images
    saddle_shift = run_config.saddle_shift
    both_saddle_sides = run_config.both_saddle_sides
    seeds_per_image = run_config.seeds_per_image
    rattle_all = run_config.rattle_all
    perturb_radius = run_config.perturb_radius
    min_rattle = run_config.min_rattle
    max_rattle = run_config.max_rattle
    distance_scale = run_config.distance_scale
    distance_floor = run_config.distance_floor
    sampling_seed = run_config.sampling_seed
    write_base_structures = run_config.write_base_structures
    write_extxyz = run_config.write_extxyz
    output_dir_raw = run_config.output_dir_raw

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

    # -- Stage shared DFT reference files -------------------------------------
    if run_config.dft_setup is not None:
        print("\nStaging DFT reference files …")
        run_config.dft_setup.stage(output_dir)
        summary["dft_setup_staged"] = True
    else:
        summary["dft_setup_staged"] = False

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
