"""End-to-end simulation that exercises the full pipeline for 3 cases.

Cases
-----
1. Single Mg vacancy — unique mode, no rattling, writes base POSCAR.
2. Single Se vacancy — seeded mode, writes rattled seeds.
3. Single Mg vacancy — migration mode with rattled seeds per image:
   direct + interstitial path families, each image rattled n_seeds times.

Run with:
    /path/to/python tests/simulation.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Setup: make vacancy_generator importable regardless of install state
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pymatgen.core import Structure
from vacancy_generator.io import (
    build_base_record,
    build_seed_record,
    build_migration_record,
    build_migration_seed_record,
    ensure_directory,
    migration_filename,
    save_json,
    save_metadata_csv,
    structure_filename,
    write_poscar,
)
from vacancy_generator.migration import (
    build_migration_image,
    enumerate_migration_assignments,
    expand_migration_target_families,
)
from vacancy_generator.recipe import (
    defect_class_label,
    generate_raw_removal_combinations,
    parse_removal_recipe,
)
from vacancy_generator.structure_ops import (
    classify_vacancy_topology,
    make_rattled_copy_near_vacancies,
    remove_sites_from_structure,
    structure_passes_min_distance_filter,
)
from vacancy_generator.records import ComboContext
from vacancy_generator.structure_utils import get_species_counts
from vacancy_generator.symmetry import (
    build_parent_symmetry_maps,
    canonicalize_combo,
    keep_only_unique_combinations_by_hash,
    make_combo_hash,
)

POSCAR_PATH = REPO_ROOT / "vacancy_generator" / "POSCAR"
SYMPREC = 1e-3
MATCH_TOL = 1e-4
SCALING = (1, 1, 1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_structure() -> Structure:
    structure = Structure.from_file(str(POSCAR_PATH))
    print(f"  Loaded {POSCAR_PATH.name}: {len(structure)} sites, "
          f"species={dict(get_species_counts(structure))}")
    return structure


def _build_symmetry(structure: Structure):
    print("  Building symmetry maps …")
    maps = build_parent_symmetry_maps(structure, symprec=SYMPREC, match_tol=MATCH_TOL)
    print(f"  {len(maps)} symmetry operations found.")
    return maps


def _unique_combos(structure, recipe, maps):
    raw = generate_raw_removal_combinations(structure, recipe)
    unique, combo_to_hash, combo_to_canonical = keep_only_unique_combinations_by_hash(
        raw, structure, recipe, maps, SCALING
    )
    print(f"  Raw combos: {len(raw)} | Symmetry-unique: {len(unique)}")
    return raw, unique, combo_to_hash, combo_to_canonical


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Case 1 — Single Mg vacancy, unique mode, writes base POSCARs
# ---------------------------------------------------------------------------

def case_1_single_mg_vacancy_unique(output_dir: str) -> dict:
    _section("Case 1: Single Mg vacancy — unique mode")

    structure = _load_structure()
    recipe = parse_removal_recipe("Mg:1", get_species_counts(structure))
    class_label = defect_class_label(recipe)
    print(f"  Recipe: {recipe}  |  Class label: {class_label}")

    maps = _build_symmetry(structure)
    raw, unique, combo_to_hash, combo_to_canonical = _unique_combos(structure, recipe, maps)

    records = []
    for combo_index, combo in enumerate(unique, start=1):
        defect = remove_sites_from_structure(structure, combo)
        removed_species = [structure[i].specie.symbol for i in combo]
        removed_frac = [np.round(structure[i].frac_coords, 8).tolist() for i in combo]
        topology = classify_vacancy_topology(structure, combo)
        combo_hash = combo_to_hash[combo]
        canonical = combo_to_canonical[combo]

        fname = structure_filename("POSCAR", class_label, combo_index, None, ".vasp")
        fpath = os.path.join(output_dir, fname)
        write_poscar(defect, fpath)
        assert os.path.isfile(fpath), f"Expected file not created: {fpath}"

        ctx = ComboContext(
            class_label=class_label, combo_index=combo_index, combo=combo,
            canonical_combo=canonical, combo_hash=combo_hash,
            removed_species=removed_species, removed_frac_coords=removed_frac,
            vacancy_topology=topology, scaling=SCALING,
        )
        records.append(build_base_record(ctx, fname, None))
        print(f"  [{combo_index}] combo={combo}, topology={topology}, hash={combo_hash}")

    save_metadata_csv(records, os.path.join(output_dir, "metadata.csv"))
    save_json(records, os.path.join(output_dir, "metadata.json"))

    summary = {
        "case": 1,
        "description": "Single Mg vacancy, unique mode",
        "raw_combos": len(raw),
        "unique_combos": len(unique),
        "files_written": len(records),
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))
    print(f"\n  -> {len(records)} base POSCAR(s) written to {output_dir}/")

    # Assertions
    assert len(unique) >= 1, "Expected at least one unique Mg vacancy"
    assert all(r.record_type == "base" for r in records)
    print("  Assertions passed.")
    return summary


# ---------------------------------------------------------------------------
# Case 2 — Single Se vacancy, seeded mode, writes rattled structures
# ---------------------------------------------------------------------------

def case_2_single_se_vacancy_seeded(output_dir: str) -> dict:
    _section("Case 2: Single Se vacancy — seeded mode (rattled seeds)")

    structure = _load_structure()
    recipe = parse_removal_recipe("Se:1", get_species_counts(structure))
    class_label = defect_class_label(recipe)
    print(f"  Recipe: {recipe}  |  Class label: {class_label}")

    maps = _build_symmetry(structure)
    raw, unique, combo_to_hash, combo_to_canonical = _unique_combos(structure, recipe, maps)

    # Use only the first unique combo to keep the simulation quick
    combos_to_run = unique[:1]

    SEEDS_PER_BASE = 4
    PERTURB_RADIUS = 4.0
    MIN_RATTLE = 0.02
    MAX_RATTLE = 0.08
    DISTANCE_SCALE = 0.55
    DISTANCE_FLOOR = 0.80

    from numpy.random import SeedSequence
    ss = SeedSequence(42)
    combo_sequences = ss.spawn(len(combos_to_run))

    records = []
    written = 0
    rejected = 0

    for combo_index, combo in enumerate(combos_to_run, start=1):
        defect = remove_sites_from_structure(structure, combo)
        removed_species = [structure[i].specie.symbol for i in combo]
        removed_frac = [np.round(structure[i].frac_coords, 8).tolist() for i in combo]
        topology = classify_vacancy_topology(structure, combo)
        combo_hash = combo_to_hash[combo]
        canonical = combo_to_canonical[combo]

        # Write base structure
        base_fname = structure_filename("POSCAR", class_label, combo_index, None, ".vasp")
        write_poscar(defect, os.path.join(output_dir, base_fname))
        ctx = ComboContext(
            class_label=class_label, combo_index=combo_index, combo=combo,
            canonical_combo=canonical, combo_hash=combo_hash,
            removed_species=removed_species, removed_frac_coords=removed_frac,
            vacancy_topology=topology, scaling=SCALING,
        )
        records.append(build_base_record(ctx, base_fname, None))

        amplitudes = np.linspace(MIN_RATTLE, MAX_RATTLE, SEEDS_PER_BASE)
        seed_sequences = combo_sequences[combo_index - 1].spawn(SEEDS_PER_BASE)

        for seed_offset, (amplitude, seed_seq) in enumerate(zip(amplitudes, seed_sequences), 1):
            rng = np.random.default_rng(seed_seq)
            rattled, perturbed_indices = make_rattled_copy_near_vacancies(
                defect, structure, combo, PERTURB_RADIUS, float(amplitude), rng
            )
            passed, reason = structure_passes_min_distance_filter(
                rattled, radius_scale=DISTANCE_SCALE, absolute_floor=DISTANCE_FLOOR
            )
            if not passed:
                rejected += 1
                records.append(build_seed_record(
                    ctx, seed_offset, PERTURB_RADIUS, amplitude, perturbed_indices,
                    False, reason, "", None,
                ))
                print(f"  Seed {seed_offset} REJECTED: {reason}")
                continue

            seed_fname = structure_filename("POSCAR", class_label, combo_index, seed_offset, ".vasp")
            write_poscar(rattled, os.path.join(output_dir, seed_fname))
            written += 1
            records.append(build_seed_record(
                ctx, seed_offset, PERTURB_RADIUS, amplitude, perturbed_indices,
                True, None, seed_fname, None,
            ))
            print(f"  Seed {seed_offset}: amplitude={amplitude:.4f} Å, "
                  f"{len(perturbed_indices)} atoms perturbed -> ACCEPTED")

    save_metadata_csv(records, os.path.join(output_dir, "metadata.csv"))
    save_json(records, os.path.join(output_dir, "metadata.json"))

    summary = {
        "case": 2,
        "description": "Single Se vacancy, seeded mode",
        "unique_combos": len(unique),
        "combos_processed": len(combos_to_run),
        "seeds_written": written,
        "seeds_rejected": rejected,
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))
    print(f"\n  -> {written} seed(s) written, {rejected} rejected.")

    # Assertions
    assert written > 0, "Expected at least one seed structure to be written"
    seed_records = [r for r in records if r.record_type == "seed"]
    assert all(r.distance_filter_passed for r in seed_records)
    print("  Assertions passed.")
    return summary


# ---------------------------------------------------------------------------
# Case 3 — Single Mg vacancy, migration mode with rattled seeds per image
# ---------------------------------------------------------------------------

def case_3_single_mg_vacancy_migration(output_dir: str) -> dict:
    _section("Case 3: Single Mg vacancy — migration mode with rattled seeds per image")

    structure = _load_structure()
    recipe = parse_removal_recipe("Mg:1", get_species_counts(structure))
    class_label = defect_class_label(recipe)
    print(f"  Recipe: {recipe}  |  Class label: {class_label}")

    maps = _build_symmetry(structure)
    raw, unique, combo_to_hash, combo_to_canonical = _unique_combos(structure, recipe, maps)

    # Use first unique combo and direct family only to keep the simulation quick
    combos_to_run = unique[:1]

    PATH_IMAGES = 5        # n: displacement steps along the path
    SEEDS_PER_IMAGE = 3    # n_2: rattled copies of each displacement image
    SADDLE_SHIFT = 0.15
    HOP_CUTOFF = 6.0
    MAX_CANDIDATES = 4
    MAX_ASSIGNMENTS = 2
    TARGET_FAMILIES = expand_migration_target_families("direct")
    PERTURB_RADIUS = 4.0
    MIN_RATTLE = 0.02
    MAX_RATTLE = 0.08
    DISTANCE_SCALE = 0.55
    DISTANCE_FLOOR = 0.80
    SAMPLING_SEED = 42

    from numpy.random import SeedSequence

    records = []
    images_written = 0
    seeds_written = 0
    seeds_rejected = 0

    for combo_index, combo in enumerate(combos_to_run, start=1):
        defect = remove_sites_from_structure(structure, combo)
        removed_species = [structure[i].specie.symbol for i in combo]
        removed_frac = [np.round(structure[i].frac_coords, 8).tolist() for i in combo]
        topology = classify_vacancy_topology(structure, combo)
        combo_hash = combo_to_hash[combo]
        canonical = combo_to_canonical[combo]

        # Write base (vacancy) structure
        base_fname = structure_filename("POSCAR", class_label, combo_index, None, ".vasp")
        write_poscar(defect, os.path.join(output_dir, base_fname))
        ctx = ComboContext(
            class_label=class_label, combo_index=combo_index, combo=combo,
            canonical_combo=canonical, combo_hash=combo_hash,
            removed_species=removed_species, removed_frac_coords=removed_frac,
            vacancy_topology=topology, scaling=SCALING,
        )
        records.append(build_base_record(ctx, base_fname, None))

        assignments = enumerate_migration_assignments(
            structure, combo,
            max_candidates_per_vacancy=MAX_CANDIDATES,
            cutoff=HOP_CUTOFF,
            max_assignments=MAX_ASSIGNMENTS,
        )
        print(f"  combo={combo}: {len(assignments)} migration assignment(s) found.")

        if not assignments:
            print("  WARNING: No migration candidates found, skipping path images.")
            continue

        path_fractions = np.linspace(0.0, 1.0, PATH_IMAGES)
        amplitudes = np.linspace(MIN_RATTLE, MAX_RATTLE, SEEDS_PER_IMAGE)

        for target_family in TARGET_FAMILIES:
            for path_idx, assignment in enumerate(assignments, start=1):
                for image_idx, fraction in enumerate(path_fractions):

                    # Build the clean interpolated image
                    img_struct = build_migration_image(
                        defect_structure=defect,
                        parent_structure=structure,
                        vacancy_indices=combo,
                        source_indices=assignment,
                        image_fraction=float(fraction),
                        saddle_shift=SADDLE_SHIFT,
                        saddle_sign=1,
                        target_family=target_family,
                    )

                    if image_idx == 0:
                        rtype = "path_start"
                    elif image_idx == len(path_fractions) - 1:
                        rtype = "path_end"
                    elif abs(float(fraction) - 0.5) < 1e-9:
                        rtype = "path_saddle"
                    else:
                        rtype = "path_image"

                    # Write the clean image
                    img_fname = migration_filename(
                        "POSCAR", class_label, combo_index,
                        target_family, path_idx, image_idx, 1, ".vasp",
                    )
                    write_poscar(img_struct, os.path.join(output_dir, img_fname))
                    assert os.path.isfile(os.path.join(output_dir, img_fname))
                    images_written += 1
                    records.append(build_migration_record(
                        ctx,
                        poscar_name=img_fname, extxyz_name=None,
                        path_index=path_idx, path_fraction=float(fraction),
                        source_indices=assignment, vacancy_indices=combo,
                        target_family=target_family, target_points=None,
                        saddle_sign=1, record_type=rtype,
                    ))

                    # Rattle the image SEEDS_PER_IMAGE times
                    for seed_offset, amplitude in enumerate(amplitudes, start=1):
                        rng = np.random.default_rng(
                            SeedSequence([SAMPLING_SEED, combo_index, path_idx, image_idx, seed_offset])
                        )
                        rattled, perturbed_indices = make_rattled_copy_near_vacancies(
                            img_struct, structure, combo,
                            PERTURB_RADIUS, float(amplitude), rng,
                        )
                        passed, reason = structure_passes_min_distance_filter(
                            rattled, radius_scale=DISTANCE_SCALE, absolute_floor=DISTANCE_FLOOR,
                        )

                        if not passed:
                            seeds_rejected += 1
                            records.append(build_migration_seed_record(
                                ctx, seed_offset=seed_offset,
                                poscar_name="", extxyz_name=None,
                                path_index=path_idx, image_index=image_idx,
                                path_fraction=float(fraction), source_indices=assignment,
                                vacancy_indices=combo, target_family=target_family,
                                target_points=None, saddle_sign=1,
                                base_record_type=rtype, perturb_radius=PERTURB_RADIUS,
                                amplitude=float(amplitude), perturbed_indices=perturbed_indices,
                                passed=False, reason=reason,
                            ))
                            continue

                        seed_fname = migration_filename(
                            "POSCAR", class_label, combo_index,
                            target_family, path_idx, image_idx, 1, ".vasp",
                            seed_index=seed_offset,
                        )
                        write_poscar(rattled, os.path.join(output_dir, seed_fname))
                        seeds_written += 1
                        records.append(build_migration_seed_record(
                            ctx, seed_offset=seed_offset,
                            poscar_name=seed_fname, extxyz_name=None,
                            path_index=path_idx, image_index=image_idx,
                            path_fraction=float(fraction), source_indices=assignment,
                            vacancy_indices=combo, target_family=target_family,
                            target_points=None, saddle_sign=1,
                            base_record_type=rtype, perturb_radius=PERTURB_RADIUS,
                            amplitude=float(amplitude), perturbed_indices=perturbed_indices,
                            passed=True, reason=None,
                        ))

                print(f"  {target_family} | path {path_idx}: "
                      f"{len(path_fractions)} images × {SEEDS_PER_IMAGE} seeds")

    save_metadata_csv(records, os.path.join(output_dir, "metadata.csv"))
    save_json(records, os.path.join(output_dir, "metadata.json"))

    summary = {
        "case": 3,
        "description": "Single Mg vacancy, migration mode with rattled seeds per image",
        "unique_combos": len(unique),
        "path_images_written": images_written,
        "seeds_per_image": SEEDS_PER_IMAGE,
        "seeds_written": seeds_written,
        "seeds_rejected": seeds_rejected,
        "target_families": TARGET_FAMILIES,
        "expected_seeds": images_written * SEEDS_PER_IMAGE,
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))
    print(f"\n  -> {images_written} clean image(s), "
          f"{seeds_written} rattled seed(s), {seeds_rejected} rejected.")

    # Assertions
    assert images_written > 0, "Expected at least one migration image"
    assert seeds_written > 0, "Expected at least one rattled seed per image"
    # Every clean image should have generated at least one accepted seed
    assert seeds_written + seeds_rejected == images_written * SEEDS_PER_IMAGE, (
        "Total seed attempts must equal images × seeds_per_image"
    )
    # All accepted seed records must pass the distance filter
    seed_records = [r for r in records if r.record_type.endswith("_seed")
                    and not r.record_type.startswith("rejected")]
    assert all(r.distance_filter_passed for r in seed_records)
    # JSON serialisability
    import json
    json.dumps([r.to_dict() for r in records])
    print("  Assertions passed.")
    return summary


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vacancy_sim_") as root:
        print(f"\nSimulation output directory: {root}")
        results = []

        dir1 = os.path.join(root, "case1_Mg1_unique")
        dir2 = os.path.join(root, "case2_Se1_seeded")
        dir3 = os.path.join(root, "case3_Mg1_migration")

        for d in (dir1, dir2, dir3):
            ensure_directory(d)

        results.append(case_1_single_mg_vacancy_unique(dir1))
        results.append(case_2_single_se_vacancy_seeded(dir2))
        results.append(case_3_single_mg_vacancy_migration(dir3))

        _section("Simulation complete — summary")
        for r in results:
            print(f"  Case {r['case']}: {r['description']}")
            for k, v in r.items():
                if k not in ("case", "description"):
                    print(f"    {k}: {v}")

        print("\nAll 3 simulation cases completed successfully.")


if __name__ == "__main__":
    main()
