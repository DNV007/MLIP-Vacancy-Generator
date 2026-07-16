"""Interactive vacancy-generation workflow — orchestration and CLI entry point."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import SeedSequence
from pymatgen.core import Structure

from ._compat import ASE_ADAPTOR_AVAILABLE, ASE_IO_AVAILABLE, DSCRIBE_AVAILABLE
from .diversity import (
    cap_items_randomly,
    select_diverse_structures_by_soap,
)
from .input_helpers import ask_choice, ask_float, ask_int, ask_string, ask_yes_no
from .io import (
    build_base_record,
    build_migration_record,
    build_migration_seed_record,
    build_seed_record,
    ensure_directory,
    maybe_write_extxyz,
    migration_filename,
    save_json,
    save_metadata_csv,
    structure_filename,
    write_poscar,
)
from .migration import (
    build_migration_image,
    enumerate_migration_assignments,
    estimate_interstitial_target_point,
    expand_migration_target_families,
)
from .recipe import (
    ask_removal_recipe,
    defect_class_label,
    generate_raw_removal_combinations,
)
from .reporting import build_dataset_report, build_soap_distance_report, describe_output_summary
from .structure_ops import (
    classify_vacancy_topology,
    make_rattled_copy_near_vacancies,
    remove_sites_from_structure,
    structure_passes_min_distance_filter,
)
from .structure_utils import (
    get_species_counts,
    maybe_build_supercell,
    print_equivalent_site_groups,
    print_space_group_info,
    print_species_counts,
)
from .symmetry import (
    build_parent_symmetry_maps,
    canonicalize_combo,
    keep_only_unique_combinations_by_hash,
    make_combo_hash,
)


# ---------------------------------------------------------------------------
# Internal orchestration helpers
# ---------------------------------------------------------------------------

def _fill_raw_combo_maps(
    combos: List[Tuple[int, ...]],
    structure: Structure,
    recipe: Dict[str, int],
    scaling: Tuple[int, int, int],
    symprec: float,
    match_tol: float,
    symmetry_maps: Optional[List[Dict[int, int]]],
    combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
) -> List[Dict[int, int]]:
    """Populate hash and canonical-combo look-up tables for a list of combos."""
    if symmetry_maps is None:
        symmetry_maps = build_parent_symmetry_maps(structure, symprec, match_tol)
    for combo in combos:
        canonical = canonicalize_combo(combo, symmetry_maps)
        h = make_combo_hash(structure, recipe, canonical, scaling)
        raw_combo_to_hash[combo] = h
        combo_to_hash[combo] = h
        combo_to_canonical[combo] = canonical
    return symmetry_maps


def _build_base_combos(
    raw_combos: List[Tuple[int, ...]],
    unique_combos: List[Tuple[int, ...]],
    symmetry_maps: Optional[List[Dict[int, int]]],
    working_structure: Structure,
    recipe: Dict[str, int],
    scaling: Tuple[int, int, int],
    symprec: float,
    match_tol: float,
    mode: str,
    combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
) -> Tuple[List[Tuple[int, ...]], str]:
    """Select the base set of combinations depending on the chosen mode."""
    if mode == "raw":
        base_combos = list(raw_combos)
        _fill_raw_combo_maps(
            base_combos, working_structure, recipe, scaling, symprec, match_tol,
            symmetry_maps, combo_to_hash, combo_to_canonical, raw_combo_to_hash,
        )
        return base_combos, "raw"

    if mode in {"unique", "seeded", "migration"}:
        return list(unique_combos), "unique"

    base_source = ask_choice(
        "For mlip mode, use raw or unique base defects",
        choices=["raw", "unique"],
        default="unique",
    )
    if base_source == "raw":
        base_combos = list(raw_combos)
        _fill_raw_combo_maps(
            base_combos, working_structure, recipe, scaling, symprec, match_tol,
            symmetry_maps, combo_to_hash, combo_to_canonical, raw_combo_to_hash,
        )
        return base_combos, "raw"

    return list(unique_combos), "unique"


def _apply_diversity_pruning(
    base_combos: List[Tuple[int, ...]],
    working_structure: Structure,
    mode: str,
    python_rng: random.Random,
) -> Tuple[List[Tuple[int, ...]], Optional[int], str, bool, dict]:
    """Optionally cap and/or SOAP-prune the base combo list."""
    soap_config: dict = {
        "r_cut": 5.0,
        "n_max": 8,
        "l_max": 6,
        "sigma": 0.5,
    }

    apply_cap = ask_yes_no(
        "Apply a balancing cap to the number of base defect structures?", default=True
    )
    if not apply_cap:
        return base_combos, None, "none", False, soap_config

    max_per_class = ask_int(
        "Maximum number of base defect structures for this defect class",
        50,
        min_value=1,
    )

    if len(base_combos) <= max_per_class:
        return base_combos, max_per_class, "cap_not_needed", False, soap_config

    use_soap_pruning = False
    if DSCRIBE_AVAILABLE and ASE_ADAPTOR_AVAILABLE:
        use_soap_pruning = ask_yes_no(
            "Use defect-local SOAP to keep a diverse subset of generated unrelaxed "
            "base defects instead of random down-selection?",
            default=(mode == "mlip"),
        )
    else:
        print("SOAP pruning is unavailable because DScribe and/or ASE support is missing.")

    if use_soap_pruning:
        soap_config["r_cut"] = ask_float("Defect-local SOAP cutoff r_cut in Å", 5.0)
        soap_config["n_max"] = ask_int("SOAP n_max", 8, min_value=1)
        soap_config["l_max"] = ask_int("SOAP l_max", 6, min_value=0)
        soap_config["sigma"] = ask_float("SOAP Gaussian width sigma in Å", 0.5)

        print("Computing defect-local SOAP descriptors for diversity pruning…")
        candidate_structures = [
            remove_sites_from_structure(working_structure, combo)
            for combo in base_combos
        ]
        all_species = sorted(get_species_counts(working_structure).keys())
        selected_indices = select_diverse_structures_by_soap(
            candidate_structures,
            parent_structure=working_structure,
            removed_combos=base_combos,
            max_count=max_per_class,
            species=all_species,
            r_cut=soap_config["r_cut"],
            n_max=soap_config["n_max"],
            l_max=soap_config["l_max"],
            sigma=soap_config["sigma"],
        )
        uncapped_count = len(base_combos)
        base_combos = [base_combos[i] for i in selected_indices]
        print(f"SOAP-pruned base defects: {uncapped_count} -> {len(base_combos)}")
        return base_combos, max_per_class, "defect_local_soap_diversity", True, soap_config

    uncapped_count = len(base_combos)
    base_combos = cap_items_randomly(base_combos, max_per_class, python_rng)
    if len(base_combos) < uncapped_count:
        print(f"Randomly capped base defects: {uncapped_count} -> {len(base_combos)}")
    return base_combos, max_per_class, "random_cap", False, soap_config


def _write_structures(
    base_combos: List[Tuple[int, ...]],
    working_structure: Structure,
    combo_to_hash: Dict[Tuple[int, ...], str],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    class_label: str,
    base_name: str,
    output_dir: str,
    scaling: Tuple[int, int, int],
    mode: str,
    write_base_structures: bool,
    write_extxyz: bool,
    seeds_per_base: int,
    perturb_radius: float,
    min_rattle: float,
    max_rattle: float,
    distance_scale: float,
    distance_floor: float,
    sampling_seed: int,
    summary: dict,
    rattle_all: bool = True,
) -> List[dict]:
    """Write all POSCAR (and optionally extxyz) files and collect metadata."""
    metadata_records: List[dict] = []
    n_total = len(base_combos)

    ss = SeedSequence(sampling_seed)
    combo_sequences = ss.spawn(n_total)

    defect_structure_cache: Dict[Tuple[int, ...], Structure] = {}

    for combo_index, combo in enumerate(base_combos, start=1):
        print(f"  Writing structure {combo_index}/{n_total} …", end="\r", flush=True)

        if combo not in defect_structure_cache:
            defect_structure_cache[combo] = remove_sites_from_structure(
                working_structure, combo
            )
        defect_structure = defect_structure_cache[combo]

        removed_species = [working_structure[i].specie.symbol for i in combo]
        removed_frac_coords = [
            np.round(working_structure[i].frac_coords, 8).tolist() for i in combo
        ]
        vacancy_topology = classify_vacancy_topology(working_structure, combo)
        combo_hash = combo_to_hash.get(combo, raw_combo_to_hash.get(combo, ""))
        canonical_combo = combo_to_canonical.get(combo, combo)

        if write_base_structures:
            poscar_name = structure_filename(base_name, class_label, combo_index, None, ".vasp")
            poscar_path = os.path.join(output_dir, poscar_name)
            write_poscar(defect_structure, poscar_path)

            extxyz_name = None
            if write_extxyz:
                extxyz_name = structure_filename(
                    base_name, class_label, combo_index, None, ".extxyz"
                )
                extxyz_path = os.path.join(output_dir, extxyz_name)
                if not maybe_write_extxyz(defect_structure, extxyz_path):
                    extxyz_name = None

            metadata_records.append(
                build_base_record(
                    class_label, combo_index, combo, canonical_combo, combo_hash,
                    removed_species, removed_frac_coords, vacancy_topology, scaling,
                    poscar_name, extxyz_name,
                )
            )
            summary["base_structures_written"] += 1

        if mode in {"seeded", "mlip"} and seeds_per_base > 0:
            amplitudes = np.linspace(min_rattle, max_rattle, seeds_per_base)
            seed_sequences = combo_sequences[combo_index - 1].spawn(seeds_per_base)

            for seed_offset, (amplitude, seed_seq) in enumerate(
                zip(amplitudes, seed_sequences), start=1
            ):
                np_rng = np.random.default_rng(seed_seq)

                rattled_structure, perturbed_indices = make_rattled_copy_near_vacancies(
                    defect_structure,
                    working_structure,
                    combo,
                    perturb_radius,
                    float(amplitude),
                    np_rng,
                    rattle_all=rattle_all,
                )

                passed, reason = structure_passes_min_distance_filter(
                    rattled_structure,
                    radius_scale=distance_scale,
                    absolute_floor=distance_floor,
                )

                if not passed:
                    summary["seed_structures_rejected_by_distance_filter"] += 1
                    metadata_records.append(
                        build_seed_record(
                            class_label, combo_index, seed_offset, combo,
                            canonical_combo, combo_hash, removed_species,
                            removed_frac_coords, vacancy_topology, scaling, perturb_radius,
                            amplitude, perturbed_indices, False, reason, "", None,
                        )
                    )
                    continue

                poscar_name = structure_filename(
                    base_name, class_label, combo_index, seed_offset, ".vasp"
                )
                poscar_path = os.path.join(output_dir, poscar_name)
                write_poscar(rattled_structure, poscar_path)

                extxyz_name = None
                if write_extxyz:
                    extxyz_name = structure_filename(
                        base_name, class_label, combo_index, seed_offset, ".extxyz"
                    )
                    extxyz_path = os.path.join(output_dir, extxyz_name)
                    if not maybe_write_extxyz(rattled_structure, extxyz_path):
                        extxyz_name = None

                metadata_records.append(
                    build_seed_record(
                        class_label, combo_index, seed_offset, combo,
                        canonical_combo, combo_hash, removed_species,
                        removed_frac_coords, vacancy_topology, scaling, perturb_radius,
                        amplitude, perturbed_indices, True, None,
                        poscar_name, extxyz_name,
                    )
                )
                summary["seed_structures_written"] += 1

    print(" " * 60, end="\r")
    return metadata_records


def _write_migration_paths(
    base_combos: List[Tuple[int, ...]],
    working_structure: Structure,
    combo_to_hash: Dict[Tuple[int, ...], str],
    raw_combo_to_hash: Dict[Tuple[int, ...], str],
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]],
    class_label: str,
    base_name: str,
    output_dir: str,
    scaling: Tuple[int, int, int],
    write_base_structures: bool,
    write_extxyz: bool,
    path_images: int,
    saddle_shift: float,
    write_both_saddle_sides: bool,
    target_family_mode: str,
    hop_cutoff: float,
    max_candidates_per_vacancy: int,
    max_assignments_per_combo: int,
    seeds_per_image: int,
    perturb_radius: float,
    min_rattle: float,
    max_rattle: float,
    distance_scale: float,
    distance_floor: float,
    sampling_seed: int,
    summary: dict,
    rattle_all: bool = True,
) -> List[dict]:
    """Write migration-path images for each vacancy combination."""
    if path_images < 2:
        raise ValueError("path_images must be at least 2")

    metadata_records: List[dict] = []
    n_total = len(base_combos)
    target_families = expand_migration_target_families(target_family_mode)

    for combo_index, combo in enumerate(base_combos, start=1):
        print(f"  Writing migration paths {combo_index}/{n_total} …", end="\r", flush=True)

        defect_structure = remove_sites_from_structure(working_structure, combo)
        removed_species = [working_structure[i].specie.symbol for i in combo]
        removed_frac_coords = [
            np.round(working_structure[i].frac_coords, 8).tolist() for i in combo
        ]
        vacancy_topology = classify_vacancy_topology(working_structure, combo)
        combo_hash = combo_to_hash.get(combo, raw_combo_to_hash.get(combo, ""))
        canonical_combo = combo_to_canonical.get(combo, combo)

        if write_base_structures:
            poscar_name = structure_filename(base_name, class_label, combo_index, None, ".vasp")
            poscar_path = os.path.join(output_dir, poscar_name)
            write_poscar(defect_structure, poscar_path)
            extxyz_name = None
            if write_extxyz:
                extxyz_name = structure_filename(
                    base_name, class_label, combo_index, None, ".extxyz"
                )
                extxyz_path = os.path.join(output_dir, extxyz_name)
                if not maybe_write_extxyz(defect_structure, extxyz_path):
                    extxyz_name = None
            metadata_records.append(
                build_base_record(
                    class_label, combo_index, combo, canonical_combo, combo_hash,
                    removed_species, removed_frac_coords, vacancy_topology, scaling,
                    poscar_name, extxyz_name,
                )
            )
            summary["base_structures_written"] += 1

        assignments = enumerate_migration_assignments(
            working_structure,
            combo,
            max_candidates_per_vacancy=max_candidates_per_vacancy,
            cutoff=hop_cutoff,
            max_assignments=max_assignments_per_combo,
        )
        summary["migration_assignments_enumerated"] += len(assignments)

        if not assignments:
            continue

        path_fractions = np.linspace(0.0, 1.0, path_images)
        saddle_signs = [1, -1] if write_both_saddle_sides else [1]
        for target_family in target_families:
            path_index = 0
            for assignment in assignments:
                for saddle_sign in saddle_signs:
                    path_index += 1
                    target_points: List[np.ndarray] = []
                    if target_family != "direct":
                        for vacancy_index, source_index in zip(combo, assignment):
                            target_point = estimate_interstitial_target_point(
                                working_structure,
                                vacancy_index=vacancy_index,
                                source_index=source_index,
                                target_family=target_family,
                            )
                            if target_point is not None:
                                target_points.append(target_point)
                    amplitudes = (
                        np.linspace(min_rattle, max_rattle, seeds_per_image)
                        if seeds_per_image > 0 else np.array([])
                    )

                    for image_index, fraction in enumerate(path_fractions):
                        path_structure = build_migration_image(
                            defect_structure=defect_structure,
                            parent_structure=working_structure,
                            vacancy_indices=combo,
                            source_indices=assignment,
                            image_fraction=float(fraction),
                            saddle_shift=saddle_shift,
                            saddle_sign=saddle_sign,
                            target_family=target_family,
                        )

                        if image_index == 0:
                            record_type = "path_start"
                        elif image_index == len(path_fractions) - 1:
                            record_type = "path_end"
                        elif abs(float(fraction) - 0.5) < 1e-9:
                            record_type = "path_saddle"
                        else:
                            record_type = "path_image"

                        if write_base_structures or seeds_per_image == 0:
                            poscar_name = migration_filename(
                                base_name, class_label, combo_index,
                                target_family, path_index, image_index, saddle_sign, ".vasp",
                            )
                            poscar_path = os.path.join(output_dir, poscar_name)
                            write_poscar(path_structure, poscar_path)

                            extxyz_name = None
                            if write_extxyz:
                                extxyz_name = migration_filename(
                                    base_name, class_label, combo_index,
                                    target_family, path_index, image_index, saddle_sign, ".extxyz",
                                )
                                extxyz_path = os.path.join(output_dir, extxyz_name)
                                if not maybe_write_extxyz(path_structure, extxyz_path):
                                    extxyz_name = None

                            metadata_records.append(
                                build_migration_record(
                                    class_label=class_label,
                                    combo_index=combo_index,
                                    combo=combo,
                                    canonical_combo=canonical_combo,
                                    combo_hash=combo_hash,
                                    removed_species=removed_species,
                                    removed_frac_coords=removed_frac_coords,
                                    vacancy_topology=vacancy_topology,
                                    scaling=scaling,
                                    poscar_name=poscar_name,
                                    extxyz_name=extxyz_name,
                                    path_index=path_index,
                                    path_fraction=float(fraction),
                                    source_indices=assignment,
                                    vacancy_indices=combo,
                                    target_family=target_family,
                                    target_points=target_points,
                                    saddle_sign=saddle_sign,
                                    record_type=record_type,
                                )
                            )
                            summary["migration_path_images_written"] += 1

                        # Rattle each path image seeds_per_image times
                        for seed_offset, amplitude in enumerate(amplitudes, start=1):
                            np_rng = np.random.default_rng(
                                SeedSequence([sampling_seed, combo_index, path_index, image_index, seed_offset])
                            )
                            rattled, perturbed_indices = make_rattled_copy_near_vacancies(
                                path_structure,
                                working_structure,
                                combo,
                                perturb_radius,
                                float(amplitude),
                                np_rng,
                                rattle_all=rattle_all,
                            )
                            passed, reason = structure_passes_min_distance_filter(
                                rattled,
                                radius_scale=distance_scale,
                                absolute_floor=distance_floor,
                            )

                            seed_poscar_name = migration_filename(
                                base_name, class_label, combo_index,
                                target_family, path_index, image_index, saddle_sign, ".vasp",
                                seed_index=seed_offset,
                            )
                            seed_extxyz_name = None

                            if not passed:
                                summary["migration_seed_structures_rejected"] += 1
                                metadata_records.append(
                                    build_migration_seed_record(
                                        class_label=class_label,
                                        combo_index=combo_index,
                                        seed_offset=seed_offset,
                                        combo=combo,
                                        canonical_combo=canonical_combo,
                                        combo_hash=combo_hash,
                                        removed_species=removed_species,
                                        removed_frac_coords=removed_frac_coords,
                                        vacancy_topology=vacancy_topology,
                                        scaling=scaling,
                                        poscar_name="",
                                        extxyz_name=None,
                                        path_index=path_index,
                                        image_index=image_index,
                                        path_fraction=float(fraction),
                                        source_indices=assignment,
                                        vacancy_indices=combo,
                                        target_family=target_family,
                                        target_points=target_points,
                                        saddle_sign=saddle_sign,
                                        base_record_type=record_type,
                                        perturb_radius=perturb_radius,
                                        amplitude=float(amplitude),
                                        perturbed_indices=perturbed_indices,
                                        passed=False,
                                        reason=reason,
                                    )
                                )
                                continue

                            seed_poscar_path = os.path.join(output_dir, seed_poscar_name)
                            write_poscar(rattled, seed_poscar_path)

                            if write_extxyz:
                                seed_extxyz_name = migration_filename(
                                    base_name, class_label, combo_index,
                                    target_family, path_index, image_index, saddle_sign, ".extxyz",
                                    seed_index=seed_offset,
                                )
                                seed_extxyz_path = os.path.join(output_dir, seed_extxyz_name)
                                if not maybe_write_extxyz(rattled, seed_extxyz_path):
                                    seed_extxyz_name = None

                            metadata_records.append(
                                build_migration_seed_record(
                                    class_label=class_label,
                                    combo_index=combo_index,
                                    seed_offset=seed_offset,
                                    combo=combo,
                                    canonical_combo=canonical_combo,
                                    combo_hash=combo_hash,
                                    removed_species=removed_species,
                                    removed_frac_coords=removed_frac_coords,
                                    vacancy_topology=vacancy_topology,
                                    scaling=scaling,
                                    poscar_name=seed_poscar_name,
                                    extxyz_name=seed_extxyz_name,
                                    path_index=path_index,
                                    image_index=image_index,
                                    path_fraction=float(fraction),
                                    source_indices=assignment,
                                    vacancy_indices=combo,
                                    target_family=target_family,
                                    target_points=target_points,
                                    saddle_sign=saddle_sign,
                                    base_record_type=record_type,
                                    perturb_radius=perturb_radius,
                                    amplitude=float(amplitude),
                                    perturbed_indices=perturbed_indices,
                                    passed=True,
                                    reason=None,
                                )
                            )
                            summary["migration_seed_structures_written"] += 1
                summary["migration_path_families_written"] += 1

    print(" " * 60, end="\r")
    return metadata_records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the interactive vacancy-generation workflow."""
    print("Vacancy structure generator with exact hashes and defect-local SOAP pruning")
    print("-" * 78)

    input_filename = ask_string("Enter POSCAR filename", "POSCAR")
    parent_structure = Structure.from_file(input_filename)

    print_species_counts(parent_structure)
    working_structure, scaling = maybe_build_supercell(parent_structure)
    print_species_counts(working_structure)

    symprec = ask_float("Symmetry tolerance symprec", 1e-3)
    match_tol = ask_float("Site-matching tolerance in fractional coordinates", 1e-4)

    print_space_group_info(working_structure, symprec)
    print_equivalent_site_groups(working_structure, symprec)

    recipe = ask_removal_recipe(working_structure)
    class_label = defect_class_label(recipe)

    mode = ask_choice(
        "Choose mode",
        choices=["raw", "unique", "seeded", "mlip", "migration"],
        default="mlip",
    )

    raw_combos = generate_raw_removal_combinations(working_structure, recipe)
    print(f"\nDefect class: {class_label}")
    print(f"Total raw combinations: {len(raw_combos)}")

    symmetry_maps: Optional[List[Dict[int, int]]] = None
    unique_combos: List[Tuple[int, ...]] = []
    combo_to_hash: Dict[Tuple[int, ...], str] = {}
    combo_to_canonical: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    raw_combo_to_hash: Dict[Tuple[int, ...], str] = {}

    if mode in {"unique", "seeded", "mlip", "migration"}:
        print("Building symmetry maps … (this may take a moment for large supercells)")
        symmetry_maps = build_parent_symmetry_maps(working_structure, symprec, match_tol)
        unique_combos, combo_to_hash, combo_to_canonical = (
            keep_only_unique_combinations_by_hash(
                raw_combos,
                structure=working_structure,
                recipe=recipe,
                symmetry_maps=symmetry_maps,
                scaling=scaling,
            )
        )
        for raw_combo in raw_combos:
            canonical = canonicalize_combo(raw_combo, symmetry_maps)
            raw_combo_to_hash[raw_combo] = make_combo_hash(
                working_structure, recipe, canonical, scaling
            )
        print(
            f"Exact parent-symmetry-unique combinations (by canonical hash): "
            f"{len(unique_combos)}"
        )

    base_combos, base_source = _build_base_combos(
        raw_combos, unique_combos, symmetry_maps,
        working_structure, recipe, scaling,
        symprec, match_tol,
        mode,
        combo_to_hash, combo_to_canonical, raw_combo_to_hash,
    )

    sampling_seed = ask_int("Random seed for down-selection and perturbations", 12345, min_value=0)
    python_rng = random.Random(sampling_seed)

    base_combos, max_per_class, base_selection_method, use_soap_pruning, soap_config = (
        _apply_diversity_pruning(
            base_combos, working_structure, mode, python_rng
        )
    )

    write_base_structures = True
    if mode in {"seeded", "mlip", "migration"}:
        write_base_structures = ask_yes_no(
            "Also write the unperturbed base defect structures?", default=True
        )

    seeds_per_base = 0
    perturb_radius = 0.0
    min_rattle = 0.0
    max_rattle = 0.0
    distance_scale = 0.55
    distance_floor = 0.80
    rattle_all = True

    migration_hop_cutoff = 0.0
    migration_max_candidates = 0
    migration_max_assignments = 0
    migration_path_images = 0
    migration_saddle_shift = 0.0
    migration_write_both_sides = True
    migration_target_family_mode = "all"
    migration_seeds_per_image = 0

    if mode in {"seeded", "mlip"}:
        seeds_per_base = ask_int("How many rattled seeds per base defect", 8, min_value=0)
        rattle_all = ask_yes_no("Rattle all atoms in the structure?", default=True)
        if not rattle_all:
            perturb_radius = ask_float(
                "Perturb atoms within this radius of each vacancy site (Å)", 4.0
            )
        min_rattle = ask_float(
            "Smallest maximum random displacement used for a seed (Å)", 0.02
        )
        max_rattle = ask_float(
            "Largest maximum random displacement used for a seed (Å)", 0.08
        )
        if max_rattle < min_rattle:
            raise ValueError("Largest displacement must be >= smallest displacement.")
        distance_scale = ask_float("Minimum-distance filter scale factor", 0.55)
        distance_floor = ask_float("Absolute minimum allowed interatomic distance (Å)", 0.80)
    elif mode == "migration":
        migration_target_family_mode = ask_choice(
            "Migration target family mode",
            choices=["all", "direct", "tetrahedral", "octahedral"],
            default="all",
        )
        migration_hop_cutoff = ask_float(
            "Maximum source-to-vacancy hop distance to consider (Å)", 4.0
        )
        migration_max_candidates = ask_int(
            "Maximum source candidates per vacancy", 6, min_value=1
        )
        migration_max_assignments = ask_int(
            "Maximum source-vacancy assignments per vacancy combo", 5, min_value=1
        )
        migration_path_images = ask_int(
            "Number of images along each migration path", 5, min_value=2
        )
        migration_saddle_shift = ask_float(
            "Perpendicular saddle-point shift amplitude (Å)", 0.15
        )
        migration_write_both_sides = ask_yes_no(
            "Write both + and - saddle-side images?", default=False
        )
        migration_seeds_per_image = ask_int(
            "How many rattled seeds per path image (0 = none)", 4, min_value=0
        )
        if migration_seeds_per_image > 0:
            rattle_all = ask_yes_no("Rattle all atoms in the structure?", default=True)
            if not rattle_all:
                perturb_radius = ask_float(
                    "Perturb atoms within this radius of each vacancy site (Å)", 4.0
                )
            min_rattle = ask_float(
                "Smallest maximum random displacement used for a seed (Å)", 0.02
            )
            max_rattle = ask_float(
                "Largest maximum random displacement used for a seed (Å)", 0.08
            )
            if max_rattle < min_rattle:
                raise ValueError("Largest displacement must be >= smallest displacement.")
            distance_scale = ask_float("Minimum-distance filter scale factor", 0.55)
            distance_floor = ask_float(
                "Absolute minimum allowed interatomic distance (Å)", 0.80
            )

    write_extxyz = ask_yes_no(
        "Also write extxyz files when ASE is available?",
        default=False,
    )
    if write_extxyz and not (ASE_ADAPTOR_AVAILABLE and ASE_IO_AVAILABLE):
        print("ASE or pymatgen-to-ASE support is not available. extxyz export will be skipped.")

    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_dir = ask_string("Output directory", f"{base_name}_{class_label}_dataset")
    ensure_directory(output_dir)

    summary = {
        "input_file": input_filename,
        "supercell_scaling": scaling,
        "symprec": symprec,
        "site_match_tolerance": match_tol,
        "defect_recipe": recipe,
        "defect_class": class_label,
        "mode": mode,
        "raw_combinations": len(raw_combos),
        "base_structures_written": 0,
        "seed_structures_written": 0,
        "seed_structures_rejected_by_distance_filter": 0,
        "balancing_cap_per_class": max_per_class,
        "base_selection_method": base_selection_method,
        "defect_local_soap_diversity_used": use_soap_pruning,
        "soap_settings": soap_config if use_soap_pruning else None,
        "migration_hop_cutoff_A": migration_hop_cutoff if mode == "migration" else None,
        "migration_max_candidates_per_vacancy": migration_max_candidates if mode == "migration" else None,
        "migration_max_assignments_per_combo": migration_max_assignments if mode == "migration" else None,
        "migration_path_images": migration_path_images if mode == "migration" else None,
        "migration_saddle_shift_A": migration_saddle_shift if mode == "migration" else None,
        "migration_write_both_sides": migration_write_both_sides if mode == "migration" else None,
        "migration_target_family_mode": migration_target_family_mode if mode == "migration" else None,
        "migration_assignments_enumerated": 0,
        "migration_path_images_written": 0,
        "migration_path_families_written": 0,
        "migration_seeds_per_image": migration_seeds_per_image if mode == "migration" else None,
        "migration_seed_structures_written": 0,
        "migration_seed_structures_rejected": 0,
    }

    if mode == "migration":
        metadata_records = _write_migration_paths(
            base_combos=base_combos,
            working_structure=working_structure,
            combo_to_hash=combo_to_hash,
            raw_combo_to_hash=raw_combo_to_hash,
            combo_to_canonical=combo_to_canonical,
            class_label=class_label,
            base_name=base_name,
            output_dir=output_dir,
            scaling=scaling,
            write_base_structures=write_base_structures,
            write_extxyz=write_extxyz,
            path_images=migration_path_images,
            saddle_shift=migration_saddle_shift,
            write_both_saddle_sides=migration_write_both_sides,
            target_family_mode=migration_target_family_mode,
            hop_cutoff=migration_hop_cutoff,
            max_candidates_per_vacancy=migration_max_candidates,
            max_assignments_per_combo=migration_max_assignments,
            seeds_per_image=migration_seeds_per_image,
            perturb_radius=perturb_radius,
            min_rattle=min_rattle,
            max_rattle=max_rattle,
            distance_scale=distance_scale,
            distance_floor=distance_floor,
            sampling_seed=sampling_seed,
            summary=summary,
            rattle_all=rattle_all,
        )
    else:
        metadata_records = _write_structures(
            base_combos=base_combos,
            working_structure=working_structure,
            combo_to_hash=combo_to_hash,
            raw_combo_to_hash=raw_combo_to_hash,
            combo_to_canonical=combo_to_canonical,
            class_label=class_label,
            base_name=base_name,
            output_dir=output_dir,
            scaling=scaling,
            mode=mode,
            write_base_structures=write_base_structures,
            write_extxyz=write_extxyz,
            seeds_per_base=seeds_per_base,
            perturb_radius=perturb_radius,
            min_rattle=min_rattle,
            max_rattle=max_rattle,
            distance_scale=distance_scale,
            distance_floor=distance_floor,
            sampling_seed=sampling_seed,
            summary=summary,
            rattle_all=rattle_all,
        )

    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_json_path = os.path.join(output_dir, "metadata.json")
    summary_json_path = os.path.join(output_dir, "summary.json")
    report_json_path = os.path.join(output_dir, "report.json")
    soap_report_json_path = os.path.join(output_dir, "soap_report.json")
    soap_distances_csv_path = os.path.join(output_dir, "soap_distances.csv")

    save_metadata_csv(metadata_records, metadata_csv_path)
    save_json(metadata_records, metadata_json_path)
    save_json(summary, summary_json_path)

    base_combo_hashes = {
        combo: combo_to_hash.get(combo, raw_combo_to_hash.get(combo, ""))
        for combo in base_combos
    }
    soap_rows: List[dict] = []
    soap_stats: dict = {
        "soap_distance_pairs": 0,
        "soap_distance_min": None,
        "soap_distance_mean": None,
        "soap_distance_max": None,
    }
    if use_soap_pruning and DSCRIBE_AVAILABLE and ASE_ADAPTOR_AVAILABLE:
        soap_rows, soap_stats = build_soap_distance_report(
            working_structure=working_structure,
            base_combos=base_combos,
            combo_to_hash=base_combo_hashes,
            species=sorted(get_species_counts(working_structure).keys()),
            soap_config=soap_config,
        )
        save_metadata_csv(soap_rows, soap_distances_csv_path)

    dataset_report = build_dataset_report(metadata_records, summary)
    dataset_report.update(soap_stats)
    save_json(dataset_report, report_json_path)

    if use_soap_pruning and DSCRIBE_AVAILABLE and ASE_ADAPTOR_AVAILABLE:
        soap_report_data = {
            "selected_base_defects": [list(combo) for combo in base_combos],
            "soap_distance_stats": soap_stats,
            "soap_pairwise_rows": soap_rows,
        }
        save_json(soap_report_data, soap_report_json_path)

    describe_output_summary(metadata_records)
    print(f"\nMetadata written to: {metadata_csv_path}")
    print(f"Summary written to:  {summary_json_path}")
    print(f"Report written to:    {report_json_path}")
    if soap_rows:
        print(f"SOAP distances written to: {soap_distances_csv_path}")
        print(f"SOAP report written to:    {soap_report_json_path}")


if __name__ == "__main__":
    main()
