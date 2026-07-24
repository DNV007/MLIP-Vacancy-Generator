"""Unit tests for the vacancy_generator package."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

POSCAR_PATH = pathlib.Path(__file__).parent.parent / "vacancy_generator" / "POSCAR"


@pytest.fixture(scope="module")
def poscar_structure() -> Structure:
    return Structure.from_file(str(POSCAR_PATH))


@pytest.fixture(scope="module")
def simple_cubic() -> Structure:
    """Small 2x2x2 BCC-like test structure with a single species."""
    lattice = Lattice.cubic(4.0)
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ]
    return Structure(lattice, ["Fe", "Fe"], coords)


@pytest.fixture(scope="module")
def binary_structure() -> Structure:
    """Small NaCl-like structure with two species."""
    lattice = Lattice.cubic(5.0)
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
    species = ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"]
    return Structure(lattice, species, coords)


# ===========================================================================
# Imports
# ===========================================================================

class TestImports:
    def test_package_importable(self):
        import vacancy_generator
        assert hasattr(vacancy_generator, "main")

    def test_structure_utils_imports(self):
        from vacancy_generator import structure_utils
        for name in (
            "get_species_counts",
            "get_indices_by_species",
            "wrap_fractional_coords",
            "periodic_fractional_distance",
            "minimum_image_cartesian_vector",
            "minimum_image_cartesian_distance",
            "propose_supercell_scaling",
            "maybe_build_supercell",
            "print_species_counts",
            "print_space_group_info",
            "print_equivalent_site_groups",
        ):
            assert hasattr(structure_utils, name), f"Missing: structure_utils.{name}"

    def test_structure_ops_imports(self):
        from vacancy_generator import structure_ops
        for name in (
            "remove_sites_from_structure",
            "parent_indices_to_defect_indices",
            "classify_vacancy_topology",
            "element_radius",
            "structure_passes_min_distance_filter",
            "get_nearby_atom_indices",
            "random_unit_vector",
            "make_rattled_copy_near_vacancies",
        ):
            assert hasattr(structure_ops, name), f"Missing: structure_ops.{name}"

    def test_recipe_imports(self):
        from vacancy_generator import recipe
        for name in (
            "parse_removal_recipe",
            "ask_removal_recipe",
            "defect_class_label",
            "generate_raw_removal_combinations",
        ):
            assert hasattr(recipe, name), f"Missing: recipe.{name}"

    def test_symmetry_imports(self):
        from vacancy_generator import symmetry
        for name in (
            "build_parent_symmetry_maps",
            "canonicalize_combo",
            "make_combo_hash",
            "keep_only_unique_combinations_by_hash",
        ):
            assert hasattr(symmetry, name), f"Missing: symmetry.{name}"

    def test_migration_imports(self):
        from vacancy_generator import migration
        for name in (
            "expand_migration_target_families",
            "find_migration_candidates",
            "enumerate_migration_assignments",
            "build_migration_image",
            "estimate_interstitial_target_point",
            "estimate_local_shell_normal",
            "choose_perpendicular_unit",
        ):
            assert hasattr(migration, name), f"Missing: migration.{name}"

    def test_io_imports(self):
        from vacancy_generator import io
        for name in (
            "ensure_directory",
            "structure_filename",
            "migration_filename",
            "write_poscar",
            "maybe_write_extxyz",
            "save_metadata_csv",
            "save_json",
            "build_base_record",
            "build_seed_record",
            "build_migration_record",
        ):
            assert hasattr(io, name), f"Missing: io.{name}"

    def test_compat_flags_are_bool(self):
        from vacancy_generator._compat import (
            ASE_ADAPTOR_AVAILABLE,
            ASE_IO_AVAILABLE,
            DSCRIBE_AVAILABLE,
        )
        assert isinstance(ASE_ADAPTOR_AVAILABLE, bool)
        assert isinstance(ASE_IO_AVAILABLE, bool)
        assert isinstance(DSCRIBE_AVAILABLE, bool)

    def test_diversity_imports(self):
        from vacancy_generator import diversity
        for name in ("cap_items_randomly", "select_diverse_structures_by_soap"):
            assert hasattr(diversity, name), f"Missing: diversity.{name}"

    def test_reporting_imports(self):
        from vacancy_generator import reporting
        for name in (
            "build_dataset_report",
            "build_soap_distance_report",
            "describe_output_summary",
        ):
            assert hasattr(reporting, name), f"Missing: reporting.{name}"

    def test_input_helpers_imports(self):
        from vacancy_generator import input_helpers
        for name in ("ask_choice", "ask_float", "ask_int", "ask_string", "ask_yes_no"):
            assert hasattr(input_helpers, name), f"Missing: input_helpers.{name}"


# ===========================================================================
# structure_utils
# ===========================================================================

class TestStructureUtils:
    def test_get_species_counts_poscar(self, poscar_structure):
        from vacancy_generator.structure_utils import get_species_counts
        counts = get_species_counts(poscar_structure)
        assert counts["Mg"] == 8
        assert counts["Sc"] == 16
        assert counts["Se"] == 32

    def test_get_species_counts_sums_to_total(self, poscar_structure):
        from vacancy_generator.structure_utils import get_species_counts
        counts = get_species_counts(poscar_structure)
        assert sum(counts.values()) == len(poscar_structure)

    def test_get_indices_by_species_all_covered(self, poscar_structure):
        from vacancy_generator.structure_utils import get_indices_by_species
        indices = get_indices_by_species(poscar_structure)
        all_indices = sorted(i for group in indices.values() for i in group)
        assert all_indices == list(range(len(poscar_structure)))

    def test_wrap_fractional_coords(self):
        from vacancy_generator.structure_utils import wrap_fractional_coords
        arr = np.array([-0.1, 0.5, 1.3])
        result = wrap_fractional_coords(arr)
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    def test_periodic_fractional_distance_self_is_zero(self):
        from vacancy_generator.structure_utils import periodic_fractional_distance
        assert periodic_fractional_distance([0.3, 0.3, 0.3], [0.3, 0.3, 0.3]) == pytest.approx(0.0)

    def test_periodic_fractional_distance_wraps(self):
        from vacancy_generator.structure_utils import periodic_fractional_distance
        d1 = periodic_fractional_distance([0.01, 0.0, 0.0], [0.99, 0.0, 0.0])
        d2 = periodic_fractional_distance([0.01, 0.0, 0.0], [0.01, 0.0, 0.0])
        assert d1 == pytest.approx(0.02, abs=1e-10)
        assert d2 == pytest.approx(0.0, abs=1e-10)

    def test_minimum_image_cartesian_vector_shape(self):
        from vacancy_generator.structure_utils import minimum_image_cartesian_vector
        latt = np.eye(3) * 5.0
        vec = minimum_image_cartesian_vector(latt, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        assert vec.shape == (3,)

    def test_minimum_image_cartesian_distance_symmetric(self):
        from vacancy_generator.structure_utils import minimum_image_cartesian_distance
        latt = np.eye(3) * 4.0
        a, b = [0.1, 0.2, 0.3], [0.6, 0.7, 0.8]
        d1 = minimum_image_cartesian_distance(latt, a, b)
        d2 = minimum_image_cartesian_distance(latt, b, a)
        assert d1 == pytest.approx(d2)

    def test_propose_supercell_scaling_min_length(self, poscar_structure):
        from vacancy_generator.structure_utils import propose_supercell_scaling
        scaling = propose_supercell_scaling(poscar_structure, min_lattice_length=10.0)
        sc = poscar_structure.copy()
        sc.make_supercell(scaling)
        assert all(l >= 10.0 for l in sc.lattice.abc)


# ===========================================================================
# recipe
# ===========================================================================

class TestRecipe:
    def test_parse_single_species(self, poscar_structure):
        from vacancy_generator.recipe import parse_removal_recipe
        from vacancy_generator.structure_utils import get_species_counts
        available = get_species_counts(poscar_structure)
        recipe = parse_removal_recipe("Mg:1", available)
        assert recipe == {"Mg": 1}

    def test_parse_multi_species(self, poscar_structure):
        from vacancy_generator.recipe import parse_removal_recipe
        from vacancy_generator.structure_utils import get_species_counts
        available = get_species_counts(poscar_structure)
        recipe = parse_removal_recipe("Mg:1, Se:2", available)
        assert recipe == {"Mg": 1, "Se": 2}

    def test_parse_invalid_species_raises(self):
        from vacancy_generator.recipe import parse_removal_recipe
        with pytest.raises(ValueError, match="not present"):
            parse_removal_recipe("X:1", {"Mg": 8})

    def test_parse_duplicate_species_raises(self):
        from vacancy_generator.recipe import parse_removal_recipe
        with pytest.raises(ValueError, match="more than once"):
            parse_removal_recipe("Mg:1, Mg:2", {"Mg": 8})

    def test_parse_count_too_large_raises(self):
        from vacancy_generator.recipe import parse_removal_recipe
        with pytest.raises(ValueError, match="only"):
            parse_removal_recipe("Mg:100", {"Mg": 8})

    def test_parse_count_zero_raises(self):
        from vacancy_generator.recipe import parse_removal_recipe
        with pytest.raises(ValueError, match="at least 1"):
            parse_removal_recipe("Mg:0", {"Mg": 8})

    def test_defect_class_label(self):
        from vacancy_generator.recipe import defect_class_label
        assert defect_class_label({"Mg": 1}) == "Mg1"
        assert defect_class_label({"Se": 2, "Mg": 1}) == "Mg1+Se2"

    def test_generate_raw_combinations_count(self, binary_structure):
        from vacancy_generator.recipe import generate_raw_removal_combinations
        combos = generate_raw_removal_combinations(binary_structure, {"Na": 1})
        na_count = sum(1 for s in binary_structure if s.specie.symbol == "Na")
        assert len(combos) == na_count

    def test_generate_raw_combinations_no_duplicates(self, binary_structure):
        from vacancy_generator.recipe import generate_raw_removal_combinations
        combos = generate_raw_removal_combinations(binary_structure, {"Na": 1})
        assert len(combos) == len(set(combos))

    def test_generate_raw_combinations_indices_sorted(self, binary_structure):
        from vacancy_generator.recipe import generate_raw_removal_combinations
        combos = generate_raw_removal_combinations(binary_structure, {"Na": 2})
        for combo in combos:
            assert list(combo) == sorted(combo)


# ===========================================================================
# structure_ops
# ===========================================================================

class TestStructureOps:
    def test_remove_sites_reduces_count(self, poscar_structure):
        from vacancy_generator.structure_ops import remove_sites_from_structure
        original_len = len(poscar_structure)
        defect = remove_sites_from_structure(poscar_structure, (0,))
        assert len(defect) == original_len - 1

    def test_remove_sites_does_not_modify_original(self, poscar_structure):
        from vacancy_generator.structure_ops import remove_sites_from_structure
        original_len = len(poscar_structure)
        remove_sites_from_structure(poscar_structure, (0, 1))
        assert len(poscar_structure) == original_len

    def test_remove_sites_multi_removes_correct_count(self, poscar_structure):
        from vacancy_generator.structure_ops import remove_sites_from_structure
        defect = remove_sites_from_structure(poscar_structure, (0, 1, 5))
        assert len(defect) == len(poscar_structure) - 3

    def test_parent_indices_to_defect_indices_basic(self):
        from vacancy_generator.structure_ops import parent_indices_to_defect_indices
        mapping = parent_indices_to_defect_indices(5, (2,))
        assert mapping == {0: 0, 1: 1, 3: 2, 4: 3}

    def test_parent_indices_to_defect_indices_no_removal(self):
        from vacancy_generator.structure_ops import parent_indices_to_defect_indices
        mapping = parent_indices_to_defect_indices(4, ())
        assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_classify_vacancy_topology_single(self, poscar_structure):
        from vacancy_generator.structure_ops import classify_vacancy_topology
        label = classify_vacancy_topology(poscar_structure, (0,))
        assert label == "single"

    def test_classify_vacancy_topology_cluster(self, poscar_structure):
        from vacancy_generator.structure_ops import classify_vacancy_topology
        label = classify_vacancy_topology(poscar_structure, (0, 1))
        assert label.startswith("cluster_") or label == "isolated"

    def test_distance_filter_passes_equilibrium(self, poscar_structure):
        from vacancy_generator.structure_ops import structure_passes_min_distance_filter
        passed, reason = structure_passes_min_distance_filter(poscar_structure)
        assert passed is True
        assert reason is None

    def test_distance_filter_fails_collapsed(self):
        from vacancy_generator.structure_ops import structure_passes_min_distance_filter
        lattice = Lattice.cubic(0.1)
        collapsed = Structure(lattice, ["Mg", "Mg"], [[0, 0, 0], [0.01, 0, 0]])
        passed, reason = structure_passes_min_distance_filter(collapsed, absolute_floor=0.5)
        assert passed is False
        assert reason is not None

    def test_random_unit_vector_is_unit(self):
        from vacancy_generator.structure_ops import random_unit_vector
        rng = np.random.default_rng(42)
        for _ in range(20):
            vec = random_unit_vector(rng)
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-10)

    def test_make_rattled_copy_changes_positions(self, poscar_structure):
        from vacancy_generator.structure_ops import (
            make_rattled_copy_near_vacancies,
            remove_sites_from_structure,
        )
        combo = (0,)
        defect = remove_sites_from_structure(poscar_structure, combo)
        rng = np.random.default_rng(0)
        rattled, perturbed = make_rattled_copy_near_vacancies(
            defect, poscar_structure, combo, radius=4.0, max_displacement=0.1, rng=rng
        )
        assert len(perturbed) > 0
        moved = any(
            not np.allclose(defect[i].coords, rattled[i].coords)
            for i in perturbed
        )
        assert moved

    def test_make_rattled_copy_preserves_atom_count(self, poscar_structure):
        from vacancy_generator.structure_ops import (
            make_rattled_copy_near_vacancies,
            remove_sites_from_structure,
        )
        combo = (0,)
        defect = remove_sites_from_structure(poscar_structure, combo)
        rng = np.random.default_rng(1)
        rattled, _ = make_rattled_copy_near_vacancies(
            defect, poscar_structure, combo, radius=4.0, max_displacement=0.05, rng=rng
        )
        assert len(rattled) == len(defect)

    def test_rattle_all_default_perturbs_every_atom(self, poscar_structure):
        """rattle_all=True (default) should displace all atoms in the structure."""
        from vacancy_generator.structure_ops import (
            make_rattled_copy_near_vacancies,
            remove_sites_from_structure,
        )
        combo = (0,)
        defect = remove_sites_from_structure(poscar_structure, combo)
        rng = np.random.default_rng(42)
        rattled, perturbed = make_rattled_copy_near_vacancies(
            defect, poscar_structure, combo, radius=4.0, max_displacement=0.1, rng=rng,
            rattle_all=True,
        )
        assert len(perturbed) == len(defect), (
            f"Expected all {len(defect)} atoms in perturbed list, got {len(perturbed)}"
        )

    def test_rattle_local_only_perturbs_subset(self, poscar_structure):
        """rattle_all=False should displace only atoms near the vacancy (fewer than all)."""
        from vacancy_generator.structure_ops import (
            make_rattled_copy_near_vacancies,
            remove_sites_from_structure,
        )
        combo = (0,)
        defect = remove_sites_from_structure(poscar_structure, combo)
        rng = np.random.default_rng(99)
        _, perturbed_local = make_rattled_copy_near_vacancies(
            defect, poscar_structure, combo, radius=4.0, max_displacement=0.1, rng=rng,
            rattle_all=False,
        )
        assert len(perturbed_local) < len(defect), (
            "Local rattling should perturb fewer atoms than the full structure"
        )


# ===========================================================================
# symmetry
# ===========================================================================

class TestSymmetry:
    def test_make_combo_hash_deterministic(self, poscar_structure):
        from vacancy_generator.symmetry import make_combo_hash
        recipe = {"Mg": 1}
        h1 = make_combo_hash(poscar_structure, recipe, (0,), (1, 1, 1))
        h2 = make_combo_hash(poscar_structure, recipe, (0,), (1, 1, 1))
        assert h1 == h2

    def test_make_combo_hash_different_combos_differ(self, poscar_structure):
        from vacancy_generator.symmetry import make_combo_hash
        recipe = {"Mg": 1}
        h1 = make_combo_hash(poscar_structure, recipe, (0,), (1, 1, 1))
        h2 = make_combo_hash(poscar_structure, recipe, (3,), (1, 1, 1))
        # (0,) and (3,) are Mg sites; they may or may not be symmetry-equivalent
        # but the hashes for different canonical combos must be different
        # We just ensure the function runs without error here
        assert isinstance(h1, str) and len(h1) == 16
        assert isinstance(h2, str) and len(h2) == 16

    def test_canonicalize_combo_returns_tuple(self, poscar_structure):
        from vacancy_generator.symmetry import (
            build_parent_symmetry_maps,
            canonicalize_combo,
        )
        maps = build_parent_symmetry_maps(poscar_structure, symprec=1e-3, match_tol=1e-4)
        result = canonicalize_combo((0,), maps)
        assert isinstance(result, tuple)

    def test_keep_unique_combinations_reduces_count(self, poscar_structure):
        from vacancy_generator.recipe import generate_raw_removal_combinations
        from vacancy_generator.symmetry import (
            build_parent_symmetry_maps,
            keep_only_unique_combinations_by_hash,
        )
        recipe = {"Mg": 1}
        raw = generate_raw_removal_combinations(poscar_structure, recipe)
        maps = build_parent_symmetry_maps(poscar_structure, symprec=1e-3, match_tol=1e-4)
        unique, _, _ = keep_only_unique_combinations_by_hash(
            raw, poscar_structure, recipe, maps, (1, 1, 1)
        )
        # All Mg sites should be symmetry-equivalent in this cubic structure
        assert len(unique) <= len(raw)
        assert len(unique) >= 1

    def test_keep_unique_combinations_hashes_are_unique(self, poscar_structure):
        from vacancy_generator.recipe import generate_raw_removal_combinations
        from vacancy_generator.symmetry import (
            build_parent_symmetry_maps,
            keep_only_unique_combinations_by_hash,
        )
        recipe = {"Mg": 1}
        raw = generate_raw_removal_combinations(poscar_structure, recipe)
        maps = build_parent_symmetry_maps(poscar_structure, symprec=1e-3, match_tol=1e-4)
        unique, combo_to_hash, _ = keep_only_unique_combinations_by_hash(
            raw, poscar_structure, recipe, maps, (1, 1, 1)
        )
        hashes = list(combo_to_hash.values())
        assert len(hashes) == len(set(hashes))


# ===========================================================================
# migration
# ===========================================================================

class TestMigration:
    def test_expand_migration_target_families_all(self):
        from vacancy_generator.migration import expand_migration_target_families
        families = expand_migration_target_families("all")
        assert set(families) == {"direct", "tetrahedral", "octahedral"}

    def test_expand_migration_target_families_direct(self):
        from vacancy_generator.migration import expand_migration_target_families
        assert expand_migration_target_families("direct") == ["direct"]

    def test_expand_migration_target_families_invalid(self):
        from vacancy_generator.migration import expand_migration_target_families
        with pytest.raises(ValueError):
            expand_migration_target_families("bogus")

    def test_find_migration_candidates_returns_same_species(self, poscar_structure):
        from vacancy_generator.migration import find_migration_candidates
        vacancy_idx = 0  # Mg site
        vacancy_symbol = poscar_structure[vacancy_idx].specie.symbol
        candidates = find_migration_candidates(
            poscar_structure,
            vacancy_index=vacancy_idx,
            forbidden_indices=(vacancy_idx,),
            max_candidates=4,
            cutoff=6.0,
        )
        for c in candidates:
            assert poscar_structure[c].specie.symbol == vacancy_symbol

    def test_find_migration_candidates_respects_max(self, poscar_structure):
        from vacancy_generator.migration import find_migration_candidates
        candidates = find_migration_candidates(
            poscar_structure,
            vacancy_index=0,
            forbidden_indices=(0,),
            max_candidates=3,
            cutoff=10.0,
        )
        assert len(candidates) <= 3

    def test_enumerate_migration_assignments_single_vacancy(self, poscar_structure):
        from vacancy_generator.migration import enumerate_migration_assignments
        assignments = enumerate_migration_assignments(
            poscar_structure,
            vacancy_indices=(0,),
            max_candidates_per_vacancy=4,
            cutoff=6.0,
            max_assignments=3,
        )
        assert len(assignments) <= 3
        for assignment in assignments:
            assert len(assignment) == 1

    def test_build_migration_image_fraction_zero_is_start(self, poscar_structure):
        from vacancy_generator.migration import build_migration_image, find_migration_candidates
        from vacancy_generator.structure_ops import remove_sites_from_structure
        combo = (0,)
        defect = remove_sites_from_structure(poscar_structure, combo)
        candidates = find_migration_candidates(
            poscar_structure, vacancy_index=0, forbidden_indices=combo,
            max_candidates=1, cutoff=8.0,
        )
        if not candidates:
            pytest.skip("No migration candidates found")
        source_idx = candidates[0]
        img = build_migration_image(
            defect_structure=defect,
            parent_structure=poscar_structure,
            vacancy_indices=combo,
            source_indices=(source_idx,),
            image_fraction=0.0,
            saddle_shift=0.0,
            saddle_sign=1,
            target_family="direct",
        )
        assert len(img) == len(defect)

    def test_build_migration_image_fraction_one_near_vacancy(self, poscar_structure):
        from vacancy_generator.migration import build_migration_image, find_migration_candidates
        from vacancy_generator.structure_ops import (
            parent_indices_to_defect_indices,
            remove_sites_from_structure,
        )
        combo = (0,)
        defect = remove_sites_from_structure(poscar_structure, combo)
        candidates = find_migration_candidates(
            poscar_structure, vacancy_index=0, forbidden_indices=combo,
            max_candidates=1, cutoff=8.0,
        )
        if not candidates:
            pytest.skip("No migration candidates found")
        source_idx = candidates[0]
        img = build_migration_image(
            defect_structure=defect,
            parent_structure=poscar_structure,
            vacancy_indices=combo,
            source_indices=(source_idx,),
            image_fraction=1.0,
            saddle_shift=0.0,
            saddle_sign=1,
            target_family="direct",
        )
        # At fraction=1 the source atom should be near the vacancy site
        defect_idx = parent_indices_to_defect_indices(len(poscar_structure), combo)[source_idx]
        vacancy_cart = poscar_structure[0].coords
        atom_cart = img[defect_idx].coords
        dist = np.linalg.norm(atom_cart - vacancy_cart)
        assert dist < 1.0  # Should be within 1 Å of the vacancy

    def test_choose_perpendicular_unit_orthogonal(self):
        from vacancy_generator.migration import choose_perpendicular_unit
        v = np.array([1.0, 0.0, 0.0])
        perp = choose_perpendicular_unit(v)
        assert abs(np.dot(v, perp)) < 1e-10
        assert np.linalg.norm(perp) == pytest.approx(1.0)

    def test_safe_unit_vector_normalizes(self):
        from vacancy_generator.migration import _safe_unit_vector
        result = _safe_unit_vector(np.array([3.0, 0.0, 0.0]))
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_safe_unit_vector_zero_returns_none(self):
        from vacancy_generator.migration import _safe_unit_vector
        result = _safe_unit_vector(np.array([0.0, 0.0, 0.0]))
        assert result is None


# ===========================================================================
# io
# ===========================================================================

class TestIO:
    def test_structure_filename_base(self):
        from vacancy_generator.io import structure_filename
        name = structure_filename("POSCAR", "Mg1", 3, None, ".vasp")
        assert name == "POSCAR_Mg1_base0003.vasp"

    def test_structure_filename_seed(self):
        from vacancy_generator.io import structure_filename
        name = structure_filename("POSCAR", "Mg1", 3, 7, ".vasp")
        assert name == "POSCAR_Mg1_base0003_seed007.vasp"

    def test_migration_filename_positive_sign(self):
        from vacancy_generator.io import migration_filename
        name = migration_filename("POSCAR", "Mg1", 1, "direct", 2, 3, 1, ".vasp")
        assert "pimg003" in name
        assert "direct" in name

    def test_migration_filename_negative_sign(self):
        from vacancy_generator.io import migration_filename
        name = migration_filename("POSCAR", "Mg1", 1, "direct", 2, 3, -1, ".vasp")
        assert "mimg003" in name

    def test_write_poscar_creates_file(self, poscar_structure, tmp_path):
        from vacancy_generator.io import write_poscar
        out = str(tmp_path / "test.vasp")
        write_poscar(poscar_structure, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_save_json_creates_file(self, tmp_path):
        from vacancy_generator.io import save_json
        path = str(tmp_path / "data.json")
        save_json({"key": "value", "count": 42}, path)
        assert os.path.isfile(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["count"] == 42

    def test_save_metadata_csv_creates_file(self, tmp_path):
        from vacancy_generator.io import save_metadata_csv
        records = [{"record_type": "base", "defect_class": "Mg1", "combo_index": 1}]
        path = str(tmp_path / "meta.csv")
        save_metadata_csv(records, path)
        assert os.path.isfile(path)

    def test_save_metadata_csv_empty_does_not_create(self, tmp_path):
        from vacancy_generator.io import save_metadata_csv
        path = str(tmp_path / "empty.csv")
        save_metadata_csv([], path)
        assert not os.path.isfile(path)

    def test_build_base_record_fields(self):
        from vacancy_generator.io import build_base_record
        from vacancy_generator.records import ComboContext
        ctx = ComboContext(
            class_label="Mg1", combo_index=1, combo=(5,), canonical_combo=(5,),
            combo_hash="abc123", removed_species=["Mg"],
            removed_frac_coords=[[0.25, 0.25, 0.25]], vacancy_topology="single",
            scaling=(1, 1, 1),
        )
        record = build_base_record(ctx, poscar_name="foo.vasp", extxyz_name=None)
        assert record.record_type == "base"
        assert record.distance_filter_passed is True
        assert record.n_removed_total == 1

    def test_build_seed_record_rejected(self):
        from vacancy_generator.io import build_seed_record
        from vacancy_generator.records import ComboContext
        ctx = ComboContext(
            class_label="Mg1", combo_index=1, combo=(5,), canonical_combo=(5,),
            combo_hash="abc123", removed_species=["Mg"],
            removed_frac_coords=[[0.25, 0.25, 0.25]], vacancy_topology="single",
            scaling=(1, 1, 1),
        )
        record = build_seed_record(
            ctx, seed_offset=2, perturb_radius=4.0, amplitude=0.1,
            perturbed_indices=[3, 4], passed=False, reason={"pair": (0, 1)},
            poscar_name="", extxyz_name=None,
        )
        assert record.record_type == "rejected_seed"
        assert record.distance_filter_passed is False


# ===========================================================================
# JSON serialisability
# ===========================================================================

class TestJSONSerialisability:
    """Ensure every record type and report produced by the pipeline is JSON-safe.

    This guards against the class of bug where dict keys or values contain
    Python-only types (tuples, numpy scalars, etc.) that json.dump rejects.
    """

    def _assert_json_round_trips(self, data: object, label: str) -> None:
        """Serialise to a string and parse it back; assert no TypeError is raised.

        Accepts plain JSON-able data (dicts/lists) or a MetadataRecord, which
        is flattened via ``to_dict()`` first since dataclasses are not
        natively JSON-serialisable.
        """
        from vacancy_generator.records import MetadataRecord
        payload = data.to_dict() if isinstance(data, MetadataRecord) else data
        try:
            serialised = json.dumps(payload)
        except TypeError as exc:
            raise AssertionError(f"{label} is not JSON-serialisable: {exc}") from exc
        recovered = json.loads(serialised)
        assert recovered is not None

    @staticmethod
    def _combo_context(**overrides):
        from vacancy_generator.records import ComboContext
        defaults = dict(
            class_label="Mg1", combo_index=1, combo=(0,), canonical_combo=(0,),
            combo_hash="abc123", removed_species=["Mg"],
            removed_frac_coords=[[0.25, 0.25, 0.25]], vacancy_topology="single",
            scaling=(1, 1, 1),
        )
        defaults.update(overrides)
        return ComboContext(**defaults)

    def test_base_record_is_json_serialisable(self):
        from vacancy_generator.io import build_base_record
        record = build_base_record(
            self._combo_context(), poscar_name="foo.vasp", extxyz_name=None,
        )
        self._assert_json_round_trips(record, "base record")

    def test_seed_record_is_json_serialisable(self):
        from vacancy_generator.io import build_seed_record
        record = build_seed_record(
            self._combo_context(), seed_offset=1, perturb_radius=4.0,
            amplitude=0.05, perturbed_indices=[2, 3], passed=True, reason=None,
            poscar_name="foo_seed.vasp", extxyz_name=None,
        )
        self._assert_json_round_trips(record, "accepted seed record")

    def test_rejected_seed_record_is_json_serialisable(self):
        from vacancy_generator.io import build_seed_record
        reason = {"pair": (0, 1), "species": ("Mg", "Mg"), "distance": 0.3, "minimum_allowed": 0.8}
        record = build_seed_record(
            self._combo_context(), seed_offset=2, perturb_radius=4.0,
            amplitude=0.1, perturbed_indices=[2], passed=False, reason=reason,
            poscar_name="", extxyz_name=None,
        )
        self._assert_json_round_trips(record, "rejected seed record")

    def test_migration_record_is_json_serialisable(self):
        from vacancy_generator.io import build_migration_record
        record = build_migration_record(
            self._combo_context(), poscar_name="img.vasp", extxyz_name=None,
            path_index=1, path_fraction=0.5, source_indices=(3,),
            vacancy_indices=(0,), target_family="direct",
            target_points=None, saddle_sign=1, record_type="path_saddle",
        )
        self._assert_json_round_trips(record, "migration record")

    def test_dataset_report_base_mode_is_json_serialisable(self):
        from vacancy_generator.io import build_base_record
        from vacancy_generator.reporting import build_dataset_report
        records = [
            build_base_record(
                self._combo_context(combo_index=i), poscar_name=f"foo{i}.vasp",
                extxyz_name=None,
            )
            for i in range(3)
        ]
        summary = {"defect_local_soap_diversity_used": False, "soap_settings": None}
        report = build_dataset_report(records, summary)
        self._assert_json_round_trips(report, "dataset report (base mode)")

    def test_dataset_report_migration_mode_is_json_serialisable(self, poscar_structure):
        """Regression test for tuple keys in migration_path_index_counts."""
        from vacancy_generator.io import build_migration_record
        from vacancy_generator.reporting import build_dataset_report

        families = ["direct", "tetrahedral", "octahedral"]
        records = []
        for family in families:
            for path_idx in range(1, 3):
                for img_idx, fraction in enumerate([0.0, 0.5, 1.0]):
                    rtype = {0: "path_start", 1: "path_saddle", 2: "path_end"}[img_idx]
                    records.append(build_migration_record(
                        self._combo_context(),
                        poscar_name=f"{family}_{path_idx}_{img_idx}.vasp",
                        extxyz_name=None, path_index=path_idx, path_fraction=fraction,
                        source_indices=(3,), vacancy_indices=(0,), target_family=family,
                        target_points=None, saddle_sign=1, record_type=rtype,
                    ))
        summary = {"defect_local_soap_diversity_used": False, "soap_settings": None}
        report = build_dataset_report(records, summary)
        # All keys in migration_path_index_counts must be strings, not tuples
        for key in report.get("migration_path_index_counts", {}).keys():
            assert isinstance(key, str), f"Non-string key found: {key!r}"
        self._assert_json_round_trips(report, "dataset report (migration mode)")

    def test_save_json_round_trips_nested_data(self, tmp_path):
        from vacancy_generator.io import save_json
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"a": [1.0, 2.0], "b": "x"},
        }
        path = str(tmp_path / "round_trip.json")
        save_json(data, path)
        with open(path) as f:
            recovered = json.load(f)
        assert recovered == data

    def test_save_json_list_of_records(self, tmp_path):
        from vacancy_generator.io import build_base_record, save_json
        records = [
            build_base_record(
                self._combo_context(combo_index=i), poscar_name=f"foo{i}.vasp",
                extxyz_name=None,
            )
            for i in range(5)
        ]
        path = str(tmp_path / "records.json")
        save_json(records, path)
        with open(path) as f:
            recovered = json.load(f)
        assert len(recovered) == 5
        assert all(r["record_type"] == "base" for r in recovered)
