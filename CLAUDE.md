# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

A vacancy-defect structure generator for materials science. Given a parent
crystal structure (POSCAR), it generates candidate defect structures for DFT
labelling and MLIP (machine-learned interatomic potential) training datasets:
raw or symmetry-deduplicated vacancy combinations, randomly rattled seeds
near each defect, and interpolated migration/hop-path images between a
vacancy and a candidate diffusing atom. It writes structure files (`.vasp`,
optionally `.extxyz`) plus metadata (CSV/JSON) describing every structure it
produced. The migration runner can also stage shared DFT reference files
(POTCAR/INCAR/KPOINTS/job script) into the output tree — see `[dft_setup]`
below — but it does not run DFT calculations itself, and does not generate
per-structure INCAR/KPOINTS content. Once a user has submitted VASP jobs
from the staged output tree, `vacancy_generator.dft_extraction` (see below)
parses the finished `vasprun.xml` results back into a separate results
table — everything past that (MLIP fitting, benchmarking) is still not
built (see `next_steps.txt` for the roadmap).

The actual package lives in `vacancy_generator/`; the repo root also has
older standalone scripts (`generate_vacancy_structures_for_MLIP_datasetv3.py`,
`...v6.py`) and a separate `structure_generation/` helper that predate the
package and are not part of its import graph.

## Commands

Run all from the repo root (`/home/utzinator/Documents/Git/MLIP-Vacancy-Generator`).
There is no `pyproject.toml`/`setup.py` — this is not `pip install`-able;
run everything against the checked-out source with `PYTHONPATH`/cwd set to
the repo root.

```bash
# Interactive generator (prompts for every parameter)
python -m vacancy_generator
# or, from inside vacancy_generator/: python main.py

# Config-file-driven migration-path generation (no prompts, scriptable)
python -m vacancy_generator.migration_runner generate_migration_images.inp

# Harvest finished VASP results (vasprun.xml) from a generation output tree,
# after the user has run the staged slurm jobs; writes dft_results.csv/.json
python -m vacancy_generator.dft_extraction.runner <output_dir>

# Run the pytest suite
python3 -m pytest tests/ -q

# Run a single test
python3 -m pytest tests/test_vacancy_generator.py::TestIO::test_structure_filename_base -q

# End-to-end smoke test (3 real pipeline runs against vacancy_generator/POSCAR,
# no pytest/assertions — just exercises unique/seeded/migration modes and prints progress)
python tests/simulation.py

# Lint (ruff is used in this repo, no committed config — defaults apply)
ruff check vacancy_generator/
```

Required packages: `pymatgen`, `numpy`. Optional: `ase` (enables `.extxyz`
export), `dscribe` (enables SOAP-based diversity pruning). Availability is
probed once in `_compat.py` (`ASE_ADAPTOR_AVAILABLE`, `ASE_IO_AVAILABLE`,
`DSCRIBE_AVAILABLE`) and checked everywhere else instead of re-importing.

Both `pytest tests/ -q` and `python tests/simulation.py` are green (106 tests
pass; all 3 simulation cases complete). If you change the `build_*_record`
signatures in `io.py` or the `MetadataRecord`/`ComboContext` shape in
`records.py` again, both test files construct records via
`ComboContext(...)` + attribute access (`record.record_type`, not
`record["record_type"]`) and will need matching updates — this bit us once
already when the `ComboContext` refactor landed without touching the tests.

## Architecture

### Two entry points, one shared pipeline

- **`main.py`** — interactive CLI (`python -m vacancy_generator`). Prompts
  the user through every choice (mode, defect recipe, supercell, rattling,
  migration parameters, SOAP pruning) via `input_helpers.py`, then drives
  generation through `_write_structures` (raw/unique/seeded/mlip modes) or
  `_write_migration_paths` (migration mode).
- **`migration_runner.py`** — non-interactive, `.inp`-file-driven entry
  point (`python -m vacancy_generator.migration_runner <file>.inp`) for
  scripted/repeatable migration-path generation. Delegates config parsing
  to `config.py` (see `README_inputfile.md` for the full key reference) and
  calls into the same underlying migration-building code as `main.py`'s
  migration mode.

Both entry points converge on the same lower-level modules — there is no
separate logic path per entry point beyond argument sourcing (prompts vs.
config file). `main.py`'s interactive prompts have no equivalent of
`config.py`/`[dft_setup]` — DFT reference-file staging is currently
`migration_runner`-only.

### `config.py`: typed `.inp` parsing + DFT reference-file staging

`migration_runner.py` used to parse its `.inp` file into ~30 loose local
variables inside one function, with validation as scattered inline `if`s.
That's now `MigrationRunConfig.from_inp_file()` — a frozen dataclass built
in one place, validated once (`_validate()`), then read as attributes
everywhere downstream instead of re-threading two dozen locals.

- **`[dft_setup]`** (optional; `enabled = true` by default when the section
  is present) is modeled as a nested `Optional[DftSetup]` — `None` when the
  section is absent or `enabled = false`, so downstream code only needs one
  truthy check. `DftSetup` holds `potcar`/`incar`/`dft_job` (required) and
  `kpoints` (optional — see below).
- Parsing (`DftSetup.from_config`) and filesystem validation
  (`DftSetup.validate`) are deliberately separate: parsing never touches
  disk, so `MigrationRunConfig` stays constructible/testable without a
  populated directory. `migration_runner.run_migration_from_config` calls
  `.validate()` explicitly, before loading the structure or writing
  anything, so a missing/misconfigured DFT file aborts the run before any
  output exists.
- **KPOINTS/KSPACING fallback**: `path_to_kpoints` may be omitted only if
  the referenced INCAR contains a `KSPACING` line (checked by regex against
  the INCAR's raw text in `validate()`); otherwise validation raises.
- **`DftSetup.stage(output_dir)`** copies each configured file into
  `output_dir` **once**, under its VASP-mandated name (`POTCAR`, `INCAR`,
  `KPOINTS`) regardless of the source filename (e.g. `INCAR_MgSc2Se4` is
  copied to `output_dir/INCAR`) — the job script is the one exception and
  keeps its original basename, since VASP has no fixed naming convention
  for a submission script. It then walks every existing per-structure
  subfolder directly under `output_dir` (created earlier by
  `_write_structure_to_subfolder` in `main.py`) and adds a **relative**
  symlink (`../POTCAR`, not an absolute path) to each staged file, so the
  whole output tree keeps working after being moved/copied/rsynced
  elsewhere (e.g. to an HPC cluster).

### `dft_extraction/`: harvesting finished VASP results

A separate subpackage, run independently and later than everything above —
after a user has taken the staged output tree, submitted VASP single-point
jobs from each per-structure subfolder, and those jobs have finished.
`python -m vacancy_generator.dft_extraction.runner <output_dir>` walks
`output_dir` **one level deep**: every immediate subdirectory that has a
matching `<subfolder>.json` sidecar (the generation-time metadata record
`main.py`/`migration_runner.py` already wrote there) is treated as one
structure; subdirectories without one are silently skipped rather than
treated as failed calculations.

- **Results are a separate table, not merged into the generation-time
  CSV/JSON.** `dft_results.csv`/`.json` are written fresh into `output_dir`,
  joined to the generation metadata only by the `poscar_file` key — read
  verbatim from the sidecar JSON (`extract.read_join_key`), never
  reconstructed from folder names. This was a deliberate choice over
  updating the existing rows in place: the generation output is a completed
  artifact by the time extraction runs, DFT jobs finish at different times,
  and two keyed tables is literally the shape a future SQL schema takes
  (see "Where things are heading" below) — no rework needed at that point.
- **`DftResultRecord`** (`dft_extraction/records.py`) is a plain dataclass,
  deliberately *not* part of the `MetadataRecord` hierarchy in `records.py`
  — it only reuses that module's `_sanitise()` helper so its own `to_dict()`
  follows the same numpy-free-flattening convention and drops straight into
  the existing `save_metadata_csv`/`save_json` from `io.py` (those two
  functions duck-type on `hasattr(obj, "to_dict")` rather than
  `isinstance(_, MetadataRecord)` specifically so unrelated record types
  like this one can reuse them). It has its own hand-maintained
  `DFT_FIELD_ORDER`, passed via `save_metadata_csv`'s `field_order` param —
  `_METADATA_FIELD_ORDER` itself is untouched.
- **`status` vs. `converged` are deliberately separate fields.** `status`
  (`"ok"`/`"missing"`/`"unparsable"`/`"unconverged"`) is job state; `converged`
  is strictly the physical question of whether the SCF loop reached `EDIFF`
  (`Vasprun.converged_electronic`). `extract_one` always returns a record —
  a missing or malformed `vasprun.xml` becomes a row with an explanatory
  `error` string instead of raising, so an incomplete/failed slurm job stays
  visible in the output table rather than silently vanishing. The only case
  `extract_one` raises is an unusable sidecar JSON itself (no `poscar_file`
  key), since without a join key no meaningful row can be produced;
  `runner.extract_all` catches that specifically and continues.
- **Uses pymatgen's `Vasprun`, not ASE**, even though `ase` is available
  elsewhere in this repo for `.extxyz` export. `Vasprun` exposes
  `converged_electronic`, `.parameters` (`EDIFF`/`SIGMA`), and
  `ionic_steps[-1]["e_fr_energy"]`/`["forces"]`/`["stress"]` directly, all
  needed for the convergence check; pymatgen is also already this repo's
  hard dependency (see Commands above), so this avoids making `ase` a hard
  dependency for one submodule while `_compat.py` treats it as optional
  everywhere else. `e_fr_energy` (VASP's TOTEN / force-consistent free
  energy) is deliberately used as `energy_eV`, not the sigma→0 extrapolated
  energy — the reasoning being that VASP forces are the analytic gradient of
  TOTEN, not of the extrapolated energy, and MLIP training (MACE/ACE) wants
  the energy/forces gradient-consistent. With Gaussian smearing (`ISMEAR=0`,
  the smearing this repo's calculations use) the two energies do differ
  non-trivially, unlike tetrahedron-method (`ISMEAR=-5`) runs where the
  entropy term is zero and the distinction is moot.
- **Forces are never stored inline** in `DftResultRecord`/the CSV — an Nx3
  array in one cell isn't readable and is exactly the kind of column a SQL
  migration would normalize out anyway. `extract_one` writes them to a
  per-structure `dft_forces.json` sidecar (inside `structure_dir`, alongside
  `vasprun.xml`) and stores only the relative path in `forces_file`. That
  path is derived from `structure_dir`'s own location, not from
  `poscar_file`'s directory component — the two are not guaranteed to match
  (see the `poscar_file`-read-verbatim point above), so deriving it from the
  wrong one would silently point `forces_file` at a file that doesn't exist.
- The sidecar-write and all `Vasprun` parsing happen inside one `try/except`
  in `extract_one` — a failure at either step downgrades the row to
  `status="unparsable"` with an `error` message rather than raising, keeping
  `extract_all`'s contract that one bad structure never aborts the whole run.

### Generation pipeline (conceptual flow)

1. **`recipe.py`** parses a defect recipe (`"Mg:1, Se:2"`) into removal
   counts and enumerates every raw combination of parent-structure site
   indices satisfying it.
2. **`symmetry.py`** builds the parent structure's symmetry operations,
   canonicalizes each raw combination to a symmetry-invariant representative,
   and hashes it — this is the mechanism for **exact** deduplication ("are
   these two vacancy patterns identical under parent symmetry?"). This is
   distinct from and unrelated to SOAP-based diversity pruning.
3. **`structure_ops.py`** removes sites to build the defect structure,
   classifies vacancy topology (single vs. cluster), rattles atoms near
   vacancies, and applies a minimum-interatomic-distance sanity filter to
   reject unphysical rattled structures.
4. **`diversity.py`** optionally down-selects a large candidate set: either
   a random cap, or (if `dscribe`+`ase` are available) defect-local SOAP
   descriptors evaluated at the removed-site positions with farthest-point
   sampling, to keep a diverse subset instead of every duplicate-adjacent
   candidate.
5. **`migration.py`** (migration mode only) finds same-species atoms near
   a vacancy as hop candidates, enumerates source→vacancy assignments, and
   interpolates path images (linear, or curved through a tetrahedral/
   octahedral interstitial pocket) with an optional perpendicular
   saddle-point shift.
6. **`records.py`** + **`io.py`** turn each generated structure into a
   metadata record and write it to disk (see below).
7. **`reporting.py`** aggregates the written records into `summary.json`
   and `report.json`, and prints the terminal run summary.

`structure_utils.py` holds the shared low-level geometry helpers (fractional
coordinate wrapping, minimum-image distances, supercell-size proposals) used
throughout the pipeline.

### Metadata records: `records.py` / `io.py` split

This is the part most likely to need attention when adding a new field or a
new record type:

- **`records.py`** defines the record dataclass hierarchy — pure,
  numpy-free, no file I/O — plus `ComboContext`:
  - `MetadataRecord` → `BaseRecord` → `SeedRecord`
  - `BaseRecord` → `MigrationRecord` → `MigrationSeedRecord`
  - All `@dataclass(kw_only=True)`, so required fields can be added at any
    level without tripping Python's "non-default field after a default
    field" ordering rule. Fields genuinely inapplicable to a record type
    (e.g. `path_index` on a `BaseRecord`) default to `None` rather than a
    fake `""`/`0` sentinel — `to_dict()` then omits/nulls them correctly in
    JSON, and CSV renders `None` as an empty string.
  - `ComboContext` is a frozen dataclass bundling the ~9 values that are
    constant across every record built for one vacancy combination in a
    single loop iteration (`class_label`, `combo`, `canonical_combo`,
    `combo_hash`, `removed_species`, `removed_frac_coords`,
    `vacancy_topology`, `scaling`). Its `common_fields()` method returns the
    shared metadata dict every builder needs, avoiding re-deriving/re-passing
    the same values at each of the ~6 builder call sites per loop iteration.
- **`io.py`** holds the four builder functions (`build_base_record`,
  `build_seed_record`, `build_migration_record`, `build_migration_seed_record`)
  — each takes a `ComboContext` plus its own distinguishing fields — and the
  file-writing/serialisation layer (`write_poscar`, `maybe_write_extxyz`,
  `save_metadata_csv`, `save_json`). `save_metadata_csv`/`save_json` accept
  either record dataclass instances or plain dicts (needed because SOAP
  distance-report rows in `reporting.py` are still plain dicts) and convert
  via `.to_dict()` at the serialisation boundary — nothing upstream of that
  boundary needs to know CSV/JSON exist.
- `_METADATA_FIELD_ORDER` in `io.py` is a hand-maintained list controlling
  CSV column order; it is **not** derived from the dataclass fields, so a
  new field added to `records.py` needs a corresponding entry here or it
  will sort to the end of the CSV instead of its intended position.

`main.py`'s two orchestration loops (`_write_structures`,
`_write_migration_paths`) build one `ComboContext` per vacancy combination
and pass it into every builder call for that iteration.

Known deferred cleanups are tracked in `minor_fixables.md` at the repo root
— check it before doing a broader refactor pass in this area, since it may
already list the issue.

### Where things are heading

`next_steps.txt` documents the intended larger pipeline (this generator is
step 1 of 5). DFT calculation setup (POTCAR/INCAR/KPOINTS/SLURM staging,
`[dft_setup]`) and a first cut of data extraction (`dft_extraction/`, above)
are now built; isolated-atom reference energies, MLIP generation, and
benchmarking are still unbuilt. Also noted there: metadata records are
eventually meant to be written into a proper database rather than CSV/JSON
(the `dataset_type` field — train/validation/test split label, currently
always `"unassigned"` — exists for this future use and needs an actual
assignment strategy, e.g. a Pareto-style split function). The generation
metadata and `dft_extraction`'s results table are already two separate
CSV/JSON tables joined by `poscar_file` specifically so that migration is a
matter of pointing both at SQL tables with a foreign key, not a schema
redesign. Keep this direction in mind when touching `records.py`/`io.py`/
`dft_extraction/records.py`: the dataclass + `to_dict()` boundary was chosen
specifically so a future SQL-writing layer can consume the same objects
without another metadata-modeling rewrite.
