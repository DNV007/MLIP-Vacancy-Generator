# Migration image generator — input file guide

This guide covers the config-file-driven migration runner
(`vacancy_generator.migration_runner`).  It generates interpolated atomic
migration paths for MLIP training datasets without any interactive prompts.

## Quick start

```bash
# from the repo root
python -m vacancy_generator.migration_runner generate_migration_images.inp
```

Copy `generate_migration_images.inp`, edit the values for your system, and
re-run the same command.  Every parameter has a sensible default so you only
need to set the ones that matter for your calculation.

---

## What the runner produces

For each symmetry-unique vacancy combination the runner:

1. Finds same-species atoms within `hop_cutoff` of the vacancy (the
   diffusion candidates).
2. Enumerates up to `max_assignments` distinct source→vacancy pairings.
3. Interpolates `path_images` structures along each path (direct linear, or
   curved through a tetrahedral / octahedral interstitial pocket).
4. Optionally writes `seeds_per_image` rattled copies of every image for
   training-set diversity.

Output files land in a directory named after the POSCAR file and defect
recipe (e.g. `POSCAR_Mg1_migration_dataset/`) unless `output_dir` is set
explicitly.  Four metadata files are always written:

| File | Contents |
|---|---|
| `metadata.csv` | one row per written structure |
| `metadata.json` | same data in JSON |
| `summary.json` | run parameters and structure counts |
| `report.json` | aggregated statistics |

---

## Input file format

The file uses a standard INI format.  Lines beginning with `#` or `;` are
comments and are ignored.

```ini
[section]
key = value   # inline comments are also supported
```

---

## Section reference

### `[structure]`

Specifies the input crystal structure.

| Key | Default | Description |
|---|---|---|
| `poscar_file` | `POSCAR` | Path to the POSCAR or CONTCAR file. Relative to the working directory. |
| `symprec` | `1e-3` | Symmetry tolerance passed to spglib (Å). Increase slightly for structures with numerical noise. |
| `match_tol` | `1e-4` | Fractional-coordinate tolerance for site matching when building symmetry maps. |

---

### `[supercell]`

Controls automatic supercell construction.  Disabled by default; enable it
when the primitive cell is too small for meaningful training data.

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Set to `true` to build a supercell before generating defects. |
| `min_length` | `10.0` | Each lattice vector of the supercell will be at least this long (Å). |
| `max_atoms` | `400` | Hard cap on supercell size. The smallest valid scaling that satisfies `min_length` without exceeding this limit is chosen. |

---

### `[defect]`

Defines which atoms are removed to create the vacancy.

| Key | Default | Description |
|---|---|---|
| `recipe` | **required** | Comma-separated `species:count` pairs, e.g. `Mg:1` or `Mg:1, Se:2`. Symmetry-unique combinations of the specified atoms are removed one at a time. |

---

### `[migration]`

Controls the migration path geometry and rattling.

#### Path geometry

| Key | Default | Description |
|---|---|---|
| `target_family` | `all` | Path shape family. `direct` = straight line from source to vacancy; `tetrahedral` / `octahedral` = curved path through the nearest interstitial pocket; `all` = generate all three. |
| `hop_cutoff` | `4.0` | Maximum source-to-vacancy distance (Å). Only same-species atoms within this radius are considered as diffusion candidates. If no atoms pass the cutoff it is ignored and the nearest atom is used instead. |
| `max_candidates` | `6` | Number of nearest same-species candidates to keep per vacancy. |
| `max_assignments` | `5` | Maximum number of distinct source→vacancy pairings to generate per vacancy combination. |
| `path_images` | `5` | Number of images along each path, including start (fraction = 0) and end (fraction = 1). Must be ≥ 2. |
| `saddle_shift` | `0.15` | Amplitude (Å) of the perpendicular bump applied at the midpoint of each path to mimic the transition-state arc. Set to `0` for a perfectly straight or unperturbed Bézier path. |
| `both_saddle_sides` | `false` | If `true`, each path is written twice — once with the saddle bump pointing in the `+` direction and once in the `−` direction. Doubles the number of output structures. |
| `sampling_seed` | `12345` | Integer seed for the random-number generator used during rattling. Fix this value to reproduce the same dataset exactly. |

#### How the diffusing atom is chosen

The atom that hops into the vacancy must be the **same species** as the
removed atom and must lie within `hop_cutoff` of the vacancy site.  Among
those candidates the `max_candidates` nearest ones are retained and
combined into up to `max_assignments` injective source→vacancy pairings.
Each pairing becomes an independent migration path.

#### Path endpoint

The path always starts at the source atom's equilibrium site and ends at
the **exact lattice position of the removed atom** (the vacancy site in the
parent structure).

#### Rattling

| Key | Default | Description |
|---|---|---|
| `seeds_per_image` | `0` | Number of randomly rattled copies to generate for every path image. Set to `0` to disable rattling entirely. |
| `rattle_all` | `true` | `true` = displace every atom in the structure (recommended for MLIP training); `false` = displace only atoms within `perturb_radius` of the vacancy. |
| `perturb_radius` | `4.0` | Used only when `rattle_all = false`. Atoms further than this from any vacancy site are left undisplaced. |
| `min_rattle` | `0.02` | Smallest maximum displacement (Å) in the rattling sweep. |
| `max_rattle` | `0.08` | Largest maximum displacement (Å) in the rattling sweep. The `seeds_per_image` copies are distributed linearly between `min_rattle` and `max_rattle`. |
| `distance_scale` | `0.55` | Scale factor applied to the sum of covalent radii when checking the minimum interatomic distance. A rattled structure is rejected if any pair falls below `distance_scale × (r1 + r2)`. |
| `distance_floor` | `0.80` | Absolute minimum interatomic distance (Å) regardless of species. Rattled structures with any pair closer than this are also rejected. |

---

### `[output]`

| Key | Default | Description |
|---|---|---|
| `output_dir` | *(auto)* | Directory to write all output files. Leave empty to auto-generate from the POSCAR filename and defect recipe (e.g. `POSCAR_Mg1_migration_dataset`). |
| `write_base_structures` | `true` | Write the unperturbed vacancy structure (no atom displaced along the path) as an additional reference POSCAR. |
| `write_extxyz` | `false` | Also write `.extxyz` files alongside each `.vasp` file. Requires ASE and the pymatgen ASE adaptor. |

---

### `[dft_setup]`

Stages the shared input files needed to run DFT single-point reference
calculations on every generated structure (POTCAR, INCAR, KPOINTS, and a
job-submission script). Since these parameters are the same for every
structure in a dataset, each file is copied **once** into the output
directory (the "parent" folder) and then relative-symlinked into every
per-structure subfolder (the "child" folders) — so calculations stay
self-consistent even if the whole output directory is later moved, renamed,
or copied to another machine.

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Set to `false` to skip DFT reference-file staging entirely — no keys below are required or validated in that case. |
| `path_to_potcar` | **required** if enabled | Path to the POTCAR file. |
| `path_to_incar` | **required** if enabled | Path to the INCAR file. |
| `path_to_kpoints` | optional | Path to the KPOINTS file. May be left empty **only** if the INCAR referenced above sets `KSPACING` — the runner reads the INCAR to check. If neither is present, the run fails before any files are written. |
| `path_to_dft_job` | **required** if enabled | Path to the job-submission script (e.g. a SLURM script). Copied and symlinked under its own filename. |

All required files are validated to exist **before** any structures are
written; a missing POTCAR/INCAR/job script (or a missing KPOINTS with no
KSPACING fallback) aborts the run immediately rather than partway through.

---

## Output file naming

Files follow this pattern:

```
{POSCAR_stem}_{DefectLabel}_c{combo_index}_{family}_p{path_index}_i{image_index}_{sign}.vasp
```

Rattled seed copies append `_s{seed_index}` before the extension:

```
{...}_i{image_index}_{sign}_s{seed_index}.vasp
```

| Token | Meaning |
|---|---|
| `combo_index` | Index of the symmetry-unique vacancy combination (1-based) |
| `family` | `dir`, `tet`, or `oct` |
| `path_index` | Index of the source→vacancy assignment (1-based) |
| `image_index` | Position along the path (0 = start, N−1 = end) |
| `sign` | `p` for positive saddle direction, `n` for negative |
| `seed_index` | Rattled copy index (1-based) |

---

## Example workflow

```ini
[structure]
poscar_file = MgSe_relaxed.vasp

[supercell]
enabled    = true
min_length = 12.0
max_atoms  = 500

[defect]
recipe = Mg:1

[migration]
target_family    = direct
hop_cutoff       = 4.5
max_candidates   = 4
max_assignments  = 3
path_images      = 9
saddle_shift     = 0.10
both_saddle_sides = false
sampling_seed    = 99

seeds_per_image  = 6
rattle_all       = true
min_rattle       = 0.02
max_rattle       = 0.15
distance_scale   = 0.55
distance_floor   = 0.80

[output]
write_base_structures = true
write_extxyz          = false
```

Run:

```bash
python -m vacancy_generator.migration_runner my_run.inp
```
