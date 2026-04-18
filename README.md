# Guide to `generate_vacancy_structures_for_MLIP_datasetv3.py`

This README explains:

- what the script does
- what each terminal input means
- the exact order in which the prompts appear
- which prompts appear only in certain modes
- sensible beginner choices
- what files are written at the end

This guide is written for users who want to **run the script from the terminal without guessing what each question means**.

---

## 1. What the script does

The script generates **vacancy structures** from a crystal structure file such as a POSCAR.

A vacancy means that one or more atoms are removed from the parent crystal.

The script is designed for two main use cases:

### A. Defect structure generation
It can generate:
- all raw vacancy combinations
- only symmetry-unique vacancy combinations
- symmetry-unique vacancy combinations plus locally perturbed seeds

### B. MLIP dataset preparation
It can also help prepare **candidate structures** for later DFT calculations by:
- building a supercell
- generating vacancy patterns
- removing exact symmetry duplicates
- optionally keeping a diverse subset using **defect-local SOAP**
- generating multiple rattled structures near each vacancy
- filtering out obviously bad structures with unphysically short distances

## Important limitation
This script works at the **generation stage**.
It does **not** compare relaxed structures.
It does **not** tell you which defect is lowest in energy.
It does **not** prove that symmetry-equivalent starting defects will remain equivalent after relaxation.

---

## 2. Scientific idea behind the script

The script uses **two different ideas**, and they do different jobs.

### 2.1 Exact uniqueness: symmetry-canonical hash
This answers:

> Are these two vacancy patterns exactly the same under the symmetry of the parent crystal?

How it works:
- the script builds the symmetry operations of the parent structure
- each vacancy combination is transformed under those symmetry operations
- the lexicographically smallest transformed version is chosen as the **canonical** vacancy pattern
- a short hash is generated from that canonical pattern

If two raw combinations are symmetry-equivalent, they get the same canonical pattern and the same hash.

This is used for:
- removing exact duplicates at generation time
- labelling vacancy patterns in the metadata

### 2.2 Diversity selection: defect-local SOAP
This answers:

> Among many unrelaxed candidate defects, which ones have the most different local environments around the vacancy?

How it works:
- SOAP is evaluated at the **positions of the removed parent sites**
- this focuses on the local defect region instead of the whole supercell
- farthest-point sampling keeps a diverse subset

This is used for:
- pruning a large set of unrelaxed candidates
- keeping a more varied MLIP input set

SOAP is **not** used here to define exact uniqueness.
That job belongs to the symmetry-canonical hash.

---

## 3. Modes available in the script

The script asks you to choose one of four modes.

### `raw`
Write **all raw vacancy combinations**.

Use this when:
- you want every possible atom-removal combination
- you do not want symmetry reduction
- you want broad sampling of nominally different starting structures

### `unique`
Write only **exact parent-symmetry-unique vacancy combinations**.

Use this when:
- you want one representative for each exact vacancy pattern
- you want to avoid duplicate unrelaxed structures caused by parent symmetry

### `seeded`
Start from symmetry-unique vacancy combinations and generate **rattled seeds** near the vacancies.

Use this when:
- you want several local perturbations around each exact vacancy pattern
- you plan to relax these later

### `mlip`
Dataset-oriented mode.

Use this when:
- you are building candidate structures for DFT labelling and later MLIP fitting
- you want optional diversity pruning and metadata
- you may want either `raw` or `unique` base defects plus rattled seeds

---

## 4. Before you run the script

### Required Python packages
The script requires at least:
- `pymatgen`
- `numpy`

Optional packages:
- `ase` for `.extxyz` export
- `dscribe` for SOAP-based diversity pruning

### Input structure file
You need a structure file such as:
- `POSCAR`
- `POSCAR_292`
- another POSCAR-like structure readable by pymatgen

Put the structure file in the same directory as the script, or provide a path to it.

### Run command
Example:

```bash
python generate_vacancy_structures_for_MLIP_datasetv3.py
```

---

## 5. Exact order of terminal inputs

This section is the most important one if you want to run the script smoothly instead of treating the prompt sequence like a séance.

Not every prompt appears every time. Some only appear if you choose certain options.

## 5.1 Prompts that always appear first

### Prompt 1
```text
Enter POSCAR filename [POSCAR_292]:
```
What to type:
- the file name of your structure, for example `POSCAR_292`
- or press Enter to use the default

### Prompt 2
```text
Do you want to automatically build a supercell before creating vacancies? [y/n, default: y]:
```
What to type:
- `y` if you want a supercell
- `n` if you want to use the structure exactly as read

For defect studies and MLIP generation, `y` is usually the better choice.

---

## 5.2 If you answer `y` to supercell

You will then see these prompts:

### Prompt 3
```text
Target minimum lattice-vector length in Å [10.0]:
```
What to type:
- a number such as `10`, `12`, or `15`

Meaning:
- the script tries to enlarge the cell so each lattice direction is at least this large

Typical beginner value:
- `10.0` to `12.0`

### Prompt 4
```text
Also enforce a minimum total number of atoms? [y/n, default: n]:
```
What to type:
- `y` if you also want a minimum atom count
- `n` otherwise

### Prompt 5, only if Prompt 4 = `y`
```text
Minimum total number of atoms in the supercell [96]:
```
What to type:
- an integer such as `96`, `128`, or `192`

### Prompt 6
```text
Maximum allowed total number of atoms [400]:
```
What to type:
- an integer safety limit such as `300`, `400`, or `500`

Meaning:
- prevents the script from building a supercell that is too large to handle comfortably

---

## 5.3 If you answer `n` to supercell

Then the script skips the supercell prompts and directly uses the input structure as-is.

---

## 5.4 Symmetry settings

These prompts always appear after the working structure is finalized.

### Prompt 7
```text
Symmetry tolerance symprec [1e-3]:
```
What to type:
- usually just press Enter
- or enter something like `1e-3`, `1e-2`, or `5e-4`

Meaning:
- tolerance used by pymatgen to detect symmetry

Typical beginner value:
- `1e-3`

### Prompt 8
```text
Site-matching tolerance in fractional coordinates [1e-4]:
```
What to type:
- usually press Enter
- or enter `1e-4`, `1e-3`, etc.

Meaning:
- tolerance used when mapping sites under symmetry operations

Typical beginner value:
- `1e-4`

---

## 5.5 Defect recipe

### Prompt 9
```text
Defect recipe:
```
What to type:
- one or more `Species:Count` pairs separated by commas

Examples:
```text
Mg:1
Sc:2
Mg:1, Se:2
```

Meaning:
- `Mg:1` means remove one Mg atom
- `Sc:2` means remove two Sc atoms
- `Mg:1, Se:2` means remove one Mg atom and two Se atoms in the same defect

Rules:
- species must exist in the structure
- counts must be integers
- do not repeat the same species twice

Correct:
```text
Mg:3, Se:1
```

Wrong:
```text
Mg:1, Mg:2
```

---

## 5.6 Mode selection

### Prompt 10
```text
Choose mode [raw/unique/seeded/mlip, default: mlip]:
```
What to type:
- `raw`
- `unique`
- `seeded`
- `mlip`

Typical use:
- `raw` for exhaustive raw generation
- `unique` for symmetry-unique vacancy patterns only
- `seeded` for unique vacancy patterns plus perturbed seeds
- `mlip` for MLIP-oriented dataset generation

---

## 5.7 Additional prompt only in `mlip` mode

If mode = `mlip`, you will see:

### Prompt 11
```text
For mlip mode, use raw or unique base defects [raw/unique, default: unique]:
```
What to type:
- `raw` if you want the base set to start from all raw combinations
- `unique` if you want the base set to start from symmetry-unique combinations

Recommended for beginners:
- `unique`

Use `raw` only if you explicitly want broader sampling and you can afford more structures.

---

## 5.8 Random seed and balancing cap

### Prompt 12
```text
Random seed for down-selection and perturbations [12345]:
```
What to type:
- usually just press Enter
- or any non-negative integer

Meaning:
- controls random down-selection and random perturbations
- helps reproducibility

### Prompt 13
```text
Apply a balancing cap to the number of base defect structures? [y/n, default: y]:
```
What to type:
- `y` to limit how many base defects are kept
- `n` to keep them all

Recommended for MLIP:
- `y`

### Prompt 14, only if Prompt 13 = `y`
```text
Maximum number of base defect structures for this defect class [50]:
```
What to type:
- an integer such as `20`, `50`, or `100`

Meaning:
- limits how many base defect structures of this recipe are kept

---

## 5.9 Optional SOAP diversity pruning

This part appears only if:
- the cap is active
- there are more base structures than the cap
- DScribe and ASE support are available

### Prompt 15
```text
Use defect-local SOAP to keep a diverse subset of generated unrelaxed base defects instead of random down-selection? [y/n, default: y or n depending on mode]:
```
What to type:
- `y` to use SOAP diversity pruning
- `n` to use random down-selection instead

Recommended for MLIP:
- `y`

If you answer `y`, you will then see:

### Prompt 16
```text
Defect-local SOAP cutoff r_cut in Å [5.0]:
```
What to type:
- usually `4.0` to `6.0`

### Prompt 17
```text
SOAP n_max [8]:
```
What to type:
- usually press Enter unless you know you want to change it

### Prompt 18
```text
SOAP l_max [6]:
```
What to type:
- usually press Enter unless you know you want to change it

### Prompt 19
```text
SOAP Gaussian width sigma in Å [0.5]:
```
What to type:
- usually `0.5`

For most users, the defaults are fine.

---

## 5.10 Base structure writing and rattled seeds

This part appears only in `seeded` and `mlip` modes.

### Prompt 20
```text
Also write the unperturbed base defect structures? [y/n, default: y]:
```
What to type:
- `y` if you want the clean vacancy structures saved too
- `n` if you only want perturbed seeds

Recommended:
- `y`

### Prompt 21
```text
How many rattled seeds per base defect [8]:
```
What to type:
- an integer such as `5`, `8`, or `10`

Meaning:
- how many perturbed copies to generate from each base defect

### Prompt 22
```text
Perturb atoms within this radius of each vacancy site (Å) [4.0]:
```
What to type:
- usually `4.0` to `5.0`

Meaning:
- only atoms near each removed site are displaced

### Prompt 23
```text
Smallest maximum random displacement used for a seed (Å) [0.02]:
```
What to type:
- a number like `0.02`

### Prompt 24
```text
Largest maximum random displacement used for a seed (Å) [0.08]:
```
What to type:
- a number like `0.05`, `0.08`, or `0.10`

Meaning of Prompts 23 and 24:
- the script creates several seed amplitudes between these values
- larger values mean stronger local perturbations

### Prompt 25
```text
Minimum-distance filter scale factor [0.55]:
```
What to type:
- usually `0.55`

### Prompt 26
```text
Absolute minimum allowed interatomic distance (Å) [0.80]:
```
What to type:
- usually `0.80`

Meaning of Prompts 25 and 26:
- these control the sanity filter that rejects obviously bad rattled structures with too-short distances

---

## 5.11 Output format and output directory

These prompts appear near the end.

### Prompt 27
```text
Also write extxyz files when ASE is available? [y/n, default: n]:
```
What to type:
- `y` if you want `.extxyz` files too
- `n` if `.vasp` files are enough

Recommended:
- `y` if you use ASE-based workflows
- `n` otherwise

### Prompt 28
```text
Output directory [POSCAR_292_Mg1_dataset]:
```
What to type:
- the directory name where output files should be written
- or press Enter to use the default

---

## 6. Very short prompt map by mode

## 6.1 `raw` mode
Typical prompt flow:
1. POSCAR filename
2. supercell questions
3. `symprec`
4. site-matching tolerance
5. defect recipe
6. mode = `raw`
7. random seed
8. balancing cap questions
9. optional SOAP pruning questions if cap is active and SOAP is available
10. extxyz question
11. output directory

## 6.2 `unique` mode
Typical prompt flow:
1. POSCAR filename
2. supercell questions
3. `symprec`
4. site-matching tolerance
5. defect recipe
6. mode = `unique`
7. random seed
8. balancing cap questions
9. optional SOAP pruning questions if cap is active and SOAP is available
10. extxyz question
11. output directory

## 6.3 `seeded` mode
Typical prompt flow:
1. POSCAR filename
2. supercell questions
3. `symprec`
4. site-matching tolerance
5. defect recipe
6. mode = `seeded`
7. random seed
8. balancing cap questions
9. optional SOAP pruning questions if cap is active and SOAP is available
10. write base structures?
11. seeds per base defect
12. perturb radius
13. smallest displacement
14. largest displacement
15. distance filter scale
16. distance floor
17. extxyz question
18. output directory

## 6.4 `mlip` mode
Typical prompt flow:
1. POSCAR filename
2. supercell questions
3. `symprec`
4. site-matching tolerance
5. defect recipe
6. mode = `mlip`
7. base source = `raw` or `unique`
8. random seed
9. balancing cap questions
10. optional SOAP pruning questions
11. write base structures?
12. seeds per base defect
13. perturb radius
14. smallest displacement
15. largest displacement
16. distance filter scale
17. distance floor
18. extxyz question
19. output directory

---

## 7. Recommended choices for a beginner

If you are unsure what to enter and you want a sensible MLIP-oriented run, this is a reasonable starting point:

- supercell: `y`
- target minimum lattice-vector length: `10.0` or `12.0`
- minimum atom count: optional, often `n`
- maximum allowed atoms: `400`
- `symprec`: `1e-3`
- site-matching tolerance: `1e-4`
- defect recipe: for example `Mg:1`
- mode: `mlip`
- MLIP base source: `unique`
- random seed: `12345`
- apply balancing cap: `y`
- cap per defect class: `20` to `50`
- use defect-local SOAP pruning: `y` if many structures exist
- SOAP `r_cut`: `5.0`
- SOAP `n_max`: `8`
- SOAP `l_max`: `6`
- SOAP `sigma`: `0.5`
- write unperturbed base defects: `y`
- seeds per base defect: `5` to `10`
- perturb radius: `4.0` to `5.0 Å`
- smallest displacement: `0.02 Å`
- largest displacement: `0.05` to `0.08 Å`
- minimum-distance filter scale: `0.55`
- absolute floor: `0.80 Å`
- write extxyz: `n` unless needed

---

## 8. Example terminal session

Here is a realistic example for a beginner who wants an MLIP-oriented dataset.

User input is shown after each prompt.

```text
Enter POSCAR filename [POSCAR_292]: POSCAR_292
Do you want to automatically build a supercell before creating vacancies? [y/n, default: y]: y
Target minimum lattice-vector length in Å [10.0]: 12.0
Also enforce a minimum total number of atoms? [y/n, default: n]: n
Maximum allowed total number of atoms [400]: 400
Symmetry tolerance symprec [0.001]:
Site-matching tolerance in fractional coordinates [0.0001]:
Defect recipe: Mg:1
Choose mode [raw/unique/seeded/mlip, default: mlip]: mlip
For mlip mode, use raw or unique base defects [raw/unique, default: unique]: unique
Random seed for down-selection and perturbations [12345]:
Apply a balancing cap to the number of base defect structures? [y/n, default: y]: y
Maximum number of base defect structures for this defect class [50]: 20
Use defect-local SOAP to keep a diverse subset of generated unrelaxed base defects instead of random down-selection? [y/n, default: y]: y
Defect-local SOAP cutoff r_cut in Å [5.0]:
SOAP n_max [8]:
SOAP l_max [6]:
SOAP Gaussian width sigma in Å [0.5]:
Also write the unperturbed base defect structures? [y/n, default: y]: y
How many rattled seeds per base defect [8]: 5
Perturb atoms within this radius of each vacancy site (Å) [4.0]: 4.0
Smallest maximum random displacement used for a seed (Å) [0.02]: 0.02
Largest maximum random displacement used for a seed (Å) [0.08]: 0.08
Minimum-distance filter scale factor [0.55]:
Absolute minimum allowed interatomic distance (Å) [0.80]:
Also write extxyz files when ASE is available? [y/n, default: n]: n
Output directory [POSCAR_292_Mg1_dataset]: Mg1_dataset
```

---

## 9. What files the script writes

Inside the output directory, the script writes:

### Structure files
- `.vasp` files for base structures and/or rattled seeds
- optionally `.extxyz` files if ASE export is enabled and available

Typical naming pattern:
- `POSCAR_292_Mg1_base0001.vasp`
- `POSCAR_292_Mg1_base0001_seed001.vasp`

### Metadata files
- `metadata.csv`
- `metadata.json`
- `summary.json`

---

## 10. What the metadata means

### `metadata.csv`
Spreadsheet-friendly record of all structures.

Important columns include:
- `record_type` = `base`, `seed`, or `rejected_seed`
- `defect_class` = for example `Mg1` or `Mg1+Se2`
- `combo_index` = which base vacancy combination this record belongs to
- `seed_index` = which rattled seed it is
- `canonical_hash` = exact parent-symmetry defect identifier
- `removed_indices` = raw removed atom indices
- `canonical_removed_indices` = symmetry-canonical removed indices
- `removed_species` = removed chemical species
- `removed_frac_coords` = removed-site fractional coordinates in the working structure
- `supercell_scaling` = supercell expansion used
- `distance_filter_passed` = whether the rattled seed passed the short-distance filter

### `summary.json`
Short summary of the whole run, including:
- mode
- defect recipe
- supercell scaling
- number of raw combinations
- number of written base structures
- number of written seeds
- number of rejected seeds
- whether SOAP diversity pruning was used

---

## 11. Scientifically important notes

### 11.1 About supercells
For real defect calculations, using a supercell is often much better than using the primitive cell directly, because small cells make the defect interact strongly with its periodic images.

### 11.2 About uniqueness
The script removes **exact parent-symmetry duplicates** only.
That means:
- if two vacancy patterns are identical under the symmetry of the ideal parent structure, they are treated as the same unrelaxed pattern
- this does **not** mean they must remain equivalent after relaxation

### 11.3 About rattled seeds
Rattled seeds are meant to generate multiple local starting environments near the defect.
These are useful when you later want DFT calculations to explore different local minima.

### 11.4 About the distance filter
The minimum-distance filter is only a **sanity check**.
It helps remove obviously bad structures, but it is not a full chemical model.

### 11.5 About relaxation
If you later relax these structures in DFT and want symmetry breaking to occur naturally, you should usually avoid imposing symmetry constraints. For VASP this often means:

```text
ISYM = 0
```

---

## 12. When to use which mode

Use `raw` when:
- you want every possible combination
- you do not care about symmetry duplicates at generation time

Use `unique` when:
- you only want one representative of each exact unrelaxed defect pattern

Use `seeded` when:
- you want symmetry-unique defects plus local perturbations
- you are preparing a small set of candidate structures for later relaxation

Use `mlip` when:
- you want a more complete dataset-generation workflow
- you want optional SOAP diversity pruning
- you want metadata and multiple seeds

---

## 13. Common mistakes and how to avoid them

### Mistake 1: wrong defect recipe format
Wrong:
```text
Mg 1
```
Correct:
```text
Mg:1
```

### Mistake 2: repeating the same species twice
Wrong:
```text
Mg:1, Mg:2
```
Correct:
```text
Mg:3
```

### Mistake 3: using an unrealistically small cell
If you skip the supercell for a defect study, your defect may interact too strongly with its periodic images.

### Mistake 4: choosing too many structures without a cap
If the raw combinatorics are large, you can generate an unmanageable number of outputs.
Use the balancing cap.

### Mistake 5: using large perturbations blindly
Very large random displacements can create many rejected structures or bad starting geometries.
For most cases, stay around `0.02` to `0.08 Å`.

---

## 14. One-sentence summary

This script builds vacancy structures from a parent crystal, removes exact duplicates using **parent-symmetry canonical hashing**, optionally keeps a **diverse subset using defect-local SOAP**, generates **local rattled seeds**, filters obviously bad ones, and writes the result with metadata so the dataset remains usable instead of turning into a folder-based cautionary tale.
