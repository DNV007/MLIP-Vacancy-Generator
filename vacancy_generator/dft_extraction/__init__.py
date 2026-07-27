"""Post-DFT results extraction (pipeline step 2).

Runs independently of, and after, the structure generation in
:mod:`vacancy_generator.main` / :mod:`vacancy_generator.migration_runner`.
Parses ``vasprun.xml`` files from the per-structure subfolders of a generation
run and writes a *separate* results table (``dft_results.csv`` /
``dft_results.json``) joined to the generation metadata by ``poscar_file``.
"""

from __future__ import annotations

from .extract import extract_one
from .records import DftResultRecord

__all__ = ["DftResultRecord", "extract_one"]
