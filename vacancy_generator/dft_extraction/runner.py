"""CLI entry point: harvest finished VASP results from a generation output tree.

Usage::

    python -m vacancy_generator.dft_extraction.runner <output_dir>

``output_dir`` is the directory a previous generation run wrote: one immediate
subdirectory per structure, each containing ``POSCAR``, a ``<subfolder>.json``
sidecar with that structure's generation metadata, possibly symlinked shared DFT
inputs, and — once the job has run — ``vasprun.xml``.

Writes ``dft_results.csv`` and ``dft_results.json`` into ``output_dir``. These
are a *separate* table from the generation metadata, joined by ``poscar_file``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple

from ..io import save_json, save_metadata_csv
from .extract import extract_one
from .records import DFT_FIELD_ORDER, DftResultRecord

RESULTS_CSV = "dft_results.csv"
RESULTS_JSON = "dft_results.json"


def find_structure_dirs(output_dir: str) -> List[Tuple[str, str]]:
    """Return ``(structure_dir, sidecar_json_path)`` pairs found in ``output_dir``.

    Scans exactly one level deep: every immediate subdirectory holding a sidecar
    named ``<subdirectory name>.json`` is a structure folder. Subdirectories
    without that sidecar are unrelated (or leftover) and are silently skipped
    rather than treated as failed calculations.
    """
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"not a directory: {output_dir}")

    pairs: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(output_dir)):
        structure_dir = os.path.join(output_dir, name)
        if not os.path.isdir(structure_dir):
            continue
        sidecar = os.path.join(structure_dir, f"{name}.json")
        if os.path.isfile(sidecar):
            pairs.append((structure_dir, sidecar))
    return pairs


def extract_all(output_dir: str) -> List[DftResultRecord]:
    """Extract a result record for every structure folder under ``output_dir``."""
    records: List[DftResultRecord] = []
    for structure_dir, sidecar in find_structure_dirs(output_dir):
        try:
            records.append(extract_one(structure_dir, sidecar))
        except ValueError as exc:
            # Unusable sidecar: the join key is unknown, so no meaningful row
            # can be written. Warn and continue instead of aborting the run.
            print(f"  warning: skipping {structure_dir}: {exc}")
    return records


def describe_extraction_summary(records: List[DftResultRecord]) -> None:
    """Print a short summary of what was extracted."""
    print("\nDFT extraction summary:")
    if not records:
        print("  No structure folders with generation metadata were found.")
        return

    by_status = Counter(record.status for record in records)
    for status in sorted(by_status):
        print(f"  {status}: {by_status[status]}")
    print(f"  Total structures scanned: {len(records)}")


def run_extraction(output_dir: str) -> List[DftResultRecord]:
    """Extract every result under ``output_dir`` and write the results table."""
    records = extract_all(output_dir)
    if records:
        save_metadata_csv(
            records, os.path.join(output_dir, RESULTS_CSV), field_order=DFT_FIELD_ORDER
        )
        save_json(records, os.path.join(output_dir, RESULTS_JSON))
        print(f"\nWrote {RESULTS_CSV} and {RESULTS_JSON} to {output_dir}")
    describe_extraction_summary(records)
    return records


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run the extraction. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m vacancy_generator.dft_extraction.runner",
        description="Collect finished VASP results from a generation output directory.",
    )
    parser.add_argument(
        "output_dir",
        help="Generation output directory containing one subfolder per structure.",
    )
    args = parser.parse_args(argv)

    try:
        run_extraction(args.output_dir)
    except NotADirectoryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
