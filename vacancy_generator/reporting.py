"""Output summaries, dataset statistics, and SOAP distance reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure

from ._compat import ASE_ADAPTOR_AVAILABLE, DSCRIBE_AVAILABLE
from .diversity import build_soap_descriptor, defect_local_soap_vector
from .structure_ops import remove_sites_from_structure
from .structure_utils import get_species_counts


def describe_output_summary(records: List[dict]) -> None:
    """Print a short summary of what was written to disk."""
    print("\nGeneration summary:")
    if not records:
        print("  No structures were written.")
        return

    by_type: Dict[str, int] = defaultdict(int)
    for record in records:
        by_type[record["record_type"]] += 1

    for key in sorted(by_type):
        print(f"  {key}: {by_type[key]}")

    written = sum(1 for r in records if not r["record_type"].startswith("rejected_"))
    print(f"  Total written structures: {written}")


def build_dataset_report(records: List[dict], summary: dict) -> dict:
    """Build a compact report of uniqueness and diversity statistics."""
    report: dict = {
        "total_records": len(records),
        "written_records": sum(1 for r in records if not r["record_type"].startswith("rejected_")),
        "record_type_counts": dict(Counter(r["record_type"] for r in records)),
        "unique_canonical_hashes": len(
            {r["canonical_hash"] for r in records if r.get("canonical_hash")}
        ),
        "canonical_hash_counts": dict(
            Counter(r["canonical_hash"] for r in records if r.get("canonical_hash"))
        ),
        "vacancy_topology_counts": dict(
            Counter(r.get("vacancy_topology", "") for r in records if r.get("vacancy_topology"))
        ),
        "seeded_or_mlip_with_soap": bool(summary.get("defect_local_soap_diversity_used")),
        "soap_settings": summary.get("soap_settings"),
    }

    migration_records = [r for r in records if r["record_type"].startswith("path_")]
    if migration_records:
        report["migration_target_family_counts"] = dict(
            Counter(
                r.get("migration_target_family", "")
                for r in migration_records
                if r.get("migration_target_family")
            )
        )
        report["migration_path_index_counts"] = dict(
            Counter(
                f"{r.get('migration_target_family', '')}_{r.get('path_index', '')}"
                for r in migration_records
            )
        )
    return report


def build_soap_distance_report(
    working_structure: Structure,
    base_combos: List[Tuple[int, ...]],
    combo_to_hash: Dict[Tuple[int, ...], str],
    species: Sequence[str],
    soap_config: dict,
) -> Tuple[List[dict], dict]:
    """Compute pairwise SOAP distances for selected base defects."""
    empty_stats: dict = {
        "soap_distance_pairs": 0,
        "soap_distance_min": None,
        "soap_distance_mean": None,
        "soap_distance_max": None,
    }

    if len(base_combos) < 2:
        return [], empty_stats
    if not (DSCRIBE_AVAILABLE and ASE_ADAPTOR_AVAILABLE):
        return [], empty_stats

    defect_structures = [remove_sites_from_structure(working_structure, combo)
                         for combo in base_combos]
    soap = build_soap_descriptor(
        species,
        r_cut=soap_config["r_cut"],
        n_max=soap_config["n_max"],
        l_max=soap_config["l_max"],
        sigma=soap_config["sigma"],
    )
    vectors = [
        defect_local_soap_vector(defect_structure, working_structure, combo, soap)
        for defect_structure, combo in zip(defect_structures, base_combos)
    ]

    rows: List[dict] = []
    distances: List[float] = []
    for i in range(len(base_combos)):
        for j in range(i + 1, len(base_combos)):
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            distances.append(dist)
            rows.append({
                "index_i": i,
                "index_j": j,
                "combo_i": list(base_combos[i]),
                "combo_j": list(base_combos[j]),
                "canonical_hash_i": combo_to_hash.get(base_combos[i], ""),
                "canonical_hash_j": combo_to_hash.get(base_combos[j], ""),
                "soap_distance": dist,
            })

    stats = {
        "soap_distance_pairs": len(rows),
        "soap_distance_min": float(min(distances)) if distances else None,
        "soap_distance_mean": float(np.mean(distances)) if distances else None,
        "soap_distance_max": float(max(distances)) if distances else None,
    }
    return rows, stats
