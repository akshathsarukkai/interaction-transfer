"""Audit the Koplev deposit and emit the countable dataset description.

    python scripts/prepare_koplev.py

Writes ``results/phase2/dataset_audit.json``: provenance, the table-usability
re-derivation, the publication-statistic reproduction, per-screen counts, and
the connectivity of the coverage splits the experiment will use. Nothing here
fits a model; it exists so that the numbers in ``docs/phase2_dataset.md`` are
generated rather than typed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.splits import (connectivity_report,
                                                   make_coverage_splits)
from intervention_algebra.real_data.sweep import COVERAGES, SCREENS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=koplev.DEFAULT_RAW_DIR)
    ap.add_argument("--out", type=Path,
                    default=Path("results/phase2/dataset_audit.json"))
    ap.add_argument("--split-seeds", type=int, default=8)
    args = ap.parse_args()

    audit = {
        "provenance": {
            "paper_doi": koplev.KOPLEV_PAPER, "data_doi": koplev.KOPLEV_DOI,
            "license": koplev.KOPLEV_LICENSE, "acquired": koplev.ACQUIRED,
            "files": {k: {"filename": s.filename, "sha256": s.sha256,
                          "size": s.size,
                          "deposit_description": s.deposit_description}
                      for k, s in koplev.SOURCES.items()},
        },
        "table_usability": koplev.verify_raw(args.raw),
        "publication_statistics": koplev.reproduce_publication_stats(args.raw),
        "uncertainty_reference": koplev.measurement_noise_sd(args.raw),
        "coverage_grid": list(COVERAGES),
        "screens": {},
    }

    for label in SCREENS:
        screen = koplev.load_screen(label, args.raw)
        entry = koplev.audit_screen(screen)
        conn = []
        for seed in range(args.split_seeds):
            splits = make_coverage_splits(screen.frame, screen.n_drugs,
                                          COVERAGES, split_seed=seed)
            for cov, sp in sorted(splits.items()):
                conn.append(connectivity_report(sp, screen.n_drugs))
        entry["split_connectivity"] = conn
        audit["screens"][label] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n")

    for label, e in audit["screens"].items():
        print(f"{label}: {e['rows_raw']} raw rows -> {e['rows_used']} used "
              f"({e['rows_self_pairs_removed']} self-combinations removed), "
              f"{e['n_drugs']} drugs, {e['n_unordered_pairs']} unordered pairs, "
              f"{e['n_pairs_with_both_directions']} with both directions, "
              f"antisymmetric variance share {e['var_antisymmetric_share']:.3f}")
    ps = audit["publication_statistics"]
    print(f"publication counts reproduce: {ps['all_counts_match']} "
          f"({ps['total_high_confidence']}/{ps['total_high_confidence_published']} "
          f"high-confidence, {ps['fraction_of_all_combinations']:.3f} of all)")
    print(f"wrote {args.out}")
    return 0 if ps["all_counts_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
