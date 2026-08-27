#!/usr/bin/env python
"""Run the Phase 3 entity-level OOD sweep.

The unit held out is a DRUG. No Koplev measurement involving a test drug enters
training, validation, feature preprocessing, target scaling, hyperparameter
selection or shrinkage calibration; the only thing the model knows about it is
its molecular structure (or, in the secondary arm, its annotated targets).

    python scripts/run_phase3_entity_ood.py --counts
    python scripts/run_phase3_entity_ood.py --part smoke --workers 2
    python scripts/run_phase3_entity_ood.py --part primary
    python scripts/run_phase3_entity_ood.py --part controls,targets,coverage
    python scripts/run_phase3_entity_ood.py --part all

Run counts are derived from ``entity_ood.sweep.PART_GRIDS`` and printed by
``--counts``; they are deliberately not written into any docstring or document,
because that is exactly the number that goes stale.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.entity_ood import sweep

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "phase3_entity_ood"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="primary",
                    help="comma-separated: " + ", ".join(sorted(sweep.PART_GRIDS)) + ", all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--raw-dir", type=Path, default=koplev.DEFAULT_RAW_DIR)
    ap.add_argument("--mapping", type=Path, default=None,
                    help="drug mapping CSV; point at tests/fixtures/koplev_tiny/"
                         "drug_mapping.csv together with --raw-dir to run the "
                         "pipeline check without the deposit")
    ap.add_argument("--fold-geometry", type=int, nargs=3, metavar=("PARTITIONS", "TEST", "VAL"),
                    default=None, help="override n_partitions, n_test, n_val "
                                       "(the fixture has 12 drugs, not 100)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N conditions of each block (a dry run)")
    ap.add_argument("--counts", action="store_true",
                    help="print the number of conditions per part and exit")
    args = ap.parse_args(argv)

    if args.counts:
        for part, n in sweep.part_counts().items():
            print(f"{part:12s} {n:5d}")
        return 0

    # Built before the pool starts. An earlier version of the Phase 2N runner
    # built its jobs *after* first use of the variable and died instantly with an
    # UnboundLocalError; the detached process then sat dead for four hours while
    # being reported as running. The ordering is pinned by a test that stubs the
    # pool and asserts every advertised part reaches it.
    jobs: list[tuple[str, list]] = []
    for part in [p.strip() for p in args.part.split(",") if p.strip()]:
        jobs.extend(sweep.part_jobs(part))

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_failed_total = 0
    for name, specs in jobs:
        if args.fold_geometry:
            p_, t_, v_ = args.fold_geometry
            specs = [replace(s, n_partitions=p_, n_test=t_, n_val=v_) for s in specs]
        # `--limit 0` must mean "zero conditions", not "no limit". `if
        # args.limit:` reads 0 as falsy and would silently launch the whole
        # block -- which is how a dry-run flag once started a real ensemble.
        if args.limit is not None:
            specs = specs[:args.limit]
        out = args.outdir / f"{name}.jsonl"
        print(f"[{name}] {len(specs)} conditions -> {out}", flush=True)
        rows = sweep.run_entity_sweep(specs, out, raw_dir=args.raw_dir,
                                      mapping_path=args.mapping,
                                      workers=args.workers)
        nerr = sum(1 for r in rows if "error" in r)
        n_failed_total += nerr
        print(f"[{name}] wrote {len(rows)} rows, {nerr} failed", flush=True)
        if nerr:
            for r in rows:
                if "error" in r:
                    print(r["error"], flush=True)
                    break
    # A runner that reports failures and then exits 0 is how a CI step goes green
    # on a sweep in which nothing worked. Found by running the pipeline check on
    # a checkout with no deposit: every condition raised, and the exit code said
    # success.
    return 1 if n_failed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
