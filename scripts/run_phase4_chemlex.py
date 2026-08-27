#!/usr/bin/env python
"""Run the Phase 4 ChemLex entity-OOD sweep.

    python scripts/run_phase4_chemlex.py --counts
    python scripts/run_phase4_chemlex.py --part smoke        # pipeline check
    python scripts/run_phase4_chemlex.py --part all --workers 6

Parts write to separate files under ``results/phase4_chemlex/`` so a re-run of
one arm cannot silently truncate another. ``--part smoke`` writes to its own
gitignored ``smoke.jsonl``: a pipeline check is not a result.

Exits **non-zero** if any condition failed. The Phase 3 runner returned 0 after
printing "6 failed", which is exactly how a CI step goes green on a sweep in
which nothing worked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intervention_algebra.real_data.chemlex.dataset import DEFAULT_RAW_DIR
from intervention_algebra.real_data.chemlex.sweep import (PART_GRIDS,
                                                          part_counts,
                                                          part_jobs, run_sweep)

DEFAULT_OUT = Path("results/phase4_chemlex")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="primary",
                    choices=sorted(PART_GRIDS) + ["all"])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--counts", action="store_true",
                    help="print the condition counts and exit")
    args = ap.parse_args()

    if args.counts:
        for k, v in sorted(part_counts().items()):
            print(f"{k:14s} {v:5d}")
        return 0

    failed = 0
    total = 0
    for name, specs in part_jobs(args.part):
        out = args.outdir / f"{name}.jsonl"
        per_entity = (args.outdir / f"{name}_per_entity.csv"
                      if name != "smoke" else None)
        print(f"[{name}] {len(specs)} conditions -> {out}", flush=True)
        rows = run_sweep(specs, out, raw_dir=args.raw_dir,
                         workers=args.workers, per_entity_out=per_entity)
        n_err = sum(1 for r in rows if "error" in r)
        total += len(rows)
        failed += n_err
        print(f"[{name}] {len(rows)} rows, {n_err} failed", flush=True)
        for r in rows:
            if "error" in r:
                print(f"  FAILED {r.get('key')}\n"
                      f"{r['error'].rstrip()}", file=sys.stderr)

    print(f"\n{total} conditions attempted, {failed} failed")
    if failed:
        print("the sweep did not complete cleanly", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
