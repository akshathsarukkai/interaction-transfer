"""Run the residual-directionality diagnostic on the Koplev sequential screen.

    python scripts/download_koplev.py                       # once
    python scripts/run_phase2_residual.py --part main
    python scripts/run_phase2_residual.py --part controls
    python scripts/run_phase2_residual.py --part power
    python scripts/run_phase2_residual.py --part sensitivity
    python scripts/report_phase2_residual.py

Parts. Run counts are not written here: they are derived from the grid
functions by ``residual_sweep.part_counts()`` and printed by ``--counts``,
because a hard-coded total is what went stale the last time a grid grew.

    --part main         every rung x coverage x screen, 8 split seeds
    --part sensitivity  the two decision coverages with the ridge penalty chosen
                        to maximise directional removal
    --part controls     permuted-residual control (coverages 0.10/0.40/0.70) +
                        the contaminated-fit diagnostic
    --part robustness   three corrections the audit asked for: the shrinkage
                        refitted on validation pairs withheld from model
                        selection, the low-rank rung pinned to rank 2 with no
                        search, and a ridge titration proving the potential
                        rung is a working detector rather than a dead one
    --part power        positive control: known antisymmetric signal injected at
                        two strengths, so a null can be told apart from a lack of
                        power -- under both shrinkage estimators
    --part all          every part above, robustness included
    --part smoke        a two-minute pipeline check, not a result
    --counts            print the run count of every part and exit

There is no ``--part tune``. Hyperparameters are selected inside every run, on
that run's own validation pairs, from the small grid in
``residual_models.HPARAM_GRID``. Phase 2 selected once at one coverage on one
screen and reused the answer everywhere, and the audit showed that choice did
not transfer and handicapped the family under test; a separate tuning stage is
the thing that made that failure possible, so there is not one here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from intervention_algebra.real_data.koplev import DEFAULT_RAW_DIR
from intervention_algebra.real_data.residual_sweep import (
    main_grid, part_counts, part_jobs, run_residual_sweep)

OUT_DIR = Path("results/phase2_residual")

#: One file per part. The contaminated rows get their own file because they are
#: not a result -- keeping them in the same JSONL as the main grid would make
#: "filter on the contaminated flag" the only thing standing between a reader
#: and a leaked number.
OUTPUTS = {
    "main": OUT_DIR / "runs.jsonl",
    "sensitivity": OUT_DIR / "sensitivity.jsonl",
    "controls": OUT_DIR / "controls.jsonl",
    "contaminated": OUT_DIR / "contaminated_diagnostic.jsonl",
    "power": OUT_DIR / "power.jsonl",
    "honest_alpha": OUT_DIR / "honest_alpha.jsonl",
    "rank2": OUT_DIR / "rank2.jsonl",
    "titration": OUT_DIR / "ridge_titration.jsonl",
    "power_honest": OUT_DIR / "power_honest_alpha.jsonl",
    "smoke": OUT_DIR / "smoke.jsonl",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all",
                    choices=("main", "sensitivity", "controls", "power",
                             "robustness", "all", "smoke"))
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--counts", action="store_true",
                    help="print the run count of every part and exit")
    args = ap.parse_args()

    if args.counts:
        for part, n in part_counts().items():
            print(f"{part:12s} {n:5d}")
        return 0

    if args.part == "smoke":
        jobs: list[tuple[str, list, Path]] = [
            ("smoke", main_grid(screens=("A375",), coverages=(0.10,),
                                rungs=("zero", "potential", "lowrank"),
                                split_seeds=(0,), tag="smoke"),
             OUTPUTS["smoke"])]
    else:
        # Built from ``residual_sweep.PART_GRIDS`` rather than an if-ladder here,
        # so "what does --part all include" has exactly one answer in the
        # codebase and the documented totals are derived from the same object.
        jobs = [(name, specs, OUTPUTS[name])
                for name, specs in part_jobs(args.part)]

    status = 0
    for name, specs, out in jobs:
        print(f"part={name} runs={len(specs)} -> {out}", flush=True)
        t0 = time.time()
        rows = run_residual_sweep(specs, out, args.raw, args.workers)
        failed = [r for r in rows if "error" in r]
        print(f"  done in {time.time() - t0:.0f}s; {len(rows)} rows, "
              f"{len(failed)} failed", flush=True)
        if failed:
            print(failed[0]["error"][-1500:])
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
