"""Run the Phase 1 experiment suite and write machine-readable results.

    python scripts/run_phase1.py --out results/phase1.jsonl          # everything
    python scripts/run_phase1.py --part main                         # headline only
    python scripts/run_phase1.py --smoke                             # seconds, for CI

The grid itself lives in ``intervention_algebra.phase1`` -- this script only
decides which part of it to run and where the JSONL lands.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from intervention_algebra import phase1
from intervention_algebra.sweep import run_sweep

PARTS = {
    "main": phase1.main_sweep,
    "ceiling": phase1.ceiling_sweep,
    "regime": phase1.regime_sweep,
    "control": phase1.control_sweep,
    "misspec": phase1.misspecification_sweep,
    "power": phase1.power_sweep,
    "closure": phase1.closure_sweep,
    "all": phase1.all_specs,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=sorted(PARTS), default="all")
    ap.add_argument("--out", default="results/phase1.jsonl")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override the seed basis (default: whatever the chosen part declares)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid to check the pipeline end to end")
    args = ap.parse_args(argv)

    if args.smoke:
        specs = phase1.smoke_specs()
        out = Path(args.out).with_name("smoke.jsonl")
    else:
        # Only override the seed basis when the caller asks for it. Passing
        # phase1.REPORT_SEEDS unconditionally silently overrode each sweep's own
        # declared default -- `--part power` would have run seeds 0-4 instead of
        # POWER_SEEDS, and `--part closure` 5 seeds instead of 17, in both cases
        # producing a plausible-looking file with the wrong n.
        specs = (PARTS[args.part](tuple(args.seeds)) if args.seeds
                 else PARTS[args.part]())
        out = Path(args.out)

    part = "smoke" if args.smoke else args.part
    print(f"part={part} runs={len(specs)} workers={args.workers} -> {out}",
          file=sys.stderr)
    t0 = time.time()
    rows = run_sweep(specs, out_path=out, workers=args.workers)
    ok = sum("error" not in r for r in rows)
    print(f"done: {ok}/{len(rows)} succeeded in {time.time()-t0:.0f}s -> {out}",
          file=sys.stderr)

    # A run that ends at its epoch cap was still improving when the budget ran
    # out. Comparing families at such a point measures the cap, not the
    # inductive bias, so this is loud rather than a footnote.
    # Only meaningful when early stopping is enabled. Under patience ==
    # max_epochs every run reaches the cap by design and this is not a warning.
    stopping_on = any(r.get("config", {}).get("train", {}).get("patience", 0)
                      < r.get("config", {}).get("train", {}).get("max_epochs", 0)
                      for r in rows if "error" not in r)
    capped = [] if not stopping_on else [r for r in rows if "error" not in r
              and r.get("epochs_run", 0) >= r.get("config", {})
              .get("train", {}).get("max_epochs", 10**9)]
    if not stopping_on:
        print("early stopping disabled (patience == max_epochs): every run "
              "trains the full budget by design", file=sys.stderr)
    if capped:
        # Per-family counts, not a total: caps spread evenly across families are
        # a footnote, whereas caps concentrated in one family bias the headline
        # comparison in that family's disfavour.
        by_family: dict = {}
        for r in capped:
            by_family[r.get("family")] = by_family.get(r.get("family"), 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_family.items()))
        print(f"!! WARNING: {len(capped)}/{ok} runs ended at their epoch cap "
              f"({breakdown}). They were still improving. If the counts are "
              f"concentrated in one family the comparison is biased against it; "
              f"raise TRAIN_BASE['max_epochs'] or exclude whole "
              f"(coverage, seed) cells symmetrically "
              f"(analysis.flag_capped_runs).", file=sys.stderr)
    else:
        print("no run ended at its epoch cap", file=sys.stderr)
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
