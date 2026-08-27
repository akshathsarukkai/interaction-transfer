"""Run the Phase 2 real-data experiment on the Koplev sequential screen.

    python scripts/download_koplev.py            # once
    python scripts/select_phase2_hparams.py      # once (writes hparams.json)
    python scripts/run_phase2.py                 # the experiment
    python scripts/report_phase2.py              # tables and figures

Parts:
    --part main       every family x coverage x screen, 8 split seeds (default)
    --part control    the direction-shuffle control
    --part initvar    extra initialisation seeds, to size optimiser variance
    --part parity     both pair families across the learning-rate grid, so the
                      family contrast can be recomputed with each family at its
                      own validation-selected rate (writes tuning_parity.jsonl)
    --part all        main + control + initvar
    --part smoke      a two-minute pipeline check, not a result, written to
                      --out results/phase2/smoke.jsonl (gitignored)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from intervention_algebra.real_data.koplev import DEFAULT_RAW_DIR
from intervention_algebra.real_data.sweep import (COVERAGES, control_grid,
                                                  main_grid, parity_grid,
                                                  run_sweep)
from intervention_algebra.real_data.train import TrainConfig

SPLIT_SEEDS = tuple(range(8))
#: One initialisation seed in the headline grid. That is not an assumption that
#: optimiser variance is zero: every run already fits ``n_restarts`` independent
#: initialisations and keeps the best on validation loss, so init variation is
#: inside a run rather than across runs. It is *also* measured directly by
#: ``--part initvar``, which repeats one screen at four init seeds, so the claim
#: "split-seed variance dominates" is checked rather than asserted.
INIT_SEEDS = (0,)
INITVAR_SEEDS = (0, 1, 2, 3)


def load_hparams(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/select_phase2_hparams.py` "
            f"first -- running with untuned defaults would make the family "
            f"comparison a comparison of optimiser settings.")
    return json.loads(path.read_text())["hparams"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part",
                    choices=("main", "control", "initvar", "parity", "all",
                             "smoke"),
                    default="all")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--hparams", type=Path,
                    default=Path("results/phase2/hparams.json"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.part == "smoke":
        specs = main_grid(screens=("A375",), coverages=(0.10, 0.40),
                          split_seeds=(0,), init_seeds=(0,),
                          hparams={},
                          train=TrainConfig(max_epochs=200, patience=50,
                                            n_restarts=1))
        out = args.out or Path("results/phase2/smoke.jsonl")
    else:
        hp = load_hparams(args.hparams)
        specs = []
        if args.part in ("main", "all"):
            specs += main_grid(coverages=COVERAGES, split_seeds=SPLIT_SEEDS,
                               init_seeds=INIT_SEEDS, hparams=hp)
        if args.part in ("control", "all"):
            specs += control_grid(split_seeds=SPLIT_SEEDS, hparams=hp)
        if args.part in ("initvar", "all"):
            # Two rungs, but scored against the *full* nested split via
            # `pool_coverages`. The distinction is not cosmetic: an earlier
            # version passed `coverages=(0.10, 0.40)` alone, which also moved
            # `Phase2Config.coverages` and so scored these runs on a different,
            # larger evaluation pool (2,970 pairs, nothing dropped for
            # connectivity) than the headline's 848-1,003. That inflates the
            # apparent dominance of split-seed variance over init variance --
            # exactly the direction that flatters the claim this block exists to
            # check. `control_grid` hardcodes the full grid for the same reason.
            specs += main_grid(screens=("A375",), coverages=(0.10, 0.40),
                               pool_coverages=COVERAGES,
                               split_seeds=(0, 1), init_seeds=INITVAR_SEEDS,
                               hparams=hp, tag="initvar")
        if args.part == "parity":
            specs = parity_grid(split_seeds=SPLIT_SEEDS, hparams=hp)
        out = args.out or Path(
            "results/phase2/tuning_parity.jsonl" if args.part == "parity"
            else "results/phase2/runs.jsonl")

    print(f"part={args.part} runs={len(specs)} -> {out}", flush=True)
    t0 = time.time()
    rows = run_sweep(specs, out, args.raw, args.workers)
    failed = [r for r in rows if "error" in r]
    print(f"done in {time.time() - t0:.0f}s; {len(rows)} rows, {len(failed)} failed")
    if failed:
        print(failed[0]["error"][-1500:])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
