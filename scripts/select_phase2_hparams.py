"""Select Phase 2 hyperparameters on validation loss at reserved dev split seeds.

    python scripts/select_phase2_hparams.py

Writes ``results/phase2/hparams.json``, which the experiment runner reads.

Selection never touches the evaluation pool. Dev split seeds (100, 101) are
disjoint from the evaluation split seeds (0-7), and within a dev seed the score
is the *validation* loss on pairs carved out of that seed's training pool. Each
family gets its own best setting, which is the arrangement most favourable to
the baselines: the structured model is given no tuning advantage it does not
also hand to the model it is being compared against.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from intervention_algebra.real_data.experiment import Phase2Config
from intervention_algebra.real_data.sweep import COVERAGES, FAMILIES, run_sweep
from intervention_algebra.real_data.train import TrainConfig

DEV_SPLIT_SEEDS = (100, 101)
#: Tuned at the sparse coverage only. The hypothesis under test is about the
#: sparse regime, and including a dense coverage lets the cheap-to-fit dense
#: cell -- where every family does well and the settings barely matter -- outvote
#: the cell the claim lives in. Applied identically to every family, so it
#: cannot favour one.
DEV_COVERAGES = (0.10,)
# lr=1e-3 was dropped after a pilot: under the cosine schedule it was still
# improving at the end of the budget for every family, so it measured the
# budget rather than the setting.
LR = (3e-3, 1e-2, 3e-2)
WEIGHT_DECAY = (1e-4, 1e-3, 1e-2)
EMB_DIM = (16, 32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/phase2/hparams.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/koplev2017"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    specs = []
    for fam in FAMILIES:
        for lr, wd, emb in itertools.product(LR, WEIGHT_DECAY, EMB_DIM):
            for ss in DEV_SPLIT_SEEDS:
                for cov in DEV_COVERAGES:
                    specs.append(Phase2Config(
                        screen="A375", coverage=cov, family=fam, split_seed=ss,
                        init_seed=0, coverages=COVERAGES, emb_dim=emb,
                        train=TrainConfig(lr=lr, weight_decay=wd, n_restarts=1),
                        tag="hpsel"))
    print(f"{len(specs)} dev runs")
    rows = run_sweep(specs, args.out.with_suffix(".jsonl"), args.raw, args.workers)

    ok = [r for r in rows if "error" not in r]
    chosen = {}
    for fam in FAMILIES:
        best, best_val = None, float("inf")
        for lr, wd, emb in itertools.product(LR, WEIGHT_DECAY, EMB_DIM):
            v = [r["val_loss"] for r in ok
                 if r["family"] == fam and r["config"]["emb_dim"] == emb
                 and r["config"]["train"]["lr"] == lr
                 and r["config"]["train"]["weight_decay"] == wd]
            if len(v) == len(DEV_SPLIT_SEEDS) * len(DEV_COVERAGES) and np.mean(v) < best_val:
                best, best_val = {"lr": lr, "weight_decay": wd, "emb_dim": emb}, float(np.mean(v))
        if best is None:
            raise RuntimeError(f"no complete hyperparameter cell for {fam}")
        best["dev_val_loss"] = best_val
        chosen[fam] = best
        print(f"  {fam:18s} {best}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "selected_on": "mean validation loss",
        "dev_split_seeds": list(DEV_SPLIT_SEEDS),
        "dev_coverages": list(DEV_COVERAGES),
        "grid": {"lr": list(LR), "weight_decay": list(WEIGHT_DECAY),
                 "emb_dim": list(EMB_DIM)},
        "n_dev_runs": len(specs),
        "hparams": chosen,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
