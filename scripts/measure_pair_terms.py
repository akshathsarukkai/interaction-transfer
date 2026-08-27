#!/usr/bin/env python
"""How much does each fitted interaction term actually vary?

    python scripts/measure_pair_terms.py

A term that never leaves its zero initialisation reports exactly 0.0 incremental
skill, and in a results table that is indistinguishable from a genuine finding
of no benefit. The flexible comparator does exactly that on this screen, and
nothing measured it until an adversarial reviewer refitted the models by hand.

Refits the ladder on the authoritative folds -- same seed, same grid, same
selection on entity-OOD validation -- and records the standard deviation of each
rung's interaction term on the rows it is scored on. Writes
`results/phase4_chemlex/pair_terms.jsonl` and
`results/phase4_chemlex/summary/pair_terms.md`.

A separate artifact rather than a column in `primary.jsonl`, so that the
committed results stay exactly what `run()` produced and both files remain
regenerable from the committed code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intervention_algebra.real_data.chemlex import dataset as ds
from intervention_algebra.real_data.chemlex import experiment as ex
from intervention_algebra.real_data.chemlex import report as rp
from intervention_algebra.real_data.chemlex.models import ModelConfig
from intervention_algebra.real_data.chemlex.splits import SELECT_BUCKETS
from intervention_algebra.real_data.chemlex.train import (TrainConfig,
                                                          fit_scaling,
                                                          make_batch,
                                                          train_with_grid)

RUNGS = ("lowrank", "flexible", "condition_expanded_pair")


def pair_term_sd(model, sub: pd.DataFrame) -> float:
    pair = getattr(model, "pair", None)
    if pair is None:
        return float("nan")

    def _long(v):
        return torch.as_tensor(np.ascontiguousarray(v).copy(), dtype=torch.long)

    a, n, c = _long(sub["acid"]), _long(sub["amine"]), _long(sub["cond"])
    model.eval()
    with torch.no_grad():
        try:
            out = pair(a, n, c)
        except TypeError:
            out = pair(a, n)
    return float(out.std())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screens", nargs="+", default=["hatu", "all"])
    ap.add_argument("--folds", type=int, default=3,
                    help="folds per screen; the term is either dead or it is not")
    ap.add_argument("--outdir", type=Path, default=rp.RESULTS)
    args = ap.parse_args()
    torch.set_num_threads(4)
    raw = ds.load_raw()

    rows = []
    for screen in args.screens:
        for fold in range(args.folds):
            spec = ex.Spec(block="primary", screen=screen, endpoint="yield",
                           partition=0, fold=fold)
            prep = ex.prepare(spec, raw=raw)
            fo = ex.fold_of(prep, spec)
            f = prep.screen.frame
            y = f["y"].to_numpy()
            tr, va = fo.mask("train"), fo.mask(SELECT_BUCKETS)
            scaling = fit_scaling(y[tr])
            btr = make_batch(f["acid"][tr], f["amine"][tr], f["cond"][tr],
                             y[tr], scaling)
            bva = make_batch(f["acid"][va], f["amine"][va], f["cond"][va],
                             y[va], scaling)
            cfg = ModelConfig(n_acids=len(prep.screen.acids),
                              n_amines=len(prep.screen.amines),
                              n_conditions=len(prep.screen.conditions),
                              x_acid=prep.acid_fp.x, x_amine=prep.amine_fp.x)
            sub = f.loc[fo.mask("test_e1a")]
            for rung in RUNGS:
                fit = train_with_grid(rung, cfg, btr, bva, TrainConfig(),
                                      seed=ex.hash_seed(spec),
                                      grid=ex._grid_for(rung))
                sd = pair_term_sd(fit.model, sub)
                rows.append({"key": f"pair_terms/{screen}/{rung}/p0f{fold}",
                             "block": "pair_terms", "screen": screen,
                             "representation": "ecfp4", "endpoint": "yield",
                             "fold": fold, "rung": rung,
                             "pair_term_sd": sd,
                             "target_sd": float(sub["y"].std()),
                             "weight_decay": fit.hparams.get("weight_decay"),
                             "rank": fit.hparams.get("rank"),
                             "best_epoch": fit.best_epoch,
                             "val_loss": fit.val_loss})
                print(f"{screen} f{fold} {rung:24s} sd={sd:.3e}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "pair_terms.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    d = pd.DataFrame(rows)
    table = rp.table([
        {"screen": s, "rung": r,
         "folds": int(len(g)),
         "median pair-term sd": f"{g['pair_term_sd'].median():.3e}",
         "target sd": f"{g['target_sd'].mean():.4f}",
         "ratio to target": f"{g['pair_term_sd'].median() / g['target_sd'].mean():.3e}",
         # Relative, not absolute. An absolute floor called a term "alive" at
         # 3e-5 against a target of 0.30 -- four orders of magnitude below the
         # thing it is supposed to predict, which is dead in every sense that
         # matters. A term must move at least a thousandth of the target's
         # scale to be doing anything at all.
         "alive": ("yes" if g["pair_term_sd"].median()
                   > 1e-3 * g["target_sd"].mean() else "**no**")}
        for (s, r), g in d.groupby(["screen", "rung"])],
        ["screen", "rung", "folds", "median pair-term sd", "target sd",
         "ratio to target", "alive"])
    (args.outdir / "summary").mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary" / "pair_terms.md").write_text(
        "*Generated by `scripts/measure_pair_terms.py`.*\n\n"
        "Standard deviation of each fitted interaction term on the unseen-acid "
        "rows it is scored on, at the hyperparameters entity-OOD validation "
        "selected. A term whose standard deviation is many orders of magnitude "
        "below the target's has not left its initialisation, and its "
        "incremental skill of ~0.000 is an artefact rather than a finding.\n\n"
        + table + "\n")
    print(f"\nwrote {args.outdir / 'summary' / 'pair_terms.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
