"""Select hyperparameters per family on the DEV seeds, by validation MSE.

Protocol, which is the part that matters scientifically:

* the **same** grid is offered to every family, so none is handicapped by a
  setting chosen to suit another;
* selection uses only ``phase1.DEV_SEEDS`` and only the *validation* MSE, never
  the test set;
* the report seeds (``phase1.REPORT_SEEDS``) are disjoint from the dev seeds, so
  the headline numbers come from data this search never touched.

Writes the winning config per family as JSON. Paste the result into
``phase1.HPARAMS`` so the headline sweep is a single deterministic run.

    python scripts/select_hparams.py --out results/hparams.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from intervention_algebra import phase1
from intervention_algebra.sweep import run_sweep

# The grid. Deliberately small and interpretable: the two knobs that mattered in
# diagnosis were how strongly the embedding table is regularised (it is the
# component that overfits) and whether the pair map is given product features.
GRID = [
    dict(emb_dim=e, pair_mode=m, emb_weight_decay=ewd, weight_decay=wd, lr=3e-3)
    for e, m, (ewd, wd) in itertools.product(
        [8],
        ["concat", "concat_outer", "outer"],
        [(1e-2, 0.0), (1e-2, 1e-3), (3e-2, 1e-3)],
    )
]

# `emb_dim` is fixed at 8: it won the first search on validation MSE, which is
# out-of-sample with respect to everything reported.
#
# `pair_mode="outer"` is in the grid on purpose. It was originally adopted after
# a side check scored on *test* S-recovery from a single seed -- which is
# selection on a headline metric, and it flipped the sign of the very
# comparison it was used to report. That check is superseded by this search,
# which scores validation MSE only, over both dev seeds and both coverages,
# with the same grid offered to every family.

SEARCH_COVERAGES = (0.1, 0.4)


def build_specs() -> list[dict]:
    specs = []
    for gi, g in enumerate(GRID):
        for family in phase1.FAMILIES:
            for cov in SEARCH_COVERAGES:
                for seed in phase1.DEV_SEEDS:
                    specs.append(dict(
                        family=family,
                        seed=seed,
                        system=dict(phase1.BASE_SYSTEM, regime="both"),
                        split=dict(pair_coverage=cov),
                        model=dict(emb_dim=g["emb_dim"], pair_mode=g["pair_mode"]),
                        train=dict(phase1.TRAIN_BASE, lr=g["lr"],
                                   emb_weight_decay=g["emb_weight_decay"],
                                   weight_decay=g["weight_decay"]),
                        capacity_match=True,
                        tag=f"hpsearch_g{gi}",
                        meta=dict(grid_id=gi, cov=cov, **g),
                    ))
    return specs


def select(rows: list[dict]) -> dict[str, dict]:
    """Shared architecture, per-family optimiser settings.

    Two-stage protocol, and the split matters for fairness:

    1. The **architecture** (``emb_dim``, ``pair_mode``) is chosen ONCE, shared by
       every family: the setting with the lowest validation MSE averaged over
       families. Letting each family pick its own architecture would mean
       comparing, say, a 288-feature constrained model against an 80-feature
       unconstrained one -- an architecture comparison wearing the costume of an
       inductive-bias comparison. Capacity matching only equalises pair
       parameters *within* a shared feature map.
    2. The **optimiser settings** (weight decays) are then chosen per family
       within that architecture. Forcing one untuned optimiser on everyone does
       not remove the confound, it freezes it at an arbitrary point -- and the
       sign of the headline comparison was observed to flip with weight decay
       alone.

    Selection uses only dev seeds and only validation MSE; the report seeds are
    disjoint and never enter here.
    """
    def mean_val(family: str | None, gi: int) -> float | None:
        vals = [r["best_val_mse"] for r in rows
                if r.get("grid_id") == gi and "error" not in r
                and (family is None or r.get("family") == family)]
        return float(np.mean(vals)) if vals else None

    arch_of = {gi: (g["emb_dim"], g["pair_mode"]) for gi, g in enumerate(GRID)}

    # stage 1 -- shared architecture
    arch_scores: dict[tuple, list[float]] = {}
    for gi in range(len(GRID)):
        m = mean_val(None, gi)
        if m is not None:
            arch_scores.setdefault(arch_of[gi], []).append(m)
    if not arch_scores:
        raise RuntimeError("no successful hyperparameter runs")
    arch = min(arch_scores, key=lambda a: float(np.mean(arch_scores[a])))
    print(f"shared architecture: emb_dim={arch[0]} pair_mode={arch[1]} "
          f"(mean val {np.mean(arch_scores[arch]):.4f})", file=sys.stderr)
    for a in sorted(arch_scores, key=lambda a: float(np.mean(arch_scores[a]))):
        print(f"    arch {a}  mean val {np.mean(arch_scores[a]):.4f}", file=sys.stderr)

    # stage 2 -- per-family optimiser settings within that architecture
    candidates = [gi for gi in range(len(GRID)) if arch_of[gi] == arch]
    best: dict[str, dict] = {}
    for family in phase1.FAMILIES:
        scored = [(mean_val(family, gi), gi) for gi in candidates]
        scored = [(v, gi) for v, gi in scored if v is not None]
        if not scored:
            raise RuntimeError(f"no successful runs for family {family!r}")
        scored.sort()
        val, gi = scored[0]
        best[family] = dict(GRID[gi], _mean_val_mse=val, _grid_id=gi)
        print(f"{family:15s} grid#{gi:<3d} val={val:.4f}  {GRID[gi]}", file=sys.stderr)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hparams.json")
    ap.add_argument("--raw", default="results/hparam_search.jsonl")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--arch", default=None,
                    help="restrict the grid to one architecture, as "
                         "'emb_dim,pair_mode' (stage-2 refinement)")
    args = ap.parse_args(argv)

    specs = build_specs()
    if args.arch:
        emb, mode = args.arch.split(",")
        keep = {gi for gi, g in enumerate(GRID)
                if g["emb_dim"] == int(emb) and g["pair_mode"] == mode.strip()}
        specs = [s for s in specs if s["meta"]["grid_id"] in keep]
    print(f"{len(specs)} runs over {len(GRID)} configs x {len(phase1.FAMILIES)} "
          f"families", file=sys.stderr)
    rows = run_sweep(specs, out_path=args.raw, workers=args.workers)

    best = select(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
