"""Parallel execution of Phase 2 conditions, and the grids that define them.

Every run is independent, so this is a process pool over :func:`run_condition`
with results appended to a JSONL as they land. Screens are loaded once per worker
and cached -- reading a 10,000-row CSV per run dominated the runtime otherwise.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from . import koplev
from .experiment import Phase2Config, run_condition
from .train import TrainConfig

#: Coverage grid over the screen's 4,950 unordered pairs.
#:
#: The numbers here are measured, not assumed -- an earlier version of this
#: comment claimed the sparsest rung "leaves every drug with ~5 training pairs
#: and the pair graph connected" and fixed the pool at "1,485 pairs, 2,970
#: ordered rows". None of that held. At coverage 0.05 the minimum training
#: degree is 0 or 1 on all 8 split seeds (mean 4.2), the pair graph is
#: disconnected on 4 of them, and the evaluation pool is 848-1,003 pairs
#: (1,696-2,006 ordered rows) once the connectivity filter has run -- so the
#: old comment overstated n by 48-75% at exactly the rung the hypothesis lives
#: at. The rung is still sound, because `min_train_degree` guarantees every
#: scored test endpoint has >= 3 training pairs and isolated drugs are simply
#: never scored; but a reader judging power at 0.05 was being handed a number
#: wrong by up to 1.75x. Regenerate from `results/phase2/dataset_audit.json`.
#:
#: The top rung fixes the evaluation pool at the pairs excluded there, and the
#: nesting keeps that pool identical at every coverage within a split seed.
COVERAGES = (0.05, 0.10, 0.20, 0.40, 0.70)

FAMILIES = ("additive", "order_insensitive", "unrestricted", "structured")
SCREENS = ("A375", "PANC1")

#: Selected on validation loss at dev split seeds 100/101, which are disjoint
#: from the evaluation seeds. See ``results/phase2/hparams.json``.
HPARAMS: dict[str, dict] = {}

_SCREEN_CACHE: dict[tuple[str, str], koplev.Screen] = {}


def _screen(label: str, raw_dir: Path) -> koplev.Screen:
    key = (label, str(raw_dir))
    if key not in _SCREEN_CACHE:
        _SCREEN_CACHE[key] = koplev.load_screen(label, raw_dir)
    return _SCREEN_CACHE[key]


def _pin_threads() -> None:
    """One compute thread per worker.

    torch defaults to one thread per core *inside every pool worker*, so an
    8-worker pool on an 8-core machine asks for 64 threads and spends most of its
    time in the scheduler. Measured: pinning cut the sweep's wall clock by more
    than half. ``OMP_NUM_THREADS`` alone is not enough -- torch's intraop pool is
    set separately.
    """
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Only settable before the first parallel op; already-warm workers are
        # fine, the intraop setting above is the one that matters.
        pass


def _run(cfg: Phase2Config, raw_dir: str) -> dict:
    try:
        _pin_threads()
        return run_condition(cfg, Path(raw_dir), screen=_screen(cfg.screen, Path(raw_dir)))
    except Exception:                                   # noqa: BLE001
        # A failed condition must not take the sweep down, but it must not be
        # silently dropped either: it lands in the JSONL with its traceback so a
        # gap in the grid is visible rather than inferred.
        return {"tag": cfg.tag, "screen": cfg.screen, "coverage": cfg.coverage,
                "family": cfg.family, "split_seed": cfg.split_seed,
                "init_seed": cfg.init_seed, "error": traceback.format_exc()}


def apply_hparams(cfg: Phase2Config, hparams: dict) -> Phase2Config:
    """Overlay the family's selected hyperparameters onto a config."""
    h = hparams.get(cfg.family)
    if not h:
        return cfg
    return replace(cfg, emb_dim=h.get("emb_dim", cfg.emb_dim),
                   train=replace(cfg.train, lr=h.get("lr", cfg.train.lr),
                                 weight_decay=h.get("weight_decay",
                                                    cfg.train.weight_decay)))


def run_sweep(specs: list[Phase2Config], out: Path,
              raw_dir: Path = koplev.DEFAULT_RAW_DIR,
              workers: int | None = None, verbose: bool = True) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []
    # torch defaults to all cores per process; with a pool that oversubscribes
    # badly and runs slower than serial.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with open(out, "w") as fh, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, c, str(raw_dir)): c for c in specs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            if verbose and (k % 10 == 0 or k == len(specs)):
                nerr = sum(1 for r in rows if "error" in r)
                print(f"  {k}/{len(specs)} done ({nerr} failed)", flush=True)
    return rows


def main_grid(screens=SCREENS, coverages=COVERAGES, families=FAMILIES,
              split_seeds=range(8), init_seeds=(0,), hparams=None,
              train: TrainConfig | None = None,
              tag: str = "main",
              pool_coverages: tuple[float, ...] | None = None) -> list[Phase2Config]:
    """The headline grid: every family at every coverage, on both screens.

    Split seeds vary the pair partition; init seeds vary initialisation. Both
    are reported separately because they are different kinds of uncertainty --
    which pairs you happened to observe, versus where the optimiser landed.

    ``coverages`` selects which rungs to *run*; ``pool_coverages`` defines the
    nested split the runs are scored against, and defaults to ``coverages``.
    They are separate because ``Phase2Config.coverages`` is what fixes the
    evaluation pool: the top rung determines which pairs are held out, and the
    nesting keeps that pool identical at every rung below it. Passing a
    shortened grid therefore silently *moves the goalposts* rather than just
    running fewer conditions -- at ``(0.10, 0.40)`` the pool becomes 2,970 pairs
    with nothing dropped for connectivity, against the headline's 848-1,003.
    Any block that runs a subset of rungs but wants to stay comparable to the
    headline must pass ``pool_coverages=COVERAGES``. ``control_grid`` hardcodes
    the full grid for exactly this reason; this parameter lets the init-variance
    block do the same without also running all five rungs.

    ``tag`` is a parameter and not a constant for a specific reason. The
    init-variance block reuses this grid at extra init seeds on a subset of
    split seeds; emitting it under ``tag="main"`` would fold those rows into the
    headline cells, so two of the eight split seeds would be averaged over four
    initialisations and the other six over one. The aggregation would still run,
    the ``n`` would still read 8, and the headline means would quietly be
    weighted. Caught by inspection before any such row was written; the grid now
    refuses to emit under a caller-unspecified tag.
    """
    base = Phase2Config(coverages=tuple(pool_coverages or coverages),
                        train=train or TrainConfig())
    out = []
    for sc in screens:
        for cov in coverages:
            for fam in families:
                for ss in split_seeds:
                    for it in init_seeds:
                        cfg = replace(base, screen=sc, coverage=cov, family=fam,
                                      split_seed=ss, init_seed=it, tag=tag)
                        out.append(apply_hparams(cfg, hparams or HPARAMS))
    return out


#: Learning rates swept in the tuning-parity block, spanning the selection grid.
PARITY_LRS = (0.003, 0.01, 0.03)

#: Where parity is checked: the pre-registered primary cell and the cell that
#: carries the strongest negative claim, on both screens for the latter.
PARITY_CELLS = (("A375", 0.05), ("A375", 0.10), ("A375", 0.70), ("PANC1", 0.70))


def parity_grid(cells=PARITY_CELLS, families=("structured", "unrestricted"),
                lrs=PARITY_LRS, split_seeds=range(8), hparams=None,
                train: TrainConfig | None = None) -> list[Phase2Config]:
    """Every learning rate for both pair families, at the cells that matter.

    ``select_phase2_hparams.py`` tunes once, on A375 at coverage 0.10, and the
    winning setting is then reused at every coverage and on both screens. That
    is a defensible protocol, but it is only *fair* if the choice transfers --
    and here it does not. Structured drew lr=0.01 and unrestricted lr=0.03, and
    those are not equally good at the rungs the conclusion is read off:
    structured prefers 0.03 at coverage 0.10 and 0.003 at 0.70, so at the
    primary cell the shipped pairing is the least favourable of the available
    comparisons for the family under test.

    This block re-runs both families across the whole rate grid so the family
    contrast can be recomputed with each family at *its own* validation-selected
    rate. Selection is on validation loss, per split seed, never on test -- the
    point is to remove a tuning artifact from the comparison, not to shop for a
    result. The headline grid is left exactly as it was run.
    """
    base = Phase2Config(coverages=tuple(COVERAGES), train=train or TrainConfig())
    out = []
    for sc, cov in cells:
        for fam in families:
            for lr in lrs:
                for ss in split_seeds:
                    cfg = replace(base, screen=sc, coverage=cov, family=fam,
                                  split_seed=ss, init_seed=0,
                                  tag=f"parity_lr{lr:g}")
                    cfg = apply_hparams(cfg, hparams or HPARAMS)
                    # apply_hparams would put the *selected* rate back; the
                    # whole point of this grid is to override it.
                    cfg = replace(cfg, train=replace(cfg.train, lr=lr))
                    out.append(cfg)
    return out


def control_grid(screens=("A375",), coverages=(0.10, 0.40), families=FAMILIES,
                 split_seeds=range(8), hparams=None,
                 train: TrainConfig | None = None) -> list[Phase2Config]:
    """Direction-shuffle control: schedule direction destroyed in training only.

    If a family still predicts the held-out directional effect after this, it is
    reading something other than the schedule -- a leak, or an artifact of the
    evaluation. The structured family is the one that must fall to chance.
    """
    base = Phase2Config(coverages=tuple(COVERAGES), train=train or TrainConfig())
    out = []
    for sc in screens:
        for cov in coverages:
            for fam in families:
                for ss in split_seeds:
                    cfg = replace(base, screen=sc, coverage=cov, family=fam,
                                  split_seed=ss, init_seed=0,
                                  shuffle_train_direction=True,
                                  tag="direction_shuffle")
                    out.append(apply_hparams(cfg, hparams or HPARAMS))
    return out
