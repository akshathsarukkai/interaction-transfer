"""Grids for the residual-directionality diagnostic, and parallel execution.

Separate from ``real_data.sweep`` on purpose. That module's rows answer "can a
family predict ``y(i->j)``"; these answer "can anything predict ``D_res``". The
two use different targets, different nulls and different metric names, and the
one thing that must never happen is a reader averaging them together. They are
therefore written to different files under different directories, and
``residual_report.load_residual_runs`` refuses to load a frame carrying Phase 2
columns (``test_D_pearson``, ``family``) at all.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from . import koplev
from .residual_experiment import ResidualConfig, run_residual_condition
from .residual_models import LADDER_ORDER

#: The Phase 2 coverage grid, unchanged, so every cell here lines up with a cell
#: there. The evaluation pool is defined by the *top* rung, so this tuple also
#: fixes which pairs are scored -- passing a shortened grid would move the
#: goalposts rather than run fewer conditions (the bug the Phase 2 audit found
#: in the init-variance block).
COVERAGES = (0.05, 0.10, 0.20, 0.40, 0.70)
SCREENS = ("A375", "PANC1")
SPLIT_SEEDS = tuple(range(8))

#: Where the shuffle control runs: **both coverages the decision rests on**
#: (0.40 and 0.70) plus a sparse one for contrast. An earlier version ran 0.10
#: and 0.70 only, which left pre-registered criterion 5 satisfied at 0.40 by
#: extrapolation from the adjacent rung rather than by measurement -- a gap an
#: audit of the finished writeup was right to name. Only the two rungs that can
#: express a pair-specific effect are run: ``zero`` has nothing to destroy and
#: ``potential`` is covered by the main grid.
CONTROL_COVERAGES = (0.10, 0.40, 0.70)
CONTROL_RUNGS = ("lowrank", "mlp")

#: Positive-control injection strengths, in units of RMS directional effect
#: added to ``D``. For reference the observed ``D_res`` standard deviation is
#: ~0.17 (A375) and ~0.13 (PANC1), so kappa=0.10 injects a signal whose variance
#: is roughly a third of the residual's, and kappa=0.20 roughly one and a half
#: times it.
INJECT_KAPPAS = (0.10, 0.20)
INJECT_RUNGS = ("lowrank", "mlp")

_SCREEN_CACHE: dict[tuple[str, str], koplev.Screen] = {}


def _screen(label: str, raw_dir: Path) -> koplev.Screen:
    key = (label, str(raw_dir))
    if key not in _SCREEN_CACHE:
        _SCREEN_CACHE[key] = koplev.load_screen(label, raw_dir)
    return _SCREEN_CACHE[key]


def _pin_threads() -> None:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run(cfg: ResidualConfig, raw_dir: str) -> dict:
    try:
        _pin_threads()
        return run_residual_condition(cfg, Path(raw_dir),
                                      screen=_screen(cfg.screen, Path(raw_dir)))
    except Exception:                                   # noqa: BLE001
        return {"tag": cfg.tag, "screen": cfg.screen, "coverage": cfg.coverage,
                "rung": cfg.rung, "split_seed": cfg.split_seed,
                "error": traceback.format_exc()}


def run_residual_sweep(specs: list[ResidualConfig], out: Path,
                       raw_dir: Path = koplev.DEFAULT_RAW_DIR,
                       workers: int | None = None,
                       verbose: bool = True) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    rows: list[dict] = []
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


def _base(**kw) -> ResidualConfig:
    return ResidualConfig(coverages=tuple(COVERAGES), **kw)


def main_grid(screens=SCREENS, coverages=COVERAGES, rungs=LADDER_ORDER,
              split_seeds=SPLIT_SEEDS, ridge_objective: str = "y",
              tag: str = "main") -> list[ResidualConfig]:
    """Every rung at every coverage on both screens, 8 split seeds."""
    return [replace(_base(), screen=sc, coverage=cov, rung=rg, split_seed=ss,
                    ridge_objective=ridge_objective, tag=tag)
            for sc in screens for cov in coverages
            for rg in rungs for ss in split_seeds]


def sensitivity_grid(split_seeds=SPLIT_SEEDS) -> list[ResidualConfig]:
    """The main grid with the ridge penalty chosen to *maximise* removal.

    ``ridge_objective="D"`` picks the penalty minimising validation
    ``mean(D_res**2)`` instead of validation ordered MSE. It strips out as much
    directional structure as an additive model can, so whatever a rung finds
    afterwards is a floor rather than a point estimate. Run on both screens but
    only at the two coverages the decision rule reads, because it is a
    robustness check on the headline and not a second headline.
    """
    return main_grid(coverages=(0.10, 0.70), ridge_objective="D",
                     split_seeds=split_seeds, tag="ridge_D")


def control_grid(screens=SCREENS, coverages=CONTROL_COVERAGES,
                 rungs=CONTROL_RUNGS, split_seeds=SPLIT_SEEDS) -> list[ResidualConfig]:
    """Control A: ``D_res`` permuted across training and validation pairs."""
    return [replace(_base(), screen=sc, coverage=cov, rung=rg, split_seed=ss,
                    permute_train_residual=True, tag="permute_control")
            for sc in screens for cov in coverages
            for rg in rungs for ss in split_seeds]


def power_grid(screens=SCREENS, coverages=COVERAGES, rungs=INJECT_RUNGS,
               kappas=INJECT_KAPPAS, split_seeds=SPLIT_SEEDS,
               honest_alpha: bool = False) -> list[ResidualConfig]:
    """Positive control: known antisymmetric signal injected into ``D``.

    Without this block a null is uninterpretable. "No rung beat the zero
    predictor" is consistent with "there is no residual structure" and with
    "there is residual structure and 421 training pairs cannot find it", and
    only an injection of known size separates them.

    Run on **both** screens. An earlier version ran A375 only, on the reasoning
    that detectability is a property of the pair graph and the sample size,
    which the two screens share exactly. That reasoning is wrong in one respect
    that matters: ``kappa`` is an absolute RMS added to ``D``, and the residual
    it has to be seen against is not the same size on the two screens
    (sd(D_res) ~ 0.157 on A375, ~0.129 on PANC1), so the same kappa is a
    different relative signal. Applying an A375 power curve to a PANC1 null was
    an undisclosed cross-screen extrapolation underneath half the decision.

    ``honest_alpha`` reruns the block with the shrinkage fitted on validation
    pairs withheld from selection, matching the estimator the decision quotes.
    The main result uses that estimator; a power curve computed under the other
    one is answering a slightly different question, and at the sparse coverages
    -- exactly where the power argument is load-bearing -- the two estimators
    disagree most.
    """
    tag = "power_honest_k{k:g}" if honest_alpha else "power_k{k:g}"
    return [replace(_base(), screen=sc, coverage=cov, rung=rg, split_seed=ss,
                    inject_kappa=k,
                    split_validation_for_calibration=honest_alpha,
                    tag=tag.format(k=k))
            for sc in screens for cov in coverages for rg in rungs
            for k in kappas for ss in split_seeds]


def power_honest_grid(split_seeds=SPLIT_SEEDS) -> list[ResidualConfig]:
    """The power curve under the estimator the decision actually quotes.

    ``lowrank`` only: it is the primary rung, it is cheap, and the flexible
    rungs' power is a diagnostic rather than part of the argument.
    """
    return power_grid(rungs=("lowrank",), split_seeds=split_seeds,
                      honest_alpha=True)


#: The one ``lowrank`` setting the ``rank2`` block pins: 2 latent dimensions per
#: drug, 204 parameters, one antisymmetric bilinear form, no search.
RANK2_HPARAMS = ({"rank": 2, "lr": 1e-2, "weight_decay": 1e-3},)


def honest_alpha_grid(screens=SCREENS, coverages=COVERAGES,
                      rungs=("zero", "potential", "lowrank"),
                      split_seeds=SPLIT_SEEDS) -> list[ResidualConfig]:
    """The primary contrast, rerun with the shrinkage fitted on held-out validation.

    In the shipped main grid the shrinkage coefficient is fitted on the same
    validation pairs that already chose the stopping epoch, the restart and the
    grid member, which biases it upward (``residual_train.select_shrinkage``).
    Here half the validation pairs are withheld from selection entirely and only
    calibrate ``alpha``. Neither half is ever a test pair, so this is a
    correction to an estimator, not a change to the evaluation.

    Run for the three rungs the pre-registered primary contrast depends on --
    ``lowrank - potential``, against ``zero`` -- because those are what the
    decision turns on and they are cheap. The two MLP rungs are flexible
    diagnostics whose exact magnitudes are not load-bearing, and rerunning them
    would cost more than the answer is worth.
    """
    return [replace(_base(), screen=sc, coverage=cov, rung=rg, split_seed=ss,
                    split_validation_for_calibration=True, tag="honest_alpha")
            for sc in screens for cov in coverages
            for rg in rungs for ss in split_seeds]


def rank2_grid(screens=SCREENS, coverages=COVERAGES,
               split_seeds=SPLIT_SEEDS) -> list[ResidualConfig]:
    """``lowrank`` pinned to rank 2, with no hyperparameter search at all.

    At coverage 0.70 the shipped grid selects its largest rank (16) on 8 of 8
    A375 split seeds. A selection that sits on the boundary makes
    "capacity-controlled" a statement about where the grid was truncated rather
    than about the model, and it leaves open the reading that the result needs
    the capacity it was given. Pinning the smallest, most interior setting --
    204 parameters, one 2x2 antisymmetric form, one learning rate, no search --
    answers that directly. Because selection is on validation only, a truncated
    grid biases the reported skill *down*, so this block can only make the claim
    weaker or confirm it.
    """
    return [replace(_base(), screen=sc, coverage=cov, rung="lowrank",
                    split_seed=ss, force_hparams=RANK2_HPARAMS, tag="rank2")
            for sc in screens for cov in coverages for ss in split_seeds]


#: Ridge penalties the titration block forces, spanning "barely shrunk" to
#: "heavily over-shrunk". The shipped grid selects 0.01-10 on validation.
TITRATION_LAMBDAS = (3.0, 30.0, 300.0)


def titration_grid(screens=SCREENS, coverages=(0.70,),
                   rungs=("potential", "lowrank"),
                   lambdas=TITRATION_LAMBDAS,
                   split_seeds=SPLIT_SEEDS) -> list[ResidualConfig]:
    """Prove the ``potential`` rung is a working detector and not a dead one.

    The whole primary contrast rests on ``potential`` scoring ~0: if it cannot
    learn at all, "low-rank beats potential" is vacuous, and the project has
    already been bitten once by a rung that returned exactly 0.0 because every
    gradient vanished at initialisation.

    Injecting a potential does not test this -- ``0.5*kappa*(g_i - g_j)`` added
    to ``y(i->j)`` **is** an additive model contribution, so the ridge fit
    absorbs it and it never reaches ``D_res``. The test that works is to force
    the penalty instead: over-shrink the additive fit so a known fraction of the
    potential is deliberately left in the residual, and check that ``potential``
    finds exactly that and that its skill falls to zero as the fraction does.

    A titration is the right shape for this because it turns a single "the rung
    is alive" assertion into a dose-response curve: the rung must be sensitive,
    calibrated, *and* correctly null at the shipped penalty.
    """
    return [replace(_base(), screen=sc, coverage=cov, rung=rg, split_seed=ss,
                    ridge_lambdas=(lam,), tag=f"titration_lam{lam:g}")
            for sc in screens for cov in coverages for rg in rungs
            for lam in lambdas for ss in split_seeds]


def contamination_grid(split_seeds=(0, 1, 2, 3)) -> list[ResidualConfig]:
    """Control C: the additive baseline deliberately fitted on held-out pairs.

    Runs only to quantify what the leakage guard is protecting against. The rows
    it produces carry ``contaminated: true`` and are written to their own file;
    :func:`residual_report.load_residual_runs` refuses to mix them into any
    table. The guard itself is tested in ``tests/test_phase2_residual.py`` -- the
    numbers here exist so the size of the inflation can be quoted rather than
    asserted.
    """
    return [replace(_base(), screen="A375", coverage=cov, rung=rg,
                    split_seed=ss, contaminate_additive_fit=True,
                    tag="contaminated")
            for cov in (0.10,) for rg in ("zero", "lowrank")
            for ss in split_seeds]


#: Which grids each ``--part`` of ``scripts/run_phase2_residual.py`` runs, in the
#: order the runner executes them. This mapping is the **single source of
#: truth** for what a part does: the runner builds its jobs from it and
#: :func:`part_counts` derives the documented run counts from it, so a grid that
#: grows cannot leave a hard-coded number behind in prose. It has done exactly
#: that once already -- the permutation control was widened from two coverages
#: to three (64 -> 96 rows) and three documents went on quoting the old totals.
PART_GRIDS: dict[str, tuple[tuple[str, str], ...]] = {
    "main": (("main", "main_grid"),),
    "sensitivity": (("sensitivity", "sensitivity_grid"),),
    "controls": (("controls", "control_grid"),
                 ("contaminated", "contamination_grid")),
    "power": (("power", "power_grid"),
              ("power_honest", "power_honest_grid")),
    "robustness": (("honest_alpha", "honest_alpha_grid"),
                   ("rank2", "rank2_grid"),
                   ("titration", "titration_grid")),
}

#: ``--part all`` runs every part above. Spelled as a derived tuple rather than
#: a second list, so adding a part cannot leave ``all`` behind.
ALL_PARTS: tuple[str, ...] = tuple(PART_GRIDS)


def part_jobs(part: str) -> list[tuple[str, list[ResidualConfig]]]:
    """``[(output name, specs), ...]`` for one ``--part``, ``"all"`` included."""
    if part == "all":
        names: tuple[str, ...] = ALL_PARTS
    elif part in PART_GRIDS:
        names = (part,)
    else:
        raise ValueError(f"unknown part {part!r}; expected one of "
                         f"{sorted(PART_GRIDS) + ['all']}")
    out: list[tuple[str, list[ResidualConfig]]] = []
    for name in names:
        for out_name, grid_name in PART_GRIDS[name]:
            out.append((out_name, globals()[grid_name]()))
    return out


def part_counts() -> dict[str, int]:
    """Total runs per ``--part``, derived from the grids themselves.

    Documentation quotes these numbers and
    ``test_documented_run_counts_match_the_grids`` pins the quotes to this
    function, which is the only reason they can be trusted.
    """
    counts = {p: sum(len(s) for _, s in part_jobs(p)) for p in ALL_PARTS}
    counts["all"] = sum(counts[p] for p in ALL_PARTS)
    return counts
