"""One residual-directionality condition: (screen, coverage, rung, split seed).

The order of operations is the whole experiment, and it is fixed here so no
caller can reorder it:

1. build the nested pair-level split (identical to Phase 2, same function);
2. assert no pair leakage;
3. fit the two-way additive model **on training rows only**, penalty chosen on
   validation rows;
4. residualise train, validation and test rows with that fit;
5. fit the rung on the training pairs' ``D_res``, selecting hyperparameters on
   the validation pairs' ``D_res``;
6. score on the held-out pairs against the zero predictor.

Step 3 before step 4 is the leakage-critical ordering: residualising the whole
matrix first and splitting afterwards would let every held-out pair contribute
to its own baseline, and the resulting "residual" would be shrunk toward zero on
exactly the rows the conclusion is read off.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import koplev
from .residual import (RIDGE_LAMBDAS, decomposition, fit_additive,
                       ordered_residuals, residual_targets)
from .residual_models import (HPARAM_GRID, LADDER_ORDER, ResidualModelConfig,
                              apply_arch, build_residual_model, split_hparams)
from .residual_train import (ResidualTrainConfig, ordered_tensors, pair_tensors,
                             predict_d_res, select_shrinkage,
                             split_calibration_pairs, train_residual)
from .splits import assert_no_pair_leakage, connectivity_report, make_coverage_splits


@dataclass(frozen=True)
class ResidualConfig:
    screen: str = "A375"
    coverage: float = 0.10
    #: A rung of ``residual_models.LADDER``, not a Phase 2 family. The two
    #: vocabularies are kept apart on purpose -- see ``tag``.
    rung: str = "lowrank"
    split_seed: int = 0
    init_seed: int = 0
    coverages: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70)
    min_train_degree: int = 3
    min_eligible_test_pairs: int = 50
    val_fraction: float = 0.15
    ridge_lambdas: tuple[float, ...] = RIDGE_LAMBDAS
    #: ``"y"`` (primary) or ``"D"`` (sensitivity). See ``residual.fit_additive``.
    ridge_objective: str = "y"
    train: ResidualTrainConfig = ResidualTrainConfig()
    #: Control A: permute ``D_res`` across the *training* unordered pairs (and
    #: the validation ones, so selection cannot see through it). The evaluation
    #: pool is untouched. A rung that still scores after this is reading pair
    #: identity or a split artifact rather than residual structure.
    permute_train_residual: bool = False
    #: Hold half the validation pairs out of model selection entirely and fit
    #: the shrinkage coefficient only on that half. Off by default so the
    #: shipped main grid is exactly the grid the decision rule was registered
    #: against; the ``honest_alpha`` block reruns the rungs the primary contrast
    #: depends on with it on. See ``residual_train.select_shrinkage`` for why:
    #: fitting alpha on the same pairs that chose the epoch, the restart and the
    #: grid member biases it upward, which makes the sparse-coverage numbers
    #: more negative than the rung's attainable performance.
    split_validation_for_calibration: bool = False
    #: Force a single ``lowrank`` setting instead of searching the grid. Used by
    #: the ``rank2`` block: at coverage 0.70 the grid selects its largest rank on
    #: 8 of 8 A375 seeds, and a boundary selection makes "capacity-controlled"
    #: a claim about the grid rather than about the model. Pinning rank 2 -- 204
    #: parameters, one antisymmetric bilinear form, no search at all -- settles
    #: whether the result needs the capacity it was given.
    force_hparams: tuple = ()
    #: Positive control (power). Adds an exactly antisymmetric, low-rank
    #: pair-specific signal ``kappa * S_ij`` to the *directional* part of the
    #: response before anything else runs, with ``S = u K u^T``, ``K = -K^T``,
    #: ``rms(S) = 1``. It contributes nothing to the symmetric part and nothing
    #: to the per-drug potential, so it lands entirely in ``D_res``. Its purpose
    #: is to make a null interpretable: "no rung beat zero" means something only
    #: if a rung would have beaten zero had structure of this size been present.
    inject_kappa: float = 0.0
    inject_rank: int = 3
    inject_seed: int = 0
    #: Deliberately fits the additive baseline on train+val+test. Exists only so
    #: the leakage guard can be shown to fire and the size of the inflation can
    #: be quoted; never a scientific result. ``run_residual_condition`` stamps
    #: ``contaminated: true`` on any row produced this way.
    contaminate_additive_fit: bool = False
    tag: str = "main"


def inject_antisymmetric(frame: pd.DataFrame, n_drugs: int, kappa: float,
                         rank: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Add a known pair-specific antisymmetric signal of RMS ``kappa`` to ``D``.

    ``S = u K u^T`` with ``K`` antisymmetric is antisymmetric in ``(i, j)``, so
    adding ``kappa * S_ij / 2`` to ``y(i -> j)`` adds ``kappa * S_ij`` to
    ``D(i, j)`` and exactly zero to the symmetric part ``(y_ij + y_ji) / 2``.
    ``S`` is normalised to unit RMS over the off-diagonal, so ``kappa`` is
    directly the RMS of the injected directional effect in the screen's own
    units and can be compared with ``D_res_std``.

    The injected form is deliberately the same shape as the ``lowrank`` rung's
    hypothesis class. That makes the resulting power curve an **upper bound**:
    it says what the experiment could detect under the most favourable possible
    match between signal and model, which is the right bound to have when the
    conclusion is a null.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_drugs, rank))
    W = rng.normal(size=(rank, rank))
    S = u @ (W - W.T) @ u.T
    off = ~np.eye(n_drugs, dtype=bool)
    S = S / np.sqrt((S[off] ** 2).mean())
    out = frame.copy()
    out["y"] = (out["y"].to_numpy()
                + 0.5 * kappa * S[out["i"].to_numpy(), out["j"].to_numpy()])
    return out, S


def _permute_direction(pairs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute ``D_res`` across unordered pairs, leaving everything else alone.

    Keeps the marginal distribution of the residual directional effect exactly,
    and destroys only its association with the pair. ``y_f``/``y_r`` are
    rebuilt so the ordered-loss rung sees a frame consistent with the permuted
    direction: the symmetric half ``(y_f + y_r)/2`` is untouched and the
    antisymmetric half is replaced.
    """
    out = pairs.copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    d_new = out["D_res"].to_numpy()[perm]
    d_old = out["D_res"].to_numpy()
    out["D_res"] = d_new
    # y_f - y_r must move with D_res or the ordered rung would be trained on the
    # unshuffled direction while the direct rungs see the shuffled one.
    delta = 0.5 * (d_new - d_old)
    out["y_f"] = out["y_f"].to_numpy() + delta
    out["y_r"] = out["y_r"].to_numpy() - delta
    out["D_true"] = out["y_f"].to_numpy() - out["y_r"].to_numpy()
    return out


def _ordered_from_pairs(pairs: pd.DataFrame, fit) -> pd.DataFrame:
    """Ordered residual rows ``r_ij`` reconstructed from a pair frame.

    Used instead of :func:`residual.ordered_residuals` whenever the pair frame
    has been modified (the permutation control), so the ordered rung is trained
    on exactly the target the direct rungs are.
    """
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    fwd = pd.DataFrame({"i": i, "j": j, "y": pairs["y_f"].to_numpy()})
    rev = pd.DataFrame({"i": j, "j": i, "y": pairs["y_r"].to_numpy()})
    out = pd.concat([fwd, rev], ignore_index=True)
    out["r"] = out["y"].to_numpy() - fit.predict(out["i"].to_numpy(),
                                                 out["j"].to_numpy())
    return out


def residual_metrics(d_true: np.ndarray, d_pred: np.ndarray,
                     threshold: float) -> dict:
    """Everything the decision rule reads, against the zero predictor.

    ``skill = 1 - MSE_model / MSE_zero`` with ``MSE_zero = mean(D_res**2)`` on
    the *same* held-out pairs. Positive means the rung beat "there is no
    pair-specific directional effect"; zero means it tied; negative means
    predicting nothing would have been better.
    """
    err = d_pred - d_true
    mse_zero = float((d_true ** 2).mean())
    mse = float((err ** 2).mean())
    out = {
        "n_pairs": int(len(d_true)),
        "mse_zero": mse_zero,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(err).mean()),
        "mae_zero": float(np.abs(d_true).mean()),
        "skill": float(1.0 - mse / mse_zero) if mse_zero > 0 else float("nan"),
        "pred_std": float(d_pred.std()),
        "true_std": float(d_true.std()),
    }
    if len(d_true) > 2 and d_pred.std() > 0 and d_true.std() > 0:
        out["pearson"] = float(stats.pearsonr(d_true, d_pred).statistic)
        out["spearman"] = float(stats.spearmanr(d_true, d_pred).statistic)
    else:
        # The zero rung emits a constant, so a correlation with it is undefined
        # rather than zero. Reporting 0.0 would invite the reader to treat the
        # null as "uncorrelated" when it is "not a random variable".
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")

    sel = np.abs(d_true) > threshold
    out["sign_threshold"] = float(threshold)
    out["sign_n"] = int(sel.sum())
    out["sign_frac_of_pairs"] = float(sel.mean())
    if sel.sum() >= 10 and np.any(d_pred[sel] != 0):
        correct = (np.sign(d_pred[sel]) == np.sign(d_true[sel])).astype(float)
        correct[d_pred[sel] == 0.0] = 0.5      # a tie is a coin flip, as in Phase 2
        out["sign_accuracy"] = float(correct.mean())
    else:
        out["sign_accuracy"] = float("nan")
    return out


def run_residual_condition(cfg: ResidualConfig,
                           raw_dir: Path = koplev.DEFAULT_RAW_DIR,
                           screen: koplev.Screen | None = None) -> dict:
    screen = screen or koplev.load_screen(cfg.screen, raw_dir)
    frame = screen.frame
    if cfg.inject_kappa:
        frame, _ = inject_antisymmetric(frame, screen.n_drugs, cfg.inject_kappa,
                                        cfg.inject_rank, cfg.inject_seed)

    splits = make_coverage_splits(
        frame, screen.n_drugs, cfg.coverages, split_seed=cfg.split_seed,
        val_fraction=cfg.val_fraction, min_train_degree=cfg.min_train_degree,
        min_eligible_test_pairs=cfg.min_eligible_test_pairs)
    if cfg.coverage not in splits:
        raise ValueError(f"coverage {cfg.coverage} not in grid {cfg.coverages}")
    split = splits[cfg.coverage]
    assert_no_pair_leakage(split, frame)

    fit = fit_additive(split, frame, screen.n_drugs, cfg.ridge_lambdas,
                       objective=cfg.ridge_objective,
                       _contaminate=cfg.contaminate_additive_fit)

    train_rows = split.rows(frame, "train")
    val_rows = split.rows(frame, "val")
    test_rows = split.rows(frame, "test")

    tr_pairs = residual_targets(train_rows, fit)
    va_pairs = residual_targets(val_rows, fit)
    te_pairs = residual_targets(test_rows, fit)

    if cfg.permute_train_residual:
        tr_pairs = _permute_direction(tr_pairs, seed=cfg.split_seed + 7919)
        va_pairs = _permute_direction(va_pairs, seed=cfg.split_seed + 104729)
        tr_ordered = _ordered_from_pairs(tr_pairs, fit)
    else:
        tr_ordered = ordered_residuals(train_rows, fit)

    # Scale-only standardisation, from training pairs. Never shifted -- see
    # residual_train's module docstring.
    d_scale = float(np.sqrt((tr_pairs["D_res"].to_numpy() ** 2).mean()))
    r_scale = float(np.sqrt((tr_ordered["r"].to_numpy() ** 2).mean()))
    d_scale = max(d_scale, 1e-9)
    r_scale = max(r_scale, 1e-9)

    base_cfg = ResidualModelConfig(n_drugs=screen.n_drugs, seed=cfg.init_seed)

    # Which validation pairs select the model, and which calibrate the shrinkage.
    # By default the same ones do both, which is the bias documented in
    # `select_shrinkage`; with the flag on they are disjoint halves. Neither is
    # ever a test pair.
    if cfg.split_validation_for_calibration:
        sel_ix, cal_ix = split_calibration_pairs(len(va_pairs),
                                                 seed=cfg.split_seed + 31337)
        va_sel = va_pairs.iloc[sel_ix].reset_index(drop=True)
        va_cal = va_pairs.iloc[cal_ix].reset_index(drop=True)
    else:
        va_sel = va_cal = va_pairs

    grid = tuple(cfg.force_hparams) or HPARAM_GRID[cfg.rung]
    best = None
    grid_losses: list[float] = []
    for h in grid:
        arch, opt = split_hparams(h)
        mcfg = apply_arch(base_cfg, arch)
        tcfg = replace(cfg.train, **opt)
        ordered = build_residual_model(cfg.rung, mcfg).ordered_loss
        tr_t = (ordered_tensors(tr_ordered, r_scale) if ordered
                else pair_tensors(tr_pairs, d_scale))
        # The ordered rung predicts r in units of r_scale, so its D_res readout
        # is in those units too; the direct rungs are in units of d_scale.
        out_scale = r_scale if ordered else d_scale
        va_scaled = pair_tensors(va_sel, out_scale)

        def build(seed: int, mcfg=mcfg):
            return build_residual_model(cfg.rung, replace(mcfg, seed=seed))

        fitres = train_residual(build, tr_t, va_scaled, tcfg, seed=cfg.init_seed)
        grid_losses.append(fitres.val_loss)
        if best is None or fitres.val_loss < best[0].val_loss:
            best = (fitres, h, out_scale)
    fitres, hp, out_scale = best                       # type: ignore[misc]

    noise = koplev.measurement_noise_sd(raw_dir)
    threshold = noise["threshold_2sd_D"][cfg.screen]

    d_true = te_pairs["D_res"].to_numpy()
    d_pred = predict_d_res(fitres.model, te_pairs["i"].to_numpy(),
                           te_pairs["j"].to_numpy(), out_scale)
    # Calibration on validation pairs only, then applied to the test prediction.
    alpha = select_shrinkage(
        va_cal["D_res"].to_numpy(),
        predict_d_res(fitres.model, va_cal["i"].to_numpy(),
                      va_cal["j"].to_numpy(), out_scale))
    m = residual_metrics(d_true, d_pred, threshold)
    m_cal = residual_metrics(d_true, alpha * d_pred, threshold)

    # The same rung's skill against the *raw* directional effect, so figure 4
    # can put "predictable because of the potential" next to "predictable at
    # all" without a second sweep. The prediction is D_add + Dhat_res.
    raw_true = te_pairs["D_true"].to_numpy()
    raw_pred = te_pairs["D_add"].to_numpy() + alpha * d_pred
    m_raw = residual_metrics(raw_true, raw_pred, threshold)

    dec = decomposition(test_rows, fit)

    row = {
        "tag": cfg.tag, "screen": cfg.screen, "coverage": cfg.coverage,
        "rung": cfg.rung, "split_seed": cfg.split_seed, "init_seed": cfg.init_seed,
        "permute_train_residual": cfg.permute_train_residual,
        "inject_kappa": cfg.inject_kappa,
        "contaminated": cfg.contaminate_additive_fit,
        "n_drugs": screen.n_drugs,
        "n_train_pairs": len(split.train_pairs),
        "n_val_pairs": len(split.val_pairs),
        "n_test_pairs": len(split.test_pairs),
        "n_params": fitres.model.n_params(),
        "hparams": hp,
        "grid_val_losses": grid_losses,
        "grid_size": len(grid),
        "grid_argmin": int(np.argmin(grid_losses)),
        "train_loss": fitres.train_loss,
        "val_loss": fitres.val_loss,
        "best_epoch": fitres.best_epoch,
        "n_restarts": len(fitres.restarts),
        "d_scale_train": d_scale,
        "r_scale_train": r_scale,
    }
    row["shrinkage_alpha"] = alpha
    row["n_val_pairs_selection"] = int(len(va_sel))
    row["n_val_pairs_calibration"] = int(len(va_cal))
    row["split_validation_for_calibration"] = cfg.split_validation_for_calibration
    # ``heldout_`` and not ``test_``: Phase 2's rows use ``test_`` for the same
    # kind of quantity against a different target, and identical column names on
    # two experiments is how a pooled table happens. It also keeps the metric
    # names from colliding with pytest's ``test_*`` namespace, which the docs
    # integrity check parses.
    row.update({f"heldout_{k}": v for k, v in m.items()})
    row.update({f"cal_{k}": v for k, v in m_cal.items()})
    row.update({f"raw_{k}": v for k, v in m_raw.items()})
    row.update({f"dec_{k}": v for k, v in dec.items()})
    row.update({f"split_{k}": v
                for k, v in connectivity_report(split, screen.n_drugs).items()})
    row["config"] = json.loads(json.dumps(asdict(cfg), default=str))
    return row
