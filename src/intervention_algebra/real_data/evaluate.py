"""Metrics for the held-out-pair evaluation.

Three levels, because they answer different questions:

* **ordered outcome** -- can the model predict ``y(i -> j)`` at all;
* **directional effect** -- can it predict ``D_ij = y(i->j) - y(j->i)``, which is
  the part of the response that only a model with order structure can get right;
* **ordering accuracy** -- on pairs whose measured direction effect is above the
  screen's measurement noise, does it name the better-ordered schedule.

The magnitude threshold for the third metric is fixed from the deposited
``synergy_sd`` column before any model runs (see
:func:`..koplev.measurement_noise_sd`) and the accuracy-vs-threshold curve is
reported in full, so no threshold can be chosen after seeing a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _correct(d_pred: np.ndarray, d_true: np.ndarray) -> np.ndarray:
    """Per-pair ordering score, with an exact tie counted as a coin flip.

    The order-insensitive family predicts ``D = 0`` identically, and scoring that
    as *wrong* would report it at 0.000 -- which reads like anti-prediction
    rather than the no-information 0.5 it actually is. Ties are scored 0.5 for
    every family alike.
    """
    correct = (np.sign(d_pred) == np.sign(d_true)).astype(float)
    correct[d_pred == 0.0] = 0.5
    return correct


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def ordered_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {
        "n_rows": int(len(y)),
        "mae": float(np.abs(e).mean()),
        "mse": float((e ** 2).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "r2": _r2(y, p),
    }


def directional_frame(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """One row per unordered test pair, with true and predicted ``D``.

    Built by joining the two ordered rows of each pair on the canonical pair id.
    A pair missing either direction is an error, not something to impute -- the
    split guarantees both are present, and :func:`..splits.assert_no_pair_leakage`
    has already checked it.
    """
    df = frame.copy()
    df["pred"] = pred
    fwd = df[df["i"] < df["j"]].set_index(["i", "j"])
    rev = df[df["i"] > df["j"]].set_index(["j", "i"])
    rev.index.names = ["i", "j"]
    joined = fwd.join(rev, how="inner", lsuffix="_f", rsuffix="_r")
    if len(joined) != len(fwd) or len(joined) != len(rev):
        raise AssertionError(
            f"directional join is not one-to-one: {len(fwd)} forward, "
            f"{len(rev)} reverse, {len(joined)} matched")
    return pd.DataFrame({
        "i": joined.index.get_level_values(0),
        "j": joined.index.get_level_values(1),
        "D_true": joined["y_f"].to_numpy() - joined["y_r"].to_numpy(),
        "D_pred": joined["pred_f"].to_numpy() - joined["pred_r"].to_numpy(),
    })


def directional_metrics(d: pd.DataFrame, threshold: float,
                        curve_thresholds=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30)) -> dict:
    dt, dp = d["D_true"].to_numpy(), d["D_pred"].to_numpy()
    out: dict = {
        "n_pairs": int(len(dt)),
        "D_mae": float(np.abs(dp - dt).mean()),
        "D_rmse": float(np.sqrt(((dp - dt) ** 2).mean())),
        "D_true_std": float(dt.std()),
        "D_pred_std": float(dp.std()),
    }
    if len(dt) > 2 and dp.std() > 0 and dt.std() > 0:
        out["D_pearson"] = float(stats.pearsonr(dt, dp).statistic)
        out["D_spearman"] = float(stats.spearmanr(dt, dp).statistic)
    else:
        out["D_pearson"] = float("nan")
        out["D_spearman"] = float("nan")

    sel = np.abs(dt) > threshold
    out["ordering_threshold"] = float(threshold)
    out["ordering_n"] = int(sel.sum())
    out["ordering_frac_of_pairs"] = float(sel.mean())
    if sel.sum() >= 10:
        correct = _correct(dp[sel], dt[sel])
        out["ordering_accuracy"] = float(correct.mean())
        # Balanced across the two true signs: a model that always predicts the
        # majority direction cannot score well on this.
        pos, neg = dt[sel] > 0, dt[sel] < 0
        accs = [float(correct[m].mean()) for m in (pos, neg) if m.sum() > 0]
        out["ordering_balanced_accuracy"] = float(np.mean(accs)) if accs else float("nan")
    else:
        out["ordering_accuracy"] = float("nan")
        out["ordering_balanced_accuracy"] = float("nan")

    curve = {}
    for t in curve_thresholds:
        s = np.abs(dt) > t
        curve[f"{t:.2f}"] = {
            "n": int(s.sum()),
            "accuracy": float(_correct(dp[s], dt[s]).mean())
            if s.sum() >= 10 else float("nan"),
        }
    out["ordering_curve"] = curve
    return out


def antisymmetry_diagnostics(pred: np.ndarray, frame: pd.DataFrame,
                             d: pd.DataFrame) -> dict:
    """How much order-dependence the fitted model actually expresses.

    ``pred_A_rms`` is ``rms(D_pred) / 2`` -- the model's *total* antisymmetric
    output. It is a useful scale check but it is **not** the collapse detector:
    it includes the first-order term ``((a_i-b_i)-(a_j-b_j))/2`` that every
    family carries, so a run whose ``F_A`` is identically zero still reports a
    healthy value. Collapse is detected on the head alone, in
    :func:`..experiment.run_condition`, via ``pair_head_A``.
    """
    dp = d["D_pred"].to_numpy()
    dt = d["D_true"].to_numpy()
    return {
        "pred_A_rms": float(np.sqrt((dp ** 2).mean()) / 2),
        "true_A_rms": float(np.sqrt((dt ** 2).mean()) / 2),
        "pred_A_rms_ratio": float(np.sqrt((dp ** 2).mean())
                                  / max(np.sqrt((dt ** 2).mean()), 1e-12)),
    }


#: A run whose *pair head* emits an antisymmetric RMS under this fraction of the
#: head's symmetric RMS has effectively stopped modelling order through the head.
#: Chosen a priori as a round order-of-magnitude, not tuned.
COLLAPSE_RATIO = 0.05
