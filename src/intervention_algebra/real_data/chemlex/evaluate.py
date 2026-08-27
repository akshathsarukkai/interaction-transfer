"""Metrics, the incremental contrast, and the two diagnostics that police it.

The one metric everything turns on
----------------------------------
::

    incremental pair skill = 1 - MSE(pair model) / MSE(baseline)

computed from **paired predictions on the identical rows**, by the identical
fold, from models fitted on the identical training set. It is not a difference
of two separately-reported skills and it is not a skill against zero.

Phase 3 shipped a decision rule whose validity gate read skill-against-zero, and
the gate fired on a control containing no chemistry at all: the random
representation posted +0.204 against the zero predictor, because most of that
target's energy was a per-entity main effect that any smooth function of any
feature can partly fit. Measured as an *increment over the additive model* the
same control sits at -0.0007. The distinction is the whole reason this module
exists, and :func:`incremental` refuses to accept unpaired inputs.

The projection diagnostic
-------------------------
A bilinear form can hide lower-order effects inside its latent coordinates:
``z_A(x_a)^T W z_N(x_n)`` contains, as special cases, functions that depend on
the acid alone or the amine alone. So a pair model can beat the additive
baseline without having learned any pair-specific structure at all -- it may
simply have fitted the *substrate* effects better, using the pair term as spare
capacity.

:func:`additive_projection` settles it without touching an outcome. It takes the
pair model's **predictions** and fits the best possible additive surface to them
-- free per-entity intercepts, which is strictly more flexible than the additive
model's feature-derived heads -- then reports how much of the prediction cannot
be written that way, and what happens to test error when the non-additive part
is removed. If projecting away the non-additive component costs nothing, the
pair model has demonstrated no interaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Metric names produced by :func:`continuous_metrics`, in reporting order.
CONTINUOUS_METRICS: tuple[str, ...] = (
    "n", "mse", "rmse", "mae", "pearson", "spearman", "r2", "pred_sd", "true_sd")
BINARY_METRICS: tuple[str, ...] = (
    "n", "pos_rate", "auroc", "auprc", "balanced_accuracy", "log_loss", "brier")


def _finite(*arrays: np.ndarray) -> None:
    for a in arrays:
        if not np.isfinite(np.asarray(a, dtype=np.float64)).all():
            raise ValueError("non-finite values in a metric input")


def continuous_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError(f"y {y.shape} and pred {pred.shape} are not aligned")
    _finite(y, pred)
    n = int(y.size)
    if n == 0:
        return {k: float("nan") for k in CONTINUOUS_METRICS} | {"n": 0}
    err = pred - y
    mse = float((err ** 2).mean())
    var = float(y.var())
    out = {"n": n, "mse": mse, "rmse": float(np.sqrt(mse)),
           "mae": float(np.abs(err).mean()),
           "pred_sd": float(pred.std()), "true_sd": float(y.std()),
           "r2": 1.0 - mse / var if var > 0 else float("nan")}
    # Pearson/Spearman are undefined on a constant vector and scipy warns rather
    # than raising. Returning NaN is the honest answer and keeps the row parsable.
    if n >= 3 and y.std() > 0 and pred.std() > 0:
        out["pearson"] = float(stats.pearsonr(y, pred)[0])
        out["spearman"] = float(stats.spearmanr(y, pred)[0])
    else:
        out["pearson"] = out["spearman"] = float("nan")
    return out


def binary_metrics(label: np.ndarray, prob: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 roc_auc_score)
    label = np.asarray(label, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if label.shape != prob.shape:
        raise ValueError("label and prob are not aligned")
    _finite(prob)
    n = int(label.size)
    if n == 0:
        return {k: float("nan") for k in BINARY_METRICS} | {"n": 0}
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    out = {"n": n, "pos_rate": float(label.mean()),
           "log_loss": float(-(label * np.log(p) + (1 - label) * np.log(1 - p)).mean()),
           "brier": float(((p - label) ** 2).mean())}
    if 0 < label.sum() < n:
        out["auroc"] = float(roc_auc_score(label, prob))
        out["auprc"] = float(average_precision_score(label, prob))
        out["balanced_accuracy"] = float(balanced_accuracy_score(label, p >= 0.5))
    else:
        out["auroc"] = out["auprc"] = out["balanced_accuracy"] = float("nan")
    return out


def incremental(y: np.ndarray, pred_base: np.ndarray, pred_pair: np.ndarray,
                loss: str = "mse") -> float:
    """``1 - L(pair) / L(baseline)`` on one set of rows.

    The three arrays must be row-aligned: same rows, same order, same fold. A
    caller that assembles them from two separately-filtered frames will get
    silently wrong answers, so the shapes are checked and a length mismatch is
    an error rather than a broadcast.
    """
    y = np.asarray(y, dtype=np.float64)
    pb = np.asarray(pred_base, dtype=np.float64)
    pp = np.asarray(pred_pair, dtype=np.float64)
    if not (y.shape == pb.shape == pp.shape):
        raise ValueError(f"incremental skill needs paired predictions; got "
                         f"y {y.shape}, base {pb.shape}, pair {pp.shape}")
    if y.size == 0:
        return float("nan")
    if loss == "mse":
        lb = float(((pb - y) ** 2).mean())
        lp = float(((pp - y) ** 2).mean())
    elif loss == "log_loss":
        cb, cp = np.clip(pb, 1e-7, 1 - 1e-7), np.clip(pp, 1e-7, 1 - 1e-7)
        lb = float(-(y * np.log(cb) + (1 - y) * np.log(1 - cb)).mean())
        lp = float(-(y * np.log(cp) + (1 - y) * np.log(1 - cp)).mean())
    elif loss == "brier":
        lb = float(((pb - y) ** 2).mean())
        lp = float(((pp - y) ** 2).mean())
    else:
        raise ValueError(f"unknown loss {loss!r}")
    if lb <= 0:
        return float("nan")
    return 1.0 - lp / lb


def per_entity_incremental(frame: pd.DataFrame, role: str, y: np.ndarray,
                           pred_base: np.ndarray, pred_pair: np.ndarray,
                           min_rows: int = 3, loss: str = "mse") -> pd.DataFrame:
    """Incremental skill computed **within each held-out entity**.

    The inferential unit. Reaction rows sharing one held-out acid are not
    independent evidence: they share a substrate, often a plate, and on this
    screen a single acid can carry 200 rows. Treating rows as replicates would
    turn one well-behaved acid into a significant result.

    Entities with fewer than ``min_rows`` test rows are returned but flagged, so
    a caller can see how many were dropped rather than discovering a silently
    smaller n.
    """
    if role not in ("acid", "amine"):
        raise ValueError(f"role must be acid or amine, not {role!r}")
    d = pd.DataFrame({"entity": frame[role].to_numpy(), "y": y,
                      "base": pred_base, "pair": pred_pair})
    rows = []
    for ent, g in d.groupby("entity", sort=True):
        rows.append({
            "role": role, "entity": int(ent), "n_rows": len(g),
            "usable": len(g) >= min_rows,
            "mse_base": float(((g["base"] - g["y"]) ** 2).mean()),
            "mse_pair": float(((g["pair"] - g["y"]) ** 2).mean()),
            "incremental": incremental(g["y"].to_numpy(), g["base"].to_numpy(),
                                       g["pair"].to_numpy(), loss=loss),
            "true_sd": float(g["y"].std()) if len(g) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def paired_summary(values: np.ndarray, label: str = "") -> dict:
    """Mean, SD, paired 95 % CI, t, Wilcoxon and the count favouring the pair model.

    ``values`` is one number per inferential unit (entity or fold), never one per
    row. The Wilcoxon is reported alongside the t because incremental skills are
    a ratio and their tail is not Gaussian; disagreement between the two is
    itself informative and is left visible rather than resolved by picking one.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    out = {"label": label, "n": n, "mean": float("nan"), "sd": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"),
           "p_ttest": float("nan"), "p_wilcoxon": float("nan"),
           "n_positive": 0, "frac_positive": float("nan"),
           "median": float("nan")}
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    out["median"] = float(np.median(v))
    out["n_positive"] = int((v > 0).sum())
    out["frac_positive"] = float((v > 0).mean())
    if n < 2:
        return out
    out["sd"] = float(v.std(ddof=1))
    se = out["sd"] / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    out["ci_lo"] = out["mean"] - tcrit * se
    out["ci_hi"] = out["mean"] + tcrit * se
    if out["sd"] > 0:
        out["p_ttest"] = float(stats.ttest_1samp(v, 0.0).pvalue)
        if np.any(v != 0):
            try:
                out["p_wilcoxon"] = float(stats.wilcoxon(v).pvalue)
            except ValueError:
                pass
    return out


def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None = None,
                 n_boot: int = 2000, seed: int = 20260826,
                 statistic=np.mean) -> dict:
    """Percentile bootstrap, resampling ``groups`` rather than rows when given.

    ``groups`` is the congener-family id. Twenty near-identical analogues are not
    twenty independent demonstrations, and resampling entities would treat them
    as such. Phase 3 learned this the expensive way: its similarity effect
    survived an entity bootstrap and had to be re-checked under a
    congener-family bootstrap before it could be reported.
    """
    v = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(v)
    v = v[keep]
    if v.size == 0:
        return {"n": 0, "point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_units": 0}
    rng = np.random.default_rng(seed)
    if groups is None:
        units = [np.array([i]) for i in range(v.size)]
    else:
        g = np.asarray(groups)[keep]
        units = [np.flatnonzero(g == u) for u in np.unique(g)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[p] for p in pick])
        draws[b] = statistic(v[idx])
    return {"n": int(v.size), "n_units": len(units),
            "point": float(statistic(v)),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5))}


def _design(frame: pd.DataFrame, terms: tuple[str, ...]):
    """Sparse indicator design matrix for the requested additive terms.

    ``terms`` names columns or ``"acid:cond"``-style interactions. Every level
    present in the rows gets its own column, which makes the projection strictly
    more flexible than any feature-derived head -- deliberately so: the
    diagnostic is meant to give the additive hypothesis its best shot.

    Sparse, and solved iteratively, because the dense form is badly
    rank-deficient by construction: with an intercept and a full set of
    indicators per term, every block sums to the intercept column. On the
    condition-expanded projection -- ``acid + amine + cond + acid:cond +
    amine:cond``, several thousand columns over ~1,400 rows -- LAPACK's
    divide-and-conquer SVD failed to converge on one authoritative fold and took
    the whole condition down with it. A diagnostic must not be able to do that.
    """
    import scipy.sparse as sps

    n = len(frame)
    blocks = [sps.csr_matrix(np.ones((n, 1)))]
    for t in terms:
        parts = t.split(":")
        key = frame[parts[0]].astype(str)
        for extra in parts[1:]:
            key = key + "\x00" + frame[extra].astype(str)
        codes, _ = pd.factorize(key)
        blocks.append(sps.csr_matrix(
            (np.ones(n), (np.arange(n), codes)), shape=(n, codes.max() + 1)))
    return sps.hstack(blocks, format="csr")


def _project(X, y: np.ndarray) -> np.ndarray:
    """Least-squares projection of ``y`` onto the column space of ``X``.

    LSQR on the sparse design. It is stable on a rank-deficient system -- it
    converges to a least-squares solution rather than failing to factor one --
    and the *fitted values* are what this diagnostic uses, which are unique even
    when the coefficients are not.
    """
    from scipy.sparse.linalg import lsqr

    beta = lsqr(X, y, atol=1e-12, btol=1e-12, iter_lim=5000)[0]
    return X @ beta


def additive_projection(frame: pd.DataFrame, pred_pair: np.ndarray,
                        y: np.ndarray, pred_base: np.ndarray,
                        terms: tuple[str, ...] = ("acid", "amine", "cond")
                        ) -> dict:
    """Project the pair model's predictions onto free additive effects.

    **No outcome is used to fit the projection.** The design matrix and the
    target are both functions of the pair model's predictions and the row's
    entity/condition ids, so this cannot manufacture skill from the labels.

    Returns

    ``nonadditive_fraction``
        fraction of the prediction's variance that no additive surface over
        these terms can represent. 0.0 means the pair model is, on these rows,
        exactly an additive model.
    ``incremental``
        the pair model's real incremental skill, for reference.
    ``incremental_projected``
        the incremental skill of the *projected* prediction. If this equals
        ``incremental``, the gain is additive-representable and the pair term
        has demonstrated nothing pair-specific. If it collapses towards zero,
        the gain lives in the non-additive component.
    ``gain_in_nonadditive``
        ``incremental - incremental_projected``: the part of the gain that
        survives only because the prediction is not additive.
    """
    pred_pair = np.asarray(pred_pair, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pred_base = np.asarray(pred_base, dtype=np.float64)
    if not (len(frame) == pred_pair.size == y.size == pred_base.size):
        raise ValueError("additive_projection needs row-aligned inputs")
    if pred_pair.size == 0:
        return {"n": 0, "nonadditive_fraction": float("nan")}
    X = _design(frame, terms)
    try:
        proj = _project(X, pred_pair)
    except Exception as exc:                            # noqa: BLE001
        # A diagnostic must never take a result row down with it. Returning the
        # failure as data keeps the fold's primary numbers, which do not depend
        # on this, and makes the gap visible in the table instead of as a
        # missing condition.
        return {"n": int(pred_pair.size), "terms": ",".join(terms),
                "nonadditive_fraction": float("nan"),
                "projection_error": f"{type(exc).__name__}: {exc}"}
    resid = pred_pair - proj
    var = float(pred_pair.var())
    inc = incremental(y, pred_base, pred_pair)
    inc_proj = incremental(y, pred_base, proj)
    return {
        "n": int(pred_pair.size), "terms": ",".join(terms),
        "n_design_columns": int(X.shape[1]), "projection_error": "",
        "nonadditive_fraction": float(resid.var() / var) if var > 0 else float("nan"),
        "nonadditive_sd": float(resid.std()),
        "prediction_sd": float(pred_pair.std()),
        "incremental": inc, "incremental_projected": inc_proj,
        "gain_in_nonadditive": (inc - inc_proj
                                if np.isfinite(inc) and np.isfinite(inc_proj)
                                else float("nan")),
        # Does the non-additive component point where the baseline is wrong? A
        # positive correlation is what "the pair term is correcting a
        # pair-specific error" looks like.
        "corr_nonadditive_with_base_error": (
            float(stats.pearsonr(resid, y - pred_base)[0])
            if resid.std() > 0 and (y - pred_base).std() > 0 else float("nan")),
    }
