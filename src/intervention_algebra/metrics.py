"""Evaluation metrics.

Three distinct families of question are kept separate on purpose:

1. **Observable prediction** -- can the model predict the outcome of a pair it
   has never seen?  Always meaningful.  (MSE / RMSE / MAE, plus a skill score
   against the additive-only prediction.)

2. **Latent structure recovery** -- does the model's internal ``S`` / ``A``
   correlate with the true ones?  Only meaningful when the observation map is
   the identity, where the (v, S, A) decomposition is identified by the data
   (see docs/identifiability.md).  Reported as NaN otherwise, and NaN when the
   ground-truth quantity is identically zero (independent / single-regime
   systems), where a correlation has no defined value.

3. **Interaction-topology recovery** -- can the model tell which held-out pairs
   interact at all?  Scored with AUROC and AUPRC on ||S_pred|| / ||A_pred||.
   AUPRC is the more informative of the two under sparse interactions.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from .generator import ORD, SIM, SINGLE, ObservationTable, SyntheticSystem
from .models import BaseModel


MIN_PAIRS_FOR_CORR = 8


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Per-channel Pearson/Spearman correlation, averaged over channels.

    Correlating the *pooled* (pairs x channels) vectors would let differences in
    per-channel scale and mean masquerade as agreement, so each response channel
    is scored separately and the channel scores are averaged. NaN is returned
    rather than a number whenever the quantity is undefined: too few pairs, or
    either side constant (which is the case for the true A in a symmetric-only
    system, and for any prediction from the additive family).
    """
    a, b = np.atleast_2d(np.asarray(a)), np.atleast_2d(np.asarray(b))
    if a.shape[0] < MIN_PAIRS_FOR_CORR:
        return float("nan"), float("nan")
    pe, sp = [], []
    for c in range(a.shape[1]):
        x, y = a[:, c], b[:, c]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        pe.append(float(stats.pearsonr(x, y)[0]))
        sp.append(float(stats.spearmanr(x, y)[0]))
    if not pe:
        return float("nan"), float("nan")
    return float(np.mean(pe)), float(np.mean(sp))


def _auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """AUROC/AUPRC, or NaN where the question is not defined.

    A constant score vector (the additive family, whose interaction terms are
    identically zero) is *not* a ranking. sklearn would still return AUROC 0.5
    and AUPRC equal to the base rate, which reads in a results table as partial
    topology recovery by a model that has no interaction parameters at all.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max() or len(labels) < 4:
        return float("nan"), float("nan")
    if not np.isfinite(scores).all() or scores.std() < 1e-12:
        return float("nan"), float("nan")
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


@torch.no_grad()
def prediction_metrics(model: BaseModel, table: ObservationTable,
                       prefix: str = "") -> dict:
    out: dict[str, float] = {}
    all_err, all_y = [], []
    for kind, name in ((SIM, "sim"), (ORD, "ord"), (SINGLE, "single")):
        m = table.kind == kind
        if not m.any():
            continue
        i = torch.as_tensor(table.i[m], dtype=torch.long)
        j = torch.as_tensor(table.j[m], dtype=torch.long)
        y = table.y[m]
        if kind == SINGLE:
            p = model.predict_single(i).numpy()
        elif kind == SIM:
            p = model.predict_sim(i, j).numpy()
        else:
            p = model.predict_ordered(i, j).numpy()
        err = p - y
        out[f"{prefix}mse_{name}"] = float((err**2).mean())
        out[f"{prefix}mae_{name}"] = float(np.abs(err).mean())
        all_err.append(err.ravel())
        all_y.append(y.ravel())
    if all_err:
        e = np.concatenate(all_err)
        y = np.concatenate(all_y)
        out[f"{prefix}mse"] = float((e**2).mean())
        out[f"{prefix}rmse"] = float(np.sqrt((e**2).mean()))
        out[f"{prefix}mae"] = float(np.abs(e).mean())
        out[f"{prefix}r2"] = float(1.0 - (e**2).mean() / (y.var() + 1e-12))
    return out


@torch.no_grad()
def structure_metrics(model: BaseModel, system: SyntheticSystem,
                      pairs: np.ndarray, identifiable: bool = True,
                      prefix: str = "") -> dict:
    """Recovery of S, A and of the interaction topology on ``pairs`` (i < j)."""
    if len(pairs) == 0:
        return {}
    i = torch.as_tensor(pairs[:, 0], dtype=torch.long)
    j = torch.as_tensor(pairs[:, 1], dtype=torch.long)

    S_pred = model.implied_S(i, j).numpy()
    A_pred = model.implied_A(i, j).numpy()
    S_true = system.S[pairs[:, 0], pairs[:, 1]]
    A_true = system.A[pairs[:, 0], pairs[:, 1]]
    labels = system.mask[pairs[:, 0], pairs[:, 1]].astype(int)

    out: dict[str, float] = {
        f"{prefix}pred_S_rms": float(np.sqrt((S_pred**2).mean())),
        f"{prefix}pred_A_rms": float(np.sqrt((A_pred**2).mean())),
        f"{prefix}true_S_rms": float(np.sqrt((S_true**2).mean())),
        f"{prefix}true_A_rms": float(np.sqrt((A_true**2).mean())),
    }

    # Charitable alternative readout: symmetric projection of the raw simultaneous
    # term.  Identical to ``implied_S`` for every family whose simultaneous term is
    # already symmetric; reported so neither reading is hidden.
    S_proj = model.implied_S_projected(i, j).numpy()
    if identifiable and S_true.std() > 1e-12:
        out[f"{prefix}S_proj_pearson"] = _corr(S_true, S_proj)[0]
    else:
        out[f"{prefix}S_proj_pearson"] = float("nan")

    if identifiable:
        for name, t, p in (("S", S_true, S_pred), ("A", A_true, A_pred)):
            pe, sp = _corr(t, p)
            out[f"{prefix}{name}_pearson"] = pe
            out[f"{prefix}{name}_spearman"] = sp
            # Elementwise recovery error normalised by the true scale:
            # ||t - p|| / ||t||. Read it as "error relative to predicting zero":
            # exactly 1.0 is what a model that predicts no interaction at all
            # scores, so the additive family scores 1.000 by construction. Note
            # 1.0 is NOT unique to that case -- a perfectly *correlated*
            # prediction at twice the true scale also scores 1.0 -- so nrmse
            # must be read together with the correlation and with
            # pred_S_rms/true_S_rms, which separate shape from scale. Undefined
            # when the true quantity is identically zero (e.g. A in a
            # symmetric-only system) -- without this guard the ratio explodes to
            # ~1e4 and destroys any average taken over regimes.
            if t.std() < 1e-12:
                out[f"{prefix}{name}_nrmse"] = float("nan")
            else:
                denom = float((t**2).mean()) + 1e-12
                out[f"{prefix}{name}_nrmse"] = float(
                    np.sqrt(((t - p) ** 2).mean() / denom))
    else:
        for name in ("S", "A"):
            out[f"{prefix}{name}_pearson"] = float("nan")
            out[f"{prefix}{name}_spearman"] = float("nan")
            out[f"{prefix}{name}_nrmse"] = float("nan")

    for name, p in (("S", S_pred), ("A", A_pred)):
        auroc, auprc = _auc(labels, np.linalg.norm(p, axis=1))
        out[f"{prefix}topo_auroc_{name}"] = auroc
        out[f"{prefix}topo_auprc_{name}"] = auprc
    out[f"{prefix}topo_base_rate"] = float(labels.mean())

    # Top-k recovery of the strongest true interactions (by ||S_true||).  Both
    # sides must be non-degenerate: a model whose S is identically zero (the
    # additive family) would otherwise score chance-level overlap purely from
    # argsort tie-breaking, which reads as partial success.
    k = max(1, int(0.1 * len(pairs)))
    if S_true.std() > 1e-12 and S_pred.std() > 1e-12:
        true_rank = np.argsort(-np.linalg.norm(S_true, axis=1))[:k]
        pred_rank = np.argsort(-np.linalg.norm(S_pred, axis=1))[:k]
        out[f"{prefix}topk_S_overlap"] = float(
            len(set(true_rank.tolist()) & set(pred_rank.tolist())) / k)
    else:
        out[f"{prefix}topk_S_overlap"] = float("nan")
    return out


@torch.no_grad()
def invariance_diagnostics(model: BaseModel, pairs: np.ndarray) -> dict:
    """Regression guards on the architectural invariants, evaluated on real pairs.

    Read these for what they are. ``sym_residual`` and ``antisym_residual`` are
    **tautologically zero for every family**, because ``implied_S`` is
    order-invariant by construction everywhere (symmetrised in the algebra model,
    canonicalised in the others) and ``implied_A`` is defined as an
    antisymmetrised difference everywhere. They are therefore *not* evidence that
    the constrained model is special -- they are a tripwire that fires if a future
    edit breaks the definitions. The claim that the constraint is architectural is
    tested properly in ``tests/test_models.py``, which also asserts that the
    unconstrained families do **not** satisfy the forward/reverse identity.

    ``pair_net_order_asymmetry`` is rms of ``raw_sim(i,j) - raw_sim(j,i)``: the
    order-asymmetry of the underlying pair network's simultaneous output head,
    evaluated at a reversed input. For the unconstrained families this is an
    internal property that never reaches a prediction, since they canonicalise
    indices before predicting a simultaneous row -- their actual simultaneous
    predictions are exactly order-invariant. It is reported only as a descriptive
    measure of how close such a network came to learning symmetry unaided.
    """
    if len(pairs) == 0:
        return {}
    i = torch.as_tensor(pairs[:, 0], dtype=torch.long)
    j = torch.as_tensor(pairs[:, 1], dtype=torch.long)
    S_ij, S_ji = model.implied_S(i, j), model.implied_S(j, i)
    A_ij, A_ji = model.implied_A(i, j), model.implied_A(j, i)
    scale = float(S_ij.pow(2).mean().sqrt()) + 1e-9
    a_scale = float(A_ij.pow(2).mean().sqrt()) + 1e-9
    raw_ij, raw_ji = model.raw_sim_term(i, j), model.raw_sim_term(j, i)
    return {
        "sym_residual": float((S_ij - S_ji).pow(2).mean().sqrt()),
        "sym_residual_rel": float((S_ij - S_ji).pow(2).mean().sqrt()) / scale,
        "antisym_residual": float((A_ij + A_ji).pow(2).mean().sqrt()),
        "antisym_residual_rel": float((A_ij + A_ji).pow(2).mean().sqrt()) / a_scale,
        "pair_net_order_asymmetry": float((raw_ij - raw_ji).pow(2).mean().sqrt()),
    }


def interaction_variance_share(system: SyntheticSystem,
                               table: ObservationTable) -> float:
    """Fraction of the observed target variance that the interactions account for.

    The single most important number for interpreting any result on this
    benchmark. If interactions carry only a few percent of the variance then the
    additive model is close to optimal, every interaction-modelling family is
    mostly fitting noise, and a null result says nothing about the hypothesis --
    it says the benchmark was too easy for the null model. Recorded on every run
    so that the operating point is never implicit.
    """
    if len(table) == 0:
        return float("nan")
    z_full = table.z
    z_add = system.v[table.i] + system.v[table.j]
    single = table.kind == SINGLE
    z_add = z_add.copy()
    z_add[single] = system.v[table.i[single]]
    inter = z_full - z_add
    denom = float(z_full.var()) + 1e-12
    return float(inter.var() / denom)


def true_v_additive_mse(system: SyntheticSystem, table: ObservationTable) -> float:
    """MSE of the additive prediction built from the *true* ``v``, ignoring S and A.

    NOT an oracle and NOT a lower bound on additive performance.  The true ``v``
    is not the MSE-optimal additive predictor: because the interaction gate is
    correlated with the factors that generate ``v``, part of ``S`` is itself
    additively predictable, and a least-squares additive model fitted on the
    training rows can and does beat this quantity.  Empirically the trained
    ``additive`` family scores positive "skill" against it, so a positive skill
    score here does **not** demonstrate that a model has learned interactions.

    It is retained only as a fixed, model-independent reference scale for the
    size of the interaction signal.  The baseline that the headline comparison
    actually uses is the *trained* ``additive`` family, computed across runs in
    ``analysis.add_skill_vs_additive_family``.
    """
    z = system.v[table.i] + system.v[table.j]
    single = table.kind == SINGLE
    z[single] = system.v[table.i[single]]
    return float(((system.observe(z) - table.y) ** 2).mean())
