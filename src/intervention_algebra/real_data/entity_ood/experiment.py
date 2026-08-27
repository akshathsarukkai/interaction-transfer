"""One Phase 3 condition, end to end: screen x representation x model x fold.

Order of operations, which is the whole safety argument
------------------------------------------------------
Nothing is fitted before the entities are separated. The sequence is fixed and
each step can only see what the step before it allowed:

1. load the screen and form the canonical ``i < j`` directional frame. This
   touches every pair, and is the last step that does -- it is pure ingestion,
   fits nothing, and reuses Phase 2R's own
   :func:`~intervention_algebra.real_data.residual.directional_pairs` so that the
   two phases cannot disagree about what ``D`` means or which way it points;
2. build the entity fold and assert no drug leakage;
3. build the feature *view* from the **training drugs only**;
4. compute the target scale from the **training pairs only**;
5. search hyperparameters, scoring each on entity-OOD validation pairs;
6. choose the shrinkage coefficient on the same validation pairs;
7. only now touch the test pairs, and score E1 and E2 separately.

There is no residualisation step, and that absence is deliberate rather than an
omission. Phase 2R subtracts a per-drug potential fitted by ridge from the
training rows; for a drug with no training rows that ridge returns ``g_k = 0``
silently, leaving the unseen drug's entire potential inside its "residual". The
model would then be scored on a target that means something different for
held-out drugs than for training drugs. So Phase 3 predicts raw ``D`` and makes
the model earn the potential from features, where it can be measured.

What the numbers mean
---------------------
``skill = 1 - MSE_model / MSE_zero`` against the zero predictor, and
``incremental_skill = 1 - MSE_lowrank / MSE_potential`` against the feature
potential. The second is the primary quantity. Skill against zero can be large
and entirely uninformative about interaction, because most of ``D`` is potential;
incremental skill is positive only if knowing *which two* drugs are being
combined adds something to knowing what each drug tends to do on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import koplev, residual
from ..residual_experiment import residual_metrics
from ..residual_train import (SHRINKAGE, ResidualTrainConfig, select_shrinkage,
                              train_residual)
from . import features as feat
from . import models as mdl
from . import splits as sp

DEFAULT_MAPPING = Path(__file__).resolve().parents[4] / "data" / "external" / "koplev_drug_mapping.csv"

#: Held frozen for the whole phase. Longer than Phase 2R's 800 because the heads
#: here start at exactly zero and have to travel further, and because 3,160
#: full-batch steps are cheap. Fixed by watching validation loss flatten on
#: *development* folds of a partition seed that no reported result uses.
TRAIN = ResidualTrainConfig(max_epochs=1200, n_restarts=2, eval_every=5)


@dataclass(frozen=True)
class EntityConfig:
    screen: str
    model: str
    partition: int
    fold: int
    representation: str = "ecfp4"
    coverage: float = 1.0
    seed: int = 0
    tag: str = "primary"
    #: Fold geometry. Defaults are the pre-registered design; the CI fixture has
    #: 12 drugs and overrides them. Carried on the config rather than patched in
    #: by tests, so the code path CI exercises is the code path that runs.
    n_partitions: int = 3
    n_test: int = 10
    n_val: int = 10
    split_seed: int = 20260825
    control_seed: int = 20260825
    #: Positive control: replace the real target with a synthetic one generated
    #: from the same features with a known potential and a known rank-2 form.
    synthetic_target: bool = False
    synthetic_noise_sd: float = 0.05


def select_fold(cfg: EntityConfig, n_drugs: int) -> sp.DrugFold:
    """The fold named by ``(partition, fold)``, found by matching, not by index.

    Index arithmetic here would silently assume ten folds per partition and
    return the wrong fold for any other geometry -- including the CI fixture's.
    """
    folds = sp.make_drug_folds(n_drugs, n_partitions=cfg.n_partitions,
                               n_test=cfg.n_test, n_val=cfg.n_val,
                               seed=cfg.split_seed, coverage=cfg.coverage)
    for f in folds:
        if f.partition == cfg.partition and f.fold == cfg.fold:
            return f
    raise ValueError(f"no fold (partition={cfg.partition}, fold={cfg.fold}) in "
                     f"{cfg.n_partitions} partitions of {n_drugs // cfg.n_test}")


def _pair_tensors(frame: pd.DataFrame, scale: float):
    """``(i, j, D/scale)`` tensors from a canonical ``i < j`` frame.

    Phase 2R's :func:`~intervention_algebra.real_data.residual_train.pair_tensors`
    reads a column named ``D_res``. Phase 3's target is raw ``D``, not a
    residual, so a local builder is used rather than renaming the column -- a
    frame carrying raw ``D`` under the name ``D_res`` would be a standing
    invitation for a later reader to compare the two phases' numbers as though
    they were the same quantity.
    """
    from ..residual_train import PairTensors, _long

    return PairTensors(
        i=_long(frame["i"]), j=_long(frame["j"]),
        d=torch.as_tensor(frame["D_true"].to_numpy() / scale, dtype=torch.float32))


def feature_view(x: np.ndarray, train_drugs: tuple[int, ...]) -> np.ndarray:
    """Column mask: features that vary across the **training** drugs.

    A fingerprint bit that is constant over the training drugs carries no
    learnable signal -- in the potential term it cancels in ``g(x_i) - g(x_j)``,
    and in the bilinear term it is a direction the training data cannot
    constrain. Dropping such bits shrinks 2,048 columns to a few hundred, which
    matters when there are only ~3,160 training pairs.

    Fitted on training drugs only. It reads no outcome at all -- a fingerprint is
    a function of the molecular graph -- so this could defensibly have been fitted
    on all 100. It is not, because "the preprocessing never saw a test drug" is a
    property worth being able to state without qualification, and it costs
    nothing.
    """
    sub = x[list(train_drugs)]
    return sub.std(axis=0) > 0


def _synthetic_target(pairs: pd.DataFrame, x: np.ndarray, seed: int,
                      noise_sd: float, rank: int = 2) -> pd.DataFrame:
    """Positive control: a target that IS a feature potential plus a rank-2 form.

    ``D = g(x_i) - g(x_j) + z_i^T K z_j + noise`` with ``g`` and ``z`` random
    linear maps of the real fingerprints and ``K`` antisymmetric. The pipeline
    must recover this; if it cannot, a null on the real data says nothing about
    chemistry and everything about the machinery.

    The two terms are scaled to comparable RMS so that a method which recovers
    only the potential cannot post a high incremental skill by accident.
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    w = rng.normal(size=d) / np.sqrt(d)
    z = x @ (rng.normal(size=(d, rank)) / np.sqrt(d))
    W = rng.normal(size=(rank, rank))
    K = W - W.T
    g = x @ w
    i = pairs["i"].to_numpy()
    j = pairs["j"].to_numpy()
    pot = g[i] - g[j]
    inter = np.einsum("na,ab,nb->n", z[i], K, z[j])
    pot = pot / (pot.std() + 1e-12)
    inter = inter / (inter.std() + 1e-12)
    out = pairs.copy()
    out["D_true"] = pot + inter + rng.normal(scale=noise_sd, size=len(pairs))
    return out


def build_features(cfg: EntityConfig, mapping: pd.DataFrame) -> feat.DrugFeatures:
    base = feat.fingerprint_matrix(mapping)
    if cfg.representation == "ecfp4":
        return base
    if cfg.representation == "random":
        return feat.random_features(base, cfg.control_seed)
    if cfg.representation == "shuffled":
        return feat.shuffled_features(base, cfg.control_seed)
    if cfg.representation.startswith("targets"):
        from .targets import target_features

        return target_features(mapping, shuffled=cfg.representation.endswith("shuffled"),
                               seed=cfg.control_seed)
    raise ValueError(f"unknown representation {cfg.representation!r}")


def run_entity_condition(cfg: EntityConfig,
                         raw_dir: Path = koplev.DEFAULT_RAW_DIR,
                         mapping_path: Path = DEFAULT_MAPPING,
                         screen: koplev.Screen | None = None,
                         mapping: pd.DataFrame | None = None) -> dict:
    """Fit and evaluate one condition. Returns one JSONL row."""
    t0 = time.time()
    screen = screen if screen is not None else koplev.load_screen(cfg.screen, raw_dir)
    mapping = mapping if mapping is not None else pd.read_csv(mapping_path)
    n = screen.n_drugs

    pairs = residual.directional_pairs(screen.frame)      # i < j, D_true, one row per pair
    fold = select_fold(cfg, n)
    sp.assert_no_drug_leakage(fold, n)

    features = build_features(cfg, mapping)
    if features.x.shape[0] != n:
        raise ValueError("feature matrix does not match the screen's drug count")
    # The single most dangerous silent failure in this phase: if the mapping's
    # row order ever stopped matching the screen's canonical drug order, every
    # drug would be paired with another drug's structure. That is Control B --
    # the shuffled-feature negative control -- run by accident, and it would look
    # exactly like "chemistry does not transfer". Asserted here, at the point of
    # use, rather than only in a test.
    if tuple(features.labels) != tuple(screen.drugs):
        raise ValueError(
            "the drug mapping's row order does not match the screen's drug order; "
            "features would be attached to the wrong molecules")

    if cfg.synthetic_target:
        pairs = _synthetic_target(pairs, features.x, cfg.control_seed,
                                  cfg.synthetic_noise_sd)

    mask = feature_view(features.x, fold.train_drugs)
    x = np.ascontiguousarray(features.x[:, mask])

    tr = fold.rows(pairs, "train")
    va = fold.rows(pairs, "val")
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"{fold.key}: empty train or validation set")

    scale = float(np.sqrt((tr["D_true"].to_numpy() ** 2).mean()))
    tr_t, va_t = _pair_tensors(tr, scale), _pair_tensors(va, scale)

    grid = mdl.HPARAM_GRID[cfg.model]
    fits, val_losses = [], []
    for hp in grid:
        arch, opt = mdl.split_hparams(hp)

        def build(seed: int, arch=arch):
            return mdl.build_entity_model(
                cfg.model, mdl.EntityModelConfig(n_drugs=n, x=x, seed=seed, **arch))

        fit = train_residual(build, tr_t, va_t, replace(TRAIN, **opt), seed=cfg.seed)
        fits.append((fit, hp))
        val_losses.append(fit.val_loss)
    best_idx = int(np.argmin(val_losses))
    fit, best_hp = fits[best_idx]
    model = fit.model

    def predict(frame: pd.DataFrame) -> np.ndarray:
        from ..residual_train import _long

        if len(frame) == 0:
            return np.zeros(0)
        model.eval()
        with torch.no_grad():
            return model.d_res(_long(frame["i"]), _long(frame["j"])).numpy() * scale

    alpha = select_shrinkage(va["D_true"].to_numpy(), predict(va), SHRINKAGE)
    thr = koplev.measurement_noise_sd(raw_dir)["threshold_2sd_D"][cfg.screen]

    row: dict = {
        "tag": cfg.tag, "screen": cfg.screen, "model": cfg.model,
        "representation": cfg.representation, "partition": cfg.partition,
        "fold": cfg.fold, "fold_key": fold.key, "coverage": cfg.coverage,
        "seed": cfg.seed,
        "synthetic_target": cfg.synthetic_target,
        "n_params": int(model.n_params()), "feature_dim": int(mask.sum()),
        "feature_dim_full": int(features.x.shape[1]),
        "d_scale_train": scale, "alpha": float(alpha),
        "hparams": best_hp, "grid_size": len(grid),
        "grid_val_losses": [float(v) for v in val_losses],
        "best_epoch": fit.best_epoch, "val_loss": fit.val_loss,
        "train_loss": fit.train_loss,
        **sp.fold_summary(fold),
    }

    # The metal-excluded arm is a *re-scoring* of the same fit, not a second fit.
    # Excluding the four coordination complexes changes which test pairs are
    # averaged over; it does not change what the model learned, because those
    # drugs' features were available to training in exactly the same way either
    # way. Emitting both from one run keeps the arms exactly paired and costs
    # nothing -- refitting would have introduced optimiser noise between two
    # numbers whose difference is the whole point of the comparison.
    metals = set(_metal_drugs(mapping))
    row["n_metal_drugs"] = len(metals)
    for which, prefix in (("test_e1", "e1"), ("test_e2", "e2"), ("val", "va")):
        frame = fold.rows(pairs, which)
        if len(frame) == 0:
            continue
        d_true = frame["D_true"].to_numpy()
        d_pred = predict(frame) * alpha
        for k, v in residual_metrics(d_true, d_pred, thr).items():
            row[f"{prefix}_{k}"] = v
        keep = ~(frame["i"].isin(metals) | frame["j"].isin(metals))
        sub = frame.loc[keep].reset_index(drop=True)
        if len(sub) >= 10:
            for k, v in residual_metrics(sub["D_true"].to_numpy(),
                                         predict(sub) * alpha, thr).items():
                row[f"{prefix}x_{k}"] = v

    # How much of E1 is predictable with NO knowledge of the unseen drug at all?
    #
    # This turned out to be the question the experiment actually turns on. Of the
    # 900 E1 pairs in a fold, 800 pair a test drug with a *training* drug whose
    # ordering tendency the model learns perfectly well, and D(i, j) contains
    # that drug's potential with a minus sign. So a model that knows nothing
    # whatsoever about the new drug still scores well against zero -- and the
    # random-feature control proves it empirically by beating real fingerprints.
    #
    # (The remaining 100 pair a test drug with a *validation* drug, which appears
    # in no training pair either. Those rows are a second both-unseen regime
    # sitting inside E1, and are reported separately.)
    blind = _blind_metrics(model, fold, pairs, scale, alpha, thr, x)
    row.update(blind)

    # E1 is not homogeneous, and the documents said it was. A pair with exactly
    # one test endpoint lands in E1 whether its partner is a training drug or a
    # *validation* drug -- and validation drugs appear in no training pair at
    # all. So 800 of the 900 rows are test-x-trained and 100 are test-x-untrained,
    # a second both-unseen regime hidden inside the primary one. Reported
    # separately rather than left implicit, because the whole reinterpretation of
    # this phase rests on what the partner supplies.
    e1 = fold.rows(pairs, "test_e1")
    if len(e1):
        val = set(fold.val_drugs)
        touches_val = e1["i"].isin(val) | e1["j"].isin(val)
        d_pred_e1 = predict(e1) * alpha
        for mask, prefix in ((~touches_val, "e1tr"), (touches_val, "e1va")):
            sel = mask.to_numpy()
            if sel.sum() >= 10:
                for k, v in residual_metrics(e1["D_true"].to_numpy()[sel],
                                             d_pred_e1[sel], thr).items():
                    row[f"{prefix}_{k}"] = v

    # Post-hoc decomposition of the model's own predictions. Added after the
    # primary sweep revealed that the pair-only rung -- no potential head at all
    # -- scores within a hundredth of the potential model, i.e. a bilinear form
    # fits a potential perfectly well. Not part of the frozen decision rule; it
    # is what makes the word "pair-specific" checkable rather than assumed.
    e1 = fold.rows(pairs, "test_e1")
    if len(e1):
        d_true = e1["D_true"].to_numpy()
        d_pred = predict(e1) * alpha
        grad = gradient_projection(e1["i"].to_numpy(), e1["j"].to_numpy(), d_pred, n)
        curl = d_pred - grad
        denom = float((d_pred ** 2).sum())
        row["e1_pred_curl_fraction"] = float((curl ** 2).sum() / denom) if denom > 0 else 0.0
        row["e1_grad_mse"] = float(((grad - d_true) ** 2).mean())
        mse_zero = float((d_true ** 2).mean())
        row["e1_grad_skill"] = 1.0 - row["e1_grad_mse"] / mse_zero if mse_zero else float("nan")
        row["e1_curl_gain"] = (1.0 - row["e1_mse"] / row["e1_grad_mse"]
                               if row["e1_grad_mse"] > 0 else float("nan"))

    row["per_drug"] = _per_drug_e1(fold, pairs, predict, alpha, features, thr, set())
    row["elapsed_s"] = round(time.time() - t0, 2)
    return row


def _blind_metrics(model, fold, pairs: pd.DataFrame, scale: float, alpha: float,
                   thr: float, x: np.ndarray) -> dict:
    """E1 metrics when the model is given no information about the held-out drug.

    "No information" has to be defined carefully, and the obvious definition is
    wrong. Zeroing the held-out drug's feature row looks neutral and is not: it
    asserts "this drug is a molecule with **zero fingerprint bits**", a point no
    training drug occupies. Measured on the trained potential heads, the mean of
    ``g`` over drugs sits 0.49 / 0.23 standard deviations above what the zero row
    produces, so that substitution is a systematically pessimistic prediction
    rather than an uninformative one. Anything at all then beats it -- which is
    exactly what the random-feature control showed: features containing no
    chemistry scored a spurious "+0.052 attributable to the unseen drug"
    (p = 0.049) against the zero-row baseline.

    The information-free prediction is instead the **marginal over the training
    drugs**: replace the unseen endpoint by each of the 80 drugs the model
    trained on and average the prediction. That is the best a model can do
    knowing only that the drug is some drug from this population, it needs no
    linearity assumption, and it is on-distribution by construction.

    Computed exactly, not approximated by a mean feature row -- the two agree for
    a linear head and diverge once a hidden layer is involved, and 26 of 30
    lowrank folds chose the linear head but four did not.
    """
    from ..residual_train import _long

    e1 = fold.rows(pairs, "test_e1")
    if not len(e1) or not hasattr(model, "X"):
        return {}
    test = set(fold.test_drugs)
    train = np.asarray(fold.train_drugs)
    i = e1["i"].to_numpy()
    j = e1["j"].to_numpy()

    # For each E1 pair, substitute every training drug into the unseen position
    # and average. One forward pass over n_pairs * n_train index pairs.
    unseen_is_i = np.array([a in test for a in i])
    ii = np.where(unseen_is_i[:, None], train[None, :], i[:, None])
    jj = np.where(unseen_is_i[:, None], j[:, None], train[None, :])
    model.eval()
    with torch.no_grad():
        flat = model.d_res(_long(pd.Series(ii.ravel())),
                           _long(pd.Series(jj.ravel()))).numpy()
    pred = flat.reshape(ii.shape).mean(axis=1) * scale * alpha
    out = {f"e1_blind_{k}": v
           for k, v in residual_metrics(e1["D_true"].to_numpy(), pred, thr).items()}

    # The zero-row substitution is kept for the audit trail under a name that
    # says what it is -- the drug's contribution deleted -- and is not the
    # baseline any claim is made against.
    blind = x.copy()
    blind[list(fold.test_drugs)] = 0.0
    original = model.X
    try:
        model.X = torch.as_tensor(blind, dtype=torch.float32)
        with torch.no_grad():
            zpred = model.d_res(_long(e1["i"]), _long(e1["j"])).numpy() * scale * alpha
    finally:
        model.X = original
    out.update({f"e1_zeroed_{k}": v for k, v in
                residual_metrics(e1["D_true"].to_numpy(), zpred, thr).items()})
    return out


def gradient_projection(i: np.ndarray, j: np.ndarray, d_pred: np.ndarray,
                        n_drugs: int, ridge: float = 1e-6) -> np.ndarray:
    """The closest per-drug potential to a set of predictions: ``argmin_g |Dhat - (g_i - g_j)|``.

    Why this exists, and why it is not paranoia. A rank-2 antisymmetric bilinear
    form **contains the potential as a special case**: put ``z_i = (g_i, 1)`` and
    ``K = [[0, 1], [-1, 0]]`` and ``z_i' K z_j = g_i - g_j`` exactly. So "the
    low-rank model beats the potential model" does not by itself mean the pair
    term found pair structure -- the bilinear form may simply be fitting a
    *better potential*, along a different regularisation path, with no
    pair-specific content at all.

    The result files show this is a live worry rather than a theoretical one: the
    ``pair_only`` rung, which has no potential head whatsoever, reaches held-out
    skill within a hundredth of the potential model's. It is fitting a potential
    through its bilinear form.

    So this fits the best per-drug potential **to the model's own predictions**,
    using no outcome at all, and the caller reports how much skill survives the
    projection. If the full prediction still beats its own gradient projection,
    the surviving part is genuinely not expressible as "drug i tends to go
    first" -- which is the definition of pair-specific.

    The projection is deliberately generous: 99 free parameters (100 drugs, one
    gauge) against 900 E1 pairs, and it is handed the model's actual predictions.
    Anything it cannot absorb is not a potential.
    """
    n = len(d_pred)
    rows = np.repeat(np.arange(n), 2)
    cols = np.concatenate([i[:, None], j[:, None]], axis=1).ravel()
    vals = np.tile(np.array([1.0, -1.0]), n)
    X = np.zeros((n, n_drugs))
    X[rows, cols] = vals
    # Gauge: g is defined only up to a constant, so the normal equations are
    # singular. A tiny ridge selects the minimum-norm representative; the fitted
    # DIFFERENCES -- the only thing used -- are unaffected by the choice.
    g = np.linalg.solve(X.T @ X + ridge * np.eye(n_drugs), X.T @ d_pred)
    return g[i] - g[j]


def _metal_drugs(mapping: pd.DataFrame) -> list[int]:
    m = mapping.sort_values("drug_index")
    return [int(k) for k, flag in zip(m["drug_index"], m["contains_metal"]) if bool(flag)]


def _per_drug_e1(fold: sp.DrugFold, pairs: pd.DataFrame, predict, alpha: float,
                 features: feat.DrugFeatures, thr: float,
                 metals: set[int]) -> list[dict]:
    """Per-held-out-drug E1 performance, plus its distance from the training set.

    This is what makes "does it work for chemically distant drugs?" answerable
    rather than rhetorical, and it is also the guard against a fold-level average
    that is carried by one or two easy analogues. ``max_sim_to_train`` is the
    largest ECFP4 Tanimoto between the held-out drug and any *training* drug of
    this fold -- computed here, per fold, because the training set is what the
    model actually saw.
    """
    sim = feat.tanimoto_matrix(features)
    train = list(fold.train_drugs)
    out = []
    frame = fold.rows(pairs, "test_e1")
    if metals:
        keep = ~(frame["i"].isin(metals) | frame["j"].isin(metals))
        frame = frame.loc[keep].reset_index(drop=True)
    if len(frame) == 0:
        return out
    pred = predict(frame) * alpha
    i = frame["i"].to_numpy()
    j = frame["j"].to_numpy()
    d_true = frame["D_true"].to_numpy()
    for k in fold.test_drugs:
        sel = (i == k) | (j == k)
        if sel.sum() < 10:
            continue
        row = {"drug": int(k), "label": features.labels[k], "n_pairs": int(sel.sum()),
               "max_sim_to_train": float(np.nanmax(sim[k, train])),
               "median_sim_to_train": float(np.nanmedian(sim[k, train])),
               "bits_set": int(features.bits_set[k])}
        row.update({f"e1_{a}": b for a, b in
                    residual_metrics(d_true[sel], pred[sel], thr).items()})
        out.append(row)
    return out
