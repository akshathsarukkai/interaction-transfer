"""One condition of the Phase 4 sweep: one screen, one endpoint, one
representation, one fold, the whole model ladder.

Why a condition is a *fold* and not a *model*
---------------------------------------------
The load-bearing quantity is a ratio of two models' errors on identical rows.
Fitting the baseline and the pair model in separate units of work and joining
them afterwards is how a paired comparison quietly becomes an unpaired one --
different row order, a dropped fold, a mismatched grid. So a condition fits every
rung against one training set, predicts every rung on the same test rows, and
computes the contrasts before anything is written. A result row is
self-contained: it carries the paired incremental skills, not two skills for
someone else to subtract.

What a condition does *not* do
------------------------------
It never touches a test row before the fits are done, it never lets selection see
anything outside :data:`splits.SELECT_BUCKETS`, and it never re-fits for the
blind or projection diagnostics -- both are post-hoc transformations of a fitted
model's predictions, which is what makes them diagnostics rather than second
experiments.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch

from . import dataset as ds
from . import features as ft
from . import splits as sp
from .evaluate import (additive_projection, binary_metrics, continuous_metrics,
                       incremental, per_entity_incremental)
from .models import ModelConfig, build
from .train import (Batch, TrainConfig, fit_scaling, make_batch, predict,
                    train_with_grid)

#: Fold seed used while the pipeline was being built -- timing, training budget,
#: grid bracketing. Never used for an authoritative result.
DEV_SEED = 20260826
#: The authoritative fold seed. No model had been fitted on a fold drawn from it
#: before the pre-registration commit.
AUTH_SEED = 20260904

#: Registered fold geometry. k = 5 trades training rows for E2 power; the two
#: are the same quantity, and Phase 3's central limitation was an underpowered E2.
K_FOLDS = 5
N_PARTITIONS = 3

#: Weight decay is searched for **every** rung, so the baseline and the pair
#: model get the same regularisation freedom. On development folds the entity-OOD
#: validation optimum sat at 1e-2 for both, interior to this grid.
WEIGHT_DECAY_GRID: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
#: Latent width of any bilinear term.
RANK_GRID: tuple[int, ...] = (2, 4, 8)

#: The rungs a primary condition fits, in ladder order.
LADDER: tuple[str, ...] = (
    "condition_only", "additive", "lowrank", "flexible",
    "condition_expanded", "condition_expanded_pair")
#: The rungs a control condition fits. Controls only need the primary contrast;
#: running the full ladder on four control representations would quadruple the
#: sweep to answer a question nobody asked of a shuffled fingerprint.
CONTROL_LADDER: tuple[str, ...] = ("additive", "lowrank")

#: Registered contrasts, as (baseline, pair, name).
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("additive", "lowrank", "primary"),
    ("additive", "flexible", "flexible"),
    ("condition_expanded", "condition_expanded_pair", "robust"),
)

#: Buckets a primary condition reports on.
REPORT_BUCKETS: tuple[str, ...] = (
    "test_e1a", "test_e1n", "test_e2", "test_e2_mixed")

#: Which role's features the blind diagnostic replaces, per bucket.
BLIND_ROLES: dict[str, tuple[str, ...]] = {
    "test_e1a": ("acid",), "test_e1n": ("amine",),
    "test_e2": ("acid", "amine"), "test_e2_mixed": ("acid", "amine"),
}

#: Terms the additive projection is taken against, per baseline rung.
PROJECTION_TERMS: dict[str, tuple[str, ...]] = {
    "additive": ("acid", "amine", "cond"),
    "condition_expanded": ("acid", "amine", "cond", "acid:cond", "amine:cond"),
}


@dataclass(frozen=True)
class Spec:
    """One unit of work. Everything a worker needs and nothing it does not."""

    block: str                    # primary | control | positive | transductive
    screen: str = "all"
    encoding: str = "chemistry"
    endpoint: str = "yield"       # yield | feasible
    representation: str = "ecfp4"
    partition: int = 0
    fold: int = 0
    seed: int = AUTH_SEED
    k: int = K_FOLDS
    n_partitions: int = N_PARTITIONS
    ladder: tuple[str, ...] = LADDER
    #: Synthetic-target settings, used only by ``block == "positive"``.
    synthetic_rank: int = 3
    synthetic_scale: float = 1.0
    synthetic_seed: int = 7
    #: Restrict to amine entities whose N-H is a classical amine.
    classical_amines_only: bool = False
    #: Aggregate replicate rows of one (acid, amine, condition) cell to the mean.
    aggregate_cells: bool = False
    max_epochs: int = 800
    n_restarts: int = 2
    tag: str = ""

    @property
    def key(self) -> str:
        return (f"{self.block}/{self.screen}/{self.encoding}/{self.endpoint}/"
                f"{self.representation}/k{self.k}/p{self.partition}f{self.fold}"
                + (f"/{self.tag}" if self.tag else ""))


@dataclass
class Prepared:
    """Everything derived from the deposit that does not depend on the fold.

    Built once per (screen, encoding, filter) and reused across folds and
    representations. The split groups are always derived from the **real**
    fingerprints, never from a control representation -- defining the groups
    differently for the control than for the real run would make the two
    incomparable.
    """

    screen: ds.Screen
    acid_fp: ft.RoleFeatures
    amine_fp: ft.RoleFeatures
    acid_group: np.ndarray
    amine_group: np.ndarray
    folds: dict[tuple[int, int, int], sp.Fold] = field(default_factory=dict)


_CACHE: dict[tuple, Prepared] = {}


def prepare(spec: Spec, raw: pd.DataFrame | None = None) -> Prepared:
    ck = (spec.screen, spec.encoding, spec.classical_amines_only,
          spec.aggregate_cells)
    if ck in _CACHE:
        return _CACHE[ck]
    screen = ds.load_screen(spec.screen, spec.encoding, raw=raw)
    if spec.classical_amines_only or spec.aggregate_cells:
        screen = _refilter(screen, spec)
    acid_fp = ft.fingerprints(screen.acids, "acid")
    amine_fp = ft.fingerprints(screen.amines, "amine")
    prepared = Prepared(
        screen=screen, acid_fp=acid_fp, amine_fp=amine_fp,
        acid_group=sp.split_groups(screen.acids, acid_fp.x),
        amine_group=sp.split_groups(screen.amines, amine_fp.x))
    _CACHE[ck] = prepared
    return prepared


def _refilter(screen: ds.Screen, spec: Spec) -> ds.Screen:
    """Apply a registered sensitivity filter and **re-index the entities**.

    Re-indexing matters: dropping rows can drop an entity entirely, and leaving a
    gap in the index would leave a feature row that no row uses and a fold that
    holds out nothing.
    """
    f = screen.frame
    if spec.classical_amines_only:
        keep = f["amine_smiles"].map(ds.is_classical_amine).to_numpy()
        f = f.loc[keep].reset_index(drop=True)
    if spec.aggregate_cells:
        num = f.groupby(["acid", "amine", "cond"], as_index=False).agg(
            conversion=("conversion", "mean"))
        f = (f.drop_duplicates(["acid", "amine", "cond"])
               .drop(columns=["conversion", "y", "feasible"])
               .merge(num, on=["acid", "amine", "cond"], how="left")
               .reset_index(drop=True))
        f["y"] = f["conversion"] / ds.YIELD_SCALE
        f["feasible"] = (f["conversion"] >= ds.FEASIBLE_AT).astype(np.int64)
    acids = tuple(sorted(f["acid_smiles"].unique()))
    amines = tuple(sorted(f["amine_smiles"].unique()))
    ai = {s: i for i, s in enumerate(acids)}
    ni = {s: i for i, s in enumerate(amines)}
    f = f.copy()
    f["acid"] = f["acid_smiles"].map(ai).astype(np.int64)
    f["amine"] = f["amine_smiles"].map(ni).astype(np.int64)
    old = list(screen.conditions)
    present = sorted(set(f["cond"]))
    remap = {c: i for i, c in enumerate(present)}
    f["cond"] = f["cond"].map(remap).astype(np.int64)
    notes = dict(screen.notes)
    notes["filtered_rows"] = len(f)
    return ds.Screen(
        name=screen.name, encoding=screen.encoding,
        frame=f.sort_values(["acid", "amine", "cond"],
                            kind="stable").reset_index(drop=True),
        acids=acids, amines=amines,
        conditions=tuple(old[c] for c in present),
        condition_names=tuple(screen.condition_names[c] for c in present),
        n_raw_rows=screen.n_raw_rows, notes=notes)


def fold_of(prepared: Prepared, spec: Spec) -> sp.Fold:
    key = (spec.seed, spec.k, spec.n_partitions)
    if key not in prepared.folds:
        f = prepared.screen.frame
        folds = sp.make_folds(
            f["acid"].to_numpy(), f["amine"].to_numpy(),
            len(prepared.screen.acids), len(prepared.screen.amines),
            k=spec.k, n_partitions=spec.n_partitions, seed=spec.seed,
            acid_group=prepared.acid_group, amine_group=prepared.amine_group)
        prepared.folds[key] = {(fo.partition, fo.fold): fo for fo in folds}
    return prepared.folds[key][(spec.partition, spec.fold)]


def synthetic_target(prepared: Prepared, rank: int, scale: float,
                     seed: int) -> tuple[np.ndarray, dict]:
    """Additive terms plus a planted rank-``r`` interaction, on the real graph.

    Both parts are linear functions of the **real fingerprints**, so the planted
    structure is exactly the kind :class:`~.models.LowRankPair` can express and a
    failure to recover it is a failure of the evaluation geometry rather than of
    the hypothesis class. Nothing about the fold is used, so the same target
    serves every fold.
    """
    f = prepared.screen.frame
    xa, xn = prepared.acid_fp.x, prepared.amine_fp.x
    rng = np.random.default_rng(seed)
    d = xa.shape[1]

    def proj(x, cols):
        z = x @ rng.normal(size=(d, cols))
        return (z - z.mean(0)) / (z.std(0) + 1e-12)

    a = proj(xa, 1)[:, 0]
    n = proj(xn, 1)[:, 0]
    za, zn = proj(xa, rank), proj(xn, rank)
    ai = f["acid"].to_numpy()
    ni = f["amine"].to_numpy()
    ci = f["cond"].to_numpy()
    cond = rng.normal(size=len(prepared.screen.conditions))
    inter = (za[ai] * zn[ni]).sum(1)
    inter = inter / inter.std()
    base = a[ai] + n[ni] + 0.5 * cond[ci]
    noise = rng.normal(size=len(f))
    y = base + scale * inter + noise
    return y, {"planted_rank": rank, "planted_scale": scale,
               "interaction_sd_fraction": float(scale * inter.std() / y.std()),
               "additive_sd_fraction": float(base.std() / y.std())}


def _blinded(prepared: Prepared, fold: sp.Fold, roles: tuple[str, ...]
             ) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrices with the held-out entities replaced by the training marginal."""
    xa, xn = prepared.acid_fp.x, prepared.amine_fp.x
    if "acid" in roles:
        xa = ft.blind_features(prepared.acid_fp,
                               np.array(fold.train_acids),
                               np.array(fold.test_acids)).x
    if "amine" in roles:
        xn = ft.blind_features(prepared.amine_fp,
                               np.array(fold.train_amines),
                               np.array(fold.test_amines)).x
    return xa, xn


def run(spec: Spec, raw: pd.DataFrame | None = None) -> dict:
    """Fit the ladder on one fold and return one self-contained result row."""
    if spec.block == "transductive":
        return run_transductive(spec, raw=raw)
    t0 = time.time()
    torch.set_num_threads(1)
    prepared = prepare(spec, raw=raw)
    screen = prepared.screen
    frame = screen.frame
    fold = fold_of(prepared, spec)

    sp.assert_no_entity_leakage(fold, frame["acid"].to_numpy(),
                                frame["amine"].to_numpy(), len(frame))

    rep = ft.build_representation(spec.representation, prepared.acid_fp,
                                  prepared.amine_fp,
                                  seed=representation_seed(spec))

    if spec.block == "positive":
        target, synth_notes = synthetic_target(
            prepared, spec.synthetic_rank, spec.synthetic_scale,
            spec.synthetic_seed)
        binary = False
    else:
        synth_notes = {}
        binary = spec.endpoint == "feasible"
        target = (frame["feasible"].to_numpy(dtype=np.float64) if binary
                  else frame["y"].to_numpy())

    tr = fold.mask("train")
    va = fold.mask(sp.SELECT_BUCKETS)
    scaling = fit_scaling(target[tr]) if not binary else _identity_scaling()
    btr = make_batch(frame["acid"][tr], frame["amine"][tr], frame["cond"][tr],
                     target[tr], scaling)
    bva = make_batch(frame["acid"][va], frame["amine"][va], frame["cond"][va],
                     target[va], scaling)

    cfg = ModelConfig(n_acids=len(screen.acids), n_amines=len(screen.amines),
                      n_conditions=len(screen.conditions),
                      x_acid=rep.acid.x, x_amine=rep.amine.x)
    tcfg = TrainConfig(max_epochs=spec.max_epochs, n_restarts=spec.n_restarts)

    fits, selection = {}, {}
    for name in spec.ladder:
        grid = _grid_for(name)
        fit = train_with_grid(name, cfg, btr, bva, tcfg,
                              seed=hash_seed(spec), grid=grid,
                              binary=binary)
        fits[name] = fit
        selection[name] = {"hparams": fit.hparams, "val_loss": fit.val_loss,
                           "best_epoch": fit.best_epoch,
                           "n_params": fit.n_params,
                           "grid_val_losses": [g["val_loss"] for g in fit.grid]}

    row: dict = {
        "key": spec.key, "block": spec.block, "screen": spec.screen,
        "encoding": spec.encoding, "endpoint": spec.endpoint,
        "representation": spec.representation, "k": spec.k,
        "partition": spec.partition, "fold": spec.fold, "fold_key": fold.key,
        "seed": spec.seed, "binary": binary,
        "classical_amines_only": spec.classical_amines_only,
        "aggregate_cells": spec.aggregate_cells, "tag": spec.tag,
        "n_rows": len(frame), "n_acids": len(screen.acids),
        "n_amines": len(screen.amines), "n_conditions": len(screen.conditions),
        "target_mean_train": float(target[tr].mean()),
        "target_sd_train": float(target[tr].std()),
        "selection": selection, "counts": fold.counts(),
        **{f"synth_{k}": v for k, v in synth_notes.items()},
    }

    preds: dict[str, dict[str, np.ndarray]] = {}
    for bucket in REPORT_BUCKETS:
        m = fold.mask(bucket)
        if not m.any():
            continue
        sub = frame.loc[m]
        y = target[m]
        preds[bucket] = {}
        for name, fit in fits.items():
            p = predict(fit.model, sub["acid"], sub["amine"], sub["cond"],
                        scaling, binary=binary)
            preds[bucket][name] = p
            met = (binary_metrics(y.astype(np.int64), p) if binary
                   else continuous_metrics(y, p))
            for k, v in met.items():
                row[f"{bucket}_{name}_{k}"] = v

        for base, pair, cname in CONTRASTS:
            if base not in preds[bucket] or pair not in preds[bucket]:
                continue
            row[f"{bucket}_{cname}_incremental"] = incremental(
                y, preds[bucket][base], preds[bucket][pair],
                loss="log_loss" if binary else "mse")
            if binary:
                row[f"{bucket}_{cname}_incremental_brier"] = incremental(
                    y, preds[bucket][base], preds[bucket][pair], loss="brier")

        # Blind: the same substitution on baseline and pair model, no refit.
        xa_b, xn_b = _blinded(prepared, fold, BLIND_ROLES[bucket])
        for name, fit in fits.items():
            p = _predict_with_features(fit.model, xa_b, xn_b, sub, scaling,
                                       binary)
            preds[bucket][f"blind::{name}"] = p
        for base, pair, cname in CONTRASTS:
            bb, pb = f"blind::{base}", f"blind::{pair}"
            if bb not in preds[bucket] or pb not in preds[bucket]:
                continue
            inc_b = incremental(y, preds[bucket][bb], preds[bucket][pb],
                                loss="log_loss" if binary else "mse")
            row[f"{bucket}_{cname}_incremental_blind"] = inc_b
            full = row.get(f"{bucket}_{cname}_incremental")
            row[f"{bucket}_{cname}_blind_drop"] = (
                full - inc_b if full is not None and np.isfinite(inc_b)
                else float("nan"))

        # Projection: no outcome enters the fit.
        for base, pair, cname in CONTRASTS:
            if base not in preds[bucket] or pair not in preds[bucket] or binary:
                continue
            proj = additive_projection(sub, preds[bucket][pair], y,
                                       preds[bucket][base],
                                       terms=PROJECTION_TERMS[base])
            for k, v in proj.items():
                row[f"{bucket}_{cname}_proj_{k}"] = v

    row["per_entity"] = _per_entity(prepared, fold, frame, target, preds, binary)
    row["seconds"] = time.time() - t0
    return row


#: Rungs whose latent width is a real hyperparameter. The others carry a
#: ``rank`` field that no term reads, so searching it would fit the same model
#: three times and pick the luckiest -- selection noise dressed as tuning.
_HAS_RANK = frozenset({"lowrank", "condition_expanded",
                       "condition_expanded_pair", "transductive"})


def _grid_for(name: str) -> tuple[dict, ...]:
    ranks = RANK_GRID if name in _HAS_RANK else (4,)
    return tuple({"rank": r, "cond_rank": r, "weight_decay": wd}
                 for wd in WEIGHT_DECAY_GRID for r in ranks)


def representation_seed(spec: Spec) -> int:
    """The seed the control permutation is drawn from. **Fold-independent.**

    Deliberately not :func:`hash_seed`. That one keys on partition and fold --
    correctly, because a model's initialisation should differ per fold -- and
    feeding it to ``build_representation`` re-permuted the control fingerprints
    on **every fold**, which is the one thing ``features.shuffled`` says it does
    not do: "permuted once, before any split exists, so every fold sees the same
    permutation. A per-fold permutation would let the control differ from the
    real run in a second way."

    It differed in that second way for the whole first authoritative run. The
    real arm sees one fixed representation across its 15 folds and the control
    arm saw 5 different ones, so the control's fold-to-fold spread contained
    permutation variance the real arm did not have -- inflating its variance and
    making "the control collapses" easier to satisfy than registered.

    Keyed on the representation and the screen only, so the shuffle is a
    function of what is being shuffled and nothing else.
    """
    key = "|".join((spec.representation, spec.screen, spec.encoding))
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(),
                          "big") % (2 ** 31)


def hash_seed(spec: Spec) -> int:
    """A per-condition seed that is a deterministic function of the condition.

    A digest, not ``hash()``. Python salts string hashing per interpreter
    process unless ``PYTHONHASHSEED`` is set, so ``hash()`` here returned a
    different value in every worker and on every run -- which made the seed a
    nuisance parameter nobody could reproduce, in a repository whose committed
    results are supposed to be regenerable from its committed code. The
    conclusions never rested on one initialisation, but "you cannot reproduce
    this file" is not a defect to leave in place.

    ``blake2b`` rather than ``hashlib.md5`` or the like because it is fast, and
    the digest is truncated to 31 bits so it can seed both numpy and torch.
    """
    key = "|".join(str(x) for x in (spec.block, spec.screen, spec.encoding,
                                    spec.endpoint, spec.representation,
                                    spec.partition, spec.fold, spec.tag))
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(),
                          "big") % (2 ** 31)


def _identity_scaling():
    from .train import Scaling
    return Scaling(mean=0.0, scale=1.0)


def _predict_with_features(model, xa: np.ndarray, xn: np.ndarray,
                           sub: pd.DataFrame, scaling, binary: bool
                           ) -> np.ndarray:
    """Re-predict a fitted model with substituted feature buffers.

    The buffers are swapped and restored, so the fitted model is left exactly as
    it was and the blind prediction cannot contaminate anything computed after
    it. Models without feature buffers (the transductive rungs) are returned
    unchanged, which is correct: there is nothing about the entity to blind.
    """
    if not hasattr(model, "XA"):
        return predict(model, sub["acid"], sub["amine"], sub["cond"], scaling,
                       binary=binary)
    old_a, old_n = model.XA, model.XN
    try:
        model.XA = torch.as_tensor(np.ascontiguousarray(xa), dtype=torch.float32)
        model.XN = torch.as_tensor(np.ascontiguousarray(xn), dtype=torch.float32)
        return predict(model, sub["acid"], sub["amine"], sub["cond"], scaling,
                       binary=binary)
    finally:
        model.XA, model.XN = old_a, old_n


def _per_entity(prepared: Prepared, fold: sp.Fold, frame: pd.DataFrame,
                target: np.ndarray, preds: dict, binary: bool) -> list[dict]:
    """Per-held-out-entity incremental skill and similarity, the inferential unit."""
    out: list[dict] = []
    loss = "log_loss" if binary else "mse"
    for bucket, role, feat, roles, test_attr, train_attr in (
            ("test_e1a", "acid", prepared.acid_fp, fold.acid_role,
             "test_acids", "train_acids"),
            ("test_e1n", "amine", prepared.amine_fp, fold.amine_role,
             "test_amines", "train_amines")):
        if bucket not in preds:
            continue
        m = fold.mask(bucket)
        sub = frame.loc[m]
        y = target[m]
        test_ent = np.array(getattr(fold, test_attr))
        train_ent = np.array(getattr(fold, train_attr))
        sim = dict(zip(test_ent.tolist(),
                       ft.max_similarity_to(feat, test_ent, train_ent).tolist()))
        for base, pair, cname in CONTRASTS:
            if base not in preds[bucket] or pair not in preds[bucket]:
                continue
            table = per_entity_incremental(sub, role, y, preds[bucket][base],
                                           preds[bucket][pair], loss=loss)
            blind = per_entity_incremental(
                sub, role, y, preds[bucket][f"blind::{base}"],
                preds[bucket][f"blind::{pair}"], loss=loss)
            blind = blind.set_index("entity")["incremental"]
            for rec in table.to_dict("records"):
                rec.update({"bucket": bucket, "contrast": cname,
                            "fold_key": fold.key,
                            "max_similarity_to_train": sim.get(rec["entity"]),
                            "incremental_blind": float(
                                blind.get(rec["entity"], np.nan)),
                            "smiles": (prepared.screen.acids[rec["entity"]]
                                       if role == "acid"
                                       else prepared.screen.amines[rec["entity"]])})
                out.append(rec)
    return out


def run_transductive(spec: Spec, raw: pd.DataFrame | None = None) -> dict:
    """The ceiling: hold out **pairs**, let both endpoints be estimated directly.

    Answers the prior question -- is the acid-amine interaction matrix learnable
    at all when nothing has to be inferred from structure? If it is not, an
    inductive failure is uninformative, because there would be no structure to
    infer. The row is marked ``transductive`` and the report refuses to place it
    in an entity-OOD table.
    """
    t0 = time.time()
    torch.set_num_threads(1)
    prepared = prepare(spec, raw=raw)
    frame = prepared.screen.frame
    folds = sp.make_pair_folds(frame["acid"].to_numpy(),
                               frame["amine"].to_numpy(), k=spec.k,
                               n_partitions=spec.n_partitions, seed=spec.seed)
    fold = {(f.partition, f.fold): f for f in folds}[(spec.partition, spec.fold)]
    sp.assert_transductive(fold, frame["acid"].to_numpy(),
                           frame["amine"].to_numpy())

    binary = spec.endpoint == "feasible"
    target = (frame["feasible"].to_numpy(dtype=np.float64) if binary
              else frame["y"].to_numpy())
    tr, va, te = fold.mask("train"), fold.mask("val"), fold.mask("test")
    scaling = fit_scaling(target[tr]) if not binary else _identity_scaling()
    btr = make_batch(frame["acid"][tr], frame["amine"][tr], frame["cond"][tr],
                     target[tr], scaling)
    bva = make_batch(frame["acid"][va], frame["amine"][va], frame["cond"][va],
                     target[va], scaling)
    cfg = ModelConfig(n_acids=len(prepared.screen.acids),
                      n_amines=len(prepared.screen.amines),
                      n_conditions=len(prepared.screen.conditions),
                      x_acid=prepared.acid_fp.x, x_amine=prepared.amine_fp.x)
    tcfg = TrainConfig(max_epochs=spec.max_epochs, n_restarts=spec.n_restarts)

    row: dict = {"key": spec.key, "block": "transductive",
                 "screen": spec.screen, "encoding": spec.encoding,
                 "endpoint": spec.endpoint, "representation": "entity_ids",
                 "k": spec.k, "partition": spec.partition, "fold": spec.fold,
                 "fold_key": fold.key, "seed": spec.seed, "binary": binary,
                 "transductive": True, "counts": fold.counts(),
                 "n_dropped": fold.n_dropped, "tag": spec.tag}
    preds = {}
    sub = frame.loc[te]
    y = target[te]
    for name in ("transductive_additive", "transductive"):
        fit = train_with_grid(name, cfg, btr, bva, tcfg, seed=hash_seed(spec),
                              grid=_grid_for(name), binary=binary)
        preds[name] = predict(fit.model, sub["acid"], sub["amine"], sub["cond"],
                              scaling, binary=binary)
        met = (binary_metrics(y.astype(np.int64), preds[name]) if binary
               else continuous_metrics(y, preds[name]))
        for k, v in met.items():
            row[f"test_{name}_{k}"] = v
        row[f"selection_{name}"] = {"hparams": fit.hparams,
                                    "val_loss": fit.val_loss,
                                    "n_params": fit.n_params}
    row["test_primary_incremental"] = incremental(
        y, preds["transductive_additive"], preds["transductive"],
        loss="log_loss" if binary else "mse")
    if not binary:
        proj = additive_projection(sub, preds["transductive"], y,
                                   preds["transductive_additive"],
                                   terms=("acid", "amine", "cond"))
        for k, v in proj.items():
            row[f"test_primary_proj_{k}"] = v
    row["seconds"] = time.time() - t0
    return row


def _pair_term_sd(model, sub: pd.DataFrame) -> float:
    """Standard deviation of a fitted model's interaction term on these rows.

    ``nan`` for a rung that has no interaction term. For one that does, a value
    many orders of magnitude below the target scale means the term is dead and
    its incremental skill is an artefact of initialisation.
    """
    pair = getattr(model, "pair", None)
    if pair is None:
        return float("nan")
    def _long(v):
        return torch.as_tensor(np.ascontiguousarray(v).copy(), dtype=torch.long)
    a, n, c = (_long(sub["acid"]), _long(sub["amine"]), _long(sub["cond"]))
    model.eval()
    with torch.no_grad():
        try:
            out = pair(a, n, c)
        except TypeError:
            out = pair(a, n)
    return float(out.std())
