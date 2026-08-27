"""Training loop shared verbatim by every rung of the residual ladder.

Nothing here branches on the rung except ``_Residual.ordered_loss``, which
selects *which* residual the loss is computed against -- ``D_res`` on unordered
pairs, or ``r_ij`` on ordered rows -- because those are two different targets
with two different row counts. Optimiser, schedule, budget, restart count and
restart-selection rule are identical.

Scale-only standardisation
--------------------------
The direct rungs' target is divided by its training standard deviation and never
shifted. That is not a shortcut: ``D_res`` is antisymmetric, every rung is
antisymmetric, and subtracting a nonzero mean would ask an antisymmetric
function to fit a constant -- a target it cannot represent and a bias it would
pay for at every pair. The mean is small anyway (the canonical orientation
``i < j`` is drug-index order, which is alphabetical and unrelated to the
response), and it is reported as ``D_res_mean`` so the assumption is checkable
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .residual_models import _Residual


@dataclass(frozen=True)
class ResidualTrainConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-3
    #: Fixed annealed budget, as in Phase 2: the training length is a constant
    #: of the experiment rather than a per-run accident of early stopping.
    #: Shorter than Phase 2's 1500 because these targets are one-dimensional and
    #: the heads start at exactly zero; the budget was fixed by checking on
    #: *dev* split seeds (100/101) that validation loss had flattened, never on
    #: the evaluation seeds.
    max_epochs: int = 800
    n_restarts: int = 2
    grad_clip: float = 5.0
    cosine_decay: bool = True
    eval_every: int = 5


@dataclass
class ResidualFit:
    model: _Residual
    train_loss: float
    val_loss: float
    best_epoch: int
    restarts: list[dict]
    hparams: dict
    #: Validation losses of every setting in the grid, in grid order. Kept so
    #: "selection was on validation" is auditable rather than asserted, and so a
    #: grid whose best setting sits on an edge is visible.
    grid_val_losses: list[float]


@dataclass
class PairTensors:
    """Canonical-orientation pairs and their residual directional target."""
    i: torch.Tensor
    j: torch.Tensor
    d: torch.Tensor


@dataclass
class OrderedTensors:
    """Ordered rows and their additive residual ``r_ij``."""
    i: torch.Tensor
    j: torch.Tensor
    r: torch.Tensor


def _long(col) -> torch.Tensor:
    # ``.copy()`` because pandas can hand back a read-only view, which torch
    # wraps without copying and then warns about on every single call.
    return torch.as_tensor(np.ascontiguousarray(col.to_numpy()).copy(),
                           dtype=torch.long)


def pair_tensors(frame, scale: float) -> PairTensors:
    return PairTensors(
        i=_long(frame["i"]), j=_long(frame["j"]),
        d=torch.as_tensor(frame["D_res"].to_numpy() / scale, dtype=torch.float32))


def ordered_tensors(frame, scale: float) -> OrderedTensors:
    return OrderedTensors(
        i=_long(frame["i"]), j=_long(frame["j"]),
        r=torch.as_tensor(frame["r"].to_numpy() / scale, dtype=torch.float32))


def _loss(model: _Residual, t) -> torch.Tensor:
    if model.ordered_loss:
        return torch.nn.functional.mse_loss(model.r_ordered(t.i, t.j), t.r)
    return torch.nn.functional.mse_loss(model.d_res(t.i, t.j), t.d)


def _val_loss_on_pairs(model: _Residual, pairs: PairTensors) -> float:
    """Validation score, always on ``D_res`` -- for every rung alike.

    The ordered rung trains on ``r_ij`` but is *selected* on the same directional
    quantity as everything else. Selecting it on its own training loss instead
    would let it win the grid by fitting the symmetric part of the residual,
    which is not the thing under test and is where most of the residual's mass
    sits. This is the one place the two loss shapes are deliberately not
    symmetric, and it is the choice that keeps the comparison honest.
    """
    model.eval()
    with torch.no_grad():
        return float(torch.nn.functional.mse_loss(
            model.d_res(pairs.i, pairs.j), pairs.d))


def _fit_once(model: _Residual, tr, va_pairs: PairTensors,
              cfg: ResidualTrainConfig, seed: int) -> tuple[_Residual, float, float, int]:
    torch.manual_seed(seed)
    params = list(model.parameters())
    if not params:                       # the zero rung: nothing to fit
        return model, float(_loss_value(model, tr)), _val_loss_on_pairs(model, va_pairs), 0
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
             if cfg.cosine_decay else None)
    best_val, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        loss = _loss(model, tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        if epoch % cfg.eval_every and epoch != cfg.max_epochs - 1:
            continue
        vl = _val_loss_on_pairs(model, va_pairs)
        if vl < best_val - 1e-9:
            best_val, best_epoch = vl, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, float(_loss_value(model, tr)), best_val, best_epoch


def _loss_value(model: _Residual, tr) -> float:
    model.eval()
    with torch.no_grad():
        return float(_loss(model, tr))


def train_residual(build, tr, va_pairs: PairTensors,
                   cfg: ResidualTrainConfig, seed: int) -> ResidualFit:
    """Fit with restarts, selecting on validation ``D_res`` loss."""
    results = []
    n = 1 if not list(build(seed=seed).parameters()) else cfg.n_restarts
    for r in range(n):
        m = build(seed=seed * 1000 + r)
        m, tl, vl, ep = _fit_once(m, tr, va_pairs, cfg, seed=seed * 1000 + r)
        results.append({"restart": r, "train_loss": tl, "val_loss": vl,
                        "best_epoch": ep, "model": m})
    best = min(results, key=lambda d: d["val_loss"])
    return ResidualFit(
        model=best["model"], train_loss=best["train_loss"],
        val_loss=best["val_loss"], best_epoch=best["best_epoch"],
        restarts=[{k: v for k, v in d.items() if k != "model"} for d in results],
        hparams={}, grid_val_losses=[])


#: Shrinkage coefficients searched after the grid, on validation only.
#: ``0.0`` is the zero predictor exactly, so the calibrated prediction can never
#: be worse than the null *on validation*.
SHRINKAGE = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def select_shrinkage(d_val_true: np.ndarray, d_val_pred: np.ndarray,
                     grid: tuple[float, ...] = SHRINKAGE) -> float:
    """Pick ``alpha`` minimising validation MSE of ``alpha * Dhat``.

    Why this exists. At coverage 0.05 the validation set is ~35 unordered pairs.
    Selecting one setting out of six on 35 noisy pairs, then reporting the
    winner's *test* error against the zero predictor, produces negative skill
    from selection noise alone even when the rung has learned nothing harmful --
    and a table full of small negative numbers reads as "flexible models do
    active harm" (outcome D) when the truth is "there was nothing to select on"
    (outcome C). Shrinking toward the null attenuates that artifact.

    What it does **not** do, and an earlier version of this docstring claimed it
    did. It does not make a signal-free rung tie the null exactly. By default
    ``alpha`` is fitted on the same validation pairs that already chose the
    stopping epoch, the restart and the grid member, so the validation
    predictions are optimistically correlated with the validation targets and
    ``alpha`` is biased **upward** -- structurally, not because the validation
    set is small. Measured on the shipped grid: ``alpha`` is exactly 0 in only 13
    of 320 learned-rung runs, and at coverage 0.05 it averages 0.87 against a
    test-optimal value near 0.02. The guarantee is therefore one-sided:
    ``alpha <= 1`` means calibration can never manufacture skill, only fail to
    remove negative skill.

    The consequence is confined to the sparse cells and points in a known
    direction. It makes the sparse-coverage numbers *more* negative than the
    rung's attainable performance, so a negative cell there is evidence that
    this selection machinery costs skill at that sample size -- not that residual
    structure is harmful to model. At coverage 0.40 and 0.70 the fitted
    ``alpha`` already sits at or near the test-optimal value and the calibrated
    and uncalibrated numbers agree, so the headline is untouched.

    The fix, when the caller asks for it, is
    :func:`split_calibration_pairs`: hold out half the validation pairs from
    model selection entirely and fit ``alpha`` only there. That is what
    ``ResidualConfig.split_validation_for_calibration`` does, and the
    ``honest_alpha`` block reruns the rungs the primary contrast depends on
    under it. It is off by default so the shipped main grid stays exactly the
    grid the decision rule was registered against.
    """
    best_a, best_m = 0.0, float("inf")
    for a in grid:
        m = float(((a * d_val_pred - d_val_true) ** 2).mean())
        if m < best_m - 1e-15:
            best_a, best_m = float(a), m
    return best_a


def split_calibration_pairs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic halving of the validation pairs into (select, calibrate).

    Both halves are validation data -- neither is test -- so this changes which
    validation pairs do which job and nothing else. The split is by index under a
    fixed RNG rather than by position, because the validation pairs arrive in
    permutation order and taking a prefix would correlate the two halves with
    whatever ordering the split seed produced.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = n // 2
    return np.sort(perm[:k]), np.sort(perm[k:])


def predict_d_res(model: _Residual, i: np.ndarray, j: np.ndarray,
                  scale: float) -> np.ndarray:
    ti = torch.as_tensor(np.ascontiguousarray(i).copy(), dtype=torch.long)
    tj = torch.as_tensor(np.ascontiguousarray(j).copy(), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        return model.d_res(ti, tj).numpy() * scale
