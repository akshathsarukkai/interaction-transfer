"""The training loop, shared verbatim by every rung of the Phase 4 ladder.

Nothing here branches on the rung. Optimiser, schedule, budget, restart count,
restart selection and early-stopping rule are identical for the additive
baseline and for the pair model, because the load-bearing quantity is a *ratio*
of their test errors and any asymmetry in how they are fitted would land
directly in that ratio. The only thing that differs between two rows of the
primary contrast is the model class.

Target standardisation, and the one thing it must not see
---------------------------------------------------------
The target is centred and scaled by statistics computed on the **training rows
of that fold only**. Using the whole screen's mean and standard deviation would
leak the held-out entities' outcomes into the optimisation scale -- a small leak,
but one that the shuffled-feature controls could not detect, because it does not
travel through the representation. ``test_target_scaling_uses_training_rows_only``
mutates a test row's outcome and asserts the fitted scale does not move.

Features are **not** standardised. ECFP4 bits are already 0/1 and on the same
footing across entities, and a per-bit standardisation would be a second place
where a statistic of the data has to be restricted to training rows. One fewer
leakage surface is worth more here than the small conditioning improvement.

Full batch, not minibatch
-------------------------
About 6,500 training rows and at most a few thousand parameters. A full-batch
step is one 2,048-column matrix multiply over 272 acids and 231 amines, which is
faster than assembling minibatches, and it makes a run a deterministic function
of its seed with no shuffling order to reproduce.

No shrinkage calibration
------------------------
Phase 3 fitted a shrinkage coefficient on the validation set, and its own audit
found the coefficient biased upward because the same validation rows had already
chosen the stopping epoch and the grid member. It exists there because coverage
0.05 left ~35 validation pairs. Here the smallest selection bucket is ~1,000
rows, the artefact it was compensating for does not arise, and adding it would
import a known bias for no benefit. Deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .models import ModelConfig, build


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-2
    weight_decay: float = 1e-4
    #: Fixed annealed budget: training length is a constant of the experiment
    #: rather than a per-run accident. Set by watching validation loss flatten on
    #: dev folds drawn with a *different* seed stream from the authoritative
    #: ones, never on the authoritative folds themselves.
    max_epochs: int = 600
    n_restarts: int = 2
    grad_clip: float = 5.0
    cosine_decay: bool = True
    eval_every: int = 5


@dataclass(frozen=True)
class Scaling:
    """Target centring and scaling, derived from training rows alone."""

    mean: float
    scale: float

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.scale + self.mean


def fit_scaling(y_train: np.ndarray) -> Scaling:
    y = np.asarray(y_train, dtype=np.float64)
    if y.size == 0:
        raise ValueError("cannot standardise a target from no training rows")
    sd = float(y.std())
    # A constant training target is a degenerate fold, not a divide-by-zero to
    # paper over: every model would predict the constant and every skill would
    # be 0/0. Fail loudly.
    #
    # The threshold is relative, not `sd <= 0`, because a constant array does
    # not have exactly zero standard deviation in floating point:
    # ``np.full(10, 0.3).std()`` is 5.6e-17, and dividing by it produces a
    # target of order 1e16 rather than an error.
    if not np.isfinite(sd) or sd < 1e-9 * max(1.0, abs(float(y.mean()))):
        raise ValueError("the training target has zero variance; this fold is "
                         "degenerate and its skills would be 0/0")
    return Scaling(mean=float(y.mean()), scale=sd)


@dataclass
class Batch:
    a: torch.Tensor
    n: torch.Tensor
    c: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return int(self.a.shape[0])


def make_batch(acid: np.ndarray, amine: np.ndarray, cond: np.ndarray,
               y: np.ndarray, scaling: Scaling) -> Batch:
    def _long(v):
        return torch.as_tensor(np.ascontiguousarray(v).copy(), dtype=torch.long)
    return Batch(a=_long(acid), n=_long(amine), c=_long(cond),
                 y=torch.as_tensor(scaling.forward(y), dtype=torch.float32))


@dataclass
class Fit:
    model: object
    train_loss: float
    val_loss: float
    best_epoch: int
    restarts: list[dict]
    hparams: dict
    #: Validation losses of every grid setting, in grid order. Kept so
    #: "selection was on validation" is auditable rather than asserted, and so a
    #: grid whose winner sits on an edge is visible in the result row.
    grid: list[dict]
    n_params: int


def _loss(model, b: Batch, binary: bool) -> torch.Tensor:
    """MSE on the standardised target, or Bernoulli NLL on the raw logit.

    The binary endpoint is fitted as a genuine classifier -- the same
    architectures, a logistic link, a Bernoulli loss -- rather than by
    thresholding a regressor. The metric it feeds is an incremental log loss
    against the *same* additive baseline in the *same* function class, and a
    thresholded regressor would not give that.
    """
    out = model(b.a, b.n, b.c)
    if binary:
        return torch.nn.functional.binary_cross_entropy_with_logits(out, b.y)
    return torch.nn.functional.mse_loss(out, b.y)


def _eval(model, b: Batch, binary: bool) -> float:
    model.eval()
    with torch.no_grad():
        return float(_loss(model, b, binary))


def _fit_once(model, tr: Batch, va: Batch, cfg: TrainConfig,
              seed: int, binary: bool = False) -> tuple[object, float, float, int]:
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return model, _eval(model, tr, binary), _eval(model, va, binary), 0
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
             if cfg.cosine_decay else None)
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        loss = _loss(model, tr, binary)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        if epoch % cfg.eval_every and epoch != cfg.max_epochs - 1:
            continue
        vl = _eval(model, va, binary)
        if vl < best_val - 1e-12:
            best_val, best_epoch = vl, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, _eval(model, tr, binary), best_val, best_epoch


def train(name: str, cfg: ModelConfig, tr: Batch, va: Batch,
          tcfg: TrainConfig, seed: int, binary: bool = False) -> Fit:
    """Fit one model with restarts, selecting the restart on validation loss."""
    trials = []
    probe = build(name, replace(cfg, seed=seed))
    n_restarts = 1 if not list(probe.parameters()) else tcfg.n_restarts
    for r in range(n_restarts):
        m = build(name, replace(cfg, seed=seed * 1000 + r))
        m, tl, vl, ep = _fit_once(m, tr, va, tcfg, seed=seed * 1000 + r,
                                  binary=binary)
        trials.append({"restart": r, "train_loss": tl, "val_loss": vl,
                       "best_epoch": ep, "model": m})
    best = min(trials, key=lambda d: d["val_loss"])
    return Fit(model=best["model"], train_loss=best["train_loss"],
               val_loss=best["val_loss"], best_epoch=best["best_epoch"],
               restarts=[{k: v for k, v in d.items() if k != "model"}
                         for d in trials],
               hparams={}, grid=[],
               n_params=int(best["model"].n_params()))


#: Grid keys that belong to the optimiser rather than the architecture.
_TRAIN_KEYS = frozenset({"lr", "weight_decay", "max_epochs", "n_restarts",
                         "grad_clip", "cosine_decay", "eval_every"})


def train_with_grid(name: str, cfg: ModelConfig, tr: Batch, va: Batch,
                    tcfg: TrainConfig, seed: int, grid: tuple[dict, ...],
                    binary: bool = False) -> Fit:
    """Fit every setting in ``grid`` and keep the lowest **validation** loss.

    A setting may name architecture fields (``rank``, ``hidden``) and optimiser
    fields (``weight_decay``) in one dict; they are routed to the right config.
    Weight decay is in the grid for **every** rung, so the baseline and the pair
    model get identical regularisation freedom -- searching it for only one of
    them would be an asymmetry landing directly in the ratio they are compared by.

    ``va`` must be built from :data:`splits.SELECT_BUCKETS` rows and nothing
    else, because this function cannot tell what it was handed. The caller is
    responsible, and two tests check it:
    ``test_selection_never_sees_a_test_entity`` on the CI fixture, and
    ``test_the_authoritative_folds_pass_every_guard`` on the actual
    authoritative folds of both screens. The second exists because this
    docstring used to cite the first as proof about the authoritative folds,
    which it is not -- it runs on a 14-acid fixture.
    """
    if not grid:
        return train(name, cfg, tr, va, tcfg, seed, binary=binary)
    fits = []
    for setting in grid:
        model_kw = {k: v for k, v in setting.items() if k not in _TRAIN_KEYS}
        train_kw = {k: v for k, v in setting.items() if k in _TRAIN_KEYS}
        fit = train(name, replace(cfg, **model_kw), tr, va,
                    replace(tcfg, **train_kw), seed, binary=binary)
        fit.hparams = dict(setting)
        fits.append(fit)
    best = min(fits, key=lambda f: f.val_loss)
    best.grid = [{"hparams": f.hparams, "val_loss": f.val_loss,
                  "n_params": f.n_params} for f in fits]
    return best


def predict(model, acid: np.ndarray, amine: np.ndarray, cond: np.ndarray,
            scaling: Scaling, binary: bool = False) -> np.ndarray:
    """Predictions on the **original target scale**, or as probabilities.

    Every continuous metric in this phase is computed on the original scale, so
    that an MSE is comparable across folds whose training standard deviations
    differ. For the binary endpoint the model emits a logit and this returns the
    probability, which is what log loss and Brier are defined on.
    """
    def _long(v):
        return torch.as_tensor(np.ascontiguousarray(v).copy(), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model(_long(acid), _long(amine), _long(cond))
        if binary:
            return torch.sigmoid(out).numpy().astype(np.float64)
        z = out.numpy()
    return scaling.inverse(z)
