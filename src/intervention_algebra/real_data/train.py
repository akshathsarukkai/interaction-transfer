"""Training loop shared verbatim by all three Phase 2 families.

Nothing here branches on ``model.family``. The optimiser, schedule, budget,
early-stopping rule, restart count and restart-selection rule are the same for
every family, so a difference in the results is a difference in the pair
parameterisation and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .models import _Base


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-3
    #: Fixed budget with a cosine decay to zero, rather than a long constant-rate
    #: run stopped early. Measured at constant lr=1e-3 the models were still
    #: improving at epoch 4,000, so "converged" and "ran out of patience" were
    #: the same event and the budget was doing the tuning. A fixed annealed
    #: budget makes the training length a constant of the experiment instead of
    #: a per-run accident -- and it is identical for every family.
    max_epochs: int = 1500
    patience: int = 1500               # >= max_epochs: keep the best epoch, never stop early
    n_restarts: int = 2
    batch_size: int | None = None      # None -> full batch
    grad_clip: float = 5.0
    cosine_decay: bool = True
    #: Validate every k epochs rather than every epoch. A full validation
    #: forward pass costs about as much as a training step, so k=1 spends half
    #: the run on checkpoint resolution finer than the noise in the metric.
    eval_every: int = 5


@dataclass
class Tensors:
    i: torch.Tensor
    j: torch.Tensor
    y: torch.Tensor


def to_tensors(frame: pd.DataFrame, y_mean: float, y_std: float) -> Tensors:
    return Tensors(
        i=torch.as_tensor(frame["i"].to_numpy(), dtype=torch.long),
        j=torch.as_tensor(frame["j"].to_numpy(), dtype=torch.long),
        y=torch.as_tensor((frame["y"].to_numpy() - y_mean) / y_std, dtype=torch.float32),
    )


@dataclass
class Fit:
    model: _Base
    train_loss: float
    val_loss: float
    epochs_run: int
    #: RMS of the *pair head's* antisymmetric output on the training pairs, in
    #: standardised units. Near zero means the head has collapsed. See
    #: :func:`_a_rms` for why this is not ``implied_A``.
    train_head_A_rms: float
    restarts: list[dict]


def _a_rms(model: _Base, t: Tensors) -> float:
    """RMS of the *pair head's* antisymmetric output on the training pairs.

    Deliberately the head and not ``implied_A``: the latter includes the
    first-order antisymmetric term ``((a_i-b_i)-(a_j-b_j))/2``, which every
    family carries and which stays large no matter what the pair network does.
    A restart whose ``F_A`` died reads ~0.42 on ``implied_A`` and exactly 0
    here, so only this number can detect the degenerate basin.
    """
    with torch.no_grad():
        return float(model.pair_head_A(t.i, t.j).pow(2).mean().sqrt())


def _fit_once(model: _Base, tr: Tensors, va: Tensors, cfg: TrainConfig,
              seed: int) -> tuple[_Base, float, float, int]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
             if cfg.cosine_decay else None)
    best_val, best_state, best_epoch, since = float("inf"), None, 0, 0
    n = len(tr.y)
    bs = cfg.batch_size or n
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(cfg.max_epochs):
        model.train()
        if bs >= n:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(tr.i, tr.j), tr.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        else:
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, bs):
                k = perm[s:s + bs]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(tr.i[k], tr.j[k]), tr.y[k])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

        if sched is not None:
            sched.step()

        if epoch % cfg.eval_every and epoch != cfg.max_epochs - 1:
            continue

        model.eval()
        with torch.no_grad():
            vl = float(torch.nn.functional.mse_loss(model(va.i, va.j), va.y))
        if vl < best_val - 1e-6:
            best_val, best_epoch, since = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += cfg.eval_every
            if since >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tl = float(torch.nn.functional.mse_loss(model(tr.i, tr.j), tr.y))
    return model, tl, best_val, best_epoch


def train_model(build, tr: Tensors, va: Tensors, cfg: TrainConfig,
                seed: int) -> Fit:
    """Fit with restarts, selecting on *validation* loss.

    Phase 1 selected restarts on training loss because it had no held-out pool
    that was not the test set. Here validation pairs are carved out of the
    training pairs and are disjoint from the evaluation pool, so selecting on
    validation loss is both available and strictly better -- and it is applied
    identically to all three families.

    Every restart's diagnostics are retained in :attr:`Fit.restarts`, including
    the ones not selected. Antisymmetric collapse is a cost of the structured
    parameterisation and has to stay countable; discarding the losing restarts
    would hide exactly the number we need to report.
    """
    results = []
    for r in range(cfg.n_restarts):
        m = build(seed=seed * 1000 + r)
        m, tl, vl, ep = _fit_once(m, tr, va, cfg, seed=seed * 1000 + r)
        results.append({"restart": r, "train_loss": tl, "val_loss": vl,
                        "epochs": ep, "train_head_A_rms": _a_rms(m, tr),
                        "model": m})
    best = min(results, key=lambda d: d["val_loss"])
    return Fit(model=best["model"], train_loss=best["train_loss"],
               val_loss=best["val_loss"], epochs_run=best["epochs"],
               train_head_A_rms=best["train_head_A_rms"],
               restarts=[{k: v for k, v in d.items() if k != "model"}
                         for d in results])


def predict(model: _Base, frame: pd.DataFrame, y_mean: float,
            y_std: float) -> np.ndarray:
    i = torch.as_tensor(frame["i"].to_numpy(), dtype=torch.long)
    j = torch.as_tensor(frame["j"].to_numpy(), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        return model(i, j).numpy() * y_std + y_mean
