"""Training loop.

Deliberately minimal: the datasets are tiny (a few thousand rows), so training
is full-batch Adam on CPU.  Every model family gets **identical** optimizer
settings, epoch budget and early-stopping rule; the only thing that varies is
the pair term.  Anything else would confound the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from .generator import ORD, SIM, SINGLE, ObservationTable
from .models import BaseModel


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.0
    emb_weight_decay: float = 0.0   # applied to the embedding table only
    max_epochs: int = 4000
    patience: int = 400        # early stopping on validation MSE
    eval_every: int = 10
    # Number of independent initialisations per run; the one with the lowest
    # final TRAINING mse is kept. This exists because the antisymmetric
    # parameterisation A(i,j) = (F_A(phi_ij) - F_A(phi_ji))/2 has an exact
    # degenerate optimum: any F_A that is symmetric under exchanging the two
    # argument slots gives A == 0 identically, and the antisymmetric part of the
    # gradient vanishes there too, so it is self-sustaining. Plain Adam fell into
    # it on 3/25 algebra runs (A_rms exactly 0.0, training mse ~0.40 against a
    # peer median of ~0.04). Selection is on training loss only -- never
    # validation, never test -- and the same restart budget is given to every
    # family, so it cannot favour one of them.
    n_restarts: int = 1
    seed: int = 0


class Batch:
    """Pre-tensorised observation rows, grouped by observation kind."""

    def __init__(self, table: ObservationTable):
        self.groups = {}
        for kind in (SINGLE, SIM, ORD):
            m = table.kind == kind
            if not m.any():
                continue
            self.groups[kind] = (
                torch.as_tensor(table.i[m], dtype=torch.long),
                torch.as_tensor(table.j[m], dtype=torch.long),
                torch.as_tensor(table.y[m], dtype=torch.float32),
            )
        self.n_rows = len(table)

    def predict(self, model: BaseModel) -> tuple[torch.Tensor, torch.Tensor]:
        preds, targets = [], []
        for kind, (i, j, y) in self.groups.items():
            if kind == SINGLE:
                p = model.predict_single(i)
            elif kind == SIM:
                p = model.predict_sim(i, j)
            else:
                p = model.predict_ordered(i, j)
            preds.append(p)
            targets.append(y)
        return torch.cat(preds), torch.cat(targets)

    def mse(self, model: BaseModel) -> torch.Tensor:
        p, y = self.predict(model)
        return ((p - y) ** 2).mean()


def train_model(model: BaseModel, train_table: ObservationTable,
                val_table: ObservationTable | None,
                cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    train_batch = Batch(train_table)
    val_batch = Batch(val_table) if val_table is not None and len(val_table) else None

    # The embedding table is the part that overfits: it carries one free vector
    # per intervention, fitted from that intervention's handful of pairs. It gets
    # its own weight decay, identically for every family.
    emb_params = [model.emb.weight]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    groups = [{"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    if model.emb.weight.requires_grad:
        groups.append({"params": emb_params, "lr": cfg.lr,
                       "weight_decay": cfg.emb_weight_decay})
    opt = torch.optim.Adam(groups)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, stale = 0, 0
    history = []
    # Initialised so that max_epochs=0 returns an untrained model rather than
    # raising UnboundLocalError on `epoch` below (reachable from the public API,
    # e.g. an "untrained baseline" ablation).
    epoch = -1

    for epoch in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        loss = train_batch.mse(model)
        loss.backward()
        opt.step()

        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs - 1:
            model.eval()
            with torch.no_grad():
                tr = float(loss.detach())
                va = float(val_batch.mse(model)) if val_batch is not None else tr
            history.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
            if va < best_val - 1e-8:
                best_val, best_epoch, stale = va, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += cfg.eval_every
                if stale >= cfg.patience:
                    break

    # Keep the final-epoch weights alongside the best-validation ones. With
    # fixed-length training the model is selected from ~max_epochs/eval_every
    # checkpoints using only 66 held-out pairs, which is a lot of selection on a
    # small, noisy signal. Scoring the final epoch too makes that channel
    # measurable: if a family's advantage exists at the best-validation
    # checkpoint but not at the final epoch, the advantage is checkpoint
    # selection rather than learning.
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_train = float(train_batch.mse(model))
    return {
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "final_train_mse": final_train,
        "epochs_run": epoch + 1,
        "final_state": final_state,
        "history": history,
        "train_config": asdict(cfg),
    }
