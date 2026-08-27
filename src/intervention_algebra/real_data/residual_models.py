"""The ladder of predictors for the residual directional effect ``D_res``.

Every rung predicts the same held-out quantity -- ``D_res(i, j)`` for an unseen
unordered pair in canonical orientation ``i < j`` -- and every rung is
**antisymmetric by construction**, ``f(i, j) = -f(j, i)``.

That is not a modelling bias imported from Phase 1. ``D_res`` is antisymmetric by
*definition*: ``D_res(j, i) = -D_res(i, j)`` identically, for every pair, in the
data. A predictor that is not antisymmetric is therefore not a more general
hypothesis class, it is a strictly worse one -- its symmetric component is pure
error. Imposing antisymmetry here is arithmetic, not a prior.

The rungs
---------
``zero``       ``Dhat = 0``. The mandatory null: "there is no pair-specific
               directional effect to predict". Zero parameters, nothing fitted.
               Every other rung's skill is measured against this.

``potential``  ``Dhat = c_i - c_j``. One free scalar per drug and no pair term,
               so it can express *only* a per-drug ordering tendency. It is the
               diagnostic for an incomplete additive removal: the residual was
               constructed by subtracting a shrunk estimate of exactly this
               shape, so if ``potential`` scores above zero, part of what
               ``D_res`` still contains is the potential the residualisation was
               supposed to remove -- and any richer rung's skill has to be read
               against ``potential``, not against ``zero``.

``lowrank``    ``Dhat = u_i^T K u_j`` with ``K = -K^T``. The minimal
               pair-specific hypothesis: a low-dimensional antisymmetric bilinear
               form. Capacity-controlled -- ``rank`` is the only knob and it is
               small -- so a win here is a claim about structure, not about
               flexibility. This is the rung the scientific question is really
               about.

``mlp``        ``Dhat = F(phi_ij) - F(phi_ji)`` with ``F`` a two-hidden-layer MLP
               on the Phase 2 pair feature map. The flexible upper bound: if this
               cannot find residual structure, a smaller model will not either.
               Explicitly *not* capacity-matched to ``lowrank`` (see
               ``n_params`` in every result row) and reported as a diagnostic
               ceiling rather than a fair comparator.

``mlp_ordered`` fits the ordered residual ``r_ij = a'_i + b'_j + G(phi_ij)`` by
               MSE on ordered rows, then reads off ``Dhat = rhat_ij - rhat_ji``.
               Same function class as ``mlp`` for the directional readout, but a
               different *loss*: it also has to fit the symmetric part of the
               residual, which is where most of the residual's mass is. Included
               because it is the formulation a reader would reach for by default,
               and the difference between the two is informative about whether
               the direct formulation is worth its extra assumption.

Why the Phase 2 ``structured`` family is not a sixth rung
---------------------------------------------------------
Its antisymmetric component is
``A(i,j) = first_order_A + [F_A(phi_ij) - F_A(phi_ji)]/2``. On a target from
which the first-order potential has already been removed, and with the factor of
two absorbed into ``F_A``'s output layer, that is ``mlp`` exactly -- the same
function class, not an approximation of it.
``test_structured_A_head_equals_the_mlp_rung`` proves it numerically. There is
nothing separate to run, and running it anyway would report one hypothesis twice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from .models import PairFeatures, mlp


@dataclass(frozen=True)
class ResidualModelConfig:
    n_drugs: int = 100
    #: ``lowrank`` latent width; also the antisymmetric form's rank bound.
    rank: int = 4
    #: ``mlp`` / ``mlp_ordered`` embedding width and head width.
    emb_dim: int = 16
    hidden: int = 48
    n_hidden_layers: int = 2
    emb_init_std: float = 0.5
    seed: int = 0


class _Residual(nn.Module):
    name = "base"
    #: True when the rung is fitted on ordered residual rows rather than on
    #: ``D_res`` directly. Decides which loss the trainer uses.
    ordered_loss = False

    def __init__(self, cfg: ResidualModelConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg

    def d_res(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def r_ordered(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("only the ordered-loss rung fits r_ij")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Zero(_Residual):
    """``Dhat = 0``. Fitted by not fitting."""

    name = "zero"

    def d_res(self, i, j):
        return torch.zeros(i.shape, dtype=torch.float32)


class Potential(_Residual):
    """``Dhat = c_i - c_j``: leftover per-drug ordering tendency, no pair term."""

    name = "potential"

    def __init__(self, cfg: ResidualModelConfig):
        super().__init__(cfg)
        self.c = nn.Embedding(cfg.n_drugs, 1)
        nn.init.normal_(self.c.weight, std=0.01)

    def d_res(self, i, j):
        return self.c(i).squeeze(-1) - self.c(j).squeeze(-1)


class LowRank(_Residual):
    """``Dhat = u_i^T K u_j``, ``K`` antisymmetric.

    ``K`` is stored as a free matrix ``W`` and used as ``W - W^T``, so
    antisymmetry holds at every parameter setting rather than being a penalty
    that training could trade away. The diagonal of ``W`` never affects the
    output, which costs ``rank`` redundant parameters and buys a parameterisation
    with no constraint surface to fall off.
    """

    name = "lowrank"

    def __init__(self, cfg: ResidualModelConfig):
        super().__init__(cfg)
        self.u = nn.Embedding(cfg.n_drugs, cfg.rank)
        nn.init.normal_(self.u.weight, std=cfg.emb_init_std)
        # ``W`` starts at exactly zero, so a run begins at ``Dhat = 0`` -- at the
        # null -- and the bilinear form has to earn its way in. Same rule Phase 2
        # applies to every pair head, for the same reason: a randomly initialised
        # bilinear form emits O(1) noise into a target whose scale is O(0.15).
        #
        # There is deliberately no separate multiplicative ``scale`` parameter.
        # An earlier version had one, also zero-initialised, and the model was
        # **dead**: with ``scale = 0`` the gradient into ``W`` and ``u`` is zero,
        # and with ``W = 0`` the gradient into ``scale`` is zero too, so every
        # partial derivative vanished at initialisation and the rung returned
        # exactly 0.0 skill on every cell. That is indistinguishable in the
        # results table from "there is no signal", which is the conclusion this
        # experiment exists to test -- a silent zero here would have produced the
        # headline result by construction. ``test_lowrank_has_nonzero_gradients_at_init``
        # pins it.
        self.W = nn.Parameter(torch.zeros(cfg.rank, cfg.rank))

    def d_res(self, i, j):
        K = self.W - self.W.T
        ui, uj = self.u(i), self.u(j)
        return torch.einsum("...a,ab,...b->...", ui, K, uj)


class AntisymMLP(_Residual):
    """``Dhat = F(phi_ij) - F(phi_ji)``. The flexible upper bound."""

    name = "mlp"

    def __init__(self, cfg: ResidualModelConfig):
        super().__init__(cfg)
        self.emb = nn.Embedding(cfg.n_drugs, cfg.emb_dim)
        nn.init.normal_(self.emb.weight, std=cfg.emb_init_std)
        self.phi = PairFeatures(cfg.emb_dim)
        self.F = mlp(self.phi.out_dim, cfg.hidden, 1, cfg.n_hidden_layers,
                     zero_init_output=True)

    def _f(self, i, j):
        return self.F(self.phi(self.emb(i), self.emb(j))).squeeze(-1)

    def d_res(self, i, j):
        return self._f(i, j) - self._f(j, i)


class OrderedResidualMLP(_Residual):
    """``r_ij = a'_i + b'_j + G(phi_ij)``; ``Dhat = rhat_ij - rhat_ji``.

    The first-order term is kept even though the residual it is fitted to
    already had ``mu + a_i + b_j`` subtracted. Dropping it would make this rung
    a differently-constrained model rather than the Phase 2 ``unrestricted``
    family pointed at a new target, and the point of the rung is to be the
    latter. In practice it re-learns whatever the ridge penalty over-shrank,
    which is the same thing ``potential`` measures.
    """

    name = "mlp_ordered"
    ordered_loss = True

    def __init__(self, cfg: ResidualModelConfig):
        super().__init__(cfg)
        self.emb = nn.Embedding(cfg.n_drugs, cfg.emb_dim)
        nn.init.normal_(self.emb.weight, std=cfg.emb_init_std)
        self.first_a = nn.Embedding(cfg.n_drugs, 1)
        self.first_b = nn.Embedding(cfg.n_drugs, 1)
        nn.init.normal_(self.first_a.weight, std=0.01)
        nn.init.normal_(self.first_b.weight, std=0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        self.phi = PairFeatures(cfg.emb_dim)
        self.G = mlp(self.phi.out_dim, cfg.hidden, 1, cfg.n_hidden_layers,
                     zero_init_output=True)

    def r_ordered(self, i, j):
        return (self.bias + self.first_a(i).squeeze(-1)
                + self.first_b(j).squeeze(-1)
                + self.G(self.phi(self.emb(i), self.emb(j))).squeeze(-1))

    def d_res(self, i, j):
        return self.r_ordered(i, j) - self.r_ordered(j, i)


LADDER: dict[str, type[_Residual]] = {
    "zero": Zero,
    "potential": Potential,
    "lowrank": LowRank,
    "mlp": AntisymMLP,
    "mlp_ordered": OrderedResidualMLP,
}

#: Order used in every table and figure: null first, then increasing capacity.
LADDER_ORDER = ("zero", "potential", "lowrank", "mlp", "mlp_ordered")

#: Rungs whose parameter count is deliberately not matched to ``lowrank``; they
#: are reported as flexible diagnostics, not as fair comparators (see the
#: module docstring and section 15 of the task spec).
FLEXIBLE_RUNGS = ("mlp", "mlp_ordered")


def build_residual_model(name: str, cfg: ResidualModelConfig) -> _Residual:
    if name not in LADDER:
        raise ValueError(f"unknown rung {name!r}; expected {sorted(LADDER)}")
    return LADDER[name](cfg)


#: The hyperparameter grid each rung is selected over, **on validation loss
#: only**, independently within every single run.
#:
#: Phase 2 selected one setting per family once, on one screen at one coverage,
#: and reused it everywhere; the audit showed that choice did not transfer and
#: had handicapped the family under test (docs/phase2_koplev.md section 7.8).
#: The fix here is structural rather than a bigger search: selection happens
#: inside each run against that run's own validation pairs, so no setting is
#: ever carried across a screen, a coverage or a split seed. It cannot touch
#: test data because the test pairs are not loaded until after the fit.
#:
#: The grids are small on purpose. This experiment asks whether residual signal
#: exists, not how well it can be predicted; an expensive search would trade the
#: question for a benchmark.
#:
#: An earlier version of this comment said that because ``zero`` has no grid,
#: "selection can only ever help the learned rungs -- a null result survives the
#: asymmetry". **That has the sign backwards**, and it is worth stating plainly
#: because it was the argument standing between a sparse-coverage negative and
#: the conclusion "flexible models do active harm". Selecting one of six settings
#: on 35-75 noisy validation pairs is a *net penalty* out of sample: at coverage
#: 0.05-0.20 the learned rungs score below the null, and that is the cost of the
#: selection machinery at that sample size rather than evidence that modelling
#: the residual is harmful. What actually protects a null here is the permutation
#: control and the registered powered-null qualifier, not this asymmetry.
HPARAM_GRID: dict[str, tuple[dict, ...]] = {
    "zero": ({},),
    "potential": (
        {"lr": 3e-3, "weight_decay": 1e-4},
        {"lr": 1e-2, "weight_decay": 1e-4},
        {"lr": 1e-2, "weight_decay": 1e-2},
        {"lr": 3e-2, "weight_decay": 1e-3},
    ),
    "lowrank": (
        {"rank": 2, "lr": 1e-2, "weight_decay": 1e-3},
        {"rank": 4, "lr": 1e-2, "weight_decay": 1e-3},
        {"rank": 4, "lr": 3e-2, "weight_decay": 1e-3},
        {"rank": 8, "lr": 1e-2, "weight_decay": 1e-3},
        {"rank": 8, "lr": 3e-2, "weight_decay": 1e-2},
        {"rank": 16, "lr": 3e-2, "weight_decay": 1e-2},
    ),
    "mlp": (
        {"emb_dim": 16, "hidden": 48, "lr": 3e-3, "weight_decay": 1e-3},
        {"emb_dim": 16, "hidden": 48, "lr": 1e-2, "weight_decay": 1e-3},
        {"emb_dim": 16, "hidden": 48, "lr": 3e-2, "weight_decay": 1e-2},
        {"emb_dim": 16, "hidden": 48, "lr": 1e-2, "weight_decay": 1e-1},
        {"emb_dim": 32, "hidden": 48, "lr": 1e-2, "weight_decay": 1e-3},
        {"emb_dim": 8, "hidden": 24, "lr": 1e-2, "weight_decay": 1e-3},
    ),
    "mlp_ordered": (
        {"emb_dim": 16, "hidden": 48, "lr": 3e-3, "weight_decay": 1e-3},
        {"emb_dim": 16, "hidden": 48, "lr": 1e-2, "weight_decay": 1e-3},
        {"emb_dim": 16, "hidden": 48, "lr": 3e-2, "weight_decay": 1e-2},
        {"emb_dim": 16, "hidden": 48, "lr": 1e-2, "weight_decay": 1e-1},
        {"emb_dim": 32, "hidden": 48, "lr": 1e-2, "weight_decay": 1e-3},
        {"emb_dim": 8, "hidden": 24, "lr": 1e-2, "weight_decay": 1e-3},
    ),
}

#: Keys in ``HPARAM_GRID`` that configure the *architecture* rather than the
#: optimiser. Split out so a setting can never be silently dropped: anything not
#: in this set is handed to the trainer, and the trainer raises on an unknown
#: key rather than ignoring it.
ARCH_KEYS = frozenset({"rank", "emb_dim", "hidden", "n_hidden_layers"})


def split_hparams(h: dict) -> tuple[dict, dict]:
    arch = {k: v for k, v in h.items() if k in ARCH_KEYS}
    opt = {k: v for k, v in h.items() if k not in ARCH_KEYS}
    return arch, opt


def apply_arch(cfg: ResidualModelConfig, arch: dict) -> ResidualModelConfig:
    return replace(cfg, **arch)
