"""The Phase 3 model ladder: everything a drug is known by is its features.

The shape of the question
-------------------------
The target is the raw directional difference ``D(i, j) = y(i->j) - y(j->i)``,
which decomposes into a per-drug ordering tendency and a pair-specific
interaction::

    Dhat(i, j) = g(x_i) - g(x_j) + h(x_i, x_j),      h(x_i, x_j) = -h(x_j, x_i)

Phase 2R removed the first term by fitting it from the drug's own Koplev pairs
and then asked whether the second was predictable. That route is closed for an
unseen drug, so here the model has to earn *both* from features -- and the
scientifically load-bearing comparison is between the two terms, not between the
model and zero. A model can post a healthy skill against zero while knowing
nothing about interaction, because most of ``D``'s energy is potential.

Why the ladder is nested rather than merely comparable
------------------------------------------------------
:class:`FeatureLowRank` is :class:`FeaturePotential` plus a bilinear term whose
matrix is **initialised at exactly zero**. At initialisation the two models are
the same function, and the incremental skill of the low-rank model is therefore
the answer to a clean question: starting from the best feature-derived potential,
does adding a pair term help? A non-nested comparison would leave open the
possibility that the low-rank model won by fitting the potential slightly better,
which is not the claim anyone wants to make.

Zero-initialising ``W`` also matters for a reason Phase 2R learned the hard way:
a randomly initialised bilinear form emits O(1) noise into a target whose scale
is O(0.1), and the model spends its budget cancelling its own initialisation.
The counterpart failure is equally real -- an earlier Phase 2R design had a
*second* zero-initialised multiplicative scale, which made every gradient vanish
at init and produced a silent, permanent zero. There is exactly one zero-init
tensor here and ``test_lowrank_has_nonzero_gradients_at_init`` pins it.

Antisymmetry
------------
Every rung is exactly antisymmetric by construction, never by penalty:
``g(x_i) - g(x_j)`` and ``F(x_i, x_j) - F(x_j, x_i)`` are differences, and
``z_i^T K z_j`` with ``K = W - W^T`` flips sign under exchange at every parameter
setting. This is not decoration. ``D`` is antisymmetric by definition, so a model
that could emit a symmetric component would be spending capacity on a subspace
where the target is identically zero, and its held-out error would be inflated by
its own irrelevant degrees of freedom.

Features enter as a frozen buffer
---------------------------------
Each model registers the ``(n_drugs, dim)`` feature matrix as a non-trainable
buffer and looks up ``X[i]``. Two consequences worth stating. It makes these
models drop-in for Phase 2R's trainer, which indexes by drug id -- so Phase 3
reuses that optimiser, restart and selection logic unchanged rather than
reimplementing it. And it means a held-out drug's row holds *real information*
rather than a random initialisation, which is exactly the difference between this
phase and running Phase 2R's embedding model on an unseen entity, where the
held-out row would never receive a gradient and the model would emit noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..residual_models import _Residual


@dataclass(frozen=True)
class EntityModelConfig:
    n_drugs: int
    #: The frozen feature matrix, already restricted to the training-drug view.
    x: np.ndarray
    rank: int = 2
    hidden: int = 0          # 0 = linear head
    mlp_hidden: int = 32     # the unrestricted comparator's width
    seed: int = 0

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])


class _FeatureModel(_Residual):
    """Base: holds the frozen features and indexes them by drug id."""

    def __init__(self, cfg: EntityModelConfig):
        torch.manual_seed(cfg.seed)
        nn.Module.__init__(self)
        self.cfg = cfg
        self.register_buffer(
            "X", torch.as_tensor(np.ascontiguousarray(cfg.x), dtype=torch.float32),
            persistent=False)
        if self.X.shape[0] != cfg.n_drugs:
            raise ValueError("feature matrix rows must equal n_drugs")

    def features(self, idx: torch.Tensor) -> torch.Tensor:
        return self.X[idx]

    def apply_per_drug(self, head: nn.Module, idx: torch.Tensor) -> torch.Tensor:
        """``head(X)[idx]`` rather than ``head(X[idx])``.

        Mathematically identical -- the head is applied row-wise either way --
        but it evaluates the head once per *drug* instead of once per *pair
        endpoint*. There are 100 drugs and 3,160 training pairs, so the gathered
        form does the same 2,048-column matrix multiply about sixty times over
        every epoch. Pinned by ``test_per_drug_and_gathered_heads_agree``.
        """
        return head(self.X)[idx]


def _head(dim: int, out: int, hidden: int, zero_last: bool = False) -> nn.Module:
    """A linear map, or one hidden layer. No bias on the final layer.

    The bias is omitted because every use of a head here is inside a difference
    -- ``g(x_i) - g(x_j)``, or a bilinear form -- where a constant cancels
    exactly. Keeping it would add a parameter that provably cannot change any
    prediction, and would then appear in the parameter counts used for the
    capacity-fairness comparison.

    ``zero_last`` zeroes the final layer so the head starts at exactly zero.
    Which heads get it is not a matter of taste, and getting it wrong breaks the
    model in one of two opposite ways.

    *Heads whose output is added to the prediction* -- ``g`` in the potential
    model, ``F`` in the unrestricted model -- are zero-initialised. A randomly
    initialised linear map over 2,048 fingerprint bits emits O(0.25) into a
    target whose scale is O(0.1): the run would open well outside the data and
    spend its budget cancelling its own initialisation. Zeroing is safe here
    because the gradient into a zero linear layer is the input, which is not
    zero.

    *The latent encoder* ``f`` in the low-rank model is emphatically **not**
    zero-initialised, and this is the trap Phase 2R fell into once already. The
    pair term is ``z_i^T (W - W^T) z_j``. If ``W`` starts at zero, the gradient
    into ``f`` is zero; if ``f`` also started at zero then ``z = 0`` and the
    gradient into ``W`` would be zero too. Every partial derivative would vanish
    at initialisation and the pair term would stay dead for the whole run --
    reporting exactly 0.0 incremental skill, which is indistinguishable in the
    results table from the finding this phase exists to test. So exactly one of
    the two tensors in the pair term starts at zero.
    """
    if hidden <= 0:
        layer = nn.Linear(dim, out, bias=False)
        if zero_last:
            nn.init.zeros_(layer.weight)
        return layer
    stack = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, out, bias=False))
    if zero_last:
        nn.init.zeros_(stack[-1].weight)
    return stack


class FeatureZero(_Residual):
    """``Dhat = 0``. The mandatory null; no parameters, nothing fitted."""

    name = "zero"

    def __init__(self, cfg: EntityModelConfig):
        nn.Module.__init__(self)
        self.cfg = cfg

    def d_res(self, i, j):
        return torch.zeros(i.shape, dtype=torch.float32)


class FeaturePotential(_FeatureModel):
    """``Dhat = g(x_i) - g(x_j)``: transferable first/second-position tendency.

    The critical baseline. Most of ``D``'s energy is potential, so a model that
    predicts only this can look like a success at predicting directional
    interaction while containing no interaction term at all. Every richer rung is
    read against *this*, not against zero.
    """

    name = "potential"

    def __init__(self, cfg: EntityModelConfig):
        super().__init__(cfg)
        self.g = _head(cfg.dim, 1, cfg.hidden, zero_last=True)

    def potential(self, idx: torch.Tensor) -> torch.Tensor:
        return self.apply_per_drug(self.g, idx).squeeze(-1)

    def d_res(self, i, j):
        return self.potential(i) - self.potential(j)


class FeatureLowRank(FeaturePotential):
    """``Dhat = g(x_i) - g(x_j) + z_i^T K z_j``, ``K = W - W^T``, ``W`` zero-init.

    The hypothesis under test. ``z = f(x)`` maps a drug's structure into a small
    latent space and ``K`` is one antisymmetric bilinear form shared by every
    pair -- the same geometry Phase 2R found at rank 2 with free embeddings, but
    with the embedding replaced by a function of the molecule.

    Because ``W`` starts at zero this *is* :class:`FeaturePotential` at
    initialisation, so the incremental skill it earns is attributable to the pair
    term and not to a better-fitted potential.
    """

    name = "lowrank"

    def __init__(self, cfg: EntityModelConfig):
        super().__init__(cfg)
        # NOT zero-initialised -- see _head. z must be non-zero for W to
        # receive any gradient at all.
        self.f = _head(cfg.dim, cfg.rank, cfg.hidden)
        self.W = nn.Parameter(torch.zeros(cfg.rank, cfg.rank))

    def latent(self, idx: torch.Tensor) -> torch.Tensor:
        return self.apply_per_drug(self.f, idx)

    def d_res(self, i, j):
        K = self.W - self.W.T
        zi, zj = self.latent(i), self.latent(j)
        pair = torch.einsum("...a,ab,...b->...", zi, K, zj)
        return self.potential(i) - self.potential(j) + pair


class PairOnlyLowRank(_FeatureModel):
    """``Dhat = z_i^T K z_j``: the bilinear term with no potential head at all.

    A diagnostic, not a rung of the ladder. It isolates how much the pair term
    can carry *alone*. If :class:`FeatureLowRank` beats :class:`FeaturePotential`
    but this scores near zero, the gain came from the bilinear form acting as
    spare capacity for the potential rather than from genuine pair structure.

    It subclasses :class:`_FeatureModel` rather than :class:`FeatureLowRank`
    precisely so that no potential head exists. Inheriting one and declining to
    call it would leave ``dim`` parameters that no gradient ever reaches -- dead
    weights that weight decay would shrink towards zero while still counting in
    the parameter totals used for the capacity comparison.
    """

    name = "pair_only"

    def __init__(self, cfg: EntityModelConfig):
        super().__init__(cfg)
        self.f = _head(cfg.dim, cfg.rank, cfg.hidden)
        self.W = nn.Parameter(torch.zeros(cfg.rank, cfg.rank))

    def latent(self, idx: torch.Tensor) -> torch.Tensor:
        return self.apply_per_drug(self.f, idx)

    def d_res(self, i, j):
        K = self.W - self.W.T
        return torch.einsum("...a,ab,...b->...", self.latent(i), K, self.latent(j))


class AntisymFeatureMLP(_FeatureModel):
    """``Dhat = F(x_i, x_j) - F(x_j, x_i)``: the flexible upper bound.

    Antisymmetric but otherwise unrestricted, so it can express interactions of
    any rank. Its role is to say whether the low-rank restriction is *helping* or
    merely *limiting*: if it beats the low-rank model, rank 2 was too small a
    hypothesis; if the low-rank model beats it at comparable capacity, the
    restriction is a useful inductive bias rather than a convenience.

    The first layer is stored as two blocks rather than one wide matrix, which is
    an exact refactoring and a large speed-up. Writing the input as a
    concatenation makes ``W [x_i; x_j] = W1 x_i + W2 x_j``, so the two halves can
    be evaluated once per *drug* (100 rows) instead of once per *pair endpoint*
    (3,160 rows, twice, for both orientations). Same function, ~60x less work;
    ``test_antisym_mlp_matches_the_concatenated_form`` pins the equality against
    an explicit concatenated reference.
    """

    name = "antisym_mlp"

    def __init__(self, cfg: EntityModelConfig):
        super().__init__(cfg)
        h = cfg.mlp_hidden
        self.first = nn.Linear(cfg.dim, h)          # the W1 block, carries the bias
        self.second = nn.Linear(cfg.dim, h, bias=False)   # the W2 block
        self.out = nn.Linear(h, 1, bias=False)
        # Same discipline as the bilinear form: the last layer starts at zero, so
        # the model begins at Dhat = 0 rather than emitting O(1) noise into a
        # target of scale O(0.1). The first-layer blocks stay random, so the
        # gradient into `out` is non-zero and nothing is dead.
        nn.init.zeros_(self.out.weight)

    def d_res(self, i, j):
        a = self.first(self.X)                      # (n_drugs, hidden), with bias
        b = self.second(self.X)                     # (n_drugs, hidden), no bias
        fwd = self.out(torch.relu(a[i] + b[j])).squeeze(-1)
        rev = self.out(torch.relu(a[j] + b[i])).squeeze(-1)
        return fwd - rev


LADDER: dict[str, type[_Residual]] = {
    "zero": FeatureZero,
    "potential": FeaturePotential,
    "lowrank": FeatureLowRank,
    "pair_only": PairOnlyLowRank,
    "antisym_mlp": AntisymFeatureMLP,
}

#: Reported order. ``pair_only`` sits last because it is a diagnostic.
LADDER_ORDER = ("zero", "potential", "lowrank", "antisym_mlp", "pair_only")

#: Hyperparameter grids, searched on **entity-OOD validation** only. Small on
#: purpose: the primary contrast is nested and the question is whether a pair
#: term helps at all, not which of forty settings is best. Ranks stop at 8
#: because Phase 2R's structure was already fully visible at rank 2 and there is
#: no result here that would be rescued by more capacity.
HPARAM_GRID: dict[str, tuple[dict, ...]] = {
    "zero": ({},),
    "potential": (
        {"hidden": 0, "lr": 1e-2, "weight_decay": 1e-3},
        {"hidden": 0, "lr": 1e-2, "weight_decay": 1e-1},
        {"hidden": 0, "lr": 3e-3, "weight_decay": 1e-2},
        {"hidden": 32, "lr": 3e-3, "weight_decay": 1e-2},
    ),
    "lowrank": (
        {"rank": 2, "hidden": 0, "lr": 1e-2, "weight_decay": 1e-3},
        {"rank": 2, "hidden": 0, "lr": 1e-2, "weight_decay": 1e-1},
        {"rank": 4, "hidden": 0, "lr": 1e-2, "weight_decay": 1e-2},
        {"rank": 8, "hidden": 0, "lr": 1e-2, "weight_decay": 1e-2},
        {"rank": 2, "hidden": 32, "lr": 3e-3, "weight_decay": 1e-2},
    ),
    "pair_only": (
        {"rank": 2, "hidden": 0, "lr": 1e-2, "weight_decay": 1e-3},
        {"rank": 4, "hidden": 0, "lr": 1e-2, "weight_decay": 1e-2},
    ),
    "antisym_mlp": (
        {"mlp_hidden": 16, "lr": 3e-3, "weight_decay": 1e-2},
        {"mlp_hidden": 32, "lr": 3e-3, "weight_decay": 1e-2},
        {"mlp_hidden": 32, "lr": 1e-2, "weight_decay": 1e-1},
    ),
}

#: Keys that shape the architecture; the rest are optimiser settings.
ARCH_KEYS = ("rank", "hidden", "mlp_hidden")


def split_hparams(hp: dict) -> tuple[dict, dict]:
    arch = {k: v for k, v in hp.items() if k in ARCH_KEYS}
    opt = {k: v for k, v in hp.items() if k not in ARCH_KEYS}
    return arch, opt


def build_entity_model(name: str, cfg: EntityModelConfig) -> _Residual:
    return LADDER[name](cfg)
