"""The Phase 4 model ladder. Bipartite, not antisymmetric.

The shape of the question
-------------------------
A measured row is ``(acid a, amine n, condition c) -> y``, and the decomposition
under test is::

    y(a, n, c) = mu + f_A(x_a) + f_N(x_n) + f_C(c) + I_AN(a, n) + residual

with the low-dimensional hypothesis ``I_AN(a, n) = z_A(x_a)^T W z_N(x_n)``.

Acid and amine are **different entity types**. The reaction is
``R-COOH + H2N-R'``; there is no operation that exchanges them, so
``I_AN(n, a)`` does not typecheck and imposing ``I_AN(a, n) = -I_AN(n, a)``
would be meaningless rather than merely wrong. Phases 1-3 were about an
antisymmetric ``A`` because their two arguments were the same kind of object
in two positions. Nothing of that carries over except the method.

What is load-bearing is therefore not "does the model beat zero" and not
"does the model predict yield". It is::

    incremental pair skill = 1 - MSE(pair model) / MSE(additive model)

on rows whose acid, or amine, or both, the model has never trained on.

Why the ladder is nested rather than merely comparable
------------------------------------------------------
Every rung that adds a term adds it with the term's **output tensor
zero-initialised**, so at initialisation the richer model *is* the simpler one,
as a function, exactly. :class:`LowRankPair` at init is :class:`Additive`;
:class:`ConditionExpandedPair` at init is :class:`ConditionExpanded`. The
incremental skill a rung earns is then attributable to the term that was added
and not to a luckier fit of the terms that were already there. A non-nested
comparison leaves open the possibility that the pair model won by fitting the
substrate effects slightly better, which is not a claim anyone wants to make.

Exactly one tensor per interaction term starts at zero
------------------------------------------------------
The pair term is ``z_A^T W z_N``. If ``W`` starts at zero the gradient into the
``z`` encoders is zero; if the encoders also started at zero then ``z = 0`` and
the gradient into ``W`` would vanish too. Every partial derivative would be zero
at initialisation, the term would stay dead for the whole run, and the result
would be exactly 0.0 incremental skill -- indistinguishable in the table from
the finding this phase exists to test. Phase 2R shipped that bug once. So the
encoders are randomly initialised and only ``W`` is zeroed, and
``test_every_interaction_term_has_a_live_gradient_at_init`` pins it for every
rung.

Features enter as frozen buffers
--------------------------------
Each model registers the ``(n_acids, dim)`` and ``(n_amines, dim)`` matrices as
non-trainable buffers and looks up by index. A held-out entity's row therefore
holds *real information* rather than an untouched random initialisation -- which
is the entire difference between this and running the transductive model on an
unseen entity, where the held-out row would never receive a gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

#: Every rung the sweep may run. ``transductive`` is a labelled ceiling, never
#: an entity-generalisation result -- see :class:`TransductiveLowRank`.
MODELS: tuple[str, ...] = (
    "condition_only", "additive", "lowrank", "flexible",
    "condition_expanded", "condition_expanded_pair", "transductive")

#: The rungs whose contrast is the primary scientific claim, as (baseline, pair).
PRIMARY_CONTRAST: tuple[str, str] = ("additive", "lowrank")
#: The condition-confounding contrast: does pair structure survive after each
#: substrate role is allowed to interact with the condition?
ROBUST_CONTRAST: tuple[str, str] = ("condition_expanded", "condition_expanded_pair")


@dataclass(frozen=True)
class ModelConfig:
    n_acids: int
    n_amines: int
    n_conditions: int
    #: Frozen feature matrices, entity-index ordered.
    x_acid: np.ndarray
    x_amine: np.ndarray
    #: Latent width of the acid-amine bilinear form.
    rank: int = 4
    #: Latent width of the acid-condition and amine-condition forms.
    cond_rank: int = 4
    #: 0 = linear substrate heads; >0 = one hidden layer of this width.
    hidden: int = 0
    #: Width of the flexible comparator's role projections and its MLP.
    mlp_proj: int = 16
    mlp_hidden: int = 32
    seed: int = 0

    @property
    def dim(self) -> int:
        return int(self.x_acid.shape[1])


def _head(dim: int, out: int, hidden: int, zero_last: bool = False) -> nn.Module:
    """A linear map, or one hidden layer. No bias on the final layer.

    The bias is omitted because the model already carries a single global ``mu``
    and a per-condition intercept; a third additive constant would be a
    parameter that provably cannot change any prediction, and would then appear
    in the parameter counts used for the capacity comparison.

    ``zero_last`` zeroes the final layer so the head starts at exactly zero.
    Safe for a head whose output is *added* to the prediction: the gradient into
    a zero linear layer is its input, which is not zero. Never applied to a head
    whose output is *multiplied* by another zero tensor -- see the module
    docstring.
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


class _Base(nn.Module):
    """Holds the frozen features and the global intercept."""

    name = "base"

    def __init__(self, cfg: ModelConfig):
        torch.manual_seed(cfg.seed)
        super().__init__()
        self.cfg = cfg
        self.register_buffer("XA", torch.as_tensor(
            np.ascontiguousarray(cfg.x_acid), dtype=torch.float32), persistent=False)
        self.register_buffer("XN", torch.as_tensor(
            np.ascontiguousarray(cfg.x_amine), dtype=torch.float32), persistent=False)
        if self.XA.shape[0] != cfg.n_acids or self.XN.shape[0] != cfg.n_amines:
            raise ValueError("feature matrices must be entity-index ordered")
        self.mu = nn.Parameter(torch.zeros(()))

    def gather(self, head: nn.Module, X: torch.Tensor,
               idx: torch.Tensor) -> torch.Tensor:
        """``head(X)[idx]`` rather than ``head(X[idx])``.

        Mathematically identical -- the head is row-wise either way -- but it
        evaluates the head once per *entity* instead of once per *row*. There
        are 272 acids and ~6,500 training rows, so the gathered form does the
        same 2,048-column matrix multiply about twenty-four times over per
        epoch. Pinned by ``test_per_entity_and_gathered_heads_agree``.
        """
        return head(X)[idx]

    def forward(self, a: torch.Tensor, n: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def interaction_tensors(self) -> dict[str, nn.Parameter]:
        """The zero-initialised output tensor of each interaction term.

        Named so the gradient test can find them generically rather than being
        rewritten every time a rung is added.
        """
        return {}


class ConditionOnly(_Base):
    """``yhat = mu + f_C(c)``. A diagnostic reference, not a headline baseline.

    It says what a model that knows only which reagent was used can do. Useful
    because the condition main effect on this screen is large -- the exact-zero
    rate runs from 0.48 under PyBOP/DIPEA to 0.81 under TCFH/DIPEA -- and a
    reader needs to see how much of any model's raw skill is that alone.
    """

    name = "condition_only"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.fc = nn.Embedding(cfg.n_conditions, 1)
        nn.init.zeros_(self.fc.weight)

    def forward(self, a, n, c):
        return self.mu + self.fc(c).squeeze(-1)


class Additive(ConditionOnly):
    """``yhat = mu + f_A(x_a) + f_N(x_n) + f_C(c)``. **The load-bearing baseline.**

    Every substrate contributes independently of its partner. This is the model
    the pair term has to beat, and the reason the headline metric is a ratio of
    two MSEs rather than a skill against zero: a model can post a healthy R2 on
    this screen while containing no interaction whatsoever, because most of the
    outcome's variance is "is this acid reactive" plus "is this amine reactive"
    plus "is this reagent any good".
    """

    name = "additive"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.fa = _head(cfg.dim, 1, cfg.hidden, zero_last=True)
        self.fn = _head(cfg.dim, 1, cfg.hidden, zero_last=True)

    def substrate(self, a, n):
        return (self.gather(self.fa, self.XA, a).squeeze(-1)
                + self.gather(self.fn, self.XN, n).squeeze(-1))

    def forward(self, a, n, c):
        return self.mu + self.substrate(a, n) + self.fc(c).squeeze(-1)


class LowRankPair(Additive):
    """``Additive + z_A(x_a)^T W z_N(x_n)``, ``W`` zero-init. **The hypothesis.**

    ``z_A`` and ``z_N`` are separate role-specific encoders -- an acid and an
    amine do not share a latent space any more than they share a role -- and
    ``W`` is one ``rank x rank`` bilinear form shared by every pair. Because
    ``W`` starts at zero this *is* :class:`Additive` at initialisation, so the
    incremental skill it earns is attributable to the pair term.
    """

    name = "lowrank"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # NOT zero-initialised: z must be non-zero for W to receive a gradient.
        self.za = _head(cfg.dim, cfg.rank, cfg.hidden)
        self.zn = _head(cfg.dim, cfg.rank, cfg.hidden)
        self.W = nn.Parameter(torch.zeros(cfg.rank, cfg.rank))

    def pair(self, a, n):
        za = self.gather(self.za, self.XA, a)
        zn = self.gather(self.zn, self.XN, n)
        return torch.einsum("...i,ij,...j->...", za, self.W, zn)

    def forward(self, a, n, c):
        return super().forward(a, n, c) + self.pair(a, n)

    def interaction_tensors(self):
        return {"W": self.W}


class FlexiblePair(Additive):
    """``Additive + G([p_A(x_a), p_N(x_n), e_C(c)])`` with ``G``'s output zeroed.

    The capacity-controlled comparator. It asks whether the *low-rank
    restriction* is doing work or merely limiting: a bilinear form of rank ``r``
    is a strict subset of what this MLP can express, so if the MLP wins the
    restriction was too tight, and if the low-rank model wins at comparable
    capacity the restriction is a useful inductive bias rather than a
    convenience.

    Role-aware by construction: the acid and the amine go through *different*
    projections and are concatenated in a fixed order, so the network can tell
    which reactant is which. A symmetric pooling would impose an exchange
    symmetry the chemistry does not have.

    **This rung does not fit, and its result must not be read as evidence.**
    Measured on the authoritative folds, the fitted interaction term has a
    standard deviation of order 1e-19 to 1e-43 -- numerically zero -- against
    0.5-0.6 for the low-rank term on the same folds. It never leaves its
    initialisation. Its reported incremental skill of ~0.000 is therefore what
    an untrained term scores, not what a flexible model found, and the
    conclusion it was built to support -- "the low-rank restriction is the
    useful inductive bias rather than capacity" -- **is withdrawn**.

    The proximate cause is diagnosable and the fix was not: the projections read
    a 2,048-bit fingerprint with ~34 bits on, so 94 % of the MLP's output at
    initialisation is a constant that ``mu`` absorbs, leaving the
    zero-initialised output layer a gradient of order 1e-3 along the only
    direction that matters. Adding LayerNorm and removing the hidden biases
    doubled that and changed nothing measurable -- the fitted term stayed at
    1e-43. Rather than tune a secondary comparator until it agrees with the
    primary result, the rung is kept as registered, its deadness is measured
    into every result row as ``pair_term_sd``, and the claim is dropped.

    This is the trap this module's own docstring warns about for the bilinear
    term, in the one rung where nothing checked for it: a term that stays dead
    reports exactly 0.0 incremental skill, which is indistinguishable in a
    results table from a genuine finding of no benefit.
    """

    name = "flexible"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.pa = nn.Linear(cfg.dim, cfg.mlp_proj, bias=False)
        self.pn = nn.Linear(cfg.dim, cfg.mlp_proj, bias=False)
        self.ec = nn.Embedding(cfg.n_conditions, cfg.mlp_proj)
        self.g = nn.Sequential(
            nn.Linear(3 * cfg.mlp_proj, cfg.mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1, bias=False))
        nn.init.zeros_(self.g[-1].weight)

    def pair(self, a, n, c):
        h = torch.cat([self.gather(self.pa, self.XA, a),
                       self.gather(self.pn, self.XN, n),
                       self.ec(c)], dim=-1)
        return self.g(h).squeeze(-1)

    def forward(self, a, n, c):
        return super().forward(a, n, c) + self.pair(a, n, c)

    def interaction_tensors(self):
        return {"g_out": self.g[-1].weight}


class ConditionExpanded(Additive):
    """``Additive + u_A(x_a)^T V_A[c] + u_N(x_n)^T V_N[c]``.

    Each substrate role may interact with the reaction condition, at low rank.
    This exists because of a specific validity threat rather than for
    completeness: the screen is not a full factorial, reagent choice is not
    independent of substrate identity (normalised mutual information 0.16 with
    the acid and 0.20 with the amine, and 79 acids and 126 amines were run under
    a single reagent), so a learned acid-amine term could be absorbing
    substrate-by-condition compatibility that the additive baseline cannot
    express. Letting the baseline express it is the only way to find out.

    The ``V`` tables are zero-initialised; the ``u`` encoders are not.
    """

    name = "condition_expanded"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.ua = _head(cfg.dim, cfg.cond_rank, cfg.hidden)
        self.un = _head(cfg.dim, cfg.cond_rank, cfg.hidden)
        self.Va = nn.Embedding(cfg.n_conditions, cfg.cond_rank)
        self.Vn = nn.Embedding(cfg.n_conditions, cfg.cond_rank)
        nn.init.zeros_(self.Va.weight)
        nn.init.zeros_(self.Vn.weight)

    def cond_terms(self, a, n, c):
        return ((self.gather(self.ua, self.XA, a) * self.Va(c)).sum(-1)
                + (self.gather(self.un, self.XN, n) * self.Vn(c)).sum(-1))

    def forward(self, a, n, c):
        return super().forward(a, n, c) + self.cond_terms(a, n, c)

    def interaction_tensors(self):
        return {"Va": self.Va.weight, "Vn": self.Vn.weight}


class ConditionExpandedPair(ConditionExpanded):
    """``ConditionExpanded + z_A(x_a)^T W z_N(x_n)``. **The robustness contrast.**

    Scientifically the stronger of the two contrasts, because it asks whether
    acid-amine pair structure remains *after* each substrate has been allowed to
    interact with the reaction condition. If the low-rank gain over
    :class:`Additive` survives here it is not condition compatibility in
    disguise.
    """

    name = "condition_expanded_pair"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.za = _head(cfg.dim, cfg.rank, cfg.hidden)
        self.zn = _head(cfg.dim, cfg.rank, cfg.hidden)
        self.W = nn.Parameter(torch.zeros(cfg.rank, cfg.rank))

    def pair(self, a, n):
        za = self.gather(self.za, self.XA, a)
        zn = self.gather(self.zn, self.XN, n)
        return torch.einsum("...i,ij,...j->...", za, self.W, zn)

    def forward(self, a, n, c):
        return super().forward(a, n, c) + self.pair(a, n)

    def interaction_tensors(self):
        return {"W": self.W, "Va": self.Va.weight, "Vn": self.Vn.weight}


class _TransductiveAdditive(_Base):
    """``mu + a_i + b_j + f_C(c)``: free per-entity intercepts, no pair term.

    The transductive ceiling's own baseline, so the ceiling reports an
    *incremental* number on the same footing as every other row of the table
    rather than a raw skill that a per-entity intercept alone would earn.

    It is a separate class rather than :class:`TransductiveLowRank` with its pair
    term switched off, because inheriting the embedding tables and declining to
    use them would leave ``rank x (n_acids + n_amines)`` parameters that no
    gradient ever reaches -- dead weights that weight decay shrinks towards zero
    while they still count in the parameter totals used for the capacity
    comparison.
    """

    name = "transductive_additive"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.a_eff = nn.Embedding(cfg.n_acids, 1)
        self.n_eff = nn.Embedding(cfg.n_amines, 1)
        self.fc = nn.Embedding(cfg.n_conditions, 1)
        nn.init.zeros_(self.a_eff.weight)
        nn.init.zeros_(self.n_eff.weight)
        nn.init.zeros_(self.fc.weight)

    def additive_part(self, a, n, c):
        return (self.mu + self.a_eff(a).squeeze(-1) + self.n_eff(n).squeeze(-1)
                + self.fc(c).squeeze(-1))

    def forward(self, a, n, c):
        return self.additive_part(a, n, c)


class TransductiveLowRank(_TransductiveAdditive):
    """``mu + a_i + b_j + f_C(c) + u_i^T v_j`` with **free per-entity embeddings**.

    A labelled ceiling and never an entity-generalisation result. It answers a
    prior question: is the observed acid-amine interaction matrix learnable *at
    all* when each entity's latent vector may be estimated from its own
    measurements? If it is not, an inductive failure says nothing, because there
    would be no structure to infer. This is the same role Phase 2R played for
    Phase 3.

    Its embeddings are indexed by entity id, so a test entity's row receives no
    gradient and would emit its initialisation. It is therefore only ever
    evaluated in a transductive setting, where the "held out" unit is a *pair*
    and both endpoints appear elsewhere in training. The sweep marks its rows
    ``transductive=true`` and the report refuses to place it in an entity-OOD
    table.
    """

    name = "transductive"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.U = nn.Embedding(cfg.n_acids, cfg.rank)
        self.V = nn.Embedding(cfg.n_amines, cfg.rank)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.zeros_(self.V.weight)

    def pair(self, a, n):
        return (self.U(a) * self.V(n)).sum(-1)

    def forward(self, a, n, c):
        return self.additive_part(a, n, c) + self.pair(a, n)

    def interaction_tensors(self):
        return {"V": self.V.weight}


BUILDERS = {
    "condition_only": ConditionOnly,
    "additive": Additive,
    "lowrank": LowRankPair,
    "flexible": FlexiblePair,
    "condition_expanded": ConditionExpanded,
    "condition_expanded_pair": ConditionExpandedPair,
    "transductive": TransductiveLowRank,
    "transductive_additive": _TransductiveAdditive,
}


def build(name: str, cfg: ModelConfig) -> _Base:
    if name not in BUILDERS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[name](cfg)
