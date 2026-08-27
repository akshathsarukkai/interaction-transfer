"""The model families compared on the Koplev screen.

Only one thing differs between them: how the *pair* term is parameterised.
Embeddings, the first-order term, the pair feature map, depth, the optimiser and
the training budget are identical, and the pair-parameter budget is matched by
search rather than by hope (:func:`match_pair_hidden`).

    y(i -> j)  =  a_i + b_j  +  <pair term>

The first-order term is two free scalars per drug -- ``a_i`` for the drug in the
*first* position and ``b_j`` for the drug in the *second* -- held in embedding
tables, not decoded from the pair embedding by an MLP. That detail is not
cosmetic. An earlier version routed the first-order term through
``MLP(e_i) + MLP(e_j)``, which is (a) symmetric, so it could not express the
row-versus-column asymmetry at all, and (b) bottlenecked through an 8-dimensional
embedding. Every family then scored R^2 ~ 0.03 on held-out pairs while a
closed-form two-way ridge on the same split scored 0.435 -- i.e. all three
"models" were far worse than the trivial additive fit, and any comparison
between them would have been a comparison of optimisation failures. See
``docs/phase2_koplev.md``.

Families
--------
``Additive``           pair term = 0.  The two-way additive null, ``a_i + b_j``.
                       Not a competitor: it is the reference that says how much
                       of this screen is explained without any pair interaction
                       at all, and on this data that turns out to be most of it.

``OrderInsensitive``   the whole prediction is symmetrised,
                       ``(f(i,j) + f(j,i)) / 2``, so it emits the same value for
                       both schedules. Its gap to the other two is how much
                       order information the screen contains beyond what an
                       order-blind model can reach.

``Unrestricted``       pair term = ``G(phi(e_i, e_j))``. No constraint. The
                       primary baseline.

``Structured``         y(i->j) = M(i,j) + A(i,j) with, by construction,

                           M(i,j) = [ (a_i+b_j) + (a_j+b_i) ] / 2
                                    + [ F_M(phi_ij) + F_M(phi_ji) ] / 2
                           A(i,j) = [ (a_i+b_j) - (a_j+b_i) ] / 2
                                    + [ F_A(phi_ij) - F_A(phi_ji) ] / 2

                       so ``M(i,j) = M(j,i)`` and ``A(i,j) = -A(j,i)`` hold
                       exactly, at every parameter setting, not approximately
                       after training. The first-order term needs no special
                       handling: ``a_i + b_j`` splits into a symmetric and an
                       antisymmetric half on its own, and the two halves land in
                       M and A respectively.

Which family is the bigger function class -- and why it barely matters
---------------------------------------------------------------------
An earlier draft of this docstring, and of ``docs/phase2_koplev.md``, said the
unrestricted family is a strict *superset* of the structured one. **That is
backwards.** Setting ``F_M = F_A = G`` gives

    0.5*(G_ij + G_ji) + 0.5*(G_ij - G_ji)  ==  G_ij   identically,

so at equal head width ``Structured(h)`` contains ``Unrestricted(h)`` exactly
(verified to float32 roundoff in ``test_structured_contains_unrestricted``).
The symmetric/antisymmetric split is the *identity* decomposition of an
arbitrary function, not a restriction of one -- which is also why
``_Base.implied_A`` can use it as a like-for-like readout on every family.

At the widths actually run, capacity is matched by parameter count (two heads of
width 48 against one of width 86, 32,546 vs 32,423 parameters), so neither class
contains the other and both comfortably contain the truth. The hypothesis under
test is therefore **not** "does constraining the function class help" but
"does splitting the pair term into a symmetric and an antisymmetric head, each
with its own parameters, give a better optimisation and regularisation bias than
one undivided head of the same total size". A win for the structured family is
a win for parameterisation, not for expressiveness -- and equally, it cannot be
dismissed as extra capacity.

What is deliberately *not* carried over from Phase 1
----------------------------------------------------
The Phase 1 ``AlgebraModel`` tied a simultaneous-treatment row and two
single-intervention rows to the same ``S``. The Koplev study did not measure
simultaneous treatment at all -- "while simultaneous treatments were not directly
measured, and thus cannot be definitively excluded" -- and its single-agent
responses are already absorbed into the baseline that ``synergy_measure`` is
defined against, so they are not separate observations either. Importing that
structure would encode an assumption the data cannot support. ``M`` here is only
"the symmetric part of the ordered response", with no claim that it equals a
measurable co-treatment outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int = 2,
        zero_init_output: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(prev, hidden), nn.SiLU()]
        prev = hidden
    out = nn.Linear(prev, out_dim)
    if zero_init_output:
        # Start every pair head at exactly zero, so a run begins at the two-way
        # additive fit and the pair term has to earn its way in. Without this the
        # randomly initialised pair net emits O(1) noise in standardised units,
        # swamps the first-order term, memorises the training rows within two or
        # three epochs and trips early stopping before it has learned anything --
        # measured R^2 0.02 against 0.41 for the additive null on the same split.
        # Applied identically to every family's pair head(s).
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


def mlp_param_count(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int) -> int:
    if n_hidden_layers <= 0:
        return in_dim * out_dim + out_dim
    n = in_dim * hidden + hidden
    n += (n_hidden_layers - 1) * (hidden * hidden + hidden)
    n += hidden * out_dim + out_dim
    return n


@dataclass(frozen=True)
class RealModelConfig:
    n_drugs: int = 100
    emb_dim: int = 16
    pair_hidden: int = 48          # width of ONE structured head (F_M or F_A)
    n_hidden_layers: int = 2
    emb_init_std: float = 0.5
    first_order_init_std: float = 0.01
    seed: int = 0


class PairFeatures(nn.Module):
    """``phi(e_i, e_j) = [e_i, e_j, vec(e_i e_j^T)]`` -- order-dependent, shared.

    The linear block lets a pair net emit ``c_i - c_j``, which is exactly the
    "potential" form that accounts for roughly half of the antisymmetric variance
    in this screen; the outer block supplies the multiplicative features a small
    MLP would otherwise have to learn from a few hundred pairs. Both blocks go to
    every family, so neither can favour one.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = 2 * emb_dim + emb_dim * emb_dim

    def forward(self, ei: torch.Tensor, ej: torch.Tensor) -> torch.Tensor:
        outer = (ei.unsqueeze(-1) * ej.unsqueeze(-2)).flatten(-2)
        return torch.cat([ei, ej, outer], dim=-1)


class _Base(nn.Module):
    family = "base"
    #: MLP forward passes through the pair net per predicted row. The structured
    #: family needs four; reported so a compute difference is never mistaken for
    #: a parameter difference.
    pair_evals_per_row = 0

    def __init__(self, cfg: RealModelConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.n_drugs, cfg.emb_dim)
        nn.init.normal_(self.emb.weight, std=cfg.emb_init_std)
        # Free per-drug, per-role offsets. Identical in every family.
        self.first_a = nn.Embedding(cfg.n_drugs, 1)
        self.first_b = nn.Embedding(cfg.n_drugs, 1)
        nn.init.normal_(self.first_a.weight, std=cfg.first_order_init_std)
        nn.init.normal_(self.first_b.weight, std=cfg.first_order_init_std)
        self.bias = nn.Parameter(torch.zeros(1))
        self.phi = PairFeatures(cfg.emb_dim)

    def first_order(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return (self.bias + self.first_a(i).squeeze(-1)
                + self.first_b(j).squeeze(-1))

    def pair_term(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return self.first_order(i, j) + self.pair_term(i, j)

    def n_first_order_params(self) -> int:
        return (self.first_a.weight.numel() + self.first_b.weight.numel()
                + self.bias.numel())

    def implied_A(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Half the forward-minus-reverse prediction.

        Defined the same way for every family -- ``(f(i,j) - f(j,i)) / 2`` -- so
        it is a like-for-like readout and not a privileged look inside the
        structured model. Note this is the *total* antisymmetric output and
        includes the first-order term; it is **not** a collapse detector, see
        :meth:`pair_head_A`.
        """
        return 0.5 * (self.forward(i, j) - self.forward(j, i))

    def first_order_A(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """The antisymmetric half of ``a_i + b_j``, i.e. ``((a_i-b_i)-(a_j-b_j))/2``.

        Present in every family that is not fully symmetrised, and identical in
        form across them. On this screen it is not a small term -- a two-way
        additive fit alone captures roughly half the measured antisymmetric
        variance -- which is exactly why it has to be separated from the pair
        head before anything is called a collapse.
        """
        return 0.5 * (self.first_order(i, j) - self.first_order(j, i))

    def pair_head_A(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """The antisymmetric contribution of the *pair head alone*.

        This is the collapse detector, and the reason it exists is that
        :meth:`implied_A` cannot be one. ``implied_A`` includes
        :meth:`first_order_A`, which is common to every family and nonzero
        regardless of what the pair network is doing: with ``F_A`` pinned to
        identically zero, ``implied_A`` still reads ~0.42 rms while the true
        head contribution is exactly 0. A diagnostic that reports a healthy
        number for a dead head is worse than no diagnostic, because it is
        published as reassurance.
        """
        return 0.5 * (self.pair_term(i, j) - self.pair_term(j, i))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_pair_params(self) -> int:
        return 0


class Additive(_Base):
    """The two-way additive null: ``y(i->j) = a_i + b_j``. Reference, not rival."""

    family = "additive"
    pair_evals_per_row = 0

    def pair_term(self, i, j):
        return torch.zeros_like(i, dtype=torch.float32)


class OrderInsensitive(_Base):
    """Order-blind by construction: the whole prediction is symmetrised.

    Not "the structured model with A deleted" -- the *entire* function, first-
    order term included, is replaced by its symmetric projection, so the family
    cannot express any order dependence through any route.
    """

    family = "order_insensitive"
    pair_evals_per_row = 2

    def __init__(self, cfg: RealModelConfig):
        super().__init__(cfg)
        self.F_M = mlp(self.phi.out_dim, cfg.pair_hidden, 1, cfg.n_hidden_layers,
                     zero_init_output=True)

    def _f(self, i, j):
        return self.F_M(self.phi(self.emb(i), self.emb(j))).squeeze(-1)

    def pair_term(self, i, j):
        return 0.5 * (self._f(i, j) + self._f(j, i))

    def forward(self, i, j):
        sym_first = 0.5 * (self.first_order(i, j) + self.first_order(j, i))
        return sym_first + self.pair_term(i, j)

    def first_order_A(self, i, j):
        # The first-order term is symmetrised inside forward(), so it
        # contributes nothing antisymmetric.
        return torch.zeros_like(i, dtype=torch.float32)

    def n_pair_params(self) -> int:
        return sum(p.numel() for p in self.F_M.parameters())


class Unrestricted(_Base):
    family = "unrestricted"
    pair_evals_per_row = 1

    def __init__(self, cfg: RealModelConfig):
        super().__init__(cfg)
        self.G = mlp(self.phi.out_dim, cfg.pair_hidden, 1, cfg.n_hidden_layers,
                     zero_init_output=True)

    def pair_term(self, i, j):
        return self.G(self.phi(self.emb(i), self.emb(j))).squeeze(-1)

    def n_pair_params(self) -> int:
        return sum(p.numel() for p in self.G.parameters())


class Structured(_Base):
    family = "structured"
    pair_evals_per_row = 4

    def pair_head_A(self, i, j):
        """``F_A``'s contribution alone -- exactly zero if ``F_A`` has collapsed."""
        return 0.5 * (self._f(self.F_A, i, j) - self._f(self.F_A, j, i))

    def __init__(self, cfg: RealModelConfig):
        super().__init__(cfg)
        self.F_M = mlp(self.phi.out_dim, cfg.pair_hidden, 1, cfg.n_hidden_layers,
                     zero_init_output=True)
        self.F_A = mlp(self.phi.out_dim, cfg.pair_hidden, 1, cfg.n_hidden_layers,
                     zero_init_output=True)

    def _f(self, net, i, j):
        return net(self.phi(self.emb(i), self.emb(j))).squeeze(-1)

    def M(self, i, j):
        """The symmetric component, first-order term included. ``M(i,j)=M(j,i)``."""
        return (0.5 * (self.first_order(i, j) + self.first_order(j, i))
                + 0.5 * (self._f(self.F_M, i, j) + self._f(self.F_M, j, i)))

    def A(self, i, j):
        """The antisymmetric component. ``A(i,j) = -A(j,i)``."""
        return (0.5 * (self.first_order(i, j) - self.first_order(j, i))
                + 0.5 * (self._f(self.F_A, i, j) - self._f(self.F_A, j, i)))

    def pair_term(self, i, j):
        return (0.5 * (self._f(self.F_M, i, j) + self._f(self.F_M, j, i))
                + 0.5 * (self._f(self.F_A, i, j) - self._f(self.F_A, j, i)))

    def n_pair_params(self) -> int:
        return (sum(p.numel() for p in self.F_M.parameters())
                + sum(p.numel() for p in self.F_A.parameters()))


FAMILIES: dict[str, type[_Base]] = {
    "additive": Additive,
    "order_insensitive": OrderInsensitive,
    "unrestricted": Unrestricted,
    "structured": Structured,
}

#: The structured family sets the pair-parameter budget; the other two are
#: widened to match it.
REFERENCE_FAMILY = "structured"


def pair_param_count(family: str, cfg: RealModelConfig, hidden: int) -> int:
    if family == "additive":
        return 0
    one = mlp_param_count(PairFeatures(cfg.emb_dim).out_dim, hidden, 1,
                          cfg.n_hidden_layers)
    return 2 * one if family == "structured" else one


def match_pair_hidden(family: str, cfg: RealModelConfig,
                      search: range | None = None) -> int:
    """Width for ``family`` whose pair-parameter count is closest to the reference.

    The structured family runs two narrow heads; a single-head family at the same
    width would carry roughly half the pair parameters, and every result would be
    confounded by capacity. Both single-head families are widened, which can only
    help the baselines.
    """
    if family in (REFERENCE_FAMILY, "additive"):
        return cfg.pair_hidden
    ref = pair_param_count(REFERENCE_FAMILY, cfg, cfg.pair_hidden)
    best, gap = cfg.pair_hidden, float("inf")
    for h in (search or range(4, 1025)):
        g = abs(pair_param_count(family, cfg, h) - ref)
        if g < gap:
            best, gap = h, g
    return best


def build_model(family: str, cfg: RealModelConfig) -> _Base:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected {sorted(FAMILIES)}")
    return FAMILIES[family](RealModelConfig(**{
        **cfg.__dict__, "pair_hidden": match_pair_hidden(family, cfg)}))
