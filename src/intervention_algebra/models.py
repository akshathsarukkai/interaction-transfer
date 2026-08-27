"""Model families for the intervention-algebra Phase 1 benchmark.

All families share exactly the same machinery except for the pair term:

  * embeddings ``e_i``            (nn.Embedding, same size everywhere)
  * first-order head ``v_i = V(e_i)``
  * a shared pair-feature map ``phi(e_i, e_j)``  (order-dependent by construction)
  * optional shared elementwise observation head (only used when the benchmark
    applies a nonlinear observation map)

Families
--------
``AdditiveModel``        pred_sim = pred_ord = v_i + v_j            (no interactions)
``UnconstrainedModel``   pred_sim = v_i+v_j + G_sim(phi)
                         pred_ord = v_i+v_j + G_ord(phi)           (independent terms)
``SharedPairModel``      pred_sim = v_i+v_j + G_sim(phi)
                         pred_ord = v_i+v_j + G_sim(phi) + G_ord(phi)
``AlgebraModel``         pred_sim = v_i+v_j + S(i,j)
                         pred_ord = v_i+v_j + S(i,j) + A(i,j)

with, *by construction*,

  S(i,j) = ( F_S(phi(e_i,e_j)) + F_S(phi(e_j,e_i)) ) / 2   =>  S(i,j) =  S(j,i)
  A(i,j) = ( F_A(phi(e_i,e_j)) - F_A(phi(e_j,e_i)) ) / 2   =>  A(i,j) = -A(j,i)

What the algebra constraint actually buys
-----------------------------------------
Each unordered pair contributes **three** observation rows: simultaneous {i,j},
ordered i->j and ordered j->i. The algebra model explains all three with two
quantities (S, A) tied by exact symmetry/antisymmetry, so the three rows impose a
hard consistency constraint. ``UnconstrainedModel`` models the simultaneous and
ordered terms with independent functions; ``SharedPairModel`` shares the
simultaneous term but leaves the ordered residual unconstrained. The step from
it to ``AlgebraModel`` isolates the **cross-row identity**, not antisymmetry
alone -- see the note on ``SharedPairModel`` for the four things that step
actually changes.

Note on the simultaneous term of the unconstrained families: simultaneous rows
are physically order-free and are always stored canonically (i<j), so the pair is
fed to ``G_sim`` in canonical index order. This is what a competent practitioner
would do and makes the baseline strictly stronger -- it means symmetry of the
*simultaneous* term is free for the baseline, and no credit is claimed for it.
The algebra model's symmetry constraint still has teeth because its ``S`` is
also evaluated inside the two ordered rows, which appear in **both** index
orders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(prev, hidden), nn.SiLU()]
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


PairMode = Literal["concat", "concat_outer", "outer"]


@dataclass(frozen=True)
class ModelConfig:
    n_interventions: int = 48
    out_dim: int = 4
    emb_dim: int = 8
    v_hidden: int = 32
    pair_hidden: int = 48
    n_hidden_layers: int = 2
    pair_mode: PairMode = "concat_outer"
    emb_init_std: float = 0.5
    observation_head: bool = False   # learnable elementwise output map
    seed: int = 0


class PairFeatures(nn.Module):
    """Order-dependent pair feature map ``phi(e_i, e_j)``, shared by all families.

    ``concat``        -> [e_i, e_j]                       (2 * emb_dim)
    ``concat_outer``  -> [e_i, e_j, vec(e_i e_j^T)]       (2*emb_dim + emb_dim^2)
    ``outer``         -> vec(e_i e_j^T)                   (emb_dim^2)

    Why ``outer`` and not ``concat_outer``: the linear ``[e_i, e_j]`` block lets
    the pair network emit a term of the form ``c_i + c_j``.  That term is
    symmetric and is exactly the direction along which the (v, S) split is
    gauge-degenerate.  On *training* pairs it is harmless -- the single-
    intervention rows pin ``v``, so the pair term is forced to equal the true
    ``S`` there -- but on held-out pairs nothing stops the pair network
    extrapolating along it, which inflates ``pred_S_rms`` and corrupts the
    recovered ``S`` (measured on dev seed 100 at coverage 0.4:
    held-out S-recovery r 0.86-0.89 with ``outer`` vs 0.75-0.77 with
    ``concat_outer``, for both families alike --
    ``results/SUPERSEDED_featuremap_testmetric.jsonl``).  Earlier revisions of
    this docstring quoted "0.86 vs 0.39 at coverage 0.8"; no run at coverage 0.8
    exists anywhere in the repository and 0.39 appears in no artifact.  Pure
    bilinear features cannot express ``c_i + c_j`` at all.

    The outer-product block matters: the true interactions in the benchmark are
    *bilinear* forms of a latent factor, and a plain MLP on a concatenation has
    to learn multiplication from scratch from a few hundred pairs. Supplying the
    product features is a modelling choice applied **identically to every
    family**, so it cannot favour one of them.
    """

    def __init__(self, emb_dim: int, mode: PairMode = "concat_outer"):
        super().__init__()
        self.emb_dim = emb_dim
        self.mode = mode
        if mode == "concat":
            self.out_dim = 2 * emb_dim
        elif mode == "concat_outer":
            self.out_dim = 2 * emb_dim + emb_dim * emb_dim
        elif mode == "outer":
            self.out_dim = emb_dim * emb_dim
        else:
            raise ValueError(f"unknown pair_mode {mode!r}")

    def forward(self, ei: torch.Tensor, ej: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return torch.cat([ei, ej], dim=-1)
        outer = (ei.unsqueeze(-1) * ej.unsqueeze(-2)).flatten(-2)
        if self.mode == "outer":
            return outer
        return torch.cat([ei, ej, outer], dim=-1)


class ElementwiseHead(nn.Module):
    """Per-channel monotone squashing  y_c = b_c + w2_c * tanh(w1_c * z_c + b1_c).

    Shared verbatim by all model families so that a nonlinear observation map in
    the benchmark does not advantage any one of them.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(dim))
        self.b1 = nn.Parameter(torch.zeros(dim))
        self.w2 = nn.Parameter(torch.ones(dim))
        self.b2 = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.b2 + self.w2 * torch.tanh(self.w1 * z + self.b1)


class BaseModel(nn.Module):
    family: str = "base"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.n_interventions, cfg.emb_dim)
        nn.init.normal_(self.emb.weight, std=cfg.emb_init_std)
        self.v_net = mlp(cfg.emb_dim, cfg.v_hidden, cfg.out_dim, cfg.n_hidden_layers)
        self.phi = PairFeatures(cfg.emb_dim, cfg.pair_mode)
        self.head = ElementwiseHead(cfg.out_dim) if cfg.observation_head else None

    # -------------------------------------------------------------- pieces
    def v(self, idx: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.emb(idx))

    def _zeros(self, i: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(i), self.cfg.out_dim, device=self.emb.weight.device)

    def implied_S(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """The symmetric pair term the model *actually uses* for a simultaneous row."""
        return self._zeros(i)

    def implied_A(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Half the forward-minus-reverse ordered difference, as the model predicts it.

        For every family this equals
        ``(latent_ordered(i,j) - latent_ordered(j,i)) / 2`` in latent space, which
        is exactly what ``A`` means. It is therefore a like-for-like readout, not
        a privileged one.
        """
        return self._zeros(i)

    def implied_S_projected(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Charitable readout: symmetric projection of the raw simultaneous term.

        Identical to ``implied_S`` for families whose simultaneous term is already
        symmetric. Reported alongside it so neither reading is hidden.
        """
        return 0.5 * (self.implied_S(i, j) + self.implied_S(j, i))

    def raw_sim_term(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Simultaneous pair term evaluated at the *given* index order.

        Diagnostic only. ``raw_sim_term(i,j) - raw_sim_term(j,i)`` measures how
        order-dependent a family's underlying pair network is. Note that the
        unconstrained families canonicalise indices before predicting, so this
        residual is not exposed in their actual predictions -- it measures the
        network, not the prediction.
        """
        return self._zeros(i)

    # ------------------------------------------------------------ latents
    def latent_single(self, i: torch.Tensor) -> torch.Tensor:
        return self.v(i)

    def latent_sim(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def latent_ordered(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------- observations
    def _obs(self, z: torch.Tensor) -> torch.Tensor:
        return z if self.head is None else self.head(z)

    def predict_single(self, i: torch.Tensor) -> torch.Tensor:
        return self._obs(self.latent_single(i))

    def predict_sim(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return self._obs(self.latent_sim(i, j))

    def predict_ordered(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return self._obs(self.latent_ordered(i, j))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_pair_params(self) -> int:
        """Parameters devoted specifically to the pair-interaction term."""
        return 0


class AdditiveModel(BaseModel):
    family = "additive"

    def latent_sim(self, i, j):
        return self.v(i) + self.v(j)

    def latent_ordered(self, i, j):
        return self.v(i) + self.v(j)


class _CanonicalPairMixin:
    """Feeds the physically order-free simultaneous pair in canonical index order."""

    def _canon(self, i, j):
        return torch.minimum(i, j), torch.maximum(i, j)


class UnconstrainedModel(_CanonicalPairMixin, BaseModel):
    """General pair function; no algebraic structure imposed.

    ``G(phi(e_i,e_j)) -> R^{2d}``: the first ``d`` outputs are the simultaneous
    pair term, the last ``d`` the ordered pair term. The two are *independent*
    functions -- this family has no reason to relate the simultaneous outcome of
    {i,j} to the ordered outcomes of i->j and j->i.
    """

    family = "unconstrained"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.pair_net = mlp(self.phi.out_dim, cfg.pair_hidden, 2 * cfg.out_dim,
                            cfg.n_hidden_layers)

    def _pair(self, i, j):
        return self.pair_net(self.phi(self.emb(i), self.emb(j)))

    def _pair_sim(self, i, j):
        lo, hi = self._canon(i, j)
        return self._pair(lo, hi)[:, : self.cfg.out_dim]

    def _pair_ord(self, i, j):
        return self._pair(i, j)[:, self.cfg.out_dim:]

    def raw_sim_term(self, i, j):
        return self._pair(i, j)[:, : self.cfg.out_dim]

    def latent_sim(self, i, j):
        return self.v(i) + self.v(j) + self._pair_sim(i, j)

    def latent_ordered(self, i, j):
        return self.v(i) + self.v(j) + self._pair_ord(i, j)

    def implied_S(self, i, j):
        # exactly the term used by latent_sim; already symmetric via canonicalisation
        return self._pair_sim(i, j)

    def implied_A(self, i, j):
        # (latent_ordered(i,j) - latent_ordered(j,i)) / 2
        return 0.5 * (self._pair_ord(i, j) - self._pair_ord(j, i))

    def n_pair_params(self) -> int:
        return sum(p.numel() for p in self.pair_net.parameters())


class SharedPairModel(_CanonicalPairMixin, BaseModel):
    """Ablation: shares the simultaneous term with the ordered rows, but leaves the
    ordered residual unconstrained (no antisymmetry).

        pred_sim(i,j) = v_i + v_j + G_sim(i,j)
        pred_ord(i,j) = v_i + v_j + G_sim(i,j) + G_ord(i,j)

    Sits exactly between ``UnconstrainedModel`` and ``AlgebraModel``: it gets the
    "ordered outcome = simultaneous outcome + a correction" structure for free.

    **This step is not a one-variable ablation, and the README must not be read
    as saying it is.** Four things differ between this family and
    ``AlgebraModel``:

    1. *Antisymmetry.* The ordered correction becomes
       ``A(i,j) = (F_A(phi_ij) - F_A(phi_ji))/2`` instead of an unconstrained
       ``G_ord(phi_ij)``.
    2. *A free symmetric ordered term is deleted.* ``G_ord`` here carries an
       arbitrary symmetric part ``(G_ord(i,j)+G_ord(j,i))/2``, so this family's
       effective symmetric term differs between the simultaneous row (``G_sim``)
       and the ordered rows (``G_sim + sym(G_ord)``). ``AlgebraModel`` forces
       them equal. That is the *symmetry-of-S* constraint, imposed in the same
       step.
    3. *Trunk topology.* This family runs one MLP with ``2d`` outputs, so its S
       and A terms share every hidden layer; ``AlgebraModel`` runs two
       independent ``d``-output MLPs.
    4. *Width.* ``pair_hidden`` 76 here against 48 there, chosen by
       :func:`match_pair_capacity` to hold the parameter count fixed.

    By contrast ``UnconstrainedModel -> SharedPairModel`` *is* clean: identical
    architecture and width, differing only in whether ``G_sim`` is added into the
    ordered prediction. So the ladder measures "sharing" cleanly and measures
    "the cross-row identity" as a bundle.
    """

    family = "shared_pair"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.pair_net = mlp(self.phi.out_dim, cfg.pair_hidden, 2 * cfg.out_dim,
                            cfg.n_hidden_layers)

    def _pair(self, i, j):
        return self.pair_net(self.phi(self.emb(i), self.emb(j)))

    def _sim_term(self, i, j):
        lo, hi = self._canon(i, j)
        return self._pair(lo, hi)[:, : self.cfg.out_dim]

    def _ord_residual(self, i, j):
        return self._pair(i, j)[:, self.cfg.out_dim:]

    def raw_sim_term(self, i, j):
        return self._pair(i, j)[:, : self.cfg.out_dim]

    def latent_sim(self, i, j):
        return self.v(i) + self.v(j) + self._sim_term(i, j)

    def latent_ordered(self, i, j):
        return self.v(i) + self.v(j) + self._sim_term(i, j) + self._ord_residual(i, j)

    def implied_S(self, i, j):
        return self._sim_term(i, j)

    def implied_A(self, i, j):
        return 0.5 * (self._ord_residual(i, j) - self._ord_residual(j, i))

    def n_pair_params(self) -> int:
        return sum(p.numel() for p in self.pair_net.parameters())


class AlgebraModel(BaseModel):
    """Intervention Algebra model: S symmetric and A antisymmetric by construction."""

    family = "algebra"

    def __init__(self, cfg: ModelConfig, use_A: bool = True):
        super().__init__(cfg)
        self.use_A = use_A
        self.F_S = mlp(self.phi.out_dim, cfg.pair_hidden, cfg.out_dim, cfg.n_hidden_layers)
        self.F_A = mlp(self.phi.out_dim, cfg.pair_hidden, cfg.out_dim, cfg.n_hidden_layers)

    def _f(self, net, i, j):
        return net(self.phi(self.emb(i), self.emb(j)))

    def implied_S(self, i, j):
        return 0.5 * (self._f(self.F_S, i, j) + self._f(self.F_S, j, i))

    def raw_sim_term(self, i, j):
        return self.implied_S(i, j)

    def implied_A(self, i, j):
        if not self.use_A:
            return self._zeros(i)
        return 0.5 * (self._f(self.F_A, i, j) - self._f(self.F_A, j, i))

    def latent_sim(self, i, j):
        return self.v(i) + self.v(j) + self.implied_S(i, j)

    def latent_ordered(self, i, j):
        return self.v(i) + self.v(j) + self.implied_S(i, j) + self.implied_A(i, j)

    def n_pair_params(self) -> int:
        return sum(p.numel() for p in self.F_S.parameters()) + sum(
            p.numel() for p in self.F_A.parameters()
        )


class WellSpecifiedModel(BaseModel):
    """Reference *ceiling*, not a competitor: the generator's own functional form.

    Learns a per-intervention factor ``u_i`` and the bilinear forms directly,

        v_i  = W u_i
        S_ij = u_i^T B_c u_j     with B_c constrained symmetric
        A_ij = u_i^T K_c u_j     with K_c constrained antisymmetric

    which is exactly how the benchmark is generated (up to the interaction
    gate).  Its purpose is to answer "is this coverage level identifiable *at
    all*?".  A coverage at which even this model fails is a coverage where a
    null result says nothing about the hypothesis, so it must not be read as
    evidence.  It is excluded from the headline comparison because it is handed
    the true functional form, which no honest baseline gets.
    """

    family = "wellspecified"

    def __init__(self, cfg: ModelConfig, rank: int | None = None):
        super().__init__(cfg)
        k = rank or cfg.emb_dim
        self.rank = k
        self.u = nn.Parameter(torch.randn(cfg.n_interventions, k) / k**0.5)
        self.W_v = nn.Parameter(torch.randn(k, cfg.out_dim) / k**0.5)
        self.B = nn.Parameter(torch.randn(cfg.out_dim, k, k) / k)
        self.K = nn.Parameter(torch.randn(cfg.out_dim, k, k) / k)
        # The benchmark gates interactions by sigmoid(beta * (<u_i,u_j> - tau)),
        # so a purely bilinear reference would itself be misspecified and would
        # understate the achievable ceiling.
        self.gate_beta = nn.Parameter(torch.tensor(4.0))
        self.gate_tau = nn.Parameter(torch.tensor(0.0))

    def _gate(self, i, j):
        dot = (self.u[i] * self.u[j]).sum(-1, keepdim=True)
        return torch.sigmoid(self.gate_beta * (dot - self.gate_tau))

    def v(self, idx):
        return self.u[idx] @ self.W_v

    def _bilinear(self, M, i, j):
        M = M.unsqueeze(0)                       # (1, d, k, k)
        ui = self.u[i][:, None, None, :]         # (B, 1, 1, k)
        uj = self.u[j][:, None, :, None]         # (B, 1, k, 1)
        return (ui @ M @ uj).reshape(len(i), self.cfg.out_dim)

    def implied_S(self, i, j):
        # gate is symmetric in (i, j), so S stays exactly symmetric
        return self._gate(i, j) * self._bilinear(
            0.5 * (self.B + self.B.transpose(-1, -2)), i, j)

    def implied_A(self, i, j):
        # symmetric gate times antisymmetric bilinear form stays antisymmetric
        return self._gate(i, j) * self._bilinear(
            0.5 * (self.K - self.K.transpose(-1, -2)), i, j)

    def raw_sim_term(self, i, j):
        return self.implied_S(i, j)

    def latent_sim(self, i, j):
        return self.v(i) + self.v(j) + self.implied_S(i, j)

    def latent_ordered(self, i, j):
        return self.v(i) + self.v(j) + self.implied_S(i, j) + self.implied_A(i, j)

    def n_pair_params(self) -> int:
        return self.B.numel() + self.K.numel()


Family = Literal["additive", "unconstrained", "shared_pair", "algebra",
                 "wellspecified"]

FAMILIES: dict[str, type[BaseModel]] = {
    "additive": AdditiveModel,
    "unconstrained": UnconstrainedModel,
    "shared_pair": SharedPairModel,
    "algebra": AlgebraModel,
    "wellspecified": WellSpecifiedModel,
}

#: Families that take part in the headline comparison. ``wellspecified`` is a
#: ceiling reference and is deliberately excluded.
COMPARISON_FAMILIES = ("additive", "unconstrained", "shared_pair", "algebra")


def build_model(family: Family, cfg: ModelConfig) -> BaseModel:
    try:
        return FAMILIES[family](cfg)
    except KeyError:
        raise ValueError(f"unknown family {family!r}") from None


def _mlp_params(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int) -> int:
    """Parameter count of :func:`mlp`, computed analytically.

    Done in closed form rather than by instantiating models: the capacity search
    evaluates this for a thousand candidate widths, and building (and discarding)
    a thousand torch modules per run is slow enough to destabilise a process pool.
    """
    if n_hidden_layers <= 0:                       # mlp() degenerates to one Linear
        return in_dim * out_dim + out_dim
    n = in_dim * hidden + hidden                                  # first layer
    n += (n_hidden_layers - 1) * (hidden * hidden + hidden)       # remaining hidden
    n += hidden * out_dim + out_dim                               # output layer
    return n


def _pair_param_count(family: Family, cfg: ModelConfig, hidden: int) -> int:
    phi_out = PairFeatures(cfg.emb_dim, cfg.pair_mode).out_dim
    if family == "additive":
        return 0
    if family == "algebra":   # two MLPs, d outputs each
        return 2 * _mlp_params(phi_out, hidden, cfg.out_dim, cfg.n_hidden_layers)
    # unconstrained / shared_pair: one MLP with 2d outputs
    return _mlp_params(phi_out, hidden, 2 * cfg.out_dim, cfg.n_hidden_layers)


def match_pair_capacity(target_family: Family, cfg: ModelConfig,
                        reference_family: Family = "algebra",
                        search: range | None = None) -> int:
    """Pick ``pair_hidden`` for ``target_family`` matching the reference's pair params.

    The algebra model runs two pair MLPs (F_S, F_A) while the unconstrained
    families run one wider MLP with 2d outputs; without this search the two would
    differ by ~2x in pair capacity and any comparison would be confounded.
    """
    if target_family == "additive":
        return cfg.pair_hidden
    ref = _pair_param_count(reference_family, cfg, cfg.pair_hidden)
    search = search or range(4, 1025)
    best, best_gap = cfg.pair_hidden, float("inf")
    for h in search:
        gap = abs(_pair_param_count(target_family, cfg, h) - ref)
        if gap < best_gap:
            best, best_gap = h, gap
    return best
