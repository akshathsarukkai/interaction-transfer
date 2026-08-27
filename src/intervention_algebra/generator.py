"""Synthetic intervention-algebra systems with fully known ground truth.

Latent generative model
-----------------------
Each intervention ``i`` has a *true* latent factor vector ``u_i in R^k`` (never
shown to any model).  From these we build:

  v_i    = W_v u_i                                    (first-order effect, R^d)
  S_ij   = g_ij * s_scale * [ u_i^T B_c u_j ]_{c<d}   (symmetric,     B_c = B_c^T)
  A_ij   = g_ij * a_scale * [ u_i^T K_c u_j ]_{c<d}   (antisymmetric, K_c = -K_c^T)

``B_c`` symmetric  =>  S_ij = S_ji exactly.
``K_c`` antisymmetric =>  A_ij = -A_ji exactly (and A_ii = 0).

``g_ij in {0,1}`` is a symmetric sparsity mask (the interaction topology).

Latent responses
----------------
  single      i          :  z = v_i
  simultaneous {i,j}     :  z = v_i + v_j + S_ij
  ordered     i -> j     :  z = v_i + v_j + S_ij + A_ij
  ordered     j -> i     :  z = v_i + v_j + S_ij - A_ij

Observation map
---------------
  y = obs(z) + eps,  eps ~ N(0, noise_std^2)

``obs`` is either the identity (default; latent recovery is then identifiable --
see docs/identifiability.md) or an elementwise ``tanh`` squashing (latent
recovery is *not* identifiable there; only observable prediction and, with
caveats, topology recovery are meaningful).

Why the interaction tensors are *low-rank functions of u*
---------------------------------------------------------
If ``S_ij`` were drawn i.i.d. per pair, then a pair held out entirely from
training would carry no learnable signal and *no* model could beat chance --
the benchmark would be vacuous.  Generating S and A as bilinear forms of a
shared per-intervention factor ``u_i`` is what makes compositional
generalization possible in principle.  The ``sparsity_mode="random"`` option
deliberately breaks the *topology* half of that structure and is used as an
identifiability control (see ``configs``/README).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np

Regime = Literal["independent", "symmetric", "antisymmetric", "both"]

# observation-row type codes
SINGLE = 0
SIM = 1
ORD = 2


@dataclass(frozen=True)
class SystemConfig:
    """Configuration of a synthetic intervention system."""

    n_interventions: int = 48
    latent_dim: int = 4          # d: dimension of the response
    n_factors: int = 6           # k: dimension of the hidden per-intervention factor
    regime: Regime = "both"
    sparsity: float = 0.25       # target fraction of pairs that truly interact
    sparsity_mode: Literal["latent", "module", "random"] = "latent"
    gate_beta: float = 8.0       # sharpness of the latent gate
    n_modules: int = 8           # used by the "module" topology
    noise_std: float = 0.05
    v_scale: float = 1.0
    s_scale: float = 0.6         # rms of S over *interacting* pairs
    a_scale: float = 0.6         # rms of A over *interacting* pairs
    observation_map: Literal["identity", "tanh"] = "identity"
    obs_gain: float = 1.0
    # Misspecification axis. See ``D`` on SyntheticSystem: the fraction (relative
    # to s_scale) of an extra *simultaneity defect* that appears only in the
    # simultaneous row, breaking the algebraic tie between the simultaneous and
    # the ordered observations. 0.0 = the algebra holds exactly.
    simultaneity_defect: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyntheticSystem:
    """A generated system: ground truth plus an observation table."""

    config: SystemConfig
    U: np.ndarray            # (N, k) hidden factors
    v: np.ndarray            # (N, d)
    S: np.ndarray            # (N, N, d) symmetric in (i, j)
    A: np.ndarray            # (N, N, d) antisymmetric in (i, j)
    mask: np.ndarray         # (N, N) bool, symmetric, zero diagonal: "does this pair interact"
    gate: np.ndarray         # (N, N) float in [0,1], symmetric: the soft interaction strength
    D: np.ndarray            # (N, N, d) symmetric: the simultaneity defect (see below)

    # ---------------------------------------------------------------- latent
    def latent_single(self, i: np.ndarray) -> np.ndarray:
        return self.v[i]

    def latent_sim(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Simultaneous response.

        With ``simultaneity_defect > 0`` this is *not* the symmetric part of the
        two ordered responses, which is precisely the tie the algebra model
        imposes.  Note that adding a term "neither symmetric nor antisymmetric"
        to the ordered rows would not be a misspecification at all: every pair
        function splits uniquely into symmetric and antisymmetric parts, so the
        algebra model can represent any ordered pair function.  Its real
        constraint is the cross-row identity
        ``z_sim = (z_ord(i,j) + z_ord(j,i)) / 2``, and ``D`` is what breaks it.
        """
        return self.v[i] + self.v[j] + self.S[i, j] + self.D[i, j]

    def latent_ordered(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return self.v[i] + self.v[j] + self.S[i, j] + self.A[i, j]

    # ----------------------------------------------------------- observation
    def observe(self, z: np.ndarray) -> np.ndarray:
        if self.config.observation_map == "identity":
            return z
        if self.config.observation_map == "tanh":
            return np.tanh(self.config.obs_gain * z)
        raise ValueError(f"unknown observation_map {self.config.observation_map!r}")

    @property
    def n(self) -> int:
        return self.config.n_interventions

    @property
    def d(self) -> int:
        return self.config.latent_dim

    def all_pairs(self) -> np.ndarray:
        """All unordered pairs (i, j) with i < j, shape (P, 2)."""
        iu = np.triu_indices(self.n, k=1)
        return np.stack(iu, axis=1)

    def interacts(self, pairs: np.ndarray) -> np.ndarray:
        return self.mask[pairs[:, 0], pairs[:, 1]]


def _bilinear(U: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(N,k),(d,k,k) -> (N,N,d) with out[i,j,c] = u_i^T M_c u_j."""
    return np.einsum("ia,cab,jb->ijc", U, M, U)


def _latent_gate(U: np.ndarray, sparsity: float, beta: float) -> np.ndarray:
    """Smooth symmetric gate driven by the hidden factors: g_ij = sigma(beta(<u_i,u_j> - tau)).

    This is the default topology.  It is (a) symmetric, (b) sparse -- ``tau`` is
    set to the ``1 - sparsity`` quantile so a controlled fraction of pairs is
    "on" -- and, crucially, (c) a *smooth function of the same latent factors
    that drive S and A*.  Property (c) is what makes the topology of a pair the
    model has never seen predictable in principle.  Without it (see
    ``_module_mask`` / ``_random_mask``) no model can generalize to a held-out
    pair, and the benchmark would be vacuous rather than hard.
    """
    n = len(U)
    g = U @ U.T
    g = g / (g.std() + 1e-12)
    iu = np.triu_indices(n, k=1)
    tau = float(np.quantile(g[iu], 1.0 - sparsity))
    gate = 1.0 / (1.0 + np.exp(-beta * (g - tau)))
    gate = 0.5 * (gate + gate.T)
    np.fill_diagonal(gate, 0.0)
    return gate


def _module_mask(rng: np.random.Generator, n: int, n_modules: int,
                 sparsity: float) -> np.ndarray:
    """Discrete topology: pairs interact iff their (latent-factor-independent)
    modules interact.  Learnable only by inferring a discrete per-intervention
    label from a node's other training pairs -- a much harder regime, kept as a
    secondary condition rather than the default.
    """
    modules = rng.permutation(np.arange(n) % n_modules)
    # Choose an *exact* number of interacting module-pairs rather than
    # thresholding random numbers: with only O(M^2) module pairs, thresholding
    # gives a very high seed-to-seed variance in the realised pair density.
    mod_pairs = [(a, b) for a in range(n_modules) for b in range(a, n_modules)]
    n_on = int(round(sparsity * len(mod_pairs)))
    chosen = rng.permutation(len(mod_pairs))[:n_on]
    mod_mask = np.zeros((n_modules, n_modules), dtype=bool)
    for c in chosen:
        a, b = mod_pairs[c]
        mod_mask[a, b] = mod_mask[b, a] = True
    mask = mod_mask[modules[:, None], modules[None, :]]
    mask = np.triu(mask, k=1)
    return mask | mask.T


def _random_mask(rng: np.random.Generator, n: int, sparsity: float) -> np.ndarray:
    """i.i.d. topology: unpredictable for a fully held-out pair.

    Used as an *identifiability control*: in this regime no model -- constrained
    or not -- should be able to recover the topology of held-out pairs, so any
    apparent success there would indicate leakage.
    """
    up = np.triu(rng.random((n, n)) < sparsity, k=1)
    return up | up.T


def make_system(config: SystemConfig) -> SyntheticSystem:
    rng = np.random.default_rng(config.seed)
    n, k, d = config.n_interventions, config.n_factors, config.latent_dim

    U = rng.normal(size=(n, k)) / np.sqrt(k)

    W_v = rng.normal(size=(k, d))
    v = config.v_scale * (U @ W_v)
    v = v / (v.std() + 1e-12) * config.v_scale

    B = rng.normal(size=(d, k, k))
    B = 0.5 * (B + B.transpose(0, 2, 1))          # symmetric
    K = rng.normal(size=(d, k, k))
    K = 0.5 * (K - K.transpose(0, 2, 1))          # antisymmetric
    D_raw = rng.normal(size=(d, k, k))            # simultaneity defect (symmetrised below)

    S_raw = _bilinear(U, B)
    A_raw = _bilinear(U, K)

    if config.sparsity_mode == "latent":
        gate = _latent_gate(U, config.sparsity, config.gate_beta)
        mask = gate > 0.5
    elif config.sparsity_mode == "module":
        mask = _module_mask(rng, n, config.n_modules, config.sparsity)
        gate = mask.astype(float)
    elif config.sparsity_mode == "random":
        mask = _random_mask(rng, n, config.sparsity)
        gate = mask.astype(float)
    else:
        raise ValueError(f"unknown sparsity_mode {config.sparsity_mode!r}")
    np.fill_diagonal(mask, False)
    np.fill_diagonal(gate, 0.0)

    m3 = gate[:, :, None]

    def _scale_to(x: np.ndarray, target: float) -> np.ndarray:
        """Rescale so the rms over *interacting* entries equals ``target``."""
        vals = x[mask]
        rms = float(np.sqrt((vals**2).mean())) if vals.size else 0.0
        if rms < 1e-12:
            return np.zeros_like(x)
        return x / rms * target

    S = _scale_to(S_raw, config.s_scale) * m3
    A = _scale_to(A_raw, config.a_scale) * m3

    if config.regime in ("independent", "antisymmetric"):
        S = np.zeros_like(S)
    if config.regime in ("independent", "symmetric"):
        A = np.zeros_like(A)
    if config.regime == "independent":
        mask = np.zeros_like(mask)
        gate = np.zeros_like(gate)

    # enforce the algebraic invariants exactly (guards against numerical drift)
    S = 0.5 * (S + S.transpose(1, 0, 2))
    A = 0.5 * (A - A.transpose(1, 0, 2))

    # Simultaneity defect: a *symmetric* term present only in the simultaneous
    # row. It is bilinear in u (so still learnable in principle by any family)
    # but it is not shared with the ordered rows, so the identity
    # z_sim == (z_ord(i,j) + z_ord(j,i)) / 2 no longer holds. This is the axis
    # along which the algebra model becomes misspecified while the
    # unconstrained family does not.
    if config.simultaneity_defect > 0 and config.regime != "independent":
        Draw = _bilinear(U, 0.5 * (D_raw + D_raw.transpose(0, 2, 1)))
        D = _scale_to(Draw, config.s_scale * config.simultaneity_defect) * m3
        D = 0.5 * (D + D.transpose(1, 0, 2))
    else:
        D = np.zeros_like(S)

    return SyntheticSystem(config=config, U=U, v=v, S=S, A=A, mask=mask,
                           gate=gate, D=D)


@dataclass
class ObservationTable:
    """Flat observation rows the models are trained/evaluated on.

    ``kind`` uses the SINGLE/SIM/ORD codes.  For SINGLE rows ``j`` is a copy of
    ``i`` and is never used.  For SIM rows the stored ``(i, j)`` is canonical
    (i < j); the *unordered* nature of a simultaneous intervention is a property
    of the physical system, not of the storage order.  For ORD rows ``(i, j)``
    means "i applied first, then j", and both orders appear as separate rows.
    """

    i: np.ndarray
    j: np.ndarray
    kind: np.ndarray
    y: np.ndarray                 # (R, d) observed outcome
    z: np.ndarray                 # (R, d) noiseless latent response (diagnostics only)
    pair_key: np.ndarray          # (R,) index into the canonical pair list, -1 for singles

    def __len__(self) -> int:
        return len(self.i)

    def subset(self, idx: np.ndarray) -> "ObservationTable":
        return ObservationTable(self.i[idx], self.j[idx], self.kind[idx],
                                self.y[idx], self.z[idx], self.pair_key[idx])


def pair_index_map(n: int) -> dict[tuple[int, int], int]:
    pairs = np.stack(np.triu_indices(n, k=1), axis=1)
    return {(int(a), int(b)): p for p, (a, b) in enumerate(pairs)}


def generate_observations(system: SyntheticSystem, pairs: np.ndarray,
                          include_singles: bool = True,
                          rng: np.random.Generator | None = None,
                          n_replicates: int = 1) -> ObservationTable:
    """Build the observation rows for a given set of unordered pairs.

    Every pair contributes three rows: simultaneous {i,j}, ordered i->j and
    ordered j->i.  Splits are performed at the level of *pairs*, so all three
    rows of a pair always travel together (see splits.py -- this is what
    prevents reverse-order leakage).
    """
    cfg = system.config
    rng = rng or np.random.default_rng(cfg.seed + 10_000)
    pmap = pair_index_map(system.n)

    I, J, KIND, Z, PK = [], [], [], [], []

    if include_singles:
        idx = np.arange(system.n)
        for _ in range(n_replicates):
            I.append(idx); J.append(idx)
            KIND.append(np.full(system.n, SINGLE))
            Z.append(system.latent_single(idx))
            PK.append(np.full(system.n, -1))

    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        keys = np.array([pmap[(int(x), int(y))] for x, y in pairs])
        for _ in range(n_replicates):
            # simultaneous
            I.append(a); J.append(b); KIND.append(np.full(len(a), SIM))
            Z.append(system.latent_sim(a, b)); PK.append(keys)
            # ordered a -> b
            I.append(a); J.append(b); KIND.append(np.full(len(a), ORD))
            Z.append(system.latent_ordered(a, b)); PK.append(keys)
            # ordered b -> a
            I.append(b); J.append(a); KIND.append(np.full(len(a), ORD))
            Z.append(system.latent_ordered(b, a)); PK.append(keys)

    if not I:
        # No singles requested and no pairs given. Returning an empty table rather
        # than raising lets a degenerate split (e.g. a coverage/seed combination
        # that leaves no eligible validation pairs) be handled downstream instead
        # of crashing the whole sweep.
        d = system.d
        return ObservationTable(
            i=np.zeros(0, dtype=int), j=np.zeros(0, dtype=int),
            kind=np.zeros(0, dtype=int), y=np.zeros((0, d)), z=np.zeros((0, d)),
            pair_key=np.zeros(0, dtype=int))

    i = np.concatenate(I); j = np.concatenate(J)
    kind = np.concatenate(KIND); z = np.concatenate(Z)
    pk = np.concatenate(PK)

    y = system.observe(z)
    if cfg.noise_std > 0:
        y = y + rng.normal(scale=cfg.noise_std, size=y.shape)
    return ObservationTable(i=i, j=j, kind=kind, y=y, z=z, pair_key=pk)
