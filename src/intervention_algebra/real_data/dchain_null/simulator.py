"""A screen in which the true pair-specific sequential interaction is zero.

What "zero" means here, precisely
---------------------------------
The d-chain likelihood (``dchain.cpp``) gives the expected log relative cell
count of a well in which drug ``a`` was applied first (always at 1 uM) and drug
``b`` second at concentration ``c``:

    E[log x_AB(a, b, c)] = log beta_a
                         + 1[lambda_a] * log f(1.0;  theta_a)
                         + ( 1[lambda_AB(a,b)] * log f(c; theta_AB(a,b))
                             if the combination has its own parameters,
                             else 1[lambda_b] * log f(c; theta_b) )

with the two single-agent experiment types

    E[log x_A(b, c)]  = 1[lambda_b] * log f(c; theta_b)
    E[log x_A0(a, c)] = log beta_a + 1[lambda_a] * log f(c; theta_a)

and ``f(c; K, h, alpha) = (1 - alpha)/(1 + (K c)^h) + alpha``.

**Every term above is indexed by one drug, never by a pair.** ``beta_a`` and
``theta_a`` are drug-specific, ``theta_b`` is drug-specific, and the protocol
asymmetry -- first drug pinned at 1 uM, second drug titrated -- is a property of
the position, not of the partner. So setting ``lambda_AB == 0`` everywhere gives
a world whose true ordered response is *exactly separable*: a per-first-drug
multiplicative factor times the second drug's own single-agent curve. This is
:data:`STRICT`.

Under :data:`STRICT` the true synergy measure is **identically zero for every
ordered pair** -- not small, not noisy, zero -- because the authors' own measure
collapses to ``lambda_AB(a,b) * (...)`` (see :mod:`.synergy`). The true
directional matrix is the zero matrix and its cyclic content is undefined
because there is nothing to decompose. Anything the estimator produces is
therefore 100% estimator-induced, which makes ``STRICT`` the **most favourable
possible world for the artifact hypothesis**: there is no true signal for a
positive result to be attributed to.

The realism variant, and why it can only weaken the artifact
------------------------------------------------------------
The published model *does* carry combination-level nuisance parameters, and a
null in which they are switched off everywhere is a world the estimator never
sees. :data:`NUISANCE` therefore draws

    lambda_AB(a, b) ~ Bernoulli(p)          i.i.d. over ordered pairs
    theta_AB(a, b)  =  theta_b  perturbed by i.i.d. noise

so combinations *do* deviate from the second drug's own curve. The perturbation
for pair ``(a, b)`` is drawn from one fixed distribution, independently of ``a``,
of ``b``, and of every other pair. There is consequently **no reusable latent
pair structure**: the true synergy of a held-out pair is, conditional on the
per-drug parameters, statistically independent of every observed pair, so its
value is unpredictable in principle and no model can achieve positive held-out
skill on it.

Both are run, at the same size. ``STRICT`` is the primary because it is the arm
where **attribution** is unambiguous: with a true directional matrix of exactly
zero, anything the estimator produces is 100% estimator-induced.

It is *not* the arm where the artifact is largest, and an earlier version of this
docstring claimed otherwise. The argument -- that extra true pair noise only
enlarges the skill denominator -- ignores the multiplicative selector. The
measure is ``lambda_AB * (...)``, and ``NUISANCE``'s combination curves give the
estimator a reason to open the gate: measured on the same truth and settings, the
posterior selector sits at 0.21-0.23 under ``STRICT`` and 0.33 under
``NUISANCE``, and the artifact is correspondingly about twice as large. The real
deposit runs at a gate of 0.4916 (A375) / 0.4635 (PANC1), above both.

There is no way to open the gate without adding a true effect, and that is a
property of the world rather than a defect to engineer around: under exact
separability the combination data gives the estimator no reason to prefer a
private curve. A negative result under ``STRICT`` therefore says the artifact is
at most about twice that size at the estimator's real operating point, which is
what the report states.

What is forbidden
-----------------
No term whose true value depends jointly on the identity of ``{a, b}`` beyond
independent noise. In particular no ``h_ij``, no ``u_i^T K u_j``, no
pair-specific lambda drawn from anything that couples pairs, and nothing whose
parameters were chosen by looking at the real screen's residual directional
matrix. ``test_the_null_contains_no_reusable_pair_structure`` and
``test_no_real_pair_residual_information_reaches_the_simulator`` enforce this.

Parameter provenance
--------------------
Every constant below carries a provenance class, emitted into the run config so
a reader can check it without reading this file:

``source``      taken from the d-chain source or the paper
``replicate``   estimated from the authors' deposited *raw viability replicates*
                (``d-chain/data/viability_data.csv``) -- a single-well quantity,
                never a pair quantity
``prior``       a generic plausible choice, covered by a preregistered sweep
``design``      a fixed engineering choice with no scientific content
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .synergy import SYNERGY_CONC, response, synergy_index

#: The single-agent titration in the deposited example data, in uM.
#: Provenance: ``source`` (``d-chain/data/viability_data.csv``).
A_CONCENTRATIONS: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: The second drug's titration in a combination well, in uM. Provenance:
#: ``source``. Note it spans the synergy integration grid ``[0.01, 10]`` exactly.
AB_CONCENTRATIONS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

#: The first drug's concentration. Provenance: ``source`` -- ``dchain.cpp``
#: hard-codes ``logResponse(1.0, theta[a])`` for the pretreatment factor in
#: every AB well, so this is not a choice.
FIRST_CONCENTRATION: float = 1.0

#: Replicates per condition. Provenance: ``source`` -- every condition in the
#: deposited example data is a triplicate.
N_REPLICATES: int = 3

#: Log-scale observation SD. Provenance: ``replicate`` and ``source``, which
#: agree: the deposited triplicates give a pooled within-condition log SD of
#: 0.182 and a median of 0.127, and ``dchain.cpp``'s variance prior
#: ``Gamma(a=0.6, b=0.02)`` -- commented "worst case from statistics covering
#: PANC1 and A375" -- implies 0.112 read as the inverse-gamma mode and 0.183
#: read as the mean precision. 0.15 sits inside both. Swept over
#: :data:`NOISE_GRID`.
SIGMA_OBS: float = 0.15

#: The preregistered noise sweep, spanning half and double the central value.
NOISE_GRID: tuple[float, ...] = (0.075, 0.15, 0.30)

#: Prior hyperparameters, read off ``dchain.cpp``'s ``main()``. The LogNormal
#: quotient there is ``exp((1/(2*sd)) * [...])`` against a density
#: ``exp(-(log x - log mu)^2 / (2 sigma^2))``, so the struct's ``sd`` field is a
#: **variance** and the log-scale SD is its square root. Provenance: ``source``.
K_PRIOR_MEDIAN, K_PRIOR_LOGSD = 0.1, float(np.sqrt(2.0))
H_PRIOR_MEDIAN, H_PRIOR_LOGSD = 1.5, float(np.sqrt(0.5))
ALPHA_PRIOR_A, ALPHA_PRIOR_B = 1.0, 3.0
BETA_PRIOR_MEDIAN, BETA_PRIOR_LOGSD = 1.0, float(np.sqrt(0.05))

#: Probability that an ordered combination carries its own parameters, in the
#: ``NUISANCE`` variant. Provenance: ``source``. ``dchain.cpp`` defines a
#: ``BernPrior`` struct and a ``qprior(bool, bool, BernPrior)`` overload but
#: **never calls it** -- neither ``lambda`` nor ``lambda_AB`` gets a prior
#: correction in its acceptance ratio -- so the model's implicit prior on the
#: selector is Bernoulli(0.5). Using anything else would be a choice; this is
#: not.
P_LAMBDA_AB: float = 0.5

#: Spread of the combination-specific curve around the second drug's own curve,
#: in the ``NUISANCE`` variant. Provenance: ``prior``. Its only effect is to
#: enlarge the *unpredictable* part of the true synergy, which dilutes any
#: artifact; see the module docstring.
THETA_AB_LOGK_SD: float = 0.5
THETA_AB_H_SD: float = 0.2
THETA_AB_ALPHA_SD: float = 0.1

STRICT = "strict"
NUISANCE = "nuisance"
VARIANTS = (STRICT, NUISANCE)


@dataclass(frozen=True)
class NullConfig:
    """Everything that defines one simulated screen. Serialised with every run."""

    #: ``STRICT`` (lambda_AB == 0 everywhere, true synergy identically zero) or
    #: ``NUISANCE`` (independent combination nuisance, still no reusable
    #: pair structure).
    variant: str = STRICT
    n_drugs: int = 100
    #: Seed for the *generative* process. Kept separate from the estimator seed
    #: so "the data changed" and "the sampler changed" are never confounded.
    sim_seed: int = 0
    sigma_obs: float = SIGMA_OBS
    n_replicates: int = N_REPLICATES
    a_concentrations: tuple[float, ...] = A_CONCENTRATIONS
    ab_concentrations: tuple[float, ...] = AB_CONCENTRATIONS
    first_concentration: float = FIRST_CONCENTRATION
    p_lambda_ab: float = P_LAMBDA_AB
    theta_ab_logk_sd: float = THETA_AB_LOGK_SD
    theta_ab_h_sd: float = THETA_AB_H_SD
    theta_ab_alpha_sd: float = THETA_AB_ALPHA_SD
    #: Fraction of drugs with a real single-agent effect. 1.0: every drug has a
    #: curve. Provenance ``design`` -- an inert drug is a degenerate case the
    #: estimator handles by setting ``lambda_b = 0``, and including some would
    #: only add uninformative rows.
    p_lambda_single: float = 1.0
    #: Label the simulated screen is given so Phase 2R's per-screen constants
    #: (only the exploratory sign-accuracy threshold) resolve. See
    #: ``adapter.as_screen``.
    screen_label: str = "A375"
    tag: str = "main"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.n_drugs < 2:
            raise ValueError("need at least two drugs")


@dataclass(frozen=True)
class Truth:
    """The true state of the simulated world. **Never overwritten by estimates.**

    ``synergy`` is the true value of the authors' own measure evaluated at these
    parameters -- the oracle target of Null 0.
    """

    theta: np.ndarray = field(repr=False)        # (n, 3) K, h, alpha
    lam: np.ndarray = field(repr=False)          # (n,)
    beta: np.ndarray = field(repr=False)         # (n,)
    theta_AB: np.ndarray = field(repr=False)     # (n, n, 3) [first, second]
    lam_AB: np.ndarray = field(repr=False)       # (n, n)   [first, second]
    synergy: np.ndarray = field(repr=False)      # (n, n)   [first, second]
    config: NullConfig = None                    # type: ignore[assignment]

    @property
    def n_drugs(self) -> int:
        return self.theta.shape[0]

    def directional(self) -> np.ndarray:
        """``D(a,b) = synergy(a->b) - synergy(b->a)``, diagonal zeroed."""
        d = self.synergy - self.synergy.T
        np.fill_diagonal(d, 0.0)
        return d


def simulate_truth(cfg: NullConfig) -> Truth:
    """Draw the true world. Depends on ``cfg.sim_seed`` and nothing else."""
    rng = np.random.default_rng(cfg.sim_seed)
    n = cfg.n_drugs

    K = np.exp(rng.normal(np.log(K_PRIOR_MEDIAN), K_PRIOR_LOGSD, n))
    h = np.exp(rng.normal(np.log(H_PRIOR_MEDIAN), H_PRIOR_LOGSD, n))
    alpha = rng.beta(ALPHA_PRIOR_A, ALPHA_PRIOR_B, n)
    theta = np.stack([K, h, alpha], axis=1)
    beta = np.exp(rng.normal(np.log(BETA_PRIOR_MEDIAN), BETA_PRIOR_LOGSD, n))
    lam = (rng.random(n) < cfg.p_lambda_single).astype(float)

    if cfg.variant == STRICT:
        # No combination anywhere carries its own parameters. theta_AB is stored
        # at theta_b so the array is well-formed, but lambda_AB == 0 makes it
        # unreachable in both the likelihood and the synergy measure.
        lam_AB = np.zeros((n, n))
        theta_AB = np.broadcast_to(theta[None, :, :], (n, n, 3)).copy()
    else:
        lam_AB = (rng.random((n, n)) < cfg.p_lambda_ab).astype(float)
        # theta_b perturbed by noise drawn independently for each ordered pair.
        # The perturbation's distribution does not depend on a, on b, or on any
        # other pair -- that is what makes the true pair effect unpredictable.
        base = np.broadcast_to(theta[None, :, :], (n, n, 3)).copy()
        theta_AB = np.empty_like(base)
        theta_AB[..., 0] = base[..., 0] * np.exp(
            rng.normal(0.0, cfg.theta_ab_logk_sd, (n, n)))
        theta_AB[..., 1] = np.clip(
            base[..., 1] + rng.normal(0.0, cfg.theta_ab_h_sd, (n, n)), 1e-3, None)
        theta_AB[..., 2] = np.clip(
            base[..., 2] + rng.normal(0.0, cfg.theta_ab_alpha_sd, (n, n)), 0.0, 1.0)

    synergy = synergy_index(theta, lam, theta_AB, lam_AB, SYNERGY_CONC)
    return Truth(theta=theta, lam=lam, beta=beta, theta_AB=theta_AB,
                 lam_AB=lam_AB, synergy=synergy, config=cfg)


def _log_f(conc, par) -> np.ndarray:
    return np.log(response(np.atleast_1d(conc), par))


def true_log_means(cfg: NullConfig, truth: Truth) -> dict[str, np.ndarray]:
    """Noise-free expected log relative counts, by experiment type.

    ``A``  ``(n, n_a_conc)``            single agent alone
    ``A0`` ``(n,)``                     pretreatment residual at 1 uM
    ``AB`` ``(n, n, n_ab_conc)``        ``[first, second, conc]``
    """
    n = cfg.n_drugs
    a_conc = np.asarray(cfg.a_concentrations, float)
    ab_conc = np.asarray(cfg.ab_concentrations, float)

    mA = truth.lam[:, None] * _log_f(a_conc, truth.theta)
    mA0 = (np.log(truth.beta)
           + truth.lam * _log_f(cfg.first_concentration, truth.theta)[:, 0])

    first = (np.log(truth.beta)
             + truth.lam * _log_f(cfg.first_concentration, truth.theta)[:, 0])
    # second interval: the combination's own curve where the selector is on,
    # the second drug's own curve where it is off.
    own = _log_f(ab_conc, truth.theta_AB)                       # (n, n, C)
    fallback = np.broadcast_to(
        (truth.lam[:, None] * _log_f(ab_conc, truth.theta))[None, :, :],
        (n, n, len(ab_conc)))
    sel = truth.lam_AB[..., None]
    mAB = first[:, None, None] + sel * own + (1.0 - sel) * fallback
    return {"A": mA, "A0": mA0, "AB": mAB}


def simulate_wells(cfg: NullConfig, truth: Truth,
                   obs_seed: int | None = None) -> pd.DataFrame:
    """Well-level data in the exact CSV schema ``dchain.cpp`` parses.

    Columns and order are fixed by ``dchain.cpp``'s ``FIELD_*`` macros:
    ``Experiment, CellLine, Run, Plate, Pretreatment, Compound, Concentration,
    RelCount``. ``Run`` and ``Plate`` are parsed but unused -- the plate and run
    offsets are commented out of ``calcSufficientStat`` -- so they are constant.

    Noise is i.i.d. log-normal with SD ``cfg.sigma_obs`` on every well. It is
    independent across wells, conditions, drugs and pairs: a *correlated* noise
    model would be a way of smuggling pair structure in through the back door.
    """
    rng = np.random.default_rng(
        cfg.sim_seed if obs_seed is None else obs_seed)
    n = cfg.n_drugs
    m = true_log_means(cfg, truth)
    names = [f"Drug{k:03d}" for k in range(n)]
    R = cfg.n_replicates
    frames = []

    def block(exp, pre, comp, conc, logmean):
        k = len(comp)
        noise = rng.normal(0.0, cfg.sigma_obs, (k, R))
        rel = np.exp(np.asarray(logmean)[:, None] + noise)
        return pd.DataFrame({
            "Experiment": exp,
            "CellLine": "SIM",
            "Run": 1, "Plate": 1,
            "Pretreatment": np.repeat(pre, R),
            "Compound": np.repeat(comp, R),
            "Concentration": np.repeat(conc, R),
            "RelCount": rel.reshape(-1),
        })

    # A: every drug at every single-agent concentration.
    a_conc = np.asarray(cfg.a_concentrations, float)
    idx_d, idx_c = np.meshgrid(np.arange(n), np.arange(len(a_conc)), indexing="ij")
    frames.append(block("A", np.array(["NA"] * idx_d.size),
                        np.array(names)[idx_d.ravel()],
                        a_conc[idx_c.ravel()],
                        m["A"].reshape(-1)))

    # A0: every drug as a pretreatment, measured at the combination endpoint.
    frames.append(block("A0", np.array(["NA"] * n), np.array(names),
                        np.full(n, cfg.first_concentration), m["A0"]))

    # AB: every ordered pair, including a -> a, which the deposited tables also
    # carry. Phase 2R drops the diagonal; the estimator is given it because the
    # real one was.
    ab_conc = np.asarray(cfg.ab_concentrations, float)
    ia, ib, ic = np.meshgrid(np.arange(n), np.arange(n), np.arange(len(ab_conc)),
                             indexing="ij")
    frames.append(block("AB",
                        np.array(names)[ia.ravel()],
                        np.array(names)[ib.ravel()],
                        ab_conc[ic.ravel()],
                        m["AB"].reshape(-1)))

    out = pd.concat(frames, ignore_index=True)
    return out[["Experiment", "CellLine", "Run", "Plate", "Pretreatment",
                "Compound", "Concentration", "RelCount"]]


def config_provenance(cfg: NullConfig) -> dict:
    """Machine-readable parameter provenance, written next to every run.

    Section 14 of the brief: every simulator parameter must state where it came
    from. The classes are defined in the module docstring. Nothing here is
    derived from the real screen's *pair* quantities, which is separately
    asserted by ``test_no_real_pair_residual_information_reaches_the_simulator``.
    """
    return {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "provenance": {
            "n_drugs": "source: 100 drugs in the deposited A375/PANC1 tables",
            "a_concentrations": "source: d-chain/data/viability_data.csv",
            "ab_concentrations": "source: d-chain/data/viability_data.csv",
            "first_concentration": "source: dchain.cpp hard-codes logResponse(1.0, theta[a])",
            "n_replicates": "source: every condition in the deposited example is a triplicate",
            "sigma_obs": ("replicate+source: deposited triplicates give pooled log SD "
                          "0.182 / median 0.127; dchain.cpp's Gamma(0.6, 0.02) variance "
                          "prior implies 0.112-0.183. Swept over NOISE_GRID."),
            "K/h/alpha/beta priors": "source: dchain.cpp main(), K_prior, h_prior, alpha_prior, beta_prior",
            "p_lambda_ab": ("source: dchain.cpp defines BernPrior but never applies it to "
                            "lambda or lambda_AB, so the implicit selector prior is Bernoulli(0.5)"),
            "theta_ab_*_sd": ("prior: sets the size of the *unpredictable* true pair effect in "
                              "the NUISANCE variant only; larger values dilute any artifact"),
            "p_lambda_single": "design: every drug has a real single-agent curve",
            "REAL PAIR DATA USED": "none",
        },
    }
