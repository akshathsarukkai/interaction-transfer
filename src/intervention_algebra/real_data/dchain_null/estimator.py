"""Three estimators of the same target, at three points on a fidelity ladder.

All three end at the *same* quantity -- the authors' own ``synergy_measure``,
computed by :mod:`.synergy` -- and differ only in how the parameters that go
into it are obtained. That is the whole design: the target is held fixed so the
comparison isolates estimation.

``ORACLE``    Null 0. The measure evaluated at the **true** parameters, with no
              estimation at all. Under ``STRICT`` this is identically zero by
              construction. Its purpose is to prove the generative null does not
              secretly contain the structure being hunted; if the oracle matrix
              shows cyclic structure or held-out skill, the null is malformed and
              nothing downstream means anything.

``UNSHARED``  Null 1, and Control C. Every ordered pair is fitted against its
              **own, independently drawn** single-agent and residual
              measurements, so the error in drug ``b``'s curve is a different
              draw in every combination ``b`` appears in. Because the extra
              observations are *simulated*, the sharing is removed exactly rather
              than approximately.

              **What it is and is not.** An earlier version of this docstring
              said the only thing that changes is whether a per-drug error is
              reused. That is not true and an adversarial reviewer was right to
              say so: the joint arm is an MCMC posterior mean over 1,999 samples
              with a fractional selector, this one is a MAP point estimate with a
              hard 0/1 gate, and its estimation-error scale is about 3.3x larger.
              More to the point, its held-out skill is ``<= 0`` **by
              construction** -- every per-drug quantity is redrawn per pair, so
              the score is independent across pairs and nothing can predict it.

              So it is a pipeline sanity check, and a demonstration that
              per-pair-independent estimation error produces **no spectral
              concentration** (top-2 curl energy 0.082 against an i.i.d. floor of
              0.0747, and held-out rank-2 skill with a maximum of +0.0008 over 20
              screens). It is not a clean one-factor isolation of sharing, and it
              is not reported as one.

``JOINT``     Nulls 2 and 3, which collapse into one here because the published
              sampler is runnable: ``dchain.cpp`` itself, compiled from the
              pinned commit and executed on the simulated wells. See
              :mod:`.dchain`.

Why the fitter below is a batched Gauss-Newton and not the MCMC
---------------------------------------------------------------
``UNSHARED`` needs ~30,000 three-parameter curve fits per screen, and its job is
to answer one question -- does removing the *sharing* remove the structure --
holding the estimator's other properties fixed. Running the MCMC 10,000 times
would answer the same question at 10,000x the cost. The fitter is a MAP estimate
under ``dchain.cpp``'s own priors, on the same log scale, with the same
identifiability structure; it is not the posterior and is never described as
"the d-chain model". ``JOINT`` is the one that is.
"""

from __future__ import annotations

import numpy as np

from .simulator import (ALPHA_PRIOR_A, ALPHA_PRIOR_B, BETA_PRIOR_LOGSD,
                        BETA_PRIOR_MEDIAN, H_PRIOR_LOGSD, H_PRIOR_MEDIAN,
                        K_PRIOR_LOGSD, K_PRIOR_MEDIAN, NullConfig, Truth,
                        true_log_means)
from .synergy import SYNERGY_CONC, response, synergy_index

ORACLE, UNSHARED, JOINT = "oracle", "unshared", "joint"

#: Unconstrained coordinates the fitter works in, so the box constraints
#: ``K > 0``, ``h > 0``, ``0 < alpha < 1`` hold automatically:
#: ``(log K, log h, logit alpha)``.
_EPS = 1e-9


def _to_natural(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    out[..., 0] = np.exp(z[..., 0])
    out[..., 1] = np.exp(z[..., 1])
    out[..., 2] = 1.0 / (1.0 + np.exp(-z[..., 2]))
    return out


def _to_unconstrained(theta: np.ndarray) -> np.ndarray:
    out = np.empty_like(theta)
    out[..., 0] = np.log(np.maximum(theta[..., 0], _EPS))
    out[..., 1] = np.log(np.maximum(theta[..., 1], _EPS))
    a = np.clip(theta[..., 2], 1e-6, 1 - 1e-6)
    out[..., 2] = np.log(a / (1 - a))
    return out


def _log_response(conc: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(response(conc, _to_natural(z)), _EPS))


def _neg_log_post(z, conc, y, offset, sigma):
    """Gaussian log-likelihood on log viability plus dchain.cpp's own priors.

    ``offset`` is a per-item additive constant on the log scale -- the
    first-position factor ``log beta_a + log f(1; theta_a)`` when fitting a
    combination curve, zero otherwise.
    """
    th = _to_natural(z)
    resid = (_log_response(conc, z) + offset[..., None]) - y
    nll = 0.5 * (resid ** 2).sum(axis=-1) / sigma ** 2
    # LogNormal priors as dchain.cpp writes them: the struct's `sd` field is
    # used as a variance (its quotient divides by 2*sd, not 2*sd^2), so the
    # log-scale SD is its square root, which is what K_PRIOR_LOGSD holds.
    nll = nll + (np.log(th[..., 0]) - np.log(K_PRIOR_MEDIAN)) ** 2 / (2 * K_PRIOR_LOGSD ** 2)
    nll = nll + (np.log(th[..., 1]) - np.log(H_PRIOR_MEDIAN)) ** 2 / (2 * H_PRIOR_LOGSD ** 2)
    a = np.clip(th[..., 2], 1e-6, 1 - 1e-6)
    nll = nll - ((ALPHA_PRIOR_A - 1) * np.log(a) + (ALPHA_PRIOR_B - 1) * np.log(1 - a))
    return nll


def fit_curves(conc: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
               sigma: float = 0.15, n_iter: int = 120,
               seed: int = 0) -> np.ndarray:
    """Batched MAP fit of ``(K, h, alpha)`` to log-viability observations.

    ``conc`` ``(C,)``; ``y`` ``(B, C)`` log relative counts (replicate means are
    what the sampler uses too, via its sufficient statistics); ``offset`` ``(B,)``
    is the additive log-scale constant -- the first-position factor when fitting
    a combination curve, zero otherwise.

    Adaptive coordinate descent in the unconstrained parameterisation, with a
    per-item step size that halves whenever an item's sweep found no
    improvement, run from three initialisations. Coordinate descent rather than
    Gauss-Newton because the Hill surface has a flat direction whenever ``K``
    falls outside the measured concentration range -- which happens for a real
    fraction of drugs -- and a Jacobian solve there is ill-conditioned while a
    line search simply stops moving. Fully deterministic.
    """
    conc = np.asarray(conc, float)
    y = np.atleast_2d(np.asarray(y, float))
    B = y.shape[0]
    offset = np.zeros(B) if offset is None else np.asarray(offset, float)

    #: Three starts spanning potent / typical / weak, in (log K, log h, logit a).
    starts = (np.array([np.log(1.0), np.log(1.5), -3.0]),
              np.array([np.log(0.1), np.log(1.5), -1.0]),
              np.array([np.log(0.003), np.log(1.0), 0.0]))
    step0 = np.array([0.6, 0.4, 0.8])

    best_z = np.empty((B, 3))
    best_f = np.full(B, np.inf)
    for s0 in starts:
        z = np.broadcast_to(s0, (B, 3)).copy()
        step = np.broadcast_to(step0, (B, 3)).copy()
        f = _neg_log_post(z, conc, y, offset, sigma)
        for _ in range(n_iter):
            moved = np.zeros(B, dtype=bool)
            for k in range(3):
                for sign in (1.0, -1.0):
                    trial = z.copy()
                    trial[:, k] += sign * step[:, k]
                    ft = _neg_log_post(trial, conc, y, offset, sigma)
                    take = ft < f
                    z[take] = trial[take]
                    f[take] = ft[take]
                    moved |= take
            step = np.where(moved[:, None], step, step * 0.5)
            if step.max() < 1e-5:
                break
        take = f < best_f
        best_z[take] = z[take]
        best_f[take] = f[take]
    _ = seed          # kept for signature stability; the fit is deterministic
    return _to_natural(best_z)


def estimate_oracle(cfg: NullConfig, truth: Truth) -> dict:
    """Null 0. The measure at the true parameters -- no estimation anywhere."""
    S = synergy_index(truth.theta, truth.lam, truth.theta_AB, truth.lam_AB,
                      SYNERGY_CONC)
    return {"synergy": S, "lambda_ab": truth.lam_AB.copy(),
            "estimator": ORACLE, "diagnostics": {"n_fits": 0}}


def _replicate_means(logmean: np.ndarray, rng: np.random.Generator,
                     sigma: float, n_rep: int) -> np.ndarray:
    """Mean of ``n_rep`` log-normal replicates: adds noise of SD sigma/sqrt(n)."""
    return logmean + rng.normal(0.0, sigma / np.sqrt(n_rep), logmean.shape)


def estimate_unshared(cfg: NullConfig, truth: Truth, est_seed: int = 0) -> dict:
    """Null 1 / Control C. The same fit, with per-drug error made unshareable.

    Each ordered pair ``(a, b)`` is handed its own freshly simulated single-agent
    curve for ``b``, its own residual measurement for ``a``, and its own
    combination wells -- drawn from the same truth, at the same noise level, with
    the same replicate count as the joint fit sees. Every per-drug estimate is
    therefore an independent draw in every pair the drug appears in, and no error
    can be reused across a row or a column.

    If the joint estimator manufactures low-rank cyclic structure and this one
    does not, the difference is caused by sharing and by nothing else.
    """
    rng = np.random.default_rng(1_000_003 + est_seed)
    n = cfg.n_drugs
    a_conc = np.asarray(cfg.a_concentrations, float)
    ab_conc = np.asarray(cfg.ab_concentrations, float)
    m = true_log_means(cfg, truth)
    sig, R = cfg.sigma_obs, cfg.n_replicates

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ia, ib = ii.ravel(), jj.ravel()                       # ordered pairs, [first, second]

    # (1) the SECOND drug's single-agent curve, drawn afresh for every pair.
    yb = _replicate_means(m["A"][ib], rng, sig, R)
    theta_b = fit_curves(a_conc, yb, sigma=sig / np.sqrt(R), seed=est_seed)

    # (2) the FIRST drug's position offset u_a = log beta_a + log f(1; theta_a),
    #     drawn afresh for every pair. Its two components are structurally
    #     non-identifiable from these data -- both enter only through their sum,
    #     which is precisely why the joint fit's error in it is a single shared
    #     scalar per drug. Estimating the sum directly is exact here.
    u_a = _replicate_means(m["A0"][ia], rng, sig, R)

    # (3) the combination curve, from this pair's own wells given that offset.
    yab = _replicate_means(m["AB"][ia, ib], rng, sig, R)
    theta_ab = fit_curves(ab_conc, yab, offset=u_a, sigma=sig / np.sqrt(R),
                          seed=est_seed + 1)

    # (4) the selector, by the same comparison the sampler's Bayes factor makes:
    #     does this pair's data prefer its own curve to the second drug's?
    f_own = _neg_log_post(_to_unconstrained(theta_ab), ab_conc, yab, u_a,
                          sig / np.sqrt(R))
    f_fallback = _neg_log_post(_to_unconstrained(theta_b), ab_conc, yab, u_a,
                               sig / np.sqrt(R))
    # BIC-style penalty for the three extra parameters, at the pair's own n.
    penalty = 0.5 * 3 * np.log(len(ab_conc) * R)
    lam_ab = ((f_own + penalty) < f_fallback).astype(float).reshape(n, n)

    base = response(SYNERGY_CONC, theta_b).mean(axis=-1).reshape(n, n)
    comb = response(SYNERGY_CONC, theta_ab).mean(axis=-1).reshape(n, n)
    S = lam_ab * (base - comb)
    return {"synergy": S, "lambda_ab": lam_ab, "estimator": UNSHARED,
            "diagnostics": {"n_fits": int(2 * n * n),
                            "selector_on_fraction": float(lam_ab.mean())}}


def second_position_gain(theta: np.ndarray, lam: np.ndarray,
                         fit_conc: np.ndarray, delta: float = 0.10,
                         sigma: float = 0.05) -> np.ndarray:
    """``mtilde_j`` -- how much of a uniform log offset comes back out as synergy.

    The second factor of the artifact the reconstruction predicts
    (``docs/dchain_reconstruction.md`` §3.3). It answers: if the combination data
    for drug ``j`` is uniformly shifted by ``-delta`` on the log scale -- which is
    exactly what an error in the *first* drug's shared offset does, since the
    likelihood constrains only ``u_i + log f(c; theta_ij)`` -- how much of that
    shift reappears as apparent synergy?

    Computed by **refitting**, not by linearising. The first-order expression
    ``g_j' (J_j' J_j)^-1 J_j' 1`` is what the algebra gives, but it is badly
    ill-conditioned for the many drugs whose curve is nearly flat in one
    parameter over the four measured doses -- it returns values in the thousands
    where the quantity is bounded in roughly [0, 1]. Pushing the offset through
    the same fitter the estimator uses is both bounded and more faithful: it is
    the absorption that actually happens.

    A **central** difference, because a one-sided one at the step sizes the
    fitter can resolve is dominated by its own convergence error: one-sided at
    ``delta = 0.05`` puts a quarter of the panel outside the theoretical [0, 1]
    band, central at ``delta = 0.10`` puts none of it there.

    What is actually observed on a 100-drug panel drawn from the model's own
    priors, and what the tests pin: ``mtilde`` lands inside [0, 1] for every
    drug, with mean ~0.7 and **sd ~0.2**. The heterogeneity is the part that
    matters -- the artifact is the wedge ``eps ^ mtilde``, and a constant
    ``mtilde`` would make it additively separable and its curl exactly zero.

    The idealised limits in the reconstruction (inert -> 1, fully potent -> 0)
    are *directionally* right but do not hold sharply on a real panel, because
    "inert" there means flat over the scoring grid [0.01, 10] while the
    absorption happens at the four measured doses, and because the residual
    floor ``alpha`` matters more than the mean viability does. Reported rather
    than asserted.
    """
    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)
    fit_conc = np.asarray(fit_conc, dtype=float)
    y = np.log(np.maximum(response(fit_conc, theta), 1e-12))
    up = fit_curves(fit_conc, y - float(delta), sigma=sigma)
    down = fit_curves(fit_conc, y + float(delta), sigma=sigma)

    def meanv(th):
        base = (1.0 - lam)[:, None] + lam[:, None] * response(SYNERGY_CONC, th)
        return base.mean(axis=-1)

    return (meanv(down) - meanv(up)) / (2.0 * float(delta))


def artifact_template(eps: np.ndarray, mtilde: np.ndarray) -> np.ndarray:
    """The predicted artifact, fully specified: the wedge ``eps ^ mtilde``.

    ``T[i,j] = eps_i * mtilde_j - eps_j * mtilde_i``. A rank-2 antisymmetric
    matrix with **no free parameters** -- both factors are computed, neither is
    fitted to the thing being tested. Comparing the estimated cyclic component
    against this is the sharpest available confirm-or-refute of the mechanism.
    """
    eps = np.asarray(eps, dtype=float)
    mtilde = np.asarray(mtilde, dtype=float)
    return np.outer(eps, mtilde) - np.outer(mtilde, eps)
