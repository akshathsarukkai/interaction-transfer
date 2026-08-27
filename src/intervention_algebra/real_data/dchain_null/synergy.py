"""The Koplev synergy measure, ported line by line from the authors' own code.

Source
------
``skoplev/d-chain`` (GPL-3.0), ``post/interpretMCMC.R``,
``summaryStatisticsMCMC()``. The R is short enough to quote in full, and it is
quoted rather than paraphrased because every claim this experiment makes rests
on the port being exact::

    response = function(conc, par) {
        K = par[1]; h = par[2]; alpha = par[3]
        out = ((1 - alpha) / (1 + (K * conc)^h)) + alpha
        return(out)
    }

    s = seq(0.01, 10, length.out=10)  # evaluation points for integral
    for (a in ...) for (b in ...) for (i in 1:nsample) {
        baseline = (1 - mcmc$lambda[i, b]) + mcmc$lambda[i, b] * (response(s, mcmc$theta[i, b,]))
        resp = (1 - mcmc$lambda_AB[i, a, b]) * baseline + mcmc$lambda_AB[i, a, b] * response(s, mcmc$theta_AB[i, a, b,])
        synergy_index[i, a, b] = mean(baseline - resp)
    }
    d$synergy_index_mean[a, b] = mean(synergy_index[,a,b])
    d$synergy_index_sd[a, b]   = sd(synergy_index[,a,b])

Upstream copyright and licence
------------------------------
``post/interpretMCMC.R`` is copyright the d-chain authors and licensed
**GPL-3.0**. The block quoted above is theirs, not this repository's, and is
reproduced for verification with attribution; the upstream source is not
vendored here. The Python below is this project's own expression of the same
measure and is covered by this repository's licence. If you intend to
redistribute a work incorporating the upstream code, read its licence:
<https://github.com/skoplev/d-chain>. See THIRD_PARTY_DATA.md.

Three facts follow directly from that code and they are what make this whole
experiment possible.

**1. The baseline depends only on the second drug.** ``baseline`` is a function
of ``lambda[b]`` and ``theta[b]`` alone -- the *first* drug ``a`` does not appear
in it. The first drug's own effect enters the d-chain likelihood as a
multiplicative factor ``beta_a * f(1 uM; theta_a)`` on the whole well
(``dchain.cpp``: ``x_AB = beta * f_A * f_B``), and that factor is *absent from
the synergy measure entirely* -- it is not subtracted, it is never formed.

**2. The measure collapses to one term.** Substituting ``resp`` into
``baseline - resp`` gives

    synergy(a->b) = lambda_AB(a,b) * mean_s[ baseline_b(s) - f(s; theta_AB(a,b)) ]

exactly, with no approximation: when ``lambda_AB`` is 0 the combination *is* the
baseline and the synergy is identically 0. :func:`synergy_index` computes the
form above because it is the form the R code evaluates; :func:`synergy_index_collapsed`
computes the algebraically identical one, and they are pinned to each other in
the tests.

**3. Ordering enters through the combination parameters only.** ``synergy(a->b)``
and ``synergy(b->a)`` share *no* parameter except through ``theta``: the first
uses ``(theta_b, lambda_b, theta_AB[a,b], lambda_AB[a,b])`` and the second uses
``(theta_a, lambda_a, theta_AB[b,a], lambda_AB[b,a])``. So the directional
effect

    D(a,b) = synergy(a->b) - synergy(b->a)

is built from a *column* quantity (baseline_b vs baseline_a) and an
*ordered-pair* quantity (theta_AB). The column part is a per-drug potential and
Phase 2R removes it. What is left has to come from ``theta_AB``, and
``theta_AB[a,b]`` is fitted to the (a->b) wells whose model mean is
``log beta_a + log f(1; theta_a) + log f(c; theta_AB[a,b])`` -- so error in
drug ``a``'s *shared* single-agent curve is pushed into ``theta_AB[a,b]`` for
every ``b``, and because ``f`` saturates, how much of it lands in the synergy
integral depends on where drug ``b``'s curve sits. That product of a per-first-
drug error and a per-second-drug sensitivity is a rank-1 bilinear term whose
antisymmetric part is exactly the low-rank cyclic signature Phase 2R measures.
Whether it is *large* is an empirical question, and is what the null answers.
"""

from __future__ import annotations

import numpy as np

#: ``s = seq(0.01, 10, length.out=10)`` -- the authors' integration grid,
#: verbatim. R's ``seq(from, to, length.out=n)`` is linear, not log, spacing.
SYNERGY_CONC = np.linspace(0.01, 10.0, 10)


def response(conc: np.ndarray, par: np.ndarray) -> np.ndarray:
    """``((1 - alpha) / (1 + (K * conc)^h)) + alpha``.

    ``par`` is ``(K, h, alpha)`` and may carry leading batch dimensions; ``conc``
    is broadcast against them on the last axis.
    """
    par = np.asarray(par, dtype=float)
    K = par[..., 0, None]
    h = par[..., 1, None]
    alpha = par[..., 2, None]
    return (1.0 - alpha) / (1.0 + (K * np.asarray(conc, dtype=float)) ** h) + alpha


def synergy_index(theta: np.ndarray, lam: np.ndarray,
                  theta_AB: np.ndarray, lam_AB: np.ndarray,
                  conc: np.ndarray = SYNERGY_CONC) -> np.ndarray:
    """One MCMC sample's ``n x n`` synergy matrix, in the R code's own form.

    Parameters mirror ``interpretMCMC.R``'s names exactly:

    ``theta``     ``(n, 3)``      per-drug single-agent ``(K, h, alpha)``
    ``lam``       ``(n,)``        per-drug selector, 0/1
    ``theta_AB``  ``(n, n, 3)``   ordered-combination parameters, ``[first, second]``
    ``lam_AB``    ``(n, n)``      ordered-combination selector, ``[first, second]``

    Returns ``S`` with ``S[a, b]`` the synergy of ``a`` first, ``b`` second.
    """
    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)
    theta_AB = np.asarray(theta_AB, dtype=float)
    lam_AB = np.asarray(lam_AB, dtype=float)
    n = theta.shape[0]
    if theta.shape != (n, 3) or lam.shape != (n,):
        raise ValueError(f"theta {theta.shape} / lambda {lam.shape} are not (n,3)/(n,)")
    if theta_AB.shape != (n, n, 3) or lam_AB.shape != (n, n):
        raise ValueError(
            f"theta_AB {theta_AB.shape} / lambda_AB {lam_AB.shape} are not "
            f"(n,n,3)/(n,n); the ordered-pair index must be [first, second]")

    # baseline[b, s] -- second drug only, exactly as in the R.
    baseline = (1.0 - lam)[:, None] + lam[:, None] * response(conc, theta)
    comb = response(conc, theta_AB)                       # (n, n, len(conc))
    # resp[a, b, s]; baseline broadcast on the *second* index, which is b.
    resp = ((1.0 - lam_AB)[..., None] * baseline[None, :, :]
            + lam_AB[..., None] * comb)
    return (baseline[None, :, :] - resp).mean(axis=-1)


def synergy_index_collapsed(theta: np.ndarray, lam: np.ndarray,
                            theta_AB: np.ndarray, lam_AB: np.ndarray,
                            conc: np.ndarray = SYNERGY_CONC) -> np.ndarray:
    """``lambda_AB(a,b) * mean_s[baseline_b(s) - f(s; theta_AB(a,b))]``.

    Algebraically identical to :func:`synergy_index`; written out because the
    collapsed form is the one every argument in the documentation reasons about,
    and ``test_the_two_synergy_forms_agree`` refuses to let them drift apart.
    """
    lam = np.asarray(lam, dtype=float)
    lam_AB = np.asarray(lam_AB, dtype=float)
    baseline = (1.0 - lam)[:, None] + lam[:, None] * response(conc, np.asarray(theta, float))
    comb = response(conc, np.asarray(theta_AB, float))
    return lam_AB * (baseline[None, :, :] - comb).mean(axis=-1)


def synergy_posterior(theta: np.ndarray, lam: np.ndarray,
                      theta_AB: np.ndarray, lam_AB: np.ndarray,
                      conc: np.ndarray = SYNERGY_CONC,
                      chunk: int = 64) -> dict:
    """Posterior mean and SD of the synergy matrix over MCMC samples.

    Leading axis of every argument is the sample index. Returns ``mean`` and
    ``sd`` matrices matching the deposited ``synergy_measure`` and ``synergy_sd``
    columns; ``sd`` is the sample SD across retained samples, ``ddof=1``, which
    is what R's ``sd()`` computes.

    Chunked over samples because the intermediate is ``(n_samples, n, n, 10)``
    -- 2,000 samples at 100 drugs is 1.6e9 doubles if formed at once.

    The variance accumulates in Welford form rather than as
    ``(sum(x^2) - n*mean^2)/(n-1)``. On these inputs the two agree to 8e-17, so
    this is not a bug fix; it is that a sum-of-squares variance cancels
    catastrophically when the mean is large relative to the spread, and the
    quantity being accumulated here is a *posterior* whose mean can be large
    while its spread is small. Cheap insurance on a number that feeds a reported
    diagnostic.
    """
    theta = np.asarray(theta, dtype=float)
    n_samp = theta.shape[0]
    n = theta.shape[1]
    mean = np.zeros((n, n))
    m2 = np.zeros((n, n))
    count = 0
    for lo in range(0, n_samp, chunk):
        hi = min(lo + chunk, n_samp)
        for k in range(lo, hi):
            s = synergy_index(theta[k], lam[k], theta_AB[k], lam_AB[k], conc)
            count += 1
            delta = s - mean
            mean += delta / count
            m2 += delta * (s - mean)
    sd = (np.sqrt(m2 / (n_samp - 1)) if n_samp > 1
          else np.full((n, n), np.nan))
    return {"mean": mean, "sd": sd, "n_samples": int(n_samp)}


def synergy_index_log(theta: np.ndarray, lam: np.ndarray,
                      theta_AB: np.ndarray, lam_AB: np.ndarray,
                      conc: np.ndarray = SYNERGY_CONC) -> np.ndarray:
    """The same measure computed on **log** viability. A mechanism probe.

    Not the authors' quantity and never reported as one, and it is **not part of
    the decision rule**. It exists to separate two ways an artifact could arise,
    and its prediction is conditional rather than sharp -- which is stated here
    because the temptation is to write it the other way round.

    The reconstruction (``docs/dchain_reconstruction.md`` §3.3) has error in the
    shared first-position offset ``u_a = log beta_a + log f(1; theta_a)`` pushed
    into ``theta_AB(a,b)`` for every ``b``. **If** the fit absorbed that offset
    as an approximately multiplicative rescale of the second drug's curve, the
    log-scale score would be ``eps_a + d_b`` -- additively separable, hence a
    pure per-drug potential with **zero curl** -- and the cyclic structure would
    be revealed as an artifact of scoring on the linear scale.

    But the Hill family is not closed under scaling (``f(0) = 1`` pins it), so
    the fit absorbs the offset by whatever combination of ``(K, h, alpha)``
    minimises its loss, and that is not a rescale. A hand-injected row
    contamination that scales ``alpha`` shows **no collapse at all** on the log
    scale, while still producing exactly rank-2 cyclic structure on both scales.
    So a *surviving* log-scale curl does not refute the mechanism.

    What the probe is therefore good for: a **large** collapse would be positive
    evidence that the absorption is near-multiplicative, and no collapse leaves
    the question to the spectral test, which is the sharp one. Reported for
    completeness on the block that keeps the posterior anyway.
    """
    theta = np.asarray(theta, float)
    lam = np.asarray(lam, float)
    lam_AB = np.asarray(lam_AB, float)
    baseline = (1.0 - lam)[:, None] + lam[:, None] * response(conc, theta)
    log_base = np.log(np.maximum(baseline, 1e-12))
    log_comb = np.log(np.maximum(response(conc, np.asarray(theta_AB, float)), 1e-12))
    return lam_AB * (log_base[None, :, :] - log_comb).mean(axis=-1)


def mean_baseline_viability(theta: np.ndarray, lam: np.ndarray,
                            conc: np.ndarray = SYNERGY_CONC) -> np.ndarray:
    """``m_j``: the mean baseline viability of each drug over the score grid.

    The reconstruction predicts the artifact's second factor -- the per-drug
    "second-position gain" ``mtilde_j`` -- is a monotone function of this, going
    to 1 for an inert drug and 0 for a fully potent one. Reported so the
    prediction can be checked by regression rather than asserted.
    """
    lam = np.asarray(lam, float)
    baseline = (1.0 - lam)[:, None] + lam[:, None] * response(conc, np.asarray(theta, float))
    return baseline.mean(axis=-1)


def position_offset(theta: np.ndarray, lam: np.ndarray, beta: np.ndarray,
                    first_conc: float = 1.0) -> np.ndarray:
    """``u_i = log beta_i + 1[lambda_i] * log f(first_conc; theta_i)``.

    The shared per-first-drug quantity that every combination well involving
    drug ``i`` as the pretreatment is offset by, and the one the d-chain
    likelihood cannot separate from ``theta_AB(i, .)``. Its *error* is the first
    factor of the predicted artifact.
    """
    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)
    f1 = response(np.array([float(first_conc)]), theta)[..., 0]
    return np.log(np.asarray(beta, dtype=float)) + lam * np.log(np.maximum(f1, 1e-12))
