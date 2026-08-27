"""One null condition, end to end: simulate, estimate, decompose, run Phase 2R.

Order of operations, fixed here so no caller can reorder it:

1. draw the true world from ``sim_seed`` (:func:`simulator.simulate_truth`);
2. record the truth, **including its own directional decomposition**, before any
   estimate exists;
3. simulate wells and run the chosen estimator;
4. record the estimate separately -- truth is never overwritten;
5. Hodge-decompose the estimated directional matrix, and the true one;
6. hand the estimated matrix to Phase 2R **unchanged**, through
   ``residual_experiment.run_residual_condition``.

Step 6 is the part that must not be tampered with. The rungs, the split logic,
the coverage grid, the additive residualisation, the hyperparameter grid and the
metrics are Phase 2R's, imported, not copied. The only argument that differs
from the committed real-data blocks is the ``screen`` object.

Which Phase 2R blocks are reproduced
------------------------------------
``rank2``   the primary detector. ``ResidualConfig`` with
            ``force_hparams=residual_sweep.RANK2_HPARAMS`` -- rank 2, 204
            parameters, one 2x2 antisymmetric form, **no hyperparameter search
            at all** -- which is byte-for-byte the configuration
            ``residual_sweep.rank2_grid()`` uses on the real screens. Primary
            because it removes "a hyperparameter search found estimator noise"
            as an explanation before the question is even asked.
``honest``  the searched low-rank rung under the shrinkage estimator the Phase
            2R decision quotes (``split_validation_for_calibration=True``), i.e.
            ``residual_sweep.honest_alpha_grid()``'s configuration, together
            with the ``zero`` and ``potential`` rungs it contrasts against.

Both real-data reference values therefore have an exactly-matched null.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..residual import hodge_decomposition
from ..residual_experiment import ResidualConfig, run_residual_condition
from ..residual_sweep import RANK2_HPARAMS
from . import dchain
from .adapter import as_screen
from .estimator import (JOINT, ORACLE, UNSHARED, artifact_template,
                        estimate_oracle, estimate_unshared,
                        second_position_gain)
from .simulator import (NullConfig, config_provenance, simulate_truth,
                        simulate_wells)
from .synergy import (SYNERGY_CONC, mean_baseline_viability, position_offset,
                      synergy_index_log, synergy_posterior)

#: Coverages the *searched* block is measured at -- the two the decision rests
#: on. The fixed rank-2 block runs the full grid instead (see RANK2_COVERAGES),
#: because H_artifact-null predicts a coverage transition and two points cannot
#: show one.
#: Not a shortened version of Phase 2R's grid: ``ResidualConfig.coverages`` still
#: carries the full five, so the *splits* are identical to the real ones and only
#: the cells that are scored differ. Shortening the grid itself would move the
#: evaluation pool, which is the bug the Phase 2 audit found.
DECISION_COVERAGES: tuple[float, ...] = (0.40, 0.70)
FULL_COVERAGES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70)
#: The primary detector is cheap -- 204 parameters, no search -- so it is scored
#: at every coverage the real rank-2 block was.
RANK2_COVERAGES: tuple[float, ...] = FULL_COVERAGES


@dataclass(frozen=True)
class NullRunConfig:
    null: NullConfig = NullConfig()
    #: ``oracle`` (Null 0), ``unshared`` (Null 1 / Control C) or ``joint``
    #: (Nulls 2 and 3 -- the published sampler).
    estimator: str = JOINT
    #: MCMC seed. Distinct from ``null.sim_seed`` so "the data changed" and "the
    #: sampler changed" are never confounded. 0 is the published default engine.
    est_seed: int = 1
    iterations: int = dchain.PUBLISHED_MCMC["iterations"]
    burn: int = dchain.PUBLISHED_MCMC["burn"]
    subsample: int = dchain.PUBLISHED_MCMC["subsample"]
    init_phase: int = dchain.PUBLISHED_MCMC["init_phase"]
    coverages: tuple[float, ...] = DECISION_COVERAGES
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    #: Split seeds for the (much more expensive) searched-grid block.
    honest_split_seeds: tuple[int, ...] = (0, 1, 2, 3)
    run_honest_block: bool = True
    keep_posterior: bool = False


def _spectrum(curl: np.ndarray) -> dict:
    sv = np.linalg.svd(curl, compute_uv=False)
    energy = float((sv ** 2).sum())
    return {"top_k_energy": {str(k): float((sv[:k] ** 2).sum() / energy)
                             for k in (1, 2, 4, 8, 16, 32, 64) if k <= len(sv)},
            "singular_values_top16": [float(v) for v in sv[:16]]}


def decompose(S: np.ndarray) -> dict:
    """Hodge decomposition of ``D(a,b) = S(a,b) - S(b,a)`` plus its spectrum.

    Uses Phase 2R's own :func:`residual.hodge_decomposition` on an adapted frame
    rather than a second implementation, so "the null was measured with a
    different decomposition" is not available as an explanation.
    """
    n = S.shape[0]
    S = np.asarray(S, float)
    d0 = S - S.T
    np.fill_diagonal(d0, 0.0)
    if not np.any(d0):
        # The STRICT null's true directional matrix is *exactly* the zero
        # matrix -- not small, zero -- because the authors' measure carries a
        # multiplicative lambda_AB and every selector is off. Its potential and
        # cyclic fractions are 0/0. Reporting NaN with the flag set is the
        # honest record; hodge_decomposition would divide by zero, and any
        # convention that filled in a number here would be inventing one.
        return {"n_drugs": int(n), "D_is_identically_zero": True,
                "D_mean_square": 0.0, "grad_mean_square": 0.0,
                "curl_mean_square": 0.0,
                "grad_fraction": float("nan"), "curl_fraction": float("nan"),
                "top_k_energy": {}, "singular_values_top16": [],
                "synergy_rms": float(np.sqrt((S[~np.eye(n, dtype=bool)] ** 2).mean())),
                "synergy_std": float(S[~np.eye(n, dtype=bool)].std())}
    screen = as_screen(S)
    h = hodge_decomposition(screen.frame, n)
    h["D_is_identically_zero"] = False
    d = np.asarray(S, float) - np.asarray(S, float).T
    np.fill_diagonal(d, 0.0)
    g = d.mean(axis=1)
    curl = d - (g[:, None] - g[None, :])
    h.update(_spectrum(curl))
    off = ~np.eye(n, dtype=bool)
    h["synergy_rms"] = float(np.sqrt((np.asarray(S, float)[off] ** 2).mean()))
    h["synergy_std"] = float(np.asarray(S, float)[off].std())
    return h


def run_estimator(cfg: NullRunConfig, truth, work: Path,
                  binary: Path | None = None) -> dict:
    """Produce the estimated synergy matrix and the estimator's diagnostics."""
    if cfg.estimator == ORACLE:
        out = estimate_oracle(cfg.null, truth)
        out["diagnostics"] = dict(out["diagnostics"], converged=True)
        return out
    if cfg.estimator == UNSHARED:
        out = estimate_unshared(cfg.null, truth, est_seed=cfg.est_seed)
        out["diagnostics"] = dict(out["diagnostics"], converged=True)
        return out
    if cfg.estimator != JOINT:
        raise ValueError(f"unknown estimator {cfg.estimator!r}")

    if binary is None:
        raise ValueError("the joint estimator needs the compiled dchain binary; "
                         "run scripts/prepare_dchain_null.py")
    work.mkdir(parents=True, exist_ok=True)
    wells = simulate_wells(cfg.null, truth)
    data_csv = work / "wells.csv"
    wells.to_csv(data_csv, index=False)
    mcmc_dir = work / "mcmc"
    diag = dchain.run(binary, data_csv, mcmc_dir, cell="SIM",
                      iterations=cfg.iterations, burn=cfg.burn,
                      subsample=cfg.subsample, init_phase=cfg.init_phase,
                      seed=cfg.est_seed)
    post = dchain.load_posterior(mcmc_dir)
    sp = synergy_posterior(post.theta, post.lam, post.theta_AB, post.lam_AB)
    lam_ab_mean = post.lam_AB.mean(axis=0)

    # The two factors of the predicted artifact, both computed rather than
    # fitted. eps is the posterior's error in the shared per-first-drug offset;
    # mtilde is how much of such an offset the second drug's curve converts into
    # apparent synergy. Their wedge is a fully specified rank-2 antisymmetric
    # matrix with no free parameters -- see _template_test.
    u_hat = (np.log(post.beta_residual)
             + post.lam * np.log(np.maximum(
                 _f1(post.theta, cfg.null.first_concentration), 1e-12))).mean(axis=0)
    u_true = position_offset(truth.theta, truth.lam, truth.beta,
                             cfg.null.first_concentration)
    eps = u_hat - u_true
    mtilde = second_position_gain(truth.theta, truth.lam,
                                  np.asarray(cfg.null.ab_concentrations, float))
    diag.update({
        "converged": diag["n_samples"] == diag["n_samples_expected"],
        "offset_error_rms": float(np.sqrt((eps ** 2).mean())),
        "offset_error_sd": float(eps.std()),
        "second_position_gain_mean": float(mtilde.mean()),
        "second_position_gain_sd": float(mtilde.std()),
        # 2 * mean(posterior sd^2) / mean(D^2): how much of the directional
        # variance is MCMC/posterior noise rather than reproducible structure.
        # The real screen's is 0.205, from Data Table 4's synergy_sd column.
        "posterior_noise_fraction_of_D": _noise_fraction(sp["sd"], sp["mean"]),
        "selector_on_fraction": float(lam_ab_mean.mean()),
        "single_selector_on_fraction": float(post.lam.mean()),
        # Split-half agreement of the posterior mean synergy matrix. This is the
        # MCMC error that survives into the target, and it is the diagnostic
        # that matters here: it is independent across pairs, so it can only
        # dilute held-out skill, never create it.
        **_split_half(post),
        "identities": dchain.deposit_identities(sp["mean"], lam_ab_mean,
                                                diag["n_samples"]),
        "posterior_sd_median": float(np.median(sp["sd"])),
    })
    extra_template = {"artifact_template": _template_test(sp["mean"], eps, mtilde),
                      "eps": eps, "mtilde": mtilde}
    extra = {}
    if cfg.keep_posterior:
        # The mechanism probe. Free: it reuses the posterior this run already
        # holds, and it is only computed where the posterior is being kept
        # anyway. See synergy.synergy_index_log for what it predicts.
        extra["synergy_log_scale"] = np.mean(
            [synergy_index_log(post.theta[k], post.lam[k], post.theta_AB[k],
                               post.lam_AB[k])
             for k in range(post.n_samples)], axis=0)
        extra["mean_baseline_viability"] = np.mean(
            [mean_baseline_viability(post.theta[k], post.lam[k])
             for k in range(post.n_samples)], axis=0)
    else:
        for f in dchain.OUTPUT_FILES:
            (mcmc_dir / f).unlink(missing_ok=True)
        data_csv.unlink(missing_ok=True)
    return {"synergy": sp["mean"], "synergy_sd": sp["sd"],
            "lambda_ab": lam_ab_mean, "estimator": JOINT, "diagnostics": diag,
            **extra_template, **extra}


def _split_half(post) -> dict:
    """Posterior-mean synergy from the first and second halves of the chain."""
    h = post.n_samples // 2
    if h < 2:                                            # pragma: no cover
        return {"split_half_pearson": float("nan")}
    a = synergy_posterior(post.theta[:h], post.lam[:h], post.theta_AB[:h],
                          post.lam_AB[:h])["mean"]
    b = synergy_posterior(post.theta[h:], post.lam[h:], post.theta_AB[h:],
                          post.lam_AB[h:])["mean"]
    off = ~np.eye(a.shape[0], dtype=bool)
    da, db = a - a.T, b - b.T
    return {
        "split_half_pearson": float(np.corrcoef(a[off], b[off])[0, 1]),
        "split_half_pearson_D": float(np.corrcoef(da[off], db[off])[0, 1]),
        "split_half_rms_diff": float(np.sqrt(((a - b)[off] ** 2).mean())),
    }


def _f1(theta: np.ndarray, conc: float) -> np.ndarray:
    from .synergy import response
    return response(np.array([float(conc)]), theta)[..., 0]


def _noise_fraction(sd: np.ndarray, mean: np.ndarray) -> float:
    """``2 * mean(posterior sd^2) / mean(D^2)`` on the off-diagonal.

    D = S(i,j) - S(j,i) and the two posterior errors are treated as independent,
    which is why the 2. The comparable real number is 0.205 (A375) / 0.192
    (PANC1), computed from the deposited ``synergy_sd`` column. It is the scale
    statistic that matters, and the one the first draft of criterion D got
    wrong: ``cal_skill`` is exactly invariant to the *size* of D, but not at all
    invariant to how much of D is unreproducible noise.
    """
    off = ~np.eye(mean.shape[0], dtype=bool)
    d = mean - mean.T
    ms = float((d[off] ** 2).mean())
    return float(2.0 * (np.asarray(sd)[off] ** 2).mean() / ms) if ms > 0 else float("nan")


def _template_test(S: np.ndarray, eps: np.ndarray, mtilde: np.ndarray) -> dict:
    """How much of the estimated cyclic component is the predicted artifact?

    Builds ``T = eps ^ mtilde`` -- the reconstruction's prediction, with **no
    free parameters**, both factors computed from the truth and the posterior
    rather than fitted to the thing being explained -- takes its cyclic part, and
    regresses the estimated cyclic part on it. One scale coefficient is fitted
    and nothing else.

    This is the sharp confirm-or-refute of the mechanism, and it is reported
    rather than decided on: the verdict is about whether an artifact of the
    observed *size* is there, and this is about whether whatever is there has the
    predicted *shape*.
    """
    def curl_of(m):
        d = m - m.T
        np.fill_diagonal(d, 0.0)
        g = d.mean(axis=1)
        return d - (g[:, None] - g[None, :])

    c_hat = curl_of(np.asarray(S, float))
    c_t = curl_of(artifact_template(eps, mtilde))
    n = c_hat.shape[0]
    off = ~np.eye(n, dtype=bool)
    x, y = c_t[off], c_hat[off]
    if x.std() == 0 or y.std() == 0:                     # pragma: no cover
        return {"r2": float("nan"), "pearson": float("nan")}
    beta = float((x @ y) / (x @ x))
    resid = y - beta * x
    r2 = float(1.0 - (resid ** 2).sum() / (y ** 2).sum())
    # The subspace version: does the leading cyclic mode live where the template
    # says it should? Principal angle between span{eps, mtilde} and the leading
    # left-singular pair of the estimated curl.
    u = np.linalg.svd(c_hat)[0][:, :2]
    v, _ = np.linalg.qr(np.column_stack([eps, mtilde]))
    sv = np.linalg.svd(u.T @ v, compute_uv=False)
    return {
        "r2": r2, "scale": beta,
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "template_energy_share": float((beta ** 2) * (x ** 2).sum()
                                       / (y ** 2).sum()),
        "principal_cosines": [float(s) for s in sv],
        "subspace_overlap": float((sv ** 2).mean()),
    }


def _mechanism_probe(S: np.ndarray, m: np.ndarray) -> dict:
    """Is the cyclic part the predicted wedge of two per-drug vectors?

    The reconstruction says the artifact is ``eps ^ mtilde`` with ``mtilde_j`` a
    monotone function of drug ``j``'s mean baseline viability ``m_j``. A wedge of
    two per-drug vectors is a rank-2 antisymmetric matrix whose leading left and
    right singular vectors span the same two-dimensional subspace as
    ``{eps, mtilde}``. So: take the leading singular pair of the cyclic part and
    ask how much of ``m`` it explains. A large value supports the mechanism; a
    small one says whatever cyclic structure is present is not this.

    Reported, never used in the decision rule -- it is a description of *why* an
    artifact is there, and the verdict is about whether one is.
    """
    n = S.shape[0]
    d = S - S.T
    np.fill_diagonal(d, 0.0)
    g = d.mean(axis=1)
    curl = d - (g[:, None] - g[None, :])
    u, sv, vt = np.linalg.svd(curl)
    energy = float((sv ** 2).sum())
    basis = np.column_stack([np.ones(n), u[:, 0], u[:, 1]])
    coef, *_ = np.linalg.lstsq(basis, m, rcond=None)
    resid = m - basis @ coef
    ss = float(((m - m.mean()) ** 2).sum())
    return {
        "curl_top2_energy": float((sv[:2] ** 2).sum() / energy) if energy else float("nan"),
        # How much of the per-drug mean baseline viability the leading cyclic
        # subspace explains, over and above an intercept.
        "baseline_viability_r2_on_top2": float(1.0 - (resid ** 2).sum() / ss)
        if ss > 0 else float("nan"),
        "mean_baseline_viability_mean": float(m.mean()),
        "mean_baseline_viability_sd": float(m.std()),
    }


def _phase2r_rows(screen, cfg: NullRunConfig, raw_dir: Path | None = None) -> list[dict]:
    """Phase 2R, run on the simulated screen with the committed configurations.

    ``raw_dir`` reaches ``run_residual_condition`` only to locate the
    sign-accuracy threshold, which is exploratory and decides nothing. It is
    threaded through so the CI pipeline check can run against the tiny fixture
    instead of requiring the 1.7 MB Mendeley deposit; when omitted the default is
    unchanged, so no committed result is affected.
    """
    rows: list[dict] = []
    for cov in RANK2_COVERAGES:
        for ss in cfg.split_seeds:
            rc = ResidualConfig(screen=cfg.null.screen_label, coverage=cov,
                                rung="lowrank", split_seed=ss,
                                coverages=FULL_COVERAGES,
                                force_hparams=RANK2_HPARAMS, tag="rank2")
            row = run_residual_condition(
                rc, **({"raw_dir": raw_dir} if raw_dir else {}), screen=screen)
            row["block"] = "rank2"
            rows.append(row)
    if cfg.run_honest_block:
        for cov in cfg.coverages:
            for rung in ("zero", "potential", "lowrank"):
                for ss in cfg.honest_split_seeds:
                    rc = ResidualConfig(screen=cfg.null.screen_label,
                                        coverage=cov, rung=rung, split_seed=ss,
                                        coverages=FULL_COVERAGES,
                                        split_validation_for_calibration=True,
                                        tag="honest_alpha")
                    row = run_residual_condition(
                rc, **({"raw_dir": raw_dir} if raw_dir else {}), screen=screen)
                    row["block"] = "honest_alpha"
                    rows.append(row)
    return rows


def run_null_condition(cfg: NullRunConfig, work: Path,
                       binary: Path | None = None,
                       raw_dir: Path | None = None) -> dict:
    """One simulated screen, from generative seed to Phase 2R metrics."""
    t0 = time.time()
    truth = simulate_truth(cfg.null)

    # --- truth, recorded before an estimate exists ---------------------------
    true_block = {
        "true_synergy_rms": float(np.sqrt((truth.synergy ** 2).mean())),
        "true_pair_interaction_is_zero": bool(np.all(truth.lam_AB == 0)),
        "true_decomposition": decompose(truth.synergy),
    }

    est = run_estimator(cfg, truth, work, binary)
    S_hat = np.asarray(est["synergy"], float)

    # --- estimate, recorded separately ---------------------------------------
    est_block = {"estimated_decomposition": decompose(S_hat)}

    # Control B: what the estimator added. artifact = estimate - truth.
    artifact = S_hat - truth.synergy
    est_block["artifact_decomposition"] = decompose(artifact)
    off = ~np.eye(S_hat.shape[0], dtype=bool)
    est_block["artifact_rms"] = float(np.sqrt((artifact[off] ** 2).mean()))
    est_block["estimate_truth_pearson"] = (
        float(np.corrcoef(S_hat[off], truth.synergy[off])[0, 1])
        if truth.synergy[off].std() > 0 else float("nan"))

    if "artifact_template" in est:
        est_block["artifact_template"] = est["artifact_template"]
        # ... and the same test against the artifact matrix itself (estimate
        # minus truth), where under the strict null the two coincide because the
        # truth is zero, but under the nuisance null they do not.
        est_block["artifact_template_on_artifact"] = _template_test(
            artifact, est["eps"], est["mtilde"])
    if "synergy_log_scale" in est:
        # Does the cyclic structure survive being scored on the log scale? The
        # reconstruction predicts it should largely collapse to a potential if
        # the mechanism is the shared first-position offset; whatever survives
        # is something else.
        est_block["log_scale_decomposition"] = decompose(est["synergy_log_scale"])
        est_block["mechanism"] = _mechanism_probe(
            S_hat, np.asarray(est["mean_baseline_viability"], float))

    screen = as_screen(S_hat, label=cfg.null.screen_label,
                       lam_ab=est.get("lambda_ab"))
    rows = _phase2r_rows(screen, cfg, raw_dir)

    return {
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "provenance": config_provenance(cfg.null),
        "estimator": cfg.estimator,
        "sim_seed": cfg.null.sim_seed,
        "est_seed": cfg.est_seed,
        "variant": cfg.null.variant,
        "sigma_obs": cfg.null.sigma_obs,
        "n_drugs": cfg.null.n_drugs,
        "tag": cfg.null.tag,
        "seconds": time.time() - t0,
        "diagnostics": est["diagnostics"],
        **true_block, **est_block,
        "phase2r": rows,
    }


#: Artifact sizes, as a share of directional energy, at which the primary
#: detector's sensitivity is measured. Spans "far below anything the real
#: screens could hide" to "at the ceiling the real spectrum allows".
DETECTION_SHARES: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10,
                                       0.13, 0.20, 0.40)


def detection_curve(background: np.ndarray, shares=DETECTION_SHARES,
                    coverage: float = 0.70, split_seeds=(0, 1, 2, 3),
                    seed: int = 0, screen_label: str = "A375") -> list[dict]:
    """How large must a rank-2 cyclic artifact be before the detector sees it?

    The experiment's positive control injects a **pure** artifact (top-2 = 0.9999)
    and shows the fixed rank-2 rung recovers it at skill > 0.5. That proves the
    detector is not dead. It does **not** establish sensitivity at realistic
    artifact sizes, and a final adversarial reviewer was right that nothing in
    the experiment did -- which matters, because "the null shows no held-out
    predictability" is a statement about the world only to the extent that the
    detector would have seen predictability had it been there.

    This injects a rank-2 antisymmetric cyclic component carrying a known share
    of the directional energy into a *null-like* background, and runs the same
    fixed rank-2 rung at the same coverage. The result is a detection limit that
    can be put beside the ceiling the real spectrum allows.
    """
    B = np.asarray(background, float)
    n = B.shape[0]
    off = ~np.eye(n, dtype=bool)
    d_bg = B - B.T
    np.fill_diagonal(d_bg, 0.0)
    e_bg = float((d_bg[off] ** 2).mean())

    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 2))
    w = rng.normal(size=(2, 2))
    S = u @ (w - w.T) @ u.T                       # exactly rank-2 antisymmetric
    g = S.mean(axis=1)
    S = S - (g[:, None] - g[None, :])             # ... and purely cyclic
    S = S / np.sqrt((S[off] ** 2).mean())

    out = []
    for share in shares:
        # D_total = D_bg + k*S with k chosen so k^2*E[S^2] = share * total.
        k = 0.0 if share <= 0 else np.sqrt(share / (1.0 - share) * e_bg)
        M = B + 0.5 * k * S                       # adding to y adds k*S to D
        dec = decompose(M)
        skills = []
        for ss in split_seeds:
            rc = ResidualConfig(screen=screen_label, coverage=coverage,
                                rung="lowrank", split_seed=ss,
                                coverages=FULL_COVERAGES,
                                force_hparams=RANK2_HPARAMS, tag="detection")
            skills.append(run_residual_condition(rc, screen=as_screen(M))["cal_skill"])
        out.append({
            "artifact_share_of_D": float(share),
            "cal_skill": float(np.mean(skills)),
            "cal_skill_max": float(np.max(skills)),
            "top2": dec["top_k_energy"]["2"],
            "rank2_share_of_D": float(dec["curl_fraction"] * dec["top_k_energy"]["2"]),
            "n_split_seeds": len(split_seeds),
        })
    return out
