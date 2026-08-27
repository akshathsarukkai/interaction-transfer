"""Tests for the d-chain null: mostly, tests that the null is actually null.

The failure mode this experiment is most exposed to is not a bug that makes it
crash. It is a bug that quietly puts reusable pair structure into a world that is
supposed to have none, at which point the experiment answers a different question
and still returns a number. Most of what follows is aimed at that, and several of
the tests are written so that *inserting* the structure makes them fail --
mutation checks rather than assertions about the code as written.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import requires_deposit
from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.dchain_null import (dchain, estimator,
                                                        grids, report, synergy)
from intervention_algebra.real_data.dchain_null.adapter import (as_screen,
                                                                matrix_from_screen)
from intervention_algebra.real_data.dchain_null.experiment import (
    DECISION_COVERAGES, NullRunConfig, decompose, run_null_condition)
from intervention_algebra.real_data.dchain_null.simulator import (
    NUISANCE, STRICT, NullConfig, config_provenance, simulate_truth,
    simulate_wells, true_log_means)
from intervention_algebra.real_data.residual import hodge_decomposition
from intervention_algebra.real_data.residual_sweep import RANK2_HPARAMS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "koplev2017"
_HAVE_RAW = (RAW / "Data Table 1.csv").exists()


# --------------------------------------------------------------------------
# 1. The null is null
# --------------------------------------------------------------------------

def test_strict_null_has_exactly_zero_true_pair_interaction():
    """Not "small". Zero, for every ordered pair, on the nose.

    The authors' measure carries ``lambda_AB`` as a multiplicative factor, so a
    world with every selector off has a true synergy of exactly 0.0 everywhere.
    If this ever becomes merely small, the primary null has stopped being the
    thing the pre-registration describes.
    """
    truth = simulate_truth(NullConfig(variant=STRICT, n_drugs=30, sim_seed=3))
    assert np.all(truth.lam_AB == 0.0)
    assert np.array_equal(truth.synergy, np.zeros_like(truth.synergy))
    assert np.array_equal(truth.directional(), np.zeros_like(truth.synergy))


def test_the_null_contains_no_reusable_pair_structure():
    """The true pair effect must be unpredictable from other pairs, in principle.

    Under NUISANCE the true synergy is nonzero, so "it is zero" is not the
    argument. The argument is that its pair-specific part is drawn i.i.d., which
    means a *held-out* pair's residual is independent of every observed one.
    Operationally: after removing everything a per-drug model can express, what
    is left must have no low-rank structure -- its cyclic component must be
    spectrally indistinguishable from noise.
    """
    n = 100
    truth = simulate_truth(NullConfig(variant=NUISANCE, n_drugs=n, sim_seed=1))
    d = decompose(truth.synergy)
    # An i.i.d. antisymmetric matrix at n=100 puts ~0.076 of its cyclic energy
    # in the top two singular directions. Reusable rank-2 structure would put a
    # large multiple of that there.
    assert d["top_k_energy"]["2"] < 0.15, (
        f"the NUISANCE null's true cyclic component has "
        f"{d['top_k_energy']['2']:.3f} of its energy in two directions, against "
        f"~0.076 for noise. Something in the generative model is reusable "
        f"across pairs.")


def test_inserting_reusable_pair_structure_breaks_the_null_integrity_test():
    """Mutation check: the test above must be able to fail.

    A null-integrity test that passes on a world that *does* contain the signal
    is worse than no test. Here a rank-2 antisymmetric term -- exactly the shape
    the reconstruction predicts an artifact takes -- is injected into the true
    synergy, and the spectral criterion must reject it.
    """
    n = 100
    truth = simulate_truth(NullConfig(variant=NUISANCE, n_drugs=n, sim_seed=1))
    rng = np.random.default_rng(0)
    u, w = rng.normal(size=(n, 2)), rng.normal(size=(2, 2))
    S = u @ (w - w.T) @ u.T
    S = S / np.sqrt((S[~np.eye(n, dtype=bool)] ** 2).mean())
    contaminated = truth.synergy + 0.05 * S
    d = decompose(contaminated)
    assert d["top_k_energy"]["2"] >= 0.15, (
        "injecting a rank-2 antisymmetric pair term did not trip the spectral "
        "criterion, so that criterion cannot detect the thing it exists to "
        "detect")


def test_true_response_is_separable_under_the_strict_null():
    """The ordered response must be (per-first-drug) + (second drug's own curve).

    This is what "no pair-specific interaction" means at the level of the data
    the estimator actually sees, and it is stronger than checking the synergy
    measure: the measure could be zero while the wells still carried pair
    structure the estimator could pick up.
    """
    cfg = NullConfig(variant=STRICT, n_drugs=12, sim_seed=0)
    truth = simulate_truth(cfg)
    m = true_log_means(cfg, truth)["AB"]                 # (n, n, C)
    row = m - m.mean(axis=0, keepdims=True)              # remove the first-drug part
    col = row - row.mean(axis=1, keepdims=True)          # remove the second-drug part
    assert np.abs(col).max() < 1e-10, (
        f"the true AB log-means are not additively separable; the largest "
        f"pair-specific residual is {np.abs(col).max():.2e}")


def test_no_real_pair_residual_information_reaches_the_simulator():
    """The null's parameters must not be derived from the real screen's pairs.

    Checked two ways, because the interesting version of this failure is
    accidental. First, the simulator module must not import or read anything
    that carries real pair values. Second, a simulated screen must be
    bit-identical whether or not the real deposit is on disk.
    """
    src = (ROOT / "src" / "intervention_algebra" / "real_data" / "dchain_null"
           / "simulator.py").read_text()
    for forbidden in ("koplev", "read_csv", "np.loadtxt", "synergy_measure",
                      "data/raw", "phase2_residual", "residual_experiment",
                      "D_res", "hodge", "Data Table"):
        assert forbidden not in src, (
            f"simulator.py mentions {forbidden!r}; the null's parameters must "
            f"not be derivable from the real screen")
    prov = config_provenance(NullConfig())
    assert prov["provenance"]["REAL PAIR DATA USED"] == "none"


def test_oracle_of_the_strict_null_has_nothing_to_find():
    """Control A. The measure at the TRUE parameters must be the zero matrix."""
    cfg = NullConfig(variant=STRICT, n_drugs=20, sim_seed=2)
    truth = simulate_truth(cfg)
    out = estimator.estimate_oracle(cfg, truth)
    assert np.array_equal(out["synergy"], np.zeros((20, 20)))
    d = decompose(out["synergy"])
    assert d["D_is_identically_zero"] is True
    assert np.isnan(d["curl_fraction"])


# --------------------------------------------------------------------------
# 2. Determinism, and truth kept apart from estimates
# --------------------------------------------------------------------------

def test_simulation_is_deterministic_in_its_seed():
    for variant in (STRICT, NUISANCE):
        a = simulate_truth(NullConfig(variant=variant, n_drugs=15, sim_seed=7))
        b = simulate_truth(NullConfig(variant=variant, n_drugs=15, sim_seed=7))
        for f in ("theta", "lam", "beta", "theta_AB", "lam_AB", "synergy"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
        c = simulate_truth(NullConfig(variant=variant, n_drugs=15, sim_seed=8))
        assert not np.array_equal(a.theta, c.theta)
        cfg = NullConfig(variant=variant, n_drugs=15, sim_seed=7)
        wa = simulate_wells(cfg, a)
        wb = simulate_wells(cfg, a)
        pd.testing.assert_frame_equal(wa, wb)


@requires_deposit
def test_truth_and_estimate_are_stored_separately():
    """A run must carry both, and must not be able to pass one off as the other."""
    cfg = NullRunConfig(null=NullConfig(variant=NUISANCE, n_drugs=60, sim_seed=0),
                        estimator="unshared", coverages=(0.70,),
                        split_seeds=(0,), run_honest_block=False)
    row = run_null_condition(cfg, Path("/tmp/dchain_null_test"))
    for key in ("true_synergy_rms", "true_decomposition",
                "estimated_decomposition", "artifact_decomposition",
                "artifact_rms", "true_pair_interaction_is_zero"):
        assert key in row, key
    # Control B is a real subtraction, not a relabelling.
    assert row["artifact_rms"] > 0
    assert (row["estimated_decomposition"]["synergy_rms"]
            != row["true_decomposition"]["synergy_rms"])


def test_the_simulator_matches_the_csv_schema_dchain_parses():
    """Column names and order are fixed by dchain.cpp's FIELD_* macros."""
    cfg = NullConfig(n_drugs=4, sim_seed=0)
    w = simulate_wells(cfg, simulate_truth(cfg))
    assert list(w.columns) == ["Experiment", "CellLine", "Run", "Plate",
                               "Pretreatment", "Compound", "Concentration",
                               "RelCount"]
    assert set(w["Experiment"]) == {"A", "A0", "AB"}
    counts = w.groupby("Experiment").size()
    assert counts["A"] == 4 * len(cfg.a_concentrations) * cfg.n_replicates
    assert counts["A0"] == 4 * cfg.n_replicates
    # The diagonal is included, as the deposited tables include it.
    assert counts["AB"] == 4 * 4 * len(cfg.ab_concentrations) * cfg.n_replicates
    assert (w["RelCount"] > 0).all()


# --------------------------------------------------------------------------
# 3. The synergy measure
# --------------------------------------------------------------------------

def test_the_two_synergy_forms_agree():
    rng = np.random.default_rng(0)
    n = 7
    th = np.stack([np.exp(rng.normal(-2, 1, n)), np.exp(rng.normal(0.4, .5, n)),
                   rng.beta(1, 3, n)], axis=1)
    lam = rng.integers(0, 2, n).astype(float)
    thAB = np.stack([np.exp(rng.normal(-2, 1, (n, n))),
                     np.exp(rng.normal(0.4, .5, (n, n))),
                     rng.beta(1, 3, (n, n))], axis=-1)
    lAB = rng.integers(0, 2, (n, n)).astype(float)
    a = synergy.synergy_index(th, lam, thAB, lAB)
    b = synergy.synergy_index_collapsed(th, lam, thAB, lAB)
    assert np.allclose(a, b, atol=1e-14)


def test_synergy_matches_a_literal_transliteration_of_the_R_loop():
    """The vectorised port against a scalar loop written straight from the R.

    The R cannot be executed here (no R runtime, and no deposited posterior
    samples to feed it -- see docs/dchain_reconstruction.md §6), so the check is
    against a transliteration that is short enough to compare to the quoted
    source line by line.
    """
    rng = np.random.default_rng(1)
    n = 5
    s = synergy.SYNERGY_CONC
    th = np.stack([np.exp(rng.normal(-2, 1, n)), np.exp(rng.normal(0.4, .5, n)),
                   rng.beta(1, 3, n)], axis=1)
    lam = rng.integers(0, 2, n).astype(float)
    thAB = np.stack([np.exp(rng.normal(-2, 1, (n, n))),
                     np.exp(rng.normal(0.4, .5, (n, n))),
                     rng.beta(1, 3, (n, n))], axis=-1)
    lAB = rng.integers(0, 2, (n, n)).astype(float)

    def response(conc, par):
        K, h, alpha = par
        return ((1 - alpha) / (1 + (K * conc) ** h)) + alpha

    want = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            baseline = (1 - lam[b]) + lam[b] * response(s, th[b])
            resp = (1 - lAB[a, b]) * baseline + lAB[a, b] * response(s, thAB[a, b])
            want[a, b] = np.mean(baseline - resp)
    assert np.allclose(synergy.synergy_index(th, lam, thAB, lAB), want, atol=1e-14)


def test_zero_selector_gives_exactly_zero_synergy():
    """The identity the deposit satisfies on 191 A375 rows and 173 PANC1 rows."""
    rng = np.random.default_rng(2)
    n = 6
    th = np.stack([np.exp(rng.normal(-2, 1, n)), np.exp(rng.normal(0.4, .5, n)),
                   rng.beta(1, 3, n)], axis=1)
    thAB = np.stack([np.exp(rng.normal(-2, 1, (n, n))),
                     np.exp(rng.normal(0.4, .5, (n, n))),
                     rng.beta(1, 3, (n, n))], axis=-1)
    lAB = np.zeros((n, n))
    S = synergy.synergy_index(th, np.ones(n), thAB, lAB)
    assert np.array_equal(S, np.zeros((n, n)))


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Koplev deposit")
def test_the_deposit_still_satisfies_the_identities_the_reconstruction_rests_on():
    """If these stop holding, the reconstruction of the measure is wrong."""
    ref = dchain.deposited_reference(RAW)
    assert set(ref) == {"A375", "PANC1"}
    for label, d in ref.items():
        assert d["lambda_is_multiple_of_1_over_n"], label
        assert d["zero_lambda_implies_zero_synergy"], label
        assert d["abs_synergy_le_abs_lambda"], label
        assert d["n_lambda_zero"] > 0, label


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Koplev deposit")
def test_the_deposit_pins_the_published_mcmc_settings():
    """1,999 retained samples, and not 1,998 or 2,000.

    ``iter > 100000 && iter % 200 == 0`` over 500,000 iterations retains exactly
    1,999 samples, which is why the null runs at those settings and not at
    convenient ones.
    """
    stored = [k for k in range(dchain.PUBLISHED_MCMC["iterations"])
              if k > dchain.PUBLISHED_MCMC["burn"]
              and k % dchain.PUBLISHED_MCMC["subsample"] == 0]
    assert len(stored) == dchain.PUBLISHED_N_SAMPLES == 1999
    lam = pd.read_csv(RAW / "Data Table 1.csv")["lambda"].to_numpy()
    for n in (1998, 2000):
        q = np.abs(lam) * n
        assert np.abs(q - np.round(q)).max() > 1e-6, (
            f"|lambda| is also a multiple of 1/{n}; the sample count is not "
            f"pinned and the settings claim is weaker than documented")


# --------------------------------------------------------------------------
# 4. The adapter into Phase 2R
# --------------------------------------------------------------------------

def test_the_adapter_preserves_pair_orientation():
    """Equality on the off-diagonal, not correlation.

    A transpose here would negate every directional target and the experiment
    would still produce numbers.
    """
    rng = np.random.default_rng(0)
    n = 9
    S = rng.normal(size=(n, n))
    screen = as_screen(S)
    assert np.array_equal(matrix_from_screen(screen)[~np.eye(n, dtype=bool)],
                          S[~np.eye(n, dtype=bool)])
    r = screen.frame[(screen.frame["i"] == 2) & (screen.frame["j"] == 5)]
    assert float(r["y"].iloc[0]) == S[2, 5]


def test_the_adapter_drops_the_diagonal_and_refuses_bad_input():
    n = 6
    S = np.zeros((n, n))
    screen = as_screen(S)
    assert len(screen.frame) == n * n - n
    assert not (screen.frame["i"] == screen.frame["j"]).any()
    with pytest.raises(ValueError, match="square"):
        as_screen(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="non-finite"):
        as_screen(np.array([[0.0, np.nan], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="sign threshold"):
        as_screen(S, label="SIM")


def test_the_adapter_is_the_only_thing_between_the_null_and_phase2r():
    """The null must call Phase 2R's own entry point, not a copy of it."""
    src = (ROOT / "src" / "intervention_algebra" / "real_data" / "dchain_null"
           / "experiment.py").read_text()
    assert "from ..residual_experiment import" in src
    assert "run_residual_condition" in src
    assert "from ..residual_sweep import RANK2_HPARAMS" in src
    # and it must not have grown its own model or metric
    for forbidden in ("class ResidualConfig", "def residual_metrics",
                      "def fit_additive", "def train_residual"):
        assert forbidden not in src, f"experiment.py redefines {forbidden}"


def test_rank2_null_evaluation_uses_the_committed_fixed_configuration():
    """The primary detector must be the real block's configuration exactly."""
    src = (ROOT / "src" / "intervention_algebra" / "real_data" / "dchain_null"
           / "experiment.py").read_text()
    assert "force_hparams=RANK2_HPARAMS" in src
    assert RANK2_HPARAMS == ({"rank": 2, "lr": 1e-2, "weight_decay": 1e-3},)
    # and the splits must be built on the full coverage grid, not a truncated one
    assert "coverages=FULL_COVERAGES" in src
    from intervention_algebra.real_data.dchain_null.experiment import FULL_COVERAGES
    from intervention_algebra.real_data.residual_sweep import COVERAGES
    assert FULL_COVERAGES == COVERAGES
    assert set(DECISION_COVERAGES) <= set(COVERAGES)


@requires_deposit
def test_screen_label_affects_only_the_exploratory_sign_accuracy():
    """The label picks one threshold and nothing else the verdict reads."""
    cfg_a = NullRunConfig(
        null=NullConfig(variant=NUISANCE, n_drugs=60, sim_seed=0,
                        screen_label="A375"),
        estimator="unshared", coverages=(0.70,), split_seeds=(0,),
        run_honest_block=False)
    cfg_p = NullRunConfig(
        null=NullConfig(variant=NUISANCE, n_drugs=60, sim_seed=0,
                        screen_label="PANC1"),
        estimator="unshared", coverages=(0.70,), split_seeds=(0,),
        run_honest_block=False)
    a = run_null_condition(cfg_a, Path("/tmp/dchain_null_test_a"))["phase2r"][0]
    p = run_null_condition(cfg_p, Path("/tmp/dchain_null_test_p"))["phase2r"][0]
    for k in ("cal_skill", "cal_pearson", "cal_spearman", "heldout_skill"):
        # NaN is a legitimate value here: the shrinkage can select alpha = 0,
        # which makes the prediction constant and its correlation undefined
        # rather than zero. Both labels must reach the *same* NaN.
        if np.isnan(a[k]) or np.isnan(p[k]):
            assert np.isnan(a[k]) and np.isnan(p[k]), k
        else:
            assert a[k] == pytest.approx(p[k], abs=1e-12), k
    assert a["cal_sign_threshold"] != p["cal_sign_threshold"]


# --------------------------------------------------------------------------
# 5. Results cannot be pooled, and the summary is honest
# --------------------------------------------------------------------------

def test_null_rows_cannot_be_loaded_as_phase2r_rows():
    """The two live in different directories with different loaders, by design."""
    from intervention_algebra.real_data.residual_report import load_residual_runs
    p = Path("/tmp/dchain_null_pool_check.jsonl")
    p.write_text(json.dumps({"tag": "primary", "estimator": "joint",
                             "sim_seed": 0, "variant": "strict"}) + "\n")
    with pytest.raises(Exception):
        load_residual_runs(p)
    assert report.OUT_DIR != Path("results/phase2_residual")
    assert report.METRICS.parent.name == "dchain_null"


def test_empirical_percentile_is_computed_correctly():
    null = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    hi = report.percentile_of(0.10, null)
    assert hi["percentile"] == 100.0
    assert hi["p_one_sided"] == pytest.approx(1 / 6)
    assert hi["null_median"] == pytest.approx(0.02)
    mid = report.percentile_of(0.02, null)
    assert mid["percentile"] == pytest.approx(40.0)
    # (1 + #{null >= value}) / (n + 1): three of five are >= 0.02.
    assert mid["p_one_sided"] == pytest.approx(4 / 6)
    lo = report.percentile_of(-1.0, null)
    assert lo["percentile"] == 0.0
    assert lo["p_one_sided"] == pytest.approx(6 / 6)


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_real_reference_values_are_read_not_transcribed():
    """Every reference number must come out of the generated Phase 2R files."""
    ref = report.real_reference()
    assert set(ref) == {"A375", "PANC1"}
    # cross-check two of them against the artifacts directly
    h = json.loads((report.PHASE2R / "summary"
                    / "hodge_decomposition.json").read_text())
    assert ref["A375"]["curl_fraction"] == h["A375"]["curl_fraction"]
    r2 = pd.read_csv(report.PHASE2R / "summary" / "rank2.csv")
    want = float(r2[(r2["screen"] == "PANC1") & (r2["coverage"] == 0.70)
                    & (r2["metric"] == "cal_skill")]["mean"].iloc[0])
    assert ref["PANC1"]["rank2_skill"][0.70] == want
    src = (ROOT / "src" / "intervention_algebra" / "real_data" / "dchain_null"
           / "report.py").read_text()
    for literal in ("0.4618", "0.6017", "0.1967", "0.2372", "0.353", "0.366"):
        assert literal not in src, (
            f"report.py hard-codes the real value {literal}; it must be read "
            f"from the Phase 2R artifacts on every call")


def test_decision_thresholds_match_the_preregistration():
    """The rule the code executes must be the rule that was committed."""
    log = (ROOT / "docs" / "PREREGISTRATIONS.md").read_text()
    pre = log[log.index("## Pre-registration — the d-chain null"):]
    assert "0.5 * s_real" not in pre or True     # prose form varies; check numbers
    assert report.DECISION["artifact_fraction_of_real"] == 0.5
    assert report.DECISION["clearly_positive_skill"] == 0.02
    assert report.DECISION["max_failure_fraction"] == 0.20
    assert report.DECISION["coverages"] == (0.40, 0.70)
    for token in ("+0.119", "+0.081", "0.02", "0.980"):
        assert token in pre, (
            f"the pre-registration does not state {token}; the executed rule and "
            f"the committed rule have diverged")
    # Everything the amendment changed must be stated there, with its reason.
    amend = pre[pre.index("### Amendment to the pre-registration"):]
    assert report.DECISION["noise_floor_top2"] == 0.0747
    assert report.DECISION["min_fraction_of_planned"] == 0.80
    assert report.DECISION["min_selector_on_fraction"] == 0.10
    assert report.DECISION["min_split_half_pearson_D"] == 0.50
    for token in ("0.0747", "0.157", "0.193", "0.0875", "0.4916", "0.4635",
                  "0.10", "0.50", "80%", "28.8%", "26.7%"):
        assert token in amend, (
            f"the amendment does not state {token}; a threshold changed without "
            f"the change being recorded")


def test_the_ensemble_grid_is_the_preregistered_one():
    counts = grids.part_counts()
    assert counts["primary"] == 20
    # The realism arm was registered at 10 and raised to 20 after review; the
    # nuisance oracle follows it, so the oracle block is 20 + 20.
    assert counts["realism"] == 20
    assert counts["oracle"] == 40
    assert counts["unshared"] == 20
    assert counts["noise"] == 12
    assert counts["convergence"] == 4
    assert counts["all"] == sum(counts[p] for p in grids.ALL_PARTS)
    # generative and estimator seeds are never the same number, and never 0
    for c in grids.part_jobs("all"):
        assert c.est_seed != 0
        if c.estimator == "joint" and c.null.tag != "convergence":
            assert c.est_seed != c.null.sim_seed


# --------------------------------------------------------------------------
# 6. The patch to the published sampler
# --------------------------------------------------------------------------

def test_the_patch_matches_upstream_exactly_or_refuses():
    """Every patch hunk is asserted to hit exactly once, so drift is loud."""
    src = "int init_phase;  // relies on init_lambda to be true.\n\tstring strain;"
    with pytest.raises(RuntimeError, match="matched 0 times"):
        dchain.patch_source(src)


@pytest.mark.skipif(not (dchain.DEFAULT_DIR / "dchain.cpp").exists(),
                    reason="needs scripts/prepare_dchain_null.py")
def test_the_patched_source_changes_nothing_inside_the_sampler():
    """The likelihood, the priors and the proposals must survive the patch."""
    original = (dchain.DEFAULT_DIR / "dchain.cpp").read_text()
    patched = dchain.patch_source(original)
    for line in (
            "double out = pow(1 + pow(stat.x_mean - model, 2) * stat.n / (2 * b_post), -a_post - 0.5);",
            "var_prior.a = 0.6;",
            "var_prior.b = 0.02;",
            "K_prior.mu = 0.1;",
            "h_prior.mu = 1.5;",
            "alpha_prior.b = 3;",
            "beta_prior.sd = 0.05;",
            "double switch_prop = 0.1;",
            "logResponse(1.0, theta[a])",
            "double out = log((1 - params[2]) / (1 + pow(params[0] * conc, params[1])) + params[2]);"):
        assert line in original and line in patched, line
    # and nothing that was not in PATCHES was removed
    assert "qprior(theta[a][0], new_theta[0], K_prior)" in patched


@pytest.mark.skipif(not (dchain.DEFAULT_DIR / "dchain.cpp").exists(),
                    reason="needs scripts/prepare_dchain_null.py")
def test_the_source_digest_is_the_pinned_one():
    for name, (digest, size) in dchain.DCHAIN_FILES.items():
        p = dchain.DEFAULT_DIR / name
        if not p.exists():                               # pragma: no cover
            pytest.skip(f"{name} not fetched")
        assert dchain.sha256_of(p.read_bytes()) == digest, name
        assert p.stat().st_size == size, name


# --------------------------------------------------------------------------
# 7. The validation artifacts
# --------------------------------------------------------------------------

VALIDATION = ROOT / "results" / "dchain_null" / "summary" / "validation.json"


@pytest.mark.skipif(not VALIDATION.exists(),
                    reason="needs scripts/validate_dchain_null.py")
def test_the_committed_validation_still_supports_what_the_docs_claim():
    """Every fidelity claim in the reconstruction document is a file here.

    The document says the reconstruction is validated structurally rather than
    numerically, and names the checks. If any of them stops passing, the
    document's section 6 is a claim nobody rechecked.
    """
    v = json.loads(VALIDATION.read_text())
    assert v["source"]["commit"] == dchain.DCHAIN_COMMIT
    assert v["patch"]["byte_equivalence_enforced_at_build"] is True
    assert v["patch"]["sufficient_statistic_call_sites"] == 7
    for label, d in v["deposit_identities"].items():
        assert d["n_samples"] == 1999, label
        assert d["lambda_is_multiple_of_1_over_n"], label
        assert d["zero_lambda_implies_zero_synergy"], label
        assert d["abs_synergy_le_abs_lambda"], label
    for label, d in v["paper_counts"].items():
        assert d["exact"], f"{label}: {d['reproduced']} != {d['paper']}"
    e = v["example_run"]
    assert e["identities_hold"]
    assert e["mcmc"]["n_samples"] == e["mcmc"]["n_samples_expected"]
    # The example is two real drugs and the measure is genuinely directional on
    # them; a symmetric result would mean the ordered indexing had collapsed.
    S = np.asarray(e["synergy_mean"])
    assert S.shape == (2, 2)
    assert S[0, 1] != S[1, 0]


def test_the_reconstruction_document_does_not_overclaim_fidelity():
    """It must say plainly that a numerical check against the deposit is impossible.

    No posterior samples were deposited, so the inputs to the synergy formula do
    not exist publicly for any published value. A document that implied the port
    had been checked against published numbers would be claiming something the
    deposit cannot support.
    """
    doc = (ROOT / "docs" / "dchain_reconstruction.md").read_text()
    assert "no posterior samples" in doc.lower()
    assert "1999" in doc or "1,999" in doc
    for phrase in ("byte-identical", "40,500", "Bernoulli(0.5)"):
        assert phrase in doc, phrase


def test_smoke_rows_are_written_where_a_result_cannot_be_mistaken_for_them():
    src = (ROOT / "scripts" / "run_dchain_null.py").read_text()
    assert 'smoke.jsonl' in src
    gi = (ROOT / ".gitignore").read_text()
    assert "results/dchain_null/smoke.jsonl" in gi
    assert "results/dchain_null/simulations/" in gi


# --------------------------------------------------------------------------
# 8. A positive control for the artifact detector
# --------------------------------------------------------------------------

@requires_deposit
def test_the_predicted_artifact_is_exactly_rank_two_and_the_detector_sees_it():
    """If the mechanism is injected by hand, the diagnostics must light up.

    ``docs/dchain_reconstruction.md`` §3.3 predicts that an error in the shared
    per-first-drug offset, pushed into every combination curve in that drug's
    row, produces a cyclic component that is **exactly rank 2** -- the wedge of
    two per-drug vectors. This injects exactly that and checks the spectral
    statistic reaches 1.0.

    It matters because it makes the null's negative case interpretable: the
    statistic the verdict reads is demonstrably able to detect the artifact the
    experiment is looking for, so "the null shows a top-2 energy at the noise
    floor" means "this artifact is not there", not "this statistic is blind".

    It also fixes the scale of the comparison. The real screens put 0.340 (A375)
    and 0.321 (PANC1) of their cyclic energy in two directions, against 1.000 for
    a pure artifact and 0.076 for pure noise.
    """
    from intervention_algebra.real_data.dchain_null.synergy import (
        mean_baseline_viability, synergy_index)
    from intervention_algebra.real_data.dchain_null.experiment import _mechanism_probe

    rng = np.random.default_rng(0)
    n = 60
    theta = np.stack([np.exp(rng.normal(np.log(0.1), 1.4, n)),
                      np.exp(rng.normal(np.log(1.5), 0.7, n)),
                      rng.beta(1, 3, n)], axis=1)
    lam = np.ones(n)
    # Every combination is the second drug's own curve -- no pair interaction --
    # except that drug a's row carries a per-drug multiplicative contamination.
    theta_AB = np.broadcast_to(theta[None, :, :], (n, n, 3)).copy()
    eps = rng.normal(0.0, 0.05, n)
    theta_AB[..., 2] = np.clip(theta_AB[..., 2] * np.exp(eps[:, None]), 0.0, 1.0)
    S = synergy_index(theta, lam, theta_AB, np.ones((n, n)))

    d = decompose(S)
    assert d["top_k_energy"]["2"] > 0.99, (
        f"a per-first-drug row contamination gave top-2 cyclic energy "
        f"{d['top_k_energy']['2']:.3f}; the reconstruction predicts exactly 1.0 "
        f"and the spectral statistic is supposed to detect it")
    probe = _mechanism_probe(S, mean_baseline_viability(theta, lam))
    assert probe["curl_top2_energy"] > 0.99

    # ... and the fixed rank-2 detector must be able to predict it out of sample,
    # which is what makes it the right primary detector for this question.
    cfg = NullRunConfig(null=NullConfig(variant=STRICT, n_drugs=n, sim_seed=0),
                        estimator="oracle", coverages=(0.70,), split_seeds=(0,),
                        run_honest_block=False)
    from intervention_algebra.real_data.dchain_null.experiment import _phase2r_rows
    rows = _phase2r_rows(as_screen(S), cfg)
    dense = [r for r in rows if r["coverage"] == 0.70]
    assert dense and dense[0]["cal_skill"] > 0.5, (
        f"the rank-2 detector reached only {dense[0]['cal_skill']:.3f} at "
        f"coverage 0.70 on an injected pure artifact; it cannot be trusted to "
        f"report a null")


# --------------------------------------------------------------------------
# 9. The decision rule, exercised on synthetic ensembles
# --------------------------------------------------------------------------

def _filler(tag: str, n: int) -> list[dict]:
    """Minimal usable rows for a block, so the completeness gate is satisfied.

    verdict() refuses to classify unless 80% of the *whole* preregistered
    ensemble is present -- not just the block under test. These tests are about
    the classification logic, so they supply the rest of the ensemble rather
    than weakening the gate.
    """
    return [{"tag": tag, "estimator": "oracle", "variant": "strict",
             "sigma_obs": 0.15, "sim_seed": s, "est_seed": 900 + s,
             "true_synergy_rms": 0.0, "true_pair_interaction_is_zero": True,
             "true_decomposition": {"D_is_identically_zero": True,
                                    "curl_fraction": float("nan")},
             "estimated_decomposition": {
                 "curl_fraction": 0.97, "grad_fraction": 0.03,
                 "D_std_offdiag": 0.15, "D_mean_square": 0.0225,
                 "synergy_rms": 0.1, "top_k_energy": {"2": 0.08}},
             "artifact_decomposition": {"curl_fraction": 0.97, "top_k_energy": {}},
             "artifact_rms": 0.05, "estimate_truth_pearson": float("nan"),
             "diagnostics": {"converged": True, "n_samples": 1999,
                             "n_samples_expected": 1999},
             "phase2r": []} for s in range(n)]


def _fake_rows(skill: float, top2: float, n: int = 20, tag: str = "primary",
               d_std: float = 0.15, spread: float = 0.01,
               complete: bool = True) -> list[dict]:
    """An ensemble with a known null median, for exercising verdict()."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n):
        vals = skill + rng.normal(0, spread, 8)
        rows.append({
            "tag": tag, "estimator": "joint", "variant": "strict",
            "sigma_obs": 0.15, "sim_seed": s, "est_seed": 100 + s,
            "true_synergy_rms": 0.0, "true_pair_interaction_is_zero": True,
            "true_decomposition": {"D_is_identically_zero": True,
                                   "curl_fraction": float("nan")},
            "estimated_decomposition": {
                "curl_fraction": 0.97, "grad_fraction": 0.03,
                "D_std_offdiag": d_std, "D_mean_square": d_std ** 2,
                "synergy_rms": d_std / np.sqrt(2),
                "top_k_energy": {k: top2 for k in
                                 ("1", "2", "4", "8", "16", "32", "64")}},
            "artifact_decomposition": {"curl_fraction": 0.97, "top_k_energy": {}},
            "artifact_rms": 0.05, "estimate_truth_pearson": float("nan"),
            "diagnostics": {"converged": True, "n_samples": 1999,
                            "n_samples_expected": 1999,
                            "selector_on_fraction": 0.30,
                            "split_half_pearson_D": 0.95,
                            "posterior_noise_fraction_of_D": 1.5},
            "phase2r": [
                {"block": "rank2", "coverage": cov, "rung": "lowrank",
                 "split_seed": k, "cal_skill": float(v), "cal_pearson": 0.1,
                 "cal_spearman": 0.1, "cal_sign_accuracy": 0.5,
                 "heldout_skill": float(v), "n_params": 204}
                for cov in (0.40, 0.70) for k, v in enumerate(vals)],
        })
    if complete:
        for other, k in grids.__dict__["_PLANNED_FOR_TESTS"].items():
            if other != tag:
                rows += _filler(other, k)
    return rows


def test_verdict_refuses_to_conclude_from_an_empty_ensemble():
    """A missing comparison is a reconstruction failure, not "no artifact"."""
    v = report.verdict([], tag="primary")
    assert v["verdict"] == "INCONCLUSIVE RECONSTRUCTION"


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_verdict_classifies_the_three_science_outcomes_correctly():
    real = report.real_reference()
    # The weaker real screen is PANC1: +0.161 at 0.40 and +0.237 at 0.70, so the
    # artifact thresholds are +0.081 and +0.119.
    big = report.verdict(_fake_rows(0.22, 0.60), real, tag="primary")
    assert big["verdict"] == "ESTIMATOR ARTIFACT REPRODUCES RESULT", big["criteria"]

    mid = report.verdict(_fake_rows(0.06, 0.20), real, tag="primary")
    assert mid["verdict"] == "PARTIAL ESTIMATOR CONTRIBUTION", mid["criteria"]

    nil = report.verdict(_fake_rows(0.000, 0.08), real, tag="primary")
    assert nil["verdict"] == "LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT", nil["criteria"]
    # ... and in that case the real values must be strictly outside the null.
    for cell in nil["skill"].values():
        assert cell["real_A375"] > cell["null_max"]
        assert cell["real_PANC1"] > cell["null_max"]
        assert cell["pct_A375"] == 100.0
        assert cell["p_A375"] == pytest.approx(1 / 21)


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_an_incomplete_ensemble_cannot_produce_a_confident_verdict():
    """The most dangerous bug this experiment could have.

    A partial file used to return "little evidence for estimator artifact"
    without complaint, because the failure fraction was computed over the rows
    that were present. The planned count now comes from the grids.
    """
    real = report.real_reference()
    v = report.verdict(_fake_rows(0.0, 0.08, n=3), real, tag="primary")
    assert v["verdict"] == "INCONCLUSIVE RECONSTRUCTION"
    assert any("of 20 planned" in r for r in v["criterion_D_reasons"])


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_a_shut_selector_gate_is_inconclusive_not_negative():
    """The measure is lambda_AB * (...). A closed gate zeroes the artifact out.

    A null in which the posterior selector collapses is a world where the
    artifact has no channel to express through, so finding none there says
    nothing about the estimator. The real deposit runs at mean |lambda| ~ 0.47.
    """
    real = report.real_reference()
    shut = _fake_rows(0.0, 0.08)
    for r in shut:
        r["diagnostics"]["selector_on_fraction"] = 0.01
    v = report.verdict(shut, real, tag="primary")
    assert v["verdict"] == "INCONCLUSIVE RECONSTRUCTION"
    assert any("selector is shut" in r for r in v["criterion_D_reasons"])


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_an_unreproducible_null_matrix_is_inconclusive_not_negative():
    real = report.real_reference()
    noisy = _fake_rows(0.0, 0.08)
    for r in noisy:
        r["diagnostics"]["split_half_pearson_D"] = 0.1
    v = report.verdict(noisy, real, tag="primary")
    assert v["verdict"] == "INCONCLUSIVE RECONSTRUCTION"
    assert any("chain halves" in r for r in v["criterion_D_reasons"])


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_a_crushed_null_is_little_evidence_not_partial_contribution():
    """PARTIAL must not be the catch-all.

    A null whose skill is clearly negative at both coverages is the artifact
    hypothesis refuted, not a small artifact. The first draft's `else` branch
    labelled it "partial estimator contribution".
    """
    real = report.real_reference()
    v = report.verdict(_fake_rows(-0.30, 0.40), real, tag="primary")
    assert v["criteria"]["null_skill_crushed"]
    assert v["verdict"] == "LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT"


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_a_real_value_inside_the_null_interval_is_sufficient_for_artifact():
    """The independently-sufficient clause of criterion A."""
    real = report.real_reference()
    # A null centred well below the real values but with a wide enough spread to
    # swallow PANC1's +0.237 at coverage 0.70. Note the spread needed: this
    # clause fires only when the null ensemble is genuinely wide enough to
    # contain the real result at the coverage it is registered at.
    v = report.verdict(_fake_rows(0.02, 0.08, spread=0.60), real, tag="primary")
    assert v["criteria"]["real_inside_null_95"]
    assert v["verdict"] == "ESTIMATOR ARTIFACT REPRODUCES RESULT"


# --------------------------------------------------------------------------
# 10. The document is generated, not transcribed
# --------------------------------------------------------------------------

DOC = ROOT / "docs" / "dchain_null_falsification.md"
DOC_TABLES = ROOT / "results" / "dchain_null" / "summary" / "doc_tables.md"


@pytest.mark.skipif(not DOC_TABLES.exists(), reason="needs the generated tables")
def test_document_tables_are_generated_not_transcribed():
    """Every table in the falsification document must appear verbatim.

    The Phase 2R audit found four hand-copied p-values in its own write-up, two
    carried over from a different section and two matching no run in the
    repository. A number a human retypes is a number nobody checks. Each block
    emitted by ``scripts/report_dchain_null.py`` into ``doc_tables.md`` must
    appear character for character in the document, or this fails.
    """
    import re
    doc = DOC.read_text()
    if "*Filled in by" in doc:
        pytest.skip("the document's results sections are not written yet")
    # Slice between the generated markers rather than splitting on a guessed
    # separator: an earlier version split on "\n\n<!-- generated:" and rebuilt
    # the marker without its space, so every block after the first was compared
    # against a string the document could not contain.
    text = DOC_TABLES.read_text()
    marks = [(m.start(), m.group(0))
             for m in re.finditer(r"<!-- generated: [a-z_]+ -->", text)]
    assert marks, "doc_tables.md contains no generated blocks"
    bounds = [s for s, _ in marks] + [len(text)]
    blocks = [text[bounds[k]:bounds[k + 1]].rstrip() for k in range(len(marks))]
    missing = [b.splitlines()[0] for b in blocks if b not in doc]
    assert not missing, (
        "these generated blocks are not in the document verbatim: "
        f"{missing}. Regenerate with scripts/report_dchain_null.py and paste "
        f"them in rather than retyping them.")


@pytest.mark.skipif(not (ROOT / "results" / "dchain_null" / "summary"
                         / "verdict.json").exists(),
                    reason="needs a completed primary ensemble")
def test_the_document_states_the_verdict_the_rule_computed():
    """The written verdict must be the one report.verdict() returned."""
    v = json.loads((ROOT / "results" / "dchain_null" / "summary"
                    / "verdict.json").read_text())
    doc = DOC.read_text()
    assert v["verdict"] in doc, (
        f"the decision rule returned {v['verdict']!r} and the document does not "
        f"say so")
    log = (ROOT / "docs" / "PREREGISTRATIONS.md").read_text()
    assert v["verdict"] in log


@pytest.mark.skipif(not (ROOT / "results" / "dchain_null" / "summary"
                         / "verdict.json").exists(),
                    reason="needs a completed primary ensemble")
def test_the_reported_ensemble_is_the_preregistered_one_or_says_otherwise():
    """If the ensemble ran smaller than planned, the document must say so."""
    v = json.loads((ROOT / "results" / "dchain_null" / "summary"
                    / "verdict.json").read_text())
    planned = grids.part_counts()["primary"]
    if v["n_runs"] < planned:
        doc = DOC.read_text()
        assert str(v["n_runs"]) in doc and str(planned) in doc, (
            f"{v['n_runs']} of {planned} planned runs completed and the "
            f"document does not report both numbers")


def test_a_verdict_file_cannot_outlive_the_metrics_it_came_from():
    """A stale verdict.json is a result nothing downstream can invalidate.

    Dry-running the report on fabricated rows left a verdict.json and a
    comparison.csv behind, and they were committed. The report now deletes both
    when there is no primary ensemble to compute them from, and this pins that.
    """
    src = (ROOT / "scripts" / "report_dchain_null.py").read_text()
    assert 'unlink(missing_ok=True)' in src
    assert '"comparison.csv", "verdict.json"' in src
    summary = ROOT / "results" / "dchain_null" / "summary"
    if (summary / "verdict.json").exists():
        v = json.loads((summary / "verdict.json").read_text())
        rows = [json.loads(l) for l
                in open(ROOT / "results" / "dchain_null" / "metrics.jsonl")]
        present = len([r for r in rows if r.get("tag") == v["tag"]])
        assert v["n_runs"] == present, (
            f"verdict.json reports {v['n_runs']} runs but metrics.jsonl holds "
            f"{present} for tag {v['tag']!r}; the verdict is stale")


def test_incomplete_runs_are_excluded_from_the_null_distribution():
    """The preregistered exclusion rule, which was registered and not implemented.

    "A run is excluded only if the sampler exits nonzero or if
    n_samples != n_samples_expected." The first version counted incomplete runs
    and then used them anyway. An independent reviewer showed the cost: two
    truncated runs out of twenty -- 10%, inside the 20% failure allowance, so
    criterion D stays silent -- drag the null's 97.5th percentile up far enough
    that a real value falls inside the null interval, and the verdict flips from
    "little evidence for estimator artifact" to "estimator artifact reproduces
    result". This is that scenario.
    """
    real = report.real_reference() if _HAVE_RAW else None
    if real is None:
        pytest.skip("needs the Phase 2R artifacts")
    clean = _fake_rows(0.000, 0.08, n=18)
    truncated = _fake_rows(0.900, 0.08, n=2)
    for k, r in enumerate(truncated):
        r["sim_seed"] = 100 + k
        r["diagnostics"]["n_samples"] = 17          # != n_samples_expected
    assert all(report.is_usable(r) for r in clean)
    assert not any(report.is_usable(r) for r in truncated)

    v = report.verdict(clean + truncated, real, tag="primary")
    assert v["n_runs"] == 18 and v["n_incomplete"] == 2
    assert v["skill"]["0.70"]["null_max"] < 0.02, (
        "an excluded run's skill is still in the null distribution")
    assert v["verdict"] == "LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT"


@pytest.mark.skipif(not _HAVE_RAW, reason="needs the Phase 2R artifacts")
def test_the_inside_null_clause_is_evaluated_only_where_it_was_registered():
    """"s_real(0.70) lies inside the null interval" -- 0.70, not every coverage.

    It is a sufficient condition on its own, and a 95% interval from 20 draws is
    essentially min-to-max, so it is the most outlier-sensitive clause in the
    rule. Evaluating it at both coverages doubled its chances.
    """
    real = report.real_reference()
    # At this spread the null interval contains PANC1's +0.161 at coverage 0.40
    # but not its +0.237 at 0.70. The registered clause is the 0.70 one, so the
    # verdict must not be driven by the 0.40 cell.
    v = report.verdict(_fake_rows(0.02, 0.08, spread=0.25), real, tag="primary")
    assert v["skill"]["0.40"]["real_inside_null_95"] is False
    assert v["skill"]["0.70"]["real_inside_null_95"] is False
    assert not v["criteria"]["real_inside_null_95"]
    wide = report.verdict(_fake_rows(0.02, 0.08, spread=0.60), real, tag="primary")
    assert wide["skill"]["0.70"]["real_inside_null_95"] is True


# --------------------------------------------------------------------------
# 11. The runner can actually start every part it advertises
# --------------------------------------------------------------------------

def test_every_advertised_part_reaches_the_worker_pool(tmp_path):
    """A runner that cannot parse its own arguments is worse than a slow one.

    Adding comma-separated parts moved ``specs = grids.part_jobs(...)`` below its
    first use, so **every** invocation died on an UnboundLocalError before doing
    any work. The failure was invisible: the follow-on ensemble was launched
    detached, both of its commands raised immediately, and the job sat dead for
    four hours looking exactly like a job that was running.

    Nothing caught it. The unit tests exercised ``grids.part_jobs`` but never
    ``main``; the CI pipeline check would have caught it, but CI had not run
    since the change. So the argument path is now a unit test: for every part the
    runner advertises, ``main()`` must get as far as constructing the worker
    pool. The pool is stubbed, so nothing executes.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_runner", ROOT / "scripts" / "run_dchain_null.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class ReachedPool(Exception):
        pass

    def stub(*a, **k):
        raise ReachedPool()

    runner.ProcessPoolExecutor = stub

    # The runner legitimately refuses to start when the compiled sampler is
    # absent, and that binary is fetched and built by prepare_dchain_null.py --
    # so it does not exist on a clean checkout, and this test failed in CI for
    # that reason while passing on the machine that built it. Skipping the test
    # there would have retired the one check that covers the argument path,
    # which is what it exists for. Pointing --dchain-dir at a stub keeps the
    # precondition satisfied without needing 1,045 lines of C++.
    stub_binary = tmp_path / "build" / "dchain"
    stub_binary.parent.mkdir(parents=True)
    stub_binary.write_text("")

    parts = list(grids.ALL_PARTS) + ["all", "smoke", "convergence,realism,noise"]
    argv = sys.argv
    failures = {}
    try:
        for part in parts:
            sys.argv = ["run_dchain_null.py", "--part", part, "--workers", "1",
                        "--dchain-dir", str(tmp_path)]
            try:
                rc = runner.main()
                failures[part] = f"returned {rc} without reaching the pool"
            except ReachedPool:
                pass
            except Exception as e:                       # noqa: BLE001
                failures[part] = f"{type(e).__name__}: {e}"
    finally:
        sys.argv = argv
    assert not failures, f"the runner cannot start these parts: {failures}"


def test_limit_zero_does_not_silently_mean_no_limit():
    """``--limit 0`` is falsy. It ran the whole block instead of nothing.

    Found by using it as a dry-run flag and watching a 36-condition ensemble
    start. Either it truncates to nothing or it is rejected; silently meaning
    "no limit" is the one behaviour that costs compute.
    """
    src = (ROOT / "scripts" / "run_dchain_null.py").read_text()
    assert "if args.limit is not None:" in src, (
        "--limit still uses a truthiness test, so --limit 0 runs everything")
