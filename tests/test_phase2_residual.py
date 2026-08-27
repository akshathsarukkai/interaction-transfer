"""Invariants of the residual-directionality diagnostic.

Scoped to the ways this experiment could be quietly wrong, which are not the
same ways Phase 2 could be. The load-bearing risks here are:

* the additive baseline seeing a held-out pair, which would let every test pair
  shrink its own residual target toward zero;
* the sign of ``D_res``, which is the entire quantity under study;
* a rung that cannot learn -- a silently dead model returns exactly the null
  result this experiment is testing for, so "no signal" and "no gradient" have
  to be told apart by a test rather than by a plot;
* aggregating pairs or orientations as if they were independent replicates;
* pooling these rows with the Phase 2 rows, which measure a different target
  against a different null under the same column names.

Everything runs on the 12-drug fixture in ``tests/fixtures/koplev_tiny`` unless
marked otherwise, so CI needs no download.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.residual import (RIDGE_LAMBDAS, AdditiveFit,
                                                     assert_rows_are_train_only,
                                                     decomposition,
                                                     directional_pairs,
                                                     fit_additive,
                                                     ordered_residuals,
                                                     residual_targets)
from intervention_algebra.real_data.residual_experiment import (
    ResidualConfig, inject_antisymmetric, residual_metrics,
    run_residual_condition)
from intervention_algebra.real_data.residual_models import (
    HPARAM_GRID, LADDER, LADDER_ORDER, ResidualModelConfig,
    build_residual_model, split_hparams)
from intervention_algebra.real_data.residual_report import (
    assert_residual_grid_complete, by_split_seed, decomposition_table,
    load_residual_runs, skill_summary)
from intervention_algebra.real_data.residual_train import (SHRINKAGE,
                                                           select_shrinkage)
from intervention_algebra.real_data import residual_sweep as rs
from intervention_algebra.real_data.splits import make_coverage_splits

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "koplev_tiny"
COVERAGES = (0.30, 0.50, 0.70)


@pytest.fixture(scope="module")
def screen():
    return koplev.load_screen("A375", FIXTURE)


@pytest.fixture(scope="module")
def split(screen):
    return make_coverage_splits(screen.frame, screen.n_drugs, COVERAGES,
                                split_seed=0, min_train_degree=1,
                                min_eligible_test_pairs=5)[0.50]


@pytest.fixture(scope="module")
def fit(split, screen):
    return fit_additive(split, screen.frame, screen.n_drugs)


# --------------------------------------------------------------- leakage
def test_additive_fit_uses_training_pairs_only(split, screen, fit):
    """The design matrix must contain no validation or test pair, either way up."""
    train = split.rows(screen.frame, "train")
    assert fit.n_fit_rows == len(train)
    assert fit.n_fit_pairs == len(split.train_pairs)
    seen = set(zip(train["i"], train["j"]))
    for a, b in list(split.test_pairs) + list(split.val_pairs):
        assert (a, b) not in seen and (b, a) not in seen


def test_leakage_guard_fires_on_a_contaminated_frame(split, screen):
    """Hand the guard the thing it exists to catch, and it must refuse.

    This is the mutation test for the guard: if ``assert_rows_are_train_only``
    were weakened to a no-op -- or moved out of ``fit_additive`` into a caller
    that could forget it -- this test fails and nothing else in the suite would.
    """
    everything = screen.frame
    with pytest.raises(AssertionError, match="TEST pairs"):
        assert_rows_are_train_only(everything, split)

    # The guard is called from inside fit_additive, so the only way past it is
    # the explicit control-C flag -- not a doctored split, which would make the
    # guard *look* satisfied while leaking exactly as much.
    contaminated = fit_additive(split, screen.frame, screen.n_drugs,
                                _contaminate=True)
    clean = fit_additive(split, screen.frame, screen.n_drugs)
    assert not np.allclose(contaminated.g, clean.g), (
        "fitting on the held-out pairs changed nothing, which would mean the "
        "poisoned split was not actually used")


def test_perturbing_held_out_responses_cannot_move_the_additive_fit(split, screen):
    """The strongest form of the claim: change the test rows' ``y`` arbitrarily.

    If a single coefficient moves, some held-out row reached the solver.
    """
    base = fit_additive(split, screen.frame, screen.n_drugs)
    frame = screen.frame.copy()
    test_mask = frame["pair"].isin(set(split.test_pairs)).to_numpy()
    assert test_mask.sum() > 0
    frame.loc[test_mask, "y"] = frame.loc[test_mask, "y"].to_numpy() * 100.0 + 7.0
    moved = fit_additive(split, frame, screen.n_drugs)
    assert moved.lam == base.lam
    np.testing.assert_allclose(moved.a, base.a, atol=0, rtol=0)
    np.testing.assert_allclose(moved.b, base.b, atol=0, rtol=0)
    assert moved.mu == base.mu


def test_ridge_penalty_is_selected_on_validation_not_test(split, screen):
    """Perturbing validation rows moves the chosen penalty; test rows never do."""
    frame = screen.frame.copy()
    val_mask = frame["pair"].isin(set(split.val_pairs)).to_numpy()
    frame.loc[val_mask, "y"] = frame.loc[val_mask, "y"].to_numpy() + 50.0
    perturbed = fit_additive(split, frame, screen.n_drugs)
    base = fit_additive(split, screen.frame, screen.n_drugs)
    assert perturbed.val_score != base.val_score


# --------------------------------------------------- identifiability / sign
def test_D_add_is_gauge_invariant(fit, screen):
    """``(a+c, b-c)`` is the same model; ``g_i-g_j`` must not notice."""
    i = np.array([0, 1, 2, 3])
    j = np.array([4, 5, 6, 7])
    shifted = AdditiveFit(mu=fit.mu, a=fit.a + 3.5, b=fit.b - 3.5, lam=fit.lam,
                          objective=fit.objective, val_score=fit.val_score,
                          n_fit_rows=fit.n_fit_rows, n_fit_pairs=fit.n_fit_pairs)
    np.testing.assert_allclose(shifted.d_add(i, j), fit.d_add(i, j), atol=1e-12)
    # ...and the ordered prediction is unchanged too, since mu absorbs nothing
    # here: a_i + b_j is invariant to the same shift.
    np.testing.assert_allclose(shifted.predict(i, j), fit.predict(i, j), atol=1e-12)


def test_residual_target_is_antisymmetric_and_deterministic(split, screen, fit):
    test_rows = split.rows(screen.frame, "test")
    d = residual_targets(test_rows, fit)
    assert (d["i"].to_numpy() < d["j"].to_numpy()).all(), "not canonical orientation"

    # Swapping which schedule each measurement belongs to must negate D_true
    # exactly. That is the whole sign convention: the target is a difference
    # between two orientations, so it has exactly one degree of freedom.
    swapped = test_rows.copy()
    swapped["i"] = test_rows["j"].to_numpy()
    swapped["j"] = test_rows["i"].to_numpy()
    d1 = d.set_index(["i", "j"]).sort_index()
    d2 = residual_targets(swapped, fit).set_index(["i", "j"]).sort_index()
    assert list(d1.index) == list(d2.index)
    np.testing.assert_allclose(d2["D_true"].to_numpy(), -d1["D_true"].to_numpy(),
                               atol=1e-12)

    # And ``D_res`` evaluated in the reversed orientation is the negation of the
    # forward one -- D_res(j,i) = D(j,i) - (g_j - g_i) = -D_res(i,j) -- which is
    # what makes every rung's enforced antisymmetry arithmetic rather than a
    # prior. Checked against the fit directly, not against a re-derived frame.
    i, j = d["i"].to_numpy(), d["j"].to_numpy()
    fwd = d["D_true"].to_numpy() - fit.d_add(i, j)
    rev = (-d["D_true"].to_numpy()) - fit.d_add(j, i)
    np.testing.assert_allclose(rev, -fwd, atol=1e-12)

    again = residual_targets(test_rows, fit)
    np.testing.assert_allclose(again["D_res"].to_numpy(), d["D_res"].to_numpy(),
                               atol=0, rtol=0)


def test_ordered_and_direct_residual_formulations_share_a_target(split, screen, fit):
    """``r_ij - r_ji == D_res(i, j)``. The two rungs differ in loss, not target."""
    rows = split.rows(screen.frame, "test")
    o = ordered_residuals(rows, fit).set_index(["i", "j"])["r"]
    d = residual_targets(rows, fit)
    lhs = np.array([o.loc[(a, b)] - o.loc[(b, a)]
                    for a, b in zip(d["i"], d["j"])])
    np.testing.assert_allclose(lhs, d["D_res"].to_numpy(), atol=1e-10)


def test_both_orientations_of_every_pair_stay_in_one_split(split, screen):
    for which in ("train", "val", "test"):
        rows = split.rows(screen.frame, which)
        seen = set(zip(rows["i"], rows["j"]))
        for a, b in seen:
            assert (b, a) in seen, f"{which} holds only one orientation of ({a},{b})"


def test_directional_pairs_refuses_a_half_observed_pair(split, screen):
    rows = split.rows(screen.frame, "test")
    broken = rows.iloc[1:].reset_index(drop=True)
    with pytest.raises(AssertionError, match="not complete in both orientations"):
        directional_pairs(broken)


# ----------------------------------------------------------------- models
@pytest.mark.parametrize("rung", LADDER_ORDER)
def test_every_rung_is_exactly_antisymmetric(rung):
    cfg = ResidualModelConfig(n_drugs=12, seed=3)
    m = build_residual_model(rung, cfg)
    # Push the parameters away from their initialisation so the property is not
    # holding only because the head starts at zero.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    i = torch.tensor([0, 1, 2, 3, 4])
    j = torch.tensor([5, 6, 7, 8, 9])
    with torch.no_grad():
        f, r = m.d_res(i, j), m.d_res(j, i)
    assert torch.allclose(f, -r, atol=1e-6)


def test_lowrank_has_nonzero_gradients_at_init():
    """The dead-model regression.

    An earlier ``LowRank`` had both a zero-initialised ``W`` *and* a
    zero-initialised multiplicative ``scale``. Every partial derivative
    vanished, the rung emitted exactly 0.0 forever, and the results table showed
    skill = +0.0000 in every cell -- which is precisely the null this experiment
    is testing for. A silently dead model would have produced the headline
    conclusion by construction.
    """
    m = build_residual_model("lowrank", ResidualModelConfig(n_drugs=12, rank=4))
    i, j = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    loss = ((m.d_res(i, j) - torch.tensor([1.0, -1.0, 0.5])) ** 2).mean()
    loss.backward()
    total = sum(float(p.grad.abs().sum()) for p in m.parameters()
                if p.grad is not None)
    assert total > 1e-6, "no parameter receives gradient; the rung cannot learn"


def test_lowrank_can_express_a_pure_potential():
    """``c_i - c_j`` is inside the low-rank hypothesis class.

    Worth pinning because it is the reason the ``potential`` rung has to be
    reported next to ``lowrank``: a low-rank win is *not* by itself evidence of
    pair-specific structure, since the same family can represent leftover
    per-drug potential exactly.
    """
    c = torch.tensor([0.4, -1.2, 0.7, 2.0])
    m = build_residual_model("lowrank", ResidualModelConfig(n_drugs=4, rank=2))
    with torch.no_grad():
        m.u.weight.copy_(torch.stack([c, torch.ones(4)], dim=1))
        m.W.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))   # K = [[0,1],[-1,0]]
    i, j = torch.tensor([0, 1, 2]), torch.tensor([3, 2, 1])
    with torch.no_grad():
        got = m.d_res(i, j)
    want = c[i] - c[j]
    assert torch.allclose(got, want, atol=1e-6)


def test_structured_A_head_equals_the_mlp_rung():
    """The Phase 2 structured family is not a separate hypothesis on this target.

    Its antisymmetric component is ``first_order_A + [F_A(phi_ij) -
    F_A(phi_ji)]/2``. With the first-order potential already removed by the
    residualisation and the factor of two folded into the output layer, that is
    the ``mlp`` rung exactly -- so running it as a sixth rung would report one
    hypothesis class twice.
    """
    from intervention_algebra.real_data.models import RealModelConfig, build_model

    cfg = ResidualModelConfig(n_drugs=12, emb_dim=8, hidden=16, seed=0)
    rung = build_residual_model("mlp", cfg)
    structured = build_model("structured", RealModelConfig(
        n_drugs=12, emb_dim=8, pair_hidden=16, seed=0))
    with torch.no_grad():
        structured.emb.weight.copy_(rung.emb.weight)
        for ps, pr in zip(structured.F_A.parameters(), rung.F.parameters()):
            ps.copy_(pr)
        # A's pair contribution is (F_A(ij) - F_A(ji))/2; the rung's is the
        # undivided difference, so match the scale on the output layer.
        structured.F_A[-1].weight.mul_(2.0)
        structured.F_A[-1].bias.mul_(2.0)
        structured.first_a.weight.zero_()
        structured.first_b.weight.zero_()
        i, j = torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])
        got = rung.d_res(i, j)
        want = 2.0 * structured.pair_head_A(i, j)      # pair_head_A halves it
    assert torch.allclose(got, want, atol=1e-5)


def test_hparam_grid_keys_are_all_consumed():
    """Every grid key is either architecture or optimiser; none is dropped.

    A typo'd key would otherwise be silently ignored and the whole grid would
    collapse onto one setting, which reads in the results as "tuning changes
    nothing".
    """
    from intervention_algebra.real_data.residual_train import ResidualTrainConfig
    arch_fields = set(ResidualModelConfig().__dict__)
    opt_fields = set(ResidualTrainConfig().__dict__)
    for rung, grid in HPARAM_GRID.items():
        assert rung in LADDER
        for h in grid:
            arch, opt = split_hparams(h)
            assert set(arch) <= arch_fields, (rung, arch)
            assert set(opt) <= opt_fields, (rung, opt)


# ---------------------------------------------------------------- metrics
def test_zero_predictor_has_exactly_zero_skill():
    d = np.array([0.3, -0.1, 0.5, -0.7, 0.2, 0.0, 1.1, -0.4, 0.9, -1.3, 0.15])
    m = residual_metrics(d, np.zeros_like(d), threshold=0.05)
    assert m["skill"] == 0.0
    assert m["mse"] == m["mse_zero"]
    assert np.isnan(m["pearson"]), "a constant predictor has no correlation"


def test_skill_is_one_for_a_perfect_predictor_and_negative_for_a_harmful_one():
    d = np.array([0.3, -0.1, 0.5, -0.7, 0.2, 0.4, 1.1, -0.4, 0.9, -1.3, 0.15])
    assert residual_metrics(d, d, 0.05)["skill"] == pytest.approx(1.0)
    assert residual_metrics(d, -d, 0.05)["skill"] == pytest.approx(-3.0)


def test_shrinkage_can_only_be_chosen_on_validation_and_zero_recovers_the_null():
    true = np.array([1.0, -1.0, 0.5, -0.5])
    assert select_shrinkage(true, np.zeros_like(true)) == 0.0
    assert select_shrinkage(true, true) == 1.0
    # Pure noise in the prediction: shrinkage must pull toward the null rather
    # than leave the harmful prediction at full strength.
    rng = np.random.default_rng(0)
    noise = rng.normal(size=200) * 3.0
    assert select_shrinkage(rng.normal(size=200), noise) < 0.5
    assert 0.0 in SHRINKAGE


# --------------------------------------------------------------- controls
def test_injected_signal_is_purely_antisymmetric(screen):
    frame, S = inject_antisymmetric(screen.frame, screen.n_drugs, kappa=0.3,
                                    rank=3, seed=0)
    d0 = directional_pairs(screen.frame)
    d1 = directional_pairs(frame)
    added = d1["D_true"].to_numpy() - d0["D_true"].to_numpy()
    np.testing.assert_allclose(
        added, 0.3 * S[d1["i"].to_numpy(), d1["j"].to_numpy()], atol=1e-10)
    # The symmetric half is untouched, so the injection cannot show up as a
    # change in the response's magnitude.
    sym0 = 0.5 * (d0["y_f"].to_numpy() + d0["y_r"].to_numpy())
    sym1 = 0.5 * (d1["y_f"].to_numpy() + d1["y_r"].to_numpy())
    np.testing.assert_allclose(sym0, sym1, atol=1e-10)


def test_permutation_control_destroys_the_pair_association(split, screen, fit):
    from intervention_algebra.real_data.residual_experiment import _permute_direction

    pairs = residual_targets(split.rows(screen.frame, "train"), fit)
    shuffled = _permute_direction(pairs, seed=1)
    # Same multiset of residual effects...
    np.testing.assert_allclose(np.sort(shuffled["D_res"].to_numpy()),
                               np.sort(pairs["D_res"].to_numpy()), atol=1e-12)
    # ...attached to different pairs...
    assert not np.allclose(shuffled["D_res"].to_numpy(),
                           pairs["D_res"].to_numpy())
    # ...and the ordered rows still agree with the permuted direction, so the
    # ordered rung is trained on the same control as the direct rungs.
    np.testing.assert_allclose(
        shuffled["y_f"].to_numpy() - shuffled["y_r"].to_numpy(),
        shuffled["D_true"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(
        shuffled["y_f"].to_numpy() + shuffled["y_r"].to_numpy(),
        pairs["y_f"].to_numpy() + pairs["y_r"].to_numpy(), atol=1e-10)


# ------------------------------------------------------------- aggregation
def _fake_runs(n_seeds: int = 4) -> pd.DataFrame:
    rows = []
    for screen in ("A375", "PANC1"):
        for cov in (0.10, 0.70):
            for rung in LADDER_ORDER:
                for ss in range(n_seeds):
                    rows.append({"tag": "main", "screen": screen, "coverage": cov,
                                 "rung": rung, "split_seed": ss, "init_seed": 0,
                                 "cal_skill": 0.0 if rung == "zero" else 0.05 * ss,
                                 "heldout_skill": 0.0 if rung == "zero" else 0.04 * ss,
                                 "dec_D_mean_square": 0.05,
                                 "dec_D_res_mean_square": 0.03,
                                 "dec_frac_D_removed_by_potential": 0.4})
    df = pd.DataFrame(rows)
    df["rung"] = pd.Categorical(df["rung"], categories=list(LADDER_ORDER),
                                ordered=True)
    return df


def test_aggregation_uses_the_split_seed_as_the_unit():
    df = _fake_runs()
    s = by_split_seed(df, "cal_skill")
    assert len(s) == len(df), "one row per (screen, coverage, rung, split seed)"
    summ = skill_summary(df, "cal_skill")
    assert (summ["n_split_seeds"] == 4).all(), (
        "n must be the number of split seeds, never the number of pairs")
    # The zero rung is degenerate against a one-sample test, not significant.
    z = summ[summ["rung"] == "zero"]
    assert z["mean"].abs().max() == 0.0
    assert z["p_ttest"].isna().all()


def test_decomposition_table_counts_each_split_seed_once():
    df = _fake_runs()
    dec = decomposition_table(df)
    assert (dec["n_split_seeds"] == 4).all(), (
        "reading the decomposition off all five rungs would report the same "
        "numbers with a fivefold sample size")


def test_grid_completeness_check_catches_a_missing_seed():
    df = _fake_runs()
    assert_residual_grid_complete(df)
    holed = df.drop(df.index[3]).reset_index(drop=True)
    with pytest.raises(AssertionError, match="incomplete cells"):
        assert_residual_grid_complete(holed)


def test_residual_rows_cannot_be_loaded_as_or_pooled_with_phase2_rows(tmp_path):
    """The two experiments share a split and share nothing else."""
    p = tmp_path / "phase2ish.jsonl"
    p.write_text(json.dumps({
        "tag": "main", "screen": "A375", "coverage": 0.1, "family": "structured",
        "split_seed": 0, "test_D_pearson": 0.5, "test_head_A_over_sym": 1.0}) + "\n")
    with pytest.raises(ValueError, match="Phase 2 columns"):
        load_residual_runs(p)

    q = tmp_path / "notresidual.jsonl"
    q.write_text(json.dumps({"tag": "main", "screen": "A375", "coverage": 0.1,
                             "split_seed": 0, "cal_skill": 0.1}) + "\n")
    with pytest.raises(ValueError, match="no 'rung' column"):
        load_residual_runs(q)


def test_load_refuses_a_file_containing_a_failed_run(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps({"tag": "main", "rung": "zero", "screen": "A375",
                             "coverage": 0.1, "split_seed": 0,
                             "error": "boom"}) + "\n")
    with pytest.raises(RuntimeError, match="failed runs"):
        load_residual_runs(p)


# ------------------------------------------------------- end-to-end, tiny
def test_end_to_end_condition_runs_and_is_reproducible(screen):
    cfg = ResidualConfig(screen="A375", coverage=0.50, rung="lowrank",
                         split_seed=0, coverages=COVERAGES, min_train_degree=1,
                         min_eligible_test_pairs=5,
                         train=replace(ResidualConfig().train, max_epochs=40,
                                       n_restarts=1))
    a = run_residual_condition(cfg, FIXTURE, screen=screen)
    b = run_residual_condition(cfg, FIXTURE, screen=screen)
    assert a["heldout_skill"] == b["heldout_skill"]
    assert a["cal_skill"] == b["cal_skill"]
    assert a["n_test_pairs"] == b["n_test_pairs"]
    assert a["dec_D_res_mean_square"] > 0


def test_decomposition_is_rung_invariant(screen):
    """The additive fit does not depend on which rung is being scored.

    ``decomposition_table`` reads the decomposition off the ``zero`` rows only,
    on the strength of this identity. If it failed, that filter would be
    reporting one rung's view of the data as everyone's.
    """
    base = ResidualConfig(coverage=0.50, coverages=COVERAGES, min_train_degree=1,
                          min_eligible_test_pairs=5,
                          train=replace(ResidualConfig().train, max_epochs=20,
                                        n_restarts=1))
    a = run_residual_condition(replace(base, rung="zero"), FIXTURE, screen=screen)
    b = run_residual_condition(replace(base, rung="mlp"), FIXTURE, screen=screen)
    for k in ("dec_D_mean_square", "dec_D_res_mean_square",
              "dec_frac_D_removed_by_potential", "dec_additive_lambda"):
        assert a[k] == b[k], k
    # ...which also means the zero rung's MSE_zero is the denominator every
    # other rung's skill is measured against, so skills are exactly paired.
    assert a["heldout_mse_zero"] == b["heldout_mse_zero"]


def test_contaminated_fit_is_flagged_and_inflates_the_baseline(screen):
    """Control C: the guard is real, and what it prevents is measurable."""
    base = ResidualConfig(coverage=0.50, rung="zero", coverages=COVERAGES,
                          min_train_degree=1, min_eligible_test_pairs=5)
    clean = run_residual_condition(base, FIXTURE, screen=screen)
    dirty = run_residual_condition(replace(base, contaminate_additive_fit=True),
                                   FIXTURE, screen=screen)
    assert clean["contaminated"] is False
    assert dirty["contaminated"] is True
    # Fitting the potential on the held-out pairs removes more of their
    # directional signal than a legitimate fit can.
    assert (dirty["dec_frac_D_removed_by_potential"]
            > clean["dec_frac_D_removed_by_potential"])


# ------------------------------------------- the additive estimator, checked
REAL_RAW = Path("data/raw/koplev2017")
_HAVE_REAL = (REAL_RAW / "Data Table 1.csv").exists()


@pytest.mark.skipif(not _HAVE_REAL,
                    reason="needs the Koplev deposit; run scripts/download_koplev.py")
def test_closed_form_additive_matches_trained_additive():
    """The change of estimator does not change the baseline.

    Phase 2's ``additive`` family is the same statistical model fitted by Adam
    under the shared training budget; ``fit_additive`` solves it in closed form
    so the residual target does not inherit an optimiser seed. Cited in
    ``residual.py``'s module docstring, so it has to exist -- an uncited claim in
    a docstring is a claim nobody checks.

    The two are compared on the quantity that matters -- the *directional*
    prediction ``g_i - g_j`` on held-out pairs -- and not on coefficients, which
    are identified only up to the gauge the ridge penalty happens to pick.

    This is the one test in the file that needs the real deposit, and the reason
    is not laziness. The 12-drug fixture leaves 6 training pairs at the
    coverage used here; the Adam fit is nowhere near converged on that, and the
    two estimators agree only at r ~= 0.68 -- which would test the optimiser's
    budget rather than the model. On the real screen they agree at r = 0.90
    (coverage 0.10) to r = 1.00 (coverage 0.70), with the directional standard
    deviations within 11% of each other. It is skipped in CI rather than
    weakened to something the fixture can pass.
    """
    from intervention_algebra.real_data.models import RealModelConfig, build_model
    from intervention_algebra.real_data.train import (TrainConfig, predict,
                                                      to_tensors, train_model)

    real = koplev.load_screen("A375", REAL_RAW)
    covs = (0.05, 0.10, 0.20, 0.40, 0.70)
    sp = make_coverage_splits(real.frame, real.n_drugs, covs, split_seed=0)[0.70]
    train_rows = sp.rows(real.frame, "train")
    val_rows = sp.rows(real.frame, "val")
    test_rows = sp.rows(real.frame, "test")
    y_mean = float(train_rows["y"].mean())
    y_std = float(train_rows["y"].std())
    fit = train_model(
        lambda seed: build_model("additive", RealModelConfig(
            n_drugs=real.n_drugs, seed=seed)),
        to_tensors(train_rows, y_mean, y_std),
        to_tensors(val_rows, y_mean, y_std),
        TrainConfig(n_restarts=1), seed=0)

    i = test_rows["i"].to_numpy()
    j = test_rows["j"].to_numpy()
    rev = test_rows.copy()
    rev["i"], rev["j"] = j, i
    trained_d = predict(fit.model, test_rows, y_mean, y_std) - predict(
        fit.model, rev, y_mean, y_std)
    closed_d = fit_additive(sp, real.frame, real.n_drugs).d_add(i, j)

    r = float(np.corrcoef(trained_d, closed_d)[0, 1])
    assert r > 0.95, f"the two additive estimators disagree on direction (r={r:.3f})"
    ratio = float(np.std(closed_d) / np.std(trained_d))
    assert 0.8 < ratio < 1.25, f"directional scales differ by {ratio:.2f}x"


# ------------------------------------------------ Hodge / curl decomposition
def test_hodge_decomposition_is_exact_and_orthogonal(screen):
    from intervention_algebra.real_data.residual import hodge_decomposition

    h = hodge_decomposition(screen.frame, screen.n_drugs)
    assert h["grad_fraction"] + h["curl_fraction"] == pytest.approx(1.0, abs=1e-9)
    assert h["grad_curl_inner_product"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= h["curl_fraction"] <= 1.0
    e = h["curl_rank_energy"]
    assert e["1"] <= e["2"] <= e["4"] <= e["8"] <= e["16"] <= 1.0 + 1e-9


def test_a_pure_potential_has_zero_curl(screen):
    """The decomposition's whole meaning: a per-drug tendency is curl-free."""
    from intervention_algebra.real_data.residual import hodge_decomposition

    rng = np.random.default_rng(0)
    g = rng.normal(size=screen.n_drugs)
    frame = screen.frame.copy()
    # y(i->j) = g_i/2 - g_j/2 gives D = g_i - g_j exactly and nothing else.
    frame["y"] = 0.5 * (g[frame["i"].to_numpy()] - g[frame["j"].to_numpy()])
    h = hodge_decomposition(frame, screen.n_drugs)
    assert h["curl_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert h["grad_fraction"] == pytest.approx(1.0, abs=1e-12)


def test_a_low_rank_antisymmetric_signal_is_almost_all_curl(screen):
    from intervention_algebra.real_data.residual import hodge_decomposition

    rng = np.random.default_rng(1)
    n = screen.n_drugs
    u = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, 3))
    S = u @ (W - W.T) @ u.T
    frame = screen.frame.copy()
    frame["y"] = 0.5 * S[frame["i"].to_numpy(), frame["j"].to_numpy()]
    h = hodge_decomposition(frame, n)
    assert h["curl_fraction"] > 0.8, (
        "a bilinear antisymmetric form should be dominated by the cyclic part; "
        "if it is not, the gradient projection is wrong")


def test_hodge_refuses_an_incomplete_matrix(screen):
    from intervention_algebra.real_data.residual import hodge_decomposition

    with pytest.raises(ValueError, match="missing"):
        hodge_decomposition(screen.frame.iloc[2:], screen.n_drugs)


# ------------------------------------------------- the calibration-set split
def test_calibration_split_is_disjoint_and_deterministic():
    from intervention_algebra.real_data.residual_train import split_calibration_pairs

    a, b = split_calibration_pairs(37, seed=5)
    a2, b2 = split_calibration_pairs(37, seed=5)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    assert not set(a.tolist()) & set(b.tolist())
    assert sorted(a.tolist() + b.tolist()) == list(range(37))
    assert len(a) == 18 and len(b) == 19


def test_split_calibration_changes_alpha_but_not_the_evaluation(screen):
    """The corrected shrinkage uses different *validation* pairs, not test pairs.

    The number of scored held-out pairs, the residual target and the skill
    denominator must all be untouched -- otherwise the corrected block would not
    be comparable with the block it corrects.
    """
    base = ResidualConfig(coverage=0.50, rung="lowrank", coverages=COVERAGES,
                          min_train_degree=1, min_eligible_test_pairs=5,
                          train=replace(ResidualConfig().train, max_epochs=60,
                                        n_restarts=1))
    a = run_residual_condition(base, FIXTURE, screen=screen)
    b = run_residual_condition(replace(base, split_validation_for_calibration=True),
                               FIXTURE, screen=screen)
    assert a["n_test_pairs"] == b["n_test_pairs"]
    assert a["heldout_mse_zero"] == b["heldout_mse_zero"]
    assert a["dec_D_res_mean_square"] == b["dec_D_res_mean_square"]
    assert b["n_val_pairs_calibration"] < a["n_val_pairs_calibration"]
    assert (b["n_val_pairs_selection"] + b["n_val_pairs_calibration"]
            == a["n_val_pairs_selection"])


def test_forced_hparams_bypass_the_grid_entirely():
    from intervention_algebra.real_data.residual_sweep import (RANK2_HPARAMS,
                                                               rank2_grid)

    g = rank2_grid(screens=("A375",), coverages=(0.10,), split_seeds=(0,))
    assert len(g) == 1
    assert g[0].force_hparams == RANK2_HPARAMS
    assert g[0].tag == "rank2"


def test_rank2_run_uses_exactly_204_parameters(screen):
    """204 = 100 drugs x 2 latent dims + a 2x2 matrix. If the pin silently
    failed, the run would quietly report the searched grid's capacity under a
    tag that claims otherwise."""
    from intervention_algebra.real_data.residual_sweep import RANK2_HPARAMS

    cfg = ResidualConfig(coverage=0.50, rung="lowrank", coverages=COVERAGES,
                         min_train_degree=1, min_eligible_test_pairs=5,
                         force_hparams=RANK2_HPARAMS,
                         train=replace(ResidualConfig().train, max_epochs=30,
                                       n_restarts=1))
    row = run_residual_condition(cfg, FIXTURE, screen=screen)
    assert row["grid_size"] == 1
    assert row["hparams"]["rank"] == 2
    assert row["n_params"] == screen.n_drugs * 2 + 4


@pytest.mark.skipif(not _HAVE_REAL,
                    reason="needs the Koplev deposit; run scripts/download_koplev.py")
def test_heldout_residual_matches_the_exact_curl_at_dense_coverage():
    """The writeup calls two different constructions "the cyclic part".

    One is an exact in-sample Hodge projection of the complete matrix; the other
    is a train-only ridge residual on held-out pairs. Using one word for both is
    only honest if they coincide, so the agreement is pinned rather than
    asserted: r = 0.990 (A375) at coverage 0.70, falling to 0.908 at 0.10 where
    the additive fit is estimated from 421 pairs. The threshold here is set at
    the dense coverage, which is where the positive result lives and therefore
    where the identification has to hold.
    """
    from intervention_algebra.real_data.residual import fit_additive, residual_targets

    real = koplev.load_screen("A375", REAL_RAW)
    n = real.n_drugs
    y = np.full((n, n), np.nan)
    y[real.frame["i"].to_numpy(), real.frame["j"].to_numpy()] = real.frame["y"].to_numpy()
    d_full = y - y.T
    np.fill_diagonal(d_full, 0.0)
    g = d_full.mean(axis=1)
    curl = d_full - (g[:, None] - g[None, :])

    covs = (0.05, 0.10, 0.20, 0.40, 0.70)
    sp = make_coverage_splits(real.frame, n, covs, split_seed=0)[0.70]
    test_rows = sp.rows(real.frame, "test")
    d = residual_targets(test_rows, fit_additive(sp, real.frame, n))
    r = float(np.corrcoef(d["D_res"].to_numpy(),
                          curl[d["i"].to_numpy(), d["j"].to_numpy()])[0, 1])
    assert r > 0.97, (
        f"held-out D_res and the exact curl agree only at r={r:.3f}; the writeup "
        f"calls them the same thing")


# ------------------------------------------------- the writeup's own numbers
RESULTS = Path(__file__).resolve().parent.parent / "results" / "phase2_residual"
DOC = Path(__file__).resolve().parent.parent / "docs" / "phase2_residual_directionality.md"
_HAVE_RESULTS = (RESULTS / "summary" / "doc_tables.md").exists() and DOC.exists()


@pytest.mark.skipif(not _HAVE_RESULTS,
                    reason="needs a generated results/phase2_residual/summary")
def test_document_tables_are_generated_not_transcribed():
    """Every load-bearing table in the writeup must be the generated one.

    An adversarial audit of the finished document filed 31 findings, and the
    large majority were transcription errors in tables that had been hand-copied
    from a terminal: four p-values in the table carrying the *pre-registered
    primary contrast* did not reproduce, two of them matching no run in the
    repository, and a Pearson column labelled as coming from the corrected
    shrinkage estimator came from the uncorrected one. None of it changed the
    decision. All of it was avoidable.

    So the tables are emitted by ``scripts/report_phase2_residual.py`` into
    ``summary/doc_tables.md``, each behind a marker comment, and this test
    requires each block to appear in the document **verbatim**. Regenerating the
    report and pasting is now the only way to update them, and a stale document
    fails here rather than in a reader's hands.
    """
    gen = (RESULTS / "summary" / "doc_tables.md").read_text()
    text = DOC.read_text()
    blocks = {}
    for chunk in gen.split("<!-- generated: ")[1:]:
        name, rest = chunk.split(" -->\n", 1)
        blocks[name] = rest.strip()
    assert blocks, "no generated tables found; run scripts/report_phase2_residual.py"

    missing = []
    for name, table in blocks.items():
        marker = f"<!-- generated: {name} -->"
        if marker not in text:
            continue          # not every generated table is quoted in the doc
        if table not in text:
            missing.append(name)
    assert not missing, (
        f"these generated tables are quoted in {DOC.name} but do not match the "
        f"data: {missing}. Rerun scripts/report_phase2_residual.py and paste "
        f"the block from summary/doc_tables.md.")
    # ...and at least the two that carry the decision must actually be quoted.
    for required in ("primary_contrast_as_run", "primary_contrast_honest_alpha"):
        assert f"<!-- generated: {required} -->" in text, (
            f"{DOC.name} no longer quotes the generated {required} table")


@pytest.mark.skipif(not _HAVE_RESULTS, reason="needs the shipped results")
def test_every_shipped_block_is_complete_and_error_free():
    """No block ships with a missing cell, a duplicate run or a failed run.

    The Phase 2 audit found two separate bugs whose signature was a silently
    short cell that still printed an ``n``. Checked here for every file rather
    than for the main grid alone.
    """
    from intervention_algebra.real_data import residual_sweep as rs

    # Derived from the grid functions, not hardcoded. Hardcoding them meant that
    # widening the permutation control to a third coverage failed this test with
    # "96 rows, expected 64" -- correct behaviour, but the wrong thing to have to
    # edit: the invariant is "the file holds exactly the grid that produced it".
    expected = {
        "runs.jsonl": len(rs.main_grid()),
        "sensitivity.jsonl": len(rs.sensitivity_grid()),
        "controls.jsonl": len(rs.control_grid()),
        "power.jsonl": len(rs.power_grid()),
        "power_honest_alpha.jsonl": len(rs.power_honest_grid()),
        "honest_alpha.jsonl": len(rs.honest_alpha_grid()),
        "rank2.jsonl": len(rs.rank2_grid()),
        "ridge_titration.jsonl": len(rs.titration_grid()),
        "contaminated_diagnostic.jsonl": len(rs.contamination_grid()),
    }
    for name, n in expected.items():
        p = RESULTS / name
        if not p.exists():
            pytest.skip(f"{name} not present")
        df = load_residual_runs(p)          # raises on any row carrying an error
        assert len(df) == n, (
            f"{name}: {len(df)} rows, but its grid function specifies {n}. "
            f"Either the grid changed and the block was not rerun, or the run "
            f"did not finish.")
        key = ["tag", "screen", "coverage", "rung", "split_seed"]
        dup = df.duplicated(key).sum()
        assert dup == 0, f"{name}: {dup} duplicate {key} rows"


@pytest.mark.skipif(not _HAVE_RESULTS, reason="needs the shipped results")
def test_the_decision_is_quoted_from_the_honest_alpha_block():
    """§10 says the decision uses the corrected shrinkage. Check the numbers do.

    The audit found the criterion table labelled honest-α while quoting
    as-run Pearson and sign accuracy, which differ because the two protocols
    select different grid members. Both bounds in the document are pinned here.
    """
    h = load_residual_runs(RESULTS / "honest_alpha.jsonl")
    dense = h[(h["rung"] == "lowrank") & (h["coverage"] >= 0.40)]
    g = dense.groupby(["screen", "coverage"], observed=True)
    skill = g["cal_skill"].mean()
    pear = g["cal_pearson"].mean()
    sign = g["cal_sign_accuracy"].mean()
    text = DOC.read_text()

    assert f"{skill.min():.3f}" in text.replace("+", "")
    assert f"{skill.max():.3f}" in text.replace("+", "")
    assert f"{pear.min():.3f}" in text, (
        f"the document does not quote the honest-alpha Pearson lower bound "
        f"{pear.min():.3f}")
    assert f"{pear.max():.3f}" in text
    assert f"{sign.min():.3f}" in text and f"{sign.max():.3f}" in text


ROOT = Path(__file__).resolve().parent.parent

#: Every document that quotes a ``--part`` run count, and the parts it quotes.
#: The counts themselves are never written here -- they come from
#: ``residual_sweep.part_counts()``, which is the point.
_COUNT_DOCS = {
    ROOT / "REPRODUCIBILITY.md": ("all",),
    ROOT / "results" / "phase2_residual" / "README_PHASE2R.md": ("all",),
}


def test_documented_run_counts_match_the_grids():
    """A hard-coded total in prose is a number that goes stale silently.

    It already did: the permutation control was widened from two coverages to
    three (64 -> 96 rows) and three documents kept quoting a total of 1,112 for
    ``--part all``, which by then ran 1,560. Nothing failed, because nothing was
    checking. This checks.
    """
    counts = rs.part_counts()
    # The parts are the runner's choices, minus the two that are not grids.
    assert set(counts) == set(rs.ALL_PARTS) | {"all"}
    assert counts["all"] == sum(counts[p] for p in rs.ALL_PARTS)

    for path, parts in _COUNT_DOCS.items():
        text = path.read_text()
        for part in parts:
            n = counts[part]
            assert f"{n:,} runs" in text, (
                f"{path.name} does not quote '{n:,} runs' for --part {part}; "
                f"the grids now specify {n}. Regenerate the prose from "
                f"`python scripts/run_phase2_residual.py --counts`.")


def test_part_jobs_covers_every_grid_exactly_once():
    """``--part all`` must run every block, and no block twice.

    The bug this prevents is the one the if-ladder it replaced actually had:
    ``robustness`` was added to ``all`` in the code but not to ``all``'s
    docstring, so the documented reproduction ran ``--part all`` *and* ``--part
    robustness`` and repeated 416 runs.
    """
    all_names = [name for name, _ in rs.part_jobs("all")]
    assert len(all_names) == len(set(all_names)), f"duplicate blocks: {all_names}"
    per_part = [name for part in rs.ALL_PARTS
                for name, _ in rs.part_jobs(part)]
    assert sorted(all_names) == sorted(per_part)
    # Every named output the runner knows about is reachable from some part.
    src = (ROOT / "scripts" / "run_phase2_residual.py").read_text()
    for name in all_names:
        assert f'"{name}"' in src, f"{name} has no OUTPUTS entry in the runner"


def test_runner_docstring_does_not_hard_code_run_counts():
    """The docstring quotes ``--counts`` instead of numbers, deliberately."""
    doc = (ROOT / "scripts" / "run_phase2_residual.py").read_text()
    head = doc.split('"""')[1]
    assert "--counts" in head
    import re
    stale = re.findall(r"\(\d[\d,]* runs\)", head)
    assert not stale, f"the runner docstring hard-codes run counts: {stale}"
