"""Metrics must refuse to report a number they cannot justify.

The dangerous failure here is not a crash, it is a plausible-looking float.  A
correlation against an identically-zero ground truth, or one computed from six
pairs, or a latent-recovery score under a non-identifiable observation map, all
have no defined value -- but if the code returns 0.0 or 1.0 instead of NaN, that
value gets averaged into a headline table and manufactures a finding.

So: the NaN paths are tested as first-class behaviour, alongside the positive
control (a perfect prediction must score ~1) and the negative control (a
scrambled prediction must score ~0).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from intervention_algebra.generator import (
    SINGLE, SystemConfig, generate_observations, make_system)
from intervention_algebra.metrics import (
    MIN_PAIRS_FOR_CORR, _auc, _corr, interaction_variance_share,
    invariance_diagnostics, prediction_metrics, structure_metrics,
    true_v_additive_mse)
from intervention_algebra.models import ModelConfig, build_model

FAMILIES = ("additive", "unconstrained", "shared_pair", "algebra")
N = 24
D = 2

RNG = np.random.default_rng(20240808)


def _system(regime: str = "both", **kw):
    base = dict(n_interventions=N, latent_dim=D, n_factors=3, regime=regime,
                sparsity=0.5, noise_std=0.05, seed=0)
    base.update(kw)
    return make_system(SystemConfig(**base))


def _pairs(system) -> np.ndarray:
    return system.all_pairs()


class _StubModel:
    """Duck-typed stand-in exposing only what ``structure_metrics`` reads.

    Using a stub rather than a trained network is what makes the positive and
    negative controls exact: the "prediction" is chosen, so a Pearson of 1.0 or
    0.0 is a statement about the metric, not about an optimiser.
    """

    def __init__(self, S: np.ndarray, A: np.ndarray):
        self._S = torch.as_tensor(S, dtype=torch.float64)
        self._A = torch.as_tensor(A, dtype=torch.float64)

    def implied_S(self, i, j):
        return self._S

    def implied_A(self, i, j):
        return self._A

    def implied_S_projected(self, i, j):
        return self._S


# --------------------------------------------------------------------------
# 1. _corr must return NaN, not a number and not an exception
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_pairs", [0, 1, 2, MIN_PAIRS_FOR_CORR - 1])
def test_corr_is_nan_with_too_few_pairs(n_pairs):
    a = RNG.normal(size=(n_pairs, D))
    b = RNG.normal(size=(n_pairs, D))
    pe, sp = _corr(a, b)
    assert math.isnan(pe) and math.isnan(sp), (
        f"_corr on {n_pairs} pairs returned (pearson={pe!r}, spearman={sp!r}); "
        f"below MIN_PAIRS_FOR_CORR={MIN_PAIRS_FOR_CORR} a correlation is noise "
        f"with a plausible magnitude, and averaging it into a table would "
        f"invent structure.")


@pytest.mark.parametrize("side", ["truth", "prediction", "both"])
def test_corr_is_nan_when_a_side_is_constant(side):
    n = 64
    live = RNG.normal(size=(n, D))
    const = np.full((n, D), 1.7)
    a = const if side in ("truth", "both") else live
    b = const if side in ("prediction", "both") else live
    pe, sp = _corr(a, b)
    assert math.isnan(pe) and math.isnan(sp), (
        f"_corr with a constant {side} returned (pearson={pe!r}, "
        f"spearman={sp!r}). A constant side has zero variance, so the "
        f"correlation is 0/0. This is exactly the case of the additive family "
        f"(prediction identically zero) and of A in a symmetric-only system "
        f"(truth identically zero).")


def test_corr_is_nan_only_for_the_degenerate_channels():
    """A partially degenerate input must not poison the live channels."""
    n = 64
    x = RNG.normal(size=(n, 2))
    y = x.copy()
    y[:, 0] = 3.0                      # channel 0 dead, channel 1 perfect
    pe, sp = _corr(x, y)
    assert not math.isnan(pe), "the surviving channel should still be scored"
    assert pe == pytest.approx(1.0, abs=1e-9), (
        f"channel 1 is a perfect match and channel 0 is undefined; the average "
        f"over *scored* channels should be 1.0, got {pe}.")


def test_corr_scores_perfect_recovery_as_one():
    x = RNG.normal(size=(500, D))
    pe, sp = _corr(x, x)
    assert pe == pytest.approx(1.0, abs=1e-9), f"got pearson={pe}"
    assert sp == pytest.approx(1.0, abs=1e-9), f"got spearman={sp}"

    # A positive affine rescaling is still perfect recovery for Pearson.
    pe2, _ = _corr(x, 3.0 * x + 5.0)
    assert pe2 == pytest.approx(1.0, abs=1e-9), (
        f"Pearson is scale/shift invariant, so an affinely rescaled perfect "
        f"prediction must still score 1.0; got {pe2}.")


def test_corr_scores_a_scrambled_prediction_as_zero():
    n = 500
    x = RNG.normal(size=(n, D))
    perm = np.random.default_rng(1).permutation(n)
    pe, sp = _corr(x, x[perm])
    # Sampling std of Pearson under the null is 1/sqrt(n-3) ~ 0.045 here, so
    # 0.15 is >3 sigma: it fires on real leakage, not on chance.
    assert abs(pe) < 0.15, (
        f"a randomly permuted prediction scored pearson={pe:.3f} (limit 0.15) "
        f"on {n} pairs. A metric with a positive floor would credit every model "
        f"with partial 'structure recovery'.")
    assert abs(sp) < 0.15, f"scrambled spearman={sp:.3f}"


# --------------------------------------------------------------------------
# 2. structure_metrics: identifiability and degenerate ground truth
# --------------------------------------------------------------------------
def test_structure_metrics_returns_nan_when_not_identifiable():
    system = _system("both")
    pairs = _pairs(system)
    model = build_model("algebra", ModelConfig(n_interventions=N, out_dim=D,
                                               emb_dim=6, seed=0)).eval()
    out = structure_metrics(model, system, pairs, identifiable=False,
                            prefix="test_")
    for key in ("test_S_pearson", "test_S_spearman", "test_A_pearson",
                "test_A_spearman", "test_S_nrmse", "test_A_nrmse",
                "test_S_proj_pearson"):
        assert math.isnan(out[key]), (
            f"identifiable=False must give NaN for '{key}', got {out[key]!r}. "
            f"Under a nonlinear observation map the (v, S, A) decomposition is "
            f"not identified (docs/identifiability.md), so a latent-recovery "
            f"correlation there is not a measurement of anything.")
    # ...but observable-side quantities are still reported.
    assert not math.isnan(out["test_pred_S_rms"]), (
        "pred_S_rms is a property of the model alone and must survive "
        "identifiable=False")
    assert not math.isnan(out["test_topo_base_rate"])


@pytest.mark.parametrize("regime,dead,live", [
    ("symmetric", "A", "S"),
    ("antisymmetric", "S", "A"),
])
def test_correlation_is_nan_when_the_truth_is_identically_zero(regime, dead, live):
    system = _system(regime)
    pairs = _pairs(system)
    # Perfect recovery of both components: the *only* reason a score can be NaN
    # is the degenerate ground truth, not a bad model.
    stub = _StubModel(system.S[pairs[:, 0], pairs[:, 1]],
                      system.A[pairs[:, 0], pairs[:, 1]])
    out = structure_metrics(stub, system, pairs, identifiable=True,
                            prefix="test_")

    assert math.isnan(out[f"test_{dead}_pearson"]), (
        f"in regime='{regime}' the true {dead} is identically zero, so its "
        f"correlation has no defined value; got "
        f"{out[f'test_{dead}_pearson']!r}. A finite number here would be "
        f"averaged into the regime table as evidence of recovering something "
        f"that does not exist.")
    assert math.isnan(out[f"test_{dead}_nrmse"]), (
        f"nrmse for {dead} normalises by a zero true scale in regime="
        f"'{regime}'; it must be NaN, got {out[f'test_{dead}_nrmse']!r}.")
    assert out[f"test_{live}_pearson"] == pytest.approx(1.0, abs=1e-6), (
        f"the component that *is* present ({live}) is handed to the metric "
        f"exactly, so it must score 1.0; got {out[f'test_{live}_pearson']!r}. A "
        f"NaN guard firing too broadly would silently drop real measurements.")


def test_structure_metrics_scores_perfect_recovery_as_one():
    system = _system("both")
    pairs = _pairs(system)
    stub = _StubModel(system.S[pairs[:, 0], pairs[:, 1]],
                      system.A[pairs[:, 0], pairs[:, 1]])
    out = structure_metrics(stub, system, pairs, identifiable=True, prefix="")
    for name in ("S", "A"):
        assert out[f"{name}_pearson"] == pytest.approx(1.0, abs=1e-6), (
            f"an exact copy of the ground-truth {name} scored "
            f"{out[f'{name}_pearson']!r} instead of 1.0.")
        assert out[f"{name}_spearman"] == pytest.approx(1.0, abs=1e-6)
        assert out[f"{name}_nrmse"] == pytest.approx(0.0, abs=1e-9), (
            f"nrmse for a perfect {name} prediction should be 0, got "
            f"{out[f'{name}_nrmse']!r}.")
    assert out["topk_S_overlap"] == pytest.approx(1.0), (
        f"top-k overlap for a perfect prediction should be 1.0, got "
        f"{out['topk_S_overlap']!r}.")
    # Under sparsity_mode="latent" the interaction strength is a *soft* sigmoid
    # gate and ``mask = gate > 0.5``, so ||S_true|| does not separate the two
    # classes perfectly -- pairs just either side of the threshold overlap.
    # Near-perfect is the correct expectation here; the exact case is below.
    assert out["topo_auroc_S"] > 0.95, (
        f"ranking pairs by the exact ||S_true|| should almost perfectly recover "
        f"which pairs interact; AUROC={out['topo_auroc_S']:.4f}.")


@pytest.mark.parametrize("sparsity_mode", ["module", "random"])
def test_topology_auc_is_exactly_one_under_a_hard_interaction_gate(sparsity_mode):
    """With a discrete topology the gate is exactly 0/1, so ``S_true`` is
    identically zero off the mask and the ranking is perfectly separating.
    A value below 1.0 would mean the mask and the interaction tensor disagree."""
    system = _system("both", sparsity_mode=sparsity_mode, n_modules=6)
    pairs = _pairs(system)
    stub = _StubModel(system.S[pairs[:, 0], pairs[:, 1]],
                      system.A[pairs[:, 0], pairs[:, 1]])
    out = structure_metrics(stub, system, pairs, identifiable=True, prefix="")
    assert out["topo_auroc_S"] == pytest.approx(1.0, abs=1e-9), (
        f"[{sparsity_mode}] the mask claims which pairs interact but "
        f"||S_true|| ranks them only to AUROC={out['topo_auroc_S']:.6f}; the "
        f"topology labels and the interaction tensor are inconsistent, so "
        f"topology-recovery scores would be measured against the wrong labels.")
    assert out["topo_auprc_S"] == pytest.approx(1.0, abs=1e-9)


def test_structure_metrics_scores_a_scrambled_prediction_as_zero():
    system = _system("both", n_interventions=40)
    pairs = _pairs(system)
    S_true = system.S[pairs[:, 0], pairs[:, 1]]
    A_true = system.A[pairs[:, 0], pairs[:, 1]]
    perm = np.random.default_rng(11).permutation(len(pairs))
    stub = _StubModel(S_true[perm], A_true[perm])
    out = structure_metrics(stub, system, pairs, identifiable=True, prefix="")
    for name in ("S", "A"):
        r = out[f"{name}_pearson"]
        assert abs(r) < 0.15, (
            f"a permuted copy of the true {name} scored pearson={r:.3f} over "
            f"{len(pairs)} pairs (limit 0.15). The metric has a nonzero floor "
            f"and would credit chance as partial recovery.")


def test_topology_auc_is_nan_when_no_pair_interacts():
    """regime='independent' has an all-False mask, so AUROC/AUPRC are undefined."""
    system = _system("independent")
    pairs = _pairs(system)
    stub = _StubModel(RNG.normal(size=(len(pairs), D)),
                      RNG.normal(size=(len(pairs), D)))
    out = structure_metrics(stub, system, pairs, identifiable=True, prefix="")
    for key in ("topo_auroc_S", "topo_auprc_S", "topo_auroc_A", "topo_auprc_A"):
        assert math.isnan(out[key]), (
            f"'{key}' must be NaN when every label is the same class, got "
            f"{out[key]!r}.")
    assert out["topo_base_rate"] == 0.0


def test_auc_is_nan_on_degenerate_labels():
    scores = RNG.normal(size=32)
    assert all(math.isnan(v) for v in _auc(np.zeros(32), scores))
    assert all(math.isnan(v) for v in _auc(np.ones(32), scores))
    assert all(math.isnan(v) for v in _auc(np.array([0, 1, 0]), scores[:3]))
    # ...and a perfectly separating score is scored 1.0.
    labels = np.array([0] * 16 + [1] * 16)
    auroc, auprc = _auc(labels, labels.astype(float))
    assert auroc == pytest.approx(1.0) and auprc == pytest.approx(1.0)


def test_structure_metrics_on_empty_pairs_returns_nothing():
    system = _system("both")
    model = build_model("algebra", ModelConfig(n_interventions=N, out_dim=D,
                                               emb_dim=6, seed=0)).eval()
    assert structure_metrics(model, system, np.zeros((0, 2), dtype=int)) == {}


# --------------------------------------------------------------------------
# 3. invariance_diagnostics
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_invariance_diagnostics_reports_zero_residuals(family):
    """These residuals are architectural for every family; they are a tripwire
    that fires if a future edit breaks ``implied_S`` / ``implied_A``."""
    system = _system("both")
    pairs = _pairs(system)
    model = build_model(family, ModelConfig(n_interventions=N, out_dim=D,
                                            emb_dim=6, pair_hidden=32,
                                            seed=0)).eval()
    out = invariance_diagnostics(model, pairs)
    assert out["sym_residual"] < 1e-6, (
        f"[{family}] rms|S(i,j) - S(j,i)| = {out['sym_residual']:.3e} "
        f"(relative {out['sym_residual_rel']:.3e}); implied_S is supposed to be "
        f"order-invariant by construction in every family.")
    assert out["antisym_residual"] < 1e-6, (
        f"[{family}] rms|A(i,j) + A(j,i)| = {out['antisym_residual']:.3e} "
        f"(relative {out['antisym_residual_rel']:.3e}).")


def test_invariance_diagnostics_separates_algebra_from_the_baselines():
    """The residuals above are tautological, so on their own they prove nothing.
    ``pair_net_order_asymmetry`` is the number that actually distinguishes an
    architecturally symmetric pair network from a canonicalised one."""
    system = _system("both")
    pairs = _pairs(system)
    cfg = ModelConfig(n_interventions=N, out_dim=D, emb_dim=6, pair_hidden=32,
                      seed=0)
    alg = invariance_diagnostics(build_model("algebra", cfg).eval(), pairs)
    unc = invariance_diagnostics(build_model("unconstrained", cfg).eval(), pairs)
    assert alg["pair_net_order_asymmetry"] < 1e-7, (
        f"the algebra model's pair network should be exactly order-symmetric, "
        f"got {alg['pair_net_order_asymmetry']:.3e}.")
    assert unc["pair_net_order_asymmetry"] > 1e-4, (
        f"the unconstrained model's pair network is already order-symmetric "
        f"({unc['pair_net_order_asymmetry']:.3e}); the diagnostic then cannot "
        f"distinguish the families and the architectural claim is untestable.")


def test_invariance_diagnostics_on_empty_pairs_returns_nothing():
    model = build_model("algebra", ModelConfig(n_interventions=N, out_dim=D,
                                               seed=0)).eval()
    assert invariance_diagnostics(model, np.zeros((0, 2), dtype=int)) == {}


# --------------------------------------------------------------------------
# 4. Prediction-side metrics and reference scales
# --------------------------------------------------------------------------
def test_prediction_metrics_are_zero_for_a_perfect_predictor():
    system = _system("both", noise_std=0.0)
    table = generate_observations(system, system.all_pairs()[:40],
                                  include_singles=True)

    class _Oracle:
        def predict_single(self, i):
            return torch.as_tensor(system.v[i.numpy()])

        def predict_sim(self, i, j):
            return torch.as_tensor(system.latent_sim(i.numpy(), j.numpy()))

        def predict_ordered(self, i, j):
            return torch.as_tensor(system.latent_ordered(i.numpy(), j.numpy()))

    out = prediction_metrics(_Oracle(), table, prefix="")
    assert out["mse"] < 1e-24, (
        f"an oracle reproducing the noiseless latents exactly scored "
        f"mse={out['mse']:.3e}; the prediction metric is not comparing what it "
        f"claims to.")
    assert out["r2"] == pytest.approx(1.0, abs=1e-12)
    for kind in ("sim", "ord", "single"):
        assert out[f"mse_{kind}"] < 1e-24, f"mse_{kind}={out[f'mse_{kind}']:.3e}"


def test_interaction_variance_share_is_zero_without_interactions_and_positive_with():
    for regime, expect_zero in (("independent", True), ("both", False)):
        system = _system(regime, noise_std=0.0)
        table = generate_observations(system, system.all_pairs(),
                                      include_singles=True)
        share = interaction_variance_share(system, table)
        if expect_zero:
            assert share == pytest.approx(0.0, abs=1e-12), (
                f"regime='independent' has no interactions, so their variance "
                f"share must be 0; got {share:.3e}.")
        else:
            assert share > 0.05, (
                f"regime='both' scored interaction_variance_share={share:.4f}. "
                f"If interactions carry almost none of the target variance the "
                f"additive null is near-optimal and no comparison between "
                f"interaction families is informative.")


def test_true_v_additive_mse_is_zero_only_when_there_is_nothing_to_miss():
    noiseless_independent = _system("independent", noise_std=0.0)
    table = generate_observations(noiseless_independent,
                                  noiseless_independent.all_pairs(),
                                  include_singles=True)
    assert true_v_additive_mse(noiseless_independent, table) < 1e-24, (
        "with S = A = 0 and no noise, the true-v additive prediction is exact")

    interacting = _system("both", noise_std=0.0)
    table = generate_observations(interacting, interacting.all_pairs(),
                                  include_singles=True)
    assert true_v_additive_mse(interacting, table) > 1e-3, (
        "an additive prediction must be visibly wrong on an interacting system")


def test_metrics_do_not_mutate_the_observation_table():
    """``true_v_additive_mse`` and ``interaction_variance_share`` index into
    ``system.v`` and then assign into the result; an accidental in-place write
    would corrupt the system for every later metric in the same run."""
    system = _system("both")
    table = generate_observations(system, system.all_pairs()[:30],
                                  include_singles=True)
    v_before = system.v.copy()
    y_before = table.y.copy()
    true_v_additive_mse(system, table)
    interaction_variance_share(system, table)
    assert np.array_equal(system.v, v_before), (
        "a metric mutated system.v in place")
    assert np.array_equal(table.y, y_before), (
        "a metric mutated the observation table in place")
