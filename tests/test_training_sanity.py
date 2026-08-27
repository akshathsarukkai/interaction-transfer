"""Debug controls: is the pipeline capable of succeeding at all?

None of these are scientific results.  They are the checks that must pass before
any *comparison* between families is worth reading:

* a noiseless, densely-covered, low-rank system must be fitted essentially
  perfectly.  A failure here is an optimisation or wiring bug -- a mis-indexed
  batch, a detached gradient, a learning rate three orders of magnitude off --
  and it would show up in the headline numbers as "the algebra model does not
  help", which is a completely different conclusion;
* the benchmark must be *solvable* on held-out pairs in principle, or a flat
  coverage curve means "impossible", not "structure does not help";
* the additive null must fail the interacting system, or the task does not
  require interactions at all;
* on an interaction-free system the structured model must not manufacture
  interactions out of nothing.

Systems are deliberately tiny (24 interventions, 2 response channels, 2 latent
factors) and epoch budgets short, so the whole file is a handful of seconds.
"""

from __future__ import annotations

import pytest
import torch

from intervention_algebra.experiment import RunConfig, run_experiment, seeded
from intervention_algebra.generator import SystemConfig, generate_observations, make_system
from intervention_algebra.models import ModelConfig, build_model
from intervention_algebra.splits import SplitConfig
from intervention_algebra.train import TrainConfig, train_model

FAMILIES = ("additive", "unconstrained", "shared_pair", "algebra")
PAIR_FAMILIES = ("unconstrained", "shared_pair", "algebra")

V_SCALE = 1.0            # rms of the first-order effect in every system below
INTERACTION_SCALE = 0.6  # s_scale = a_scale

_CACHE: dict = {}


def _run(cfg: RunConfig) -> dict:
    key = repr(cfg.to_dict())
    if key not in _CACHE:
        _CACHE[key] = run_experiment(cfg)
    return _CACHE[key]


def _cfg(family: str, *, regime: str, noise_std: float, max_epochs: int,
         seed: int = 0) -> RunConfig:
    """Every family gets identical data, optimiser settings and epoch budget."""
    return seeded(RunConfig(
        family=family,
        system=SystemConfig(n_interventions=24, latent_dim=2, n_factors=2,
                            regime=regime, sparsity=1.0, noise_std=noise_std,
                            v_scale=V_SCALE, s_scale=INTERACTION_SCALE,
                            a_scale=INTERACTION_SCALE),
        split=SplitConfig(pair_coverage=0.7, eligibility_coverage=None),
        model=ModelConfig(emb_dim=3, pair_hidden=48),
        train=TrainConfig(lr=5e-3, max_epochs=max_epochs, patience=max_epochs),
        capacity_match=True,
    ), seed)


def easy(family: str) -> dict:
    """Noiseless, every pair interacts, 70% of pairs in training, rank-2 truth.

    The friendliest system this generator can produce.
    """
    return _run(_cfg(family, regime="both", noise_std=0.0, max_epochs=500))


def easy_additive_only(family: str) -> dict:
    """The same but with S = A = 0: data that lies inside *every* family's
    hypothesis class, including the additive one."""
    return _run(_cfg(family, regime="independent", noise_std=0.0,
                     max_epochs=300))


def noisy_independent(family: str, seed: int = 0) -> dict:
    return _run(_cfg(family, regime="independent", noise_std=0.05,
                     max_epochs=300, seed=seed))


# Threshold for "the optimiser did its job".  Measured values on this system:
#   regime=independent, noiseless : additive 0.000000, unconstrained 0.000369,
#                                   shared_pair 0.000518, algebra 0.000283
#   regime=both, noiseless        : unconstrained 0.00137, shared_pair 0.00169,
#                                   algebra 0.00104
# 1e-2 is ~6x the worst observed value -- loose enough to survive platform
# arithmetic differences, tight enough that any real regression (a broken
# gradient path leaves train MSE at the target variance, ~0.5-1.0 here) fires.
FIT_THRESHOLD = 1e-2


# --------------------------------------------------------------------------
# 1. Easy noiseless debug dataset -- must be FITTED
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_fits_a_noiseless_dataset_inside_its_hypothesis_class(family):
    """regime='independent' + noise_std=0: the truth is exactly v_i + v_j.

    Every family -- additive included -- can represent this exactly, so this is
    the one easy-dataset check that is meaningful for *all four*. Anything but a
    near-zero training MSE is a wiring or optimisation bug.
    """
    r = easy_additive_only(family)
    assert r["final_train_mse"] < FIT_THRESHOLD, (
        f"[{family}] failed the easy debug control: noiseless data whose truth "
        f"(v_i + v_j, no interactions) lies exactly inside this family's "
        f"hypothesis class, 70% pair coverage, {r['epochs_run']} epochs. "
        f"final_train_mse={r['final_train_mse']:.6f} (limit {FIT_THRESHOLD}), "
        f"test_mse={r['test_mse']:.6f}. This is an optimisation/wiring failure, "
        f"not a scientific result, and it invalidates every comparison in the "
        f"study.")


@pytest.mark.parametrize("family", PAIR_FAMILIES)
def test_interaction_families_fit_the_noiseless_interacting_dataset(family):
    """The same control for the families that own a pair term, on data that
    actually needs one.

    ``additive`` is excluded *by construction*, not to spare it: it has no pair
    term, so on a strongly interacting system its training MSE is floored by the
    interaction variance. That floor is the subject of the next test.
    """
    r = easy(family)
    assert r["final_train_mse"] < FIT_THRESHOLD, (
        f"[{family}] failed the easy interacting control: noiseless, every pair "
        f"interacts, rank-2 ground truth, 70% pair coverage, "
        f"{r['epochs_run']} epochs. final_train_mse={r['final_train_mse']:.6f} "
        f"(limit {FIT_THRESHOLD}), test_mse={r['test_mse']:.5f}. A family that "
        f"cannot fit data it can exactly represent is broken, and its poor "
        f"held-out numbers elsewhere would be meaningless.")


# --------------------------------------------------------------------------
# 2. The benchmark must be solvable on HELD-OUT pairs
# --------------------------------------------------------------------------
# Observed on this system: algebra test_mse = 0.0026 (S_pearson 0.99,
# A_pearson 1.00). 0.02 is ~8x that, and ~200x below the additive null's 0.53,
# so it cannot be reached by predicting first-order effects alone.
SOLVABLE_THRESHOLD = 0.02


def test_the_benchmark_is_solvable_in_principle_on_held_out_pairs():
    r = easy("algebra")
    add = easy("additive")
    assert r["test_mse"] < SOLVABLE_THRESHOLD, (
        f"the algebra family scores test_mse={r['test_mse']:.5f} on completely "
        f"held-out pairs of the easiest possible system (noiseless, rank-2, 70% "
        f"coverage), against a limit of {SOLVABLE_THRESHOLD} and an additive "
        f"null of {add['test_mse']:.4f}. If the benchmark is not solvable in "
        f"principle then a flat pair-coverage curve means 'impossible task', "
        f"not 'algebraic structure does not help', and Phase 1 answers nothing. "
        f"(S_pearson={r.get('test_S_pearson', float('nan')):.3f}, "
        f"A_pearson={r.get('test_A_pearson', float('nan')):.3f}, "
        f"train_mse={r['final_train_mse']:.5f})")
    assert r["test_S_pearson"] > 0.9 and r["test_A_pearson"] > 0.9, (
        f"held-out prediction is good but latent recovery is not: "
        f"S_pearson={r['test_S_pearson']:.3f}, "
        f"A_pearson={r['test_A_pearson']:.3f}. On an identity observation map "
        f"the (v, S, A) decomposition is identified, so a model that predicts "
        f"the rows well should recover the components too; a gap points at the "
        f"structure metrics rather than at the model.")


def test_the_additive_null_cannot_solve_the_interacting_system():
    """If it could, the task would not require interactions and no comparison
    between interaction-modelling families would mean anything."""
    add, alg = easy("additive"), easy("algebra")
    assert add["final_train_mse"] > 0.1, (
        f"the additive family -- which has no pair term at all -- reached "
        f"final_train_mse={add['final_train_mse']:.5f} on a system where every "
        f"pair interacts. The generated interactions must therefore be "
        f"negligible, or they are leaking into the first-order effects.")
    assert add["test_mse"] > 10 * alg["test_mse"], (
        f"additive test_mse={add['test_mse']:.4f} vs algebra "
        f"{alg['test_mse']:.4f} (ratio {add['test_mse'] / alg['test_mse']:.1f}x, "
        f"need >10x). A small gap means first-order effects alone nearly solve "
        f"the task.")
    assert alg["interaction_variance_share"] > 0.1, (
        f"interactions carry only "
        f"{alg['interaction_variance_share']:.1%} of the target variance on the "
        f"'easy' system; the additive null is then near-optimal by construction "
        f"and the debug control has no teeth.")


# --------------------------------------------------------------------------
# 3. Independent regime: no interactions must be invented
# --------------------------------------------------------------------------
# The structured model has an unregularised pair term that is free to drift.
# Measured pred rms on regime='independent' (v rms = 1.0):
#   noiseless, seed 0 : S 0.090, A 0.000
#   noisy 0.05        : seed 0 -> S 0.083/0.095, A 0.032/0.027
#                       seeds 1,2 -> S 0.141/0.142, A 0.041/0.037
# 0.25 * v_scale is ~1.8x the worst value seen across seeds and ~2.6x the value
# this test actually measures. It is set from observed runs rather than from
# theory because the drift has no closed form; it is still far below the
# INTERACTION_SCALE = 0.6 that a real interaction would have, so a model that
# genuinely hallucinated structure would trip it.
HALLUCINATION_THRESHOLD = 0.25 * V_SCALE


def test_ground_truth_really_is_interaction_free_in_the_independent_regime():
    r = noisy_independent("algebra")
    assert r["test_true_S_rms"] == 0.0 and r["test_true_A_rms"] == 0.0, (
        f"regime='independent' must have S = A = 0 exactly; got "
        f"true_S_rms={r['test_true_S_rms']}, true_A_rms={r['test_true_A_rms']}. "
        f"The rest of this section would be testing the wrong condition.")
    assert r["interaction_variance_share"] < 1e-9, (
        f"interaction_variance_share={r['interaction_variance_share']:.3e} on an "
        f"interaction-free system; it should be exactly zero.")


@pytest.mark.parametrize("component", ["S", "A"])
def test_algebra_does_not_invent_interactions_when_there_are_none(component):
    r = noisy_independent("algebra")
    rms = r[f"test_pred_{component}_rms"]
    assert rms < HALLUCINATION_THRESHOLD, (
        f"on an interaction-free system (regime='independent', true "
        f"{component} identically zero) the trained algebra model still emits a "
        f"{component} term with rms={rms:.4f} = {rms / V_SCALE:.0%} of the "
        f"first-order scale v_scale={V_SCALE}, and {rms / INTERACTION_SCALE:.0%} "
        f"of the interaction scale a real condition would have "
        f"({INTERACTION_SCALE}). Limit {HALLUCINATION_THRESHOLD}. A structured "
        f"model that manufactures interactions from noise would show spurious "
        f"'structure recovery' everywhere, so its successes could not be "
        f"trusted. (test_mse={r['test_mse']:.5f}, "
        f"noise floor {r['noise_floor_mse']:.5f})")


def test_additive_is_competitive_when_there_are_no_interactions():
    """The correctly-specified model must not lose to an over-parameterised one;
    if it does, the additive baseline is handicapped by something other than its
    missing pair term and is an unfair null everywhere else."""
    add, alg = noisy_independent("additive"), noisy_independent("algebra")
    assert add["test_mse"] < 1.5 * alg["test_mse"], (
        f"on an interaction-free system the additive family is the correctly "
        f"specified model, yet it scores test_mse={add['test_mse']:.5f} against "
        f"algebra's {alg['test_mse']:.5f} "
        f"(ratio {add['test_mse'] / alg['test_mse']:.2f}, need <1.5).")


# --------------------------------------------------------------------------
# 4. The architectural invariants must survive real training
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_invariants_survive_real_training(family):
    r = easy(family)
    assert r["sym_residual"] < 1e-6, (
        f"[{family}] after training, rms|S(i,j) - S(j,i)| = "
        f"{r['sym_residual']:.3e}. Symmetry of the readout is supposed to hold "
        f"by construction; a nonzero residual means it is implemented as a soft "
        f"penalty or is being bypassed.")
    assert r["antisym_residual"] < 1e-6, (
        f"[{family}] after training, rms|A(i,j) + A(j,i)| = "
        f"{r['antisym_residual']:.3e}.")


def test_training_reduces_the_loss_it_is_optimising():
    """A trivially-broken optimiser (zero lr, detached graph) would still pass
    the threshold tests if the initial loss happened to be small; require an
    actual decrease."""
    r = easy("algebra")
    # run_experiment does not return the loss history, so the claim is derived
    # from the recorded best epoch and the final loss against the target scale.
    assert r["best_epoch"] > 0, (
        f"best validation MSE was achieved at epoch {r['best_epoch']}, i.e. at "
        f"initialisation; the optimiser never improved anything.")
    assert r["final_train_mse"] < 0.01 * (INTERACTION_SCALE ** 2 + V_SCALE ** 2), (
        f"final_train_mse={r['final_train_mse']:.5f} is not small relative to "
        f"the scale of the targets ({INTERACTION_SCALE ** 2 + V_SCALE ** 2:.2f}); "
        f"training did not converge.")


# --------------------------------------------------------------------------
# 5. Degenerate but legal training budgets
# --------------------------------------------------------------------------
def _tiny_train_table():
    system = make_system(SystemConfig(n_interventions=12, latent_dim=2,
                                      n_factors=2, seed=0))
    return generate_observations(system, system.all_pairs()[:10],
                                 include_singles=True)


# Regression test. train_model used to read the loop variable `epoch` after
# `for epoch in range(cfg.max_epochs)` to build "epochs_run", so max_epochs=0
# raised UnboundLocalError instead of returning an untrained model. Reachable
# from the public API (an "untrained baseline" ablation). Fixed by initialising
# `epoch = -1` before the loop.
def test_zero_epoch_training_returns_an_untrained_model():
    model = build_model("algebra", ModelConfig(n_interventions=12, out_dim=2,
                                               emb_dim=4, seed=0))
    fit = train_model(model, _tiny_train_table(), None,
                      TrainConfig(max_epochs=0))
    assert fit["epochs_run"] == 0, (
        f"a zero-epoch budget should report 0 epochs run, got "
        f"{fit['epochs_run']}")
    assert fit["best_epoch"] == 0


def test_a_single_epoch_budget_is_handled():
    """The boundary just above the broken one, to pin down that the failure is
    specific to max_epochs=0 and not to short budgets in general."""
    model = build_model("algebra", ModelConfig(n_interventions=12, out_dim=2,
                                               emb_dim=4, seed=0))
    fit = train_model(model, _tiny_train_table(), None,
                      TrainConfig(max_epochs=1, patience=1000))
    assert fit["epochs_run"] == 1, (
        f"expected epochs_run == 1, got {fit['epochs_run']}")
    assert torch.isfinite(torch.tensor(fit["final_train_mse"])), (
        f"final_train_mse={fit['final_train_mse']} after one epoch")
