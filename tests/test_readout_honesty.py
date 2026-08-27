"""The scored S/A must be the S/A the model actually uses to predict.

This is the one property whose failure would rig the central comparison, and it
was previously verified once by hand and then cited by `docs/identifiability.md`
as if tests covered it. They did not. It is covered here.

The risk is concrete. ``implied_S`` and ``implied_A`` are read out of *every*
family and correlated against ground truth. If a family's readout returned
something other than the quantity that enters its own prediction -- a symmetric
projection it never uses, say, or a head that is trained but bypassed at
prediction time -- then the recovery comparison would be measuring different
objects for different families, and the constrained model could look better (or
worse) for reasons that have nothing to do with the inductive bias.

The invariants below tie the readouts to the predictions algebraically, so they
hold for any family, trained or untrained:

    latent_sim(i,j)     - v_i - v_j                     == implied_S(i,j)
    (latent_ordered(i,j) - latent_ordered(j,i)) / 2      == implied_A(i,j)

The second identity is what ``A`` *means*: half the forward-minus-reverse
difference of the ordered response. Any family whose ordered prediction does not
have that antisymmetric part is not being scored on a like-for-like quantity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from intervention_algebra.generator import (SystemConfig, generate_observations,
                                            make_system)
from intervention_algebra.models import FAMILIES, ModelConfig, build_model
from intervention_algebra.train import TrainConfig, train_model

TOL = 1e-5
N = 16


def _cfg(family: str) -> ModelConfig:
    return ModelConfig(n_interventions=N, out_dim=3, emb_dim=4, seed=0)


def _pairs(n_pairs: int = 60):
    rng = np.random.default_rng(0)
    i, j = [], []
    while len(i) < n_pairs:
        a, b = int(rng.integers(N)), int(rng.integers(N))
        if a != b:
            i.append(a)
            j.append(b)
    return torch.tensor(i), torch.tensor(j)


def _trained(family: str):
    """A briefly-trained model: the identities must survive optimisation."""
    system = make_system(SystemConfig(n_interventions=N, latent_dim=3, n_factors=3,
                                      regime="both", noise_std=0.0, seed=0))
    table = generate_observations(system, system.all_pairs()[:40],
                                  include_singles=True)
    model = build_model(family, _cfg(family))
    train_model(model, table, None, TrainConfig(max_epochs=40, patience=10**6, seed=0))
    return model


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("trained", [False, True], ids=["init", "trained"])
def test_implied_S_is_the_term_used_in_the_simultaneous_prediction(family, trained):
    model = _trained(family) if trained else build_model(family, _cfg(family))
    i, j = _pairs()
    with torch.no_grad():
        used = model.latent_sim(i, j) - model.v(i) - model.v(j)
        scored = model.implied_S(i, j)
    err = float((used - scored).abs().max())
    assert err < TOL, (
        f"[{family}, {'trained' if trained else 'init'}] the S that is scored "
        f"against ground truth differs from the S that enters the model's own "
        f"simultaneous prediction by up to {err:.2e}. Recovery numbers would "
        f"then be measuring a quantity the model does not use, and the "
        f"comparison across families would not be like-for-like.")


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("trained", [False, True], ids=["init", "trained"])
def test_implied_A_is_half_the_forward_minus_reverse_ordered_prediction(family, trained):
    model = _trained(family) if trained else build_model(family, _cfg(family))
    i, j = _pairs()
    with torch.no_grad():
        used = 0.5 * (model.latent_ordered(i, j) - model.latent_ordered(j, i))
        scored = model.implied_A(i, j)
    err = float((used - scored).abs().max())
    assert err < TOL, (
        f"[{family}, {'trained' if trained else 'init'}] the scored A differs "
        f"from half the model's own forward-minus-reverse ordered prediction by "
        f"up to {err:.2e}. That difference *is* the definition of A, so the "
        f"readout would not be the quantity being claimed.")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_readouts_are_not_trivially_zero_for_interaction_families(family):
    """Guard the guard: the identities above are satisfied vacuously by zeros."""
    model = _trained(family)
    i, j = _pairs()
    with torch.no_grad():
        s = float(model.implied_S(i, j).pow(2).mean().sqrt())
        a = float(model.implied_A(i, j).pow(2).mean().sqrt())
    if family == "additive":
        assert s == 0.0 and a == 0.0, (
            f"the additive family has no interaction parameters, so both "
            f"readouts must be identically zero; got S_rms={s}, A_rms={a}")
    else:
        assert s > 1e-6, (
            f"[{family}] implied_S is ~0 (rms={s:.2e}) after training on a "
            f"system with interactions. The consistency identities above would "
            f"then pass vacuously, so this asserts the readout carries signal.")
        assert a > 1e-6, (
            f"[{family}] implied_A is ~0 (rms={a:.2e}) after training. Note this "
            f"is the exact degenerate optimum the restart logic exists to escape "
            f"(see experiment.run_experiment): an F_A symmetric in its two "
            f"argument slots gives A == 0 identically.")
