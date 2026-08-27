"""The algebra constraint must be architectural, and the baseline must lack it.

Two claims underpin Phase 1 and both are testable without training anything to
convergence:

1. ``AlgebraModel`` satisfies S(i,j)=S(j,i) and A(i,j)=-A(j,i) *by construction*
   -- at random initialisation, after training, always.  If it were only a
   learned tendency, the paper would be about optimisation, not architecture.

2. The baselines do **not** get that structure for free.  If
   ``UnconstrainedModel``'s pair network were already symmetric, the "constraint"
   under test would be vacuous and the comparison would measure nothing.  (Note
   the baselines *do* canonicalise indices for the physically order-free
   simultaneous row -- that is a deliberate strengthening of the baseline, so the
   asymmetry has to be probed on the raw pair network via ``raw_sim_term``.)

Everything here is checked on all 276 real index pairs of a 24-intervention
system, in both index orders.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from intervention_algebra.generator import SystemConfig, generate_observations, make_system
from intervention_algebra.models import (
    ModelConfig, PairFeatures, build_model, match_pair_capacity)
from intervention_algebra.splits import SplitConfig, make_pair_split
from intervention_algebra.train import TrainConfig, train_model

FAMILIES = ("additive", "unconstrained", "shared_pair", "algebra")
PAIR_FAMILIES = ("unconstrained", "shared_pair", "algebra")

N = 24
OUT_DIM = 2
# float32 forward passes; the symmetrisations are exact in IEEE arithmetic
# (a+b == b+a, and 0.5*(x-y) == -(0.5*(y-x))), so the observed residual is 0.
# 1e-6 leaves room for a future implementation detail without admitting a soft
# constraint, whose residual would be orders of magnitude larger.
TOL = 1e-6


def _model_cfg(**kw) -> ModelConfig:
    base = dict(n_interventions=N, out_dim=OUT_DIM, emb_dim=8, v_hidden=32,
                pair_hidden=48, pair_mode="concat_outer", seed=17)
    base.update(kw)
    return ModelConfig(**base)


def _all_pairs() -> tuple[torch.Tensor, torch.Tensor]:
    iu = np.triu_indices(N, k=1)
    return (torch.as_tensor(iu[0], dtype=torch.long),
            torch.as_tensor(iu[1], dtype=torch.long))


@pytest.fixture(scope="module")
def train_table():
    """A small real training table, used only to move the weights off init."""
    system = make_system(SystemConfig(n_interventions=N, latent_dim=OUT_DIM,
                                      n_factors=3, regime="both", sparsity=0.5,
                                      noise_std=0.02, seed=0))
    split = make_pair_split(system, SplitConfig(pair_coverage=0.4,
                                                eligibility_coverage=0.2,
                                                seed=1000))
    return generate_observations(system, split.train_pairs, include_singles=True)


@pytest.fixture(scope="module")
def models_at_init():
    return {f: build_model(f, _model_cfg()).eval() for f in FAMILIES}


@pytest.fixture(scope="module")
def models_trained(train_table):
    out = {}
    for f in FAMILIES:
        m = build_model(f, _model_cfg())
        # 60 full-batch steps: enough that every weight has moved and the pair
        # nets are producing structured (not near-constant) outputs, cheap
        # enough to keep this file in the sub-second range.
        train_model(m, train_table, None,
                    TrainConfig(lr=5e-3, max_epochs=60, patience=10_000,
                                eval_every=10, seed=3))
        out[f] = m.eval()
    return out


def _state(which, models_at_init, models_trained):
    return models_at_init if which == "init" else models_trained


# --------------------------------------------------------------------------
# 1. The readouts every family exposes must obey the algebra
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("which", ["init", "trained"])
def test_implied_S_is_symmetric(family, which, models_at_init, models_trained):
    model = _state(which, models_at_init, models_trained)[family]
    i, j = _all_pairs()
    with torch.no_grad():
        resid = (model.implied_S(i, j) - model.implied_S(j, i)).abs().max()
    assert float(resid) < TOL, (
        f"[{family}/{which}] implied_S(i,j) != implied_S(j,i): "
        f"max residual {float(resid):.3e} over {len(i)} pairs (limit {TOL:.0e}). "
        f"``structure_metrics`` correlates this readout against the "
        f"ground-truth S, which *is* exactly symmetric; an order-dependent "
        f"readout would mean the reported S recovery depends on the arbitrary "
        f"order the test pairs happen to be stored in.")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("which", ["init", "trained"])
def test_implied_A_is_antisymmetric(family, which, models_at_init, models_trained):
    model = _state(which, models_at_init, models_trained)[family]
    i, j = _all_pairs()
    with torch.no_grad():
        resid = (model.implied_A(i, j) + model.implied_A(j, i)).abs().max()
    assert float(resid) < TOL, (
        f"[{family}/{which}] implied_A(i,j) != -implied_A(j,i): "
        f"max residual {float(resid):.3e} over {len(i)} pairs (limit {TOL:.0e}). "
        f"implied_A is defined as half the forward-minus-reverse ordered "
        f"difference; if it is not antisymmetric the definition is not being "
        f"honoured and 'A recovery' is scored against something else.")


# --------------------------------------------------------------------------
# 2. The algebra family: by construction, and consistent with its predictions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["init", "trained"])
def test_algebra_latent_sim_is_order_invariant(which, models_at_init,
                                               models_trained):
    model = _state(which, models_at_init, models_trained)["algebra"]
    i, j = _all_pairs()
    with torch.no_grad():
        resid = (model.latent_sim(i, j) - model.latent_sim(j, i)).abs().max()
    assert float(resid) < TOL, (
        f"[algebra/{which}] latent_sim(i,j) != latent_sim(j,i): max residual "
        f"{float(resid):.3e}. A simultaneous intervention has no order, and the "
        f"algebra family is supposed to encode that in the architecture rather "
        f"than by canonicalising indices at call time.")


@pytest.mark.parametrize("which", ["init", "trained"])
def test_algebra_ordered_difference_equals_twice_implied_A(which, models_at_init,
                                                           models_trained):
    """latent_ordered(i,j) - latent_ordered(j,i) == 2 A(i,j) -- the model-side
    twin of the generator identity tested in tests/test_generator.py."""
    model = _state(which, models_at_init, models_trained)["algebra"]
    i, j = _all_pairs()
    with torch.no_grad():
        lhs = model.latent_ordered(i, j) - model.latent_ordered(j, i)
        rhs = 2.0 * model.implied_A(i, j)
        resid = (lhs - rhs).abs().max()
        a_scale = float(rhs.pow(2).mean().sqrt())
    assert float(resid) < TOL, (
        f"[algebra/{which}] latent_ordered(i,j) - latent_ordered(j,i) != "
        f"2*implied_A(i,j): max residual {float(resid):.3e} "
        f"(rms of 2A = {a_scale:.4f}). implied_A would then not be the order "
        f"effect the model actually predicts.")


@pytest.mark.parametrize("which", ["init", "trained"])
def test_algebra_pair_network_is_symmetric_by_construction(which, models_at_init,
                                                           models_trained):
    model = _state(which, models_at_init, models_trained)["algebra"]
    i, j = _all_pairs()
    with torch.no_grad():
        asym = float((model.raw_sim_term(i, j)
                      - model.raw_sim_term(j, i)).pow(2).mean().sqrt())
    assert asym < TOL, (
        f"[algebra/{which}] the algebra model's raw simultaneous term is "
        f"order-dependent (rms asymmetry {asym:.3e}); its symmetry is supposed "
        f"to come from the architecture, not from canonicalising indices.")


# --------------------------------------------------------------------------
# 3. The baseline must NOT already have the constraint
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["unconstrained", "shared_pair"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unconstrained_pair_network_is_not_symmetric_at_init(family, seed):
    """If the baseline's pair network were symmetric at random init, the
    constraint being tested would be free and the comparison vacuous."""
    model = build_model(family, _model_cfg(seed=seed)).eval()
    i, j = _all_pairs()
    with torch.no_grad():
        raw_ij = model.raw_sim_term(i, j)
        raw_ji = model.raw_sim_term(j, i)
        asym = float((raw_ij - raw_ji).pow(2).mean().sqrt())
        scale = float(raw_ij.pow(2).mean().sqrt())
    rel = asym / (scale + 1e-12)
    assert asym > 1e-4, (
        f"[{family}/seed={seed}] raw_sim_term(i,j) == raw_sim_term(j,i) at "
        f"random initialisation (rms difference {asym:.3e}). The 'unconstrained' "
        f"baseline already satisfies the symmetry the algebra family is "
        f"credited with imposing, so the Phase 1 contrast would measure nothing.")
    assert rel > 0.05, (
        f"[{family}/seed={seed}] the baseline pair network is essentially "
        f"symmetric already: rms|raw(i,j) - raw(j,i)| = {asym:.4f} against an "
        f"output rms of {scale:.4f} (ratio {rel:.3f} < 0.05).")


# --------------------------------------------------------------------------
# 4. The additive family really is additive
# --------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["init", "trained"])
def test_additive_has_exactly_zero_interaction_terms(which, models_at_init,
                                                     models_trained):
    model = _state(which, models_at_init, models_trained)["additive"]
    i, j = _all_pairs()
    with torch.no_grad():
        S = model.implied_S(i, j)
        A = model.implied_A(i, j)
    assert float(S.abs().max()) == 0.0, (
        f"[additive/{which}] implied_S is not exactly zero (max "
        f"{float(S.abs().max()):.3e}). The additive family is the null model; "
        f"a nonzero pair term would make the null baseline something else.")
    assert float(A.abs().max()) == 0.0, (
        f"[additive/{which}] implied_A is not exactly zero (max "
        f"{float(A.abs().max()):.3e}).")


@pytest.mark.parametrize("which", ["init", "trained"])
def test_additive_prediction_for_a_pair_is_v_i_plus_v_j(which, models_at_init,
                                                        models_trained):
    model = _state(which, models_at_init, models_trained)["additive"]
    i, j = _all_pairs()
    with torch.no_grad():
        expected = model.predict_single(i) + model.predict_single(j)
        sim = model.predict_sim(i, j)
        ordered = model.predict_ordered(i, j)
        rev = model.predict_ordered(j, i)
    for name, got in (("sim", sim), ("ordered i->j", ordered),
                      ("ordered j->i", rev)):
        resid = float((got - expected).abs().max())
        assert resid < TOL, (
            f"[additive/{which}] predict_{name} != v_i + v_j: max residual "
            f"{resid:.3e}. The additive family has no pair term, so all three "
            f"row types must collapse to the same additive prediction.")


# --------------------------------------------------------------------------
# 5. Capacity matching
# --------------------------------------------------------------------------
CAPACITY_TOL = 0.05


@pytest.mark.parametrize("family", ["unconstrained", "shared_pair"])
# "outer" is the mode the reported sweep actually runs (configs/hparams.json).
# It was missing here, so the property the whole capacity-matched comparison
# rests on was covered at every width except the shipped one (review finding
# R9f). emb_dim=8 is the shipped embedding size.
@pytest.mark.parametrize("emb_dim,pair_mode", [(4, "concat"), (16, "concat"),
                                               (8, "concat_outer"),
                                               (16, "concat_outer"),
                                               (8, "outer"), (16, "outer")])
def test_match_pair_capacity_equalises_parameter_counts(family, emb_dim,
                                                        pair_mode):
    cfg = _model_cfg(emb_dim=emb_dim, pair_mode=pair_mode)
    reference = build_model("algebra", cfg)

    hidden = match_pair_capacity(family, cfg, reference_family="algebra")
    matched = build_model(family, replace(cfg, pair_hidden=hidden))

    ref_pair, got_pair = reference.n_pair_params(), matched.n_pair_params()
    ref_tot, got_tot = reference.n_params(), matched.n_params()
    pair_gap = abs(got_pair - ref_pair) / ref_pair
    tot_gap = abs(got_tot - ref_tot) / ref_tot

    detail = (f"[{family}, emb_dim={emb_dim}, pair_mode={pair_mode}] "
              f"chosen pair_hidden={hidden}: "
              f"pair params {got_pair} vs algebra's {ref_pair} "
              f"({pair_gap:.2%}); total params {got_tot} vs {ref_tot} "
              f"({tot_gap:.2%})")

    assert pair_gap < CAPACITY_TOL, (
        f"{detail}. The algebra model runs two pair MLPs and the baselines run "
        f"one wider MLP with 2d outputs; unmatched, they differ by ~2x and any "
        f"win could be attributed to capacity instead of structure.")
    assert tot_gap < CAPACITY_TOL, (
        f"{detail}. Pair parameters match but *total* parameters do not, so the "
        f"models are still not comparable on size.")


def test_unmatched_capacity_really_is_unmatched():
    """The control condition (capacity_match=False) must differ materially --
    otherwise it is not a control, it is a duplicate of the main condition."""
    cfg = _model_cfg(emb_dim=8, pair_mode="concat_outer", pair_hidden=48)
    ref = build_model("algebra", cfg).n_pair_params()
    nat = build_model("unconstrained", cfg).n_pair_params()
    ratio = nat / ref
    assert ratio < 0.75 or ratio > 1.25, (
        f"at the shared pair_hidden={cfg.pair_hidden} the unconstrained pair "
        f"net has {nat} parameters against algebra's {ref} (ratio {ratio:.2f}), "
        f"i.e. they are already matched, so 'control_unmatched_capacity' is not "
        f"testing anything.")


def test_match_pair_capacity_is_a_noop_for_the_additive_family():
    cfg = _model_cfg(pair_hidden=48)
    assert match_pair_capacity("additive", cfg) == cfg.pair_hidden
    assert build_model("additive", cfg).n_pair_params() == 0


# --------------------------------------------------------------------------
# 6. Shapes and finiteness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("which", ["init", "trained"])
def test_predictions_have_the_right_shape_and_are_finite(family, which,
                                                         models_at_init,
                                                         models_trained):
    model = _state(which, models_at_init, models_trained)[family]
    i, j = _all_pairs()
    n = len(i)
    with torch.no_grad():
        outputs = {
            "predict_single": model.predict_single(i),
            "predict_sim": model.predict_sim(i, j),
            "predict_ordered": model.predict_ordered(i, j),
            "implied_S": model.implied_S(i, j),
            "implied_A": model.implied_A(i, j),
            "implied_S_projected": model.implied_S_projected(i, j),
            "raw_sim_term": model.raw_sim_term(i, j),
        }
    for name, out in outputs.items():
        assert tuple(out.shape) == (n, OUT_DIM), (
            f"[{family}/{which}] {name} returned shape {tuple(out.shape)}, "
            f"expected ({n}, {OUT_DIM}).")
        assert torch.isfinite(out).all(), (
            f"[{family}/{which}] {name} produced "
            f"{int((~torch.isfinite(out)).sum())} non-finite values.")


@pytest.mark.parametrize("family", FAMILIES)
def test_build_model_reports_the_expected_family(family):
    assert build_model(family, _model_cfg()).family == family


def test_build_model_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown family"):
        build_model("not_a_family", _model_cfg())  # type: ignore[arg-type]


@pytest.mark.parametrize("pair_mode,expected", [("concat", 2 * 8),
                                                ("concat_outer", 2 * 8 + 64)])
def test_pair_feature_dimension(pair_mode, expected):
    phi = PairFeatures(8, pair_mode)
    assert phi.out_dim == expected
    ei = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
    ej = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    assert tuple(phi(ei, ej).shape) == (5, expected)


def test_model_construction_is_deterministic_in_its_seed():
    a = build_model("algebra", _model_cfg(seed=5))
    b = build_model("algebra", _model_cfg(seed=5))
    c = build_model("algebra", _model_cfg(seed=6))
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), (
            f"parameter '{na}' differs between two models built with the same "
            f"seed; paired-by-seed family comparisons assume identical init "
            f"conditions.")
    assert not torch.equal(a.emb.weight, c.emb.weight), (
        "seeds 5 and 6 produced identical embeddings; the model seed is ignored.")
