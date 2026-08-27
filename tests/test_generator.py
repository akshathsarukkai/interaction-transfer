"""The ground truth must actually be the ground truth.

Everything downstream -- the model families, the metrics, the headline claim --
is defined relative to the decomposition

    z(i)        = v_i
    z({i,j})    = v_i + v_j + S_ij
    z(i -> j)   = v_i + v_j + S_ij + A_ij
    z(j -> i)   = v_i + v_j + S_ij - A_ij

with ``S`` symmetric and ``A`` antisymmetric.  If the generator does not produce
data obeying those equations exactly, then "the algebra model imposes the true
constraint by construction" is false and every comparison in the study is
measuring something else.  These tests check the equations numerically rather
than trusting the docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from intervention_algebra.generator import (
    ORD, SIM, SINGLE, SystemConfig, generate_observations, make_system)

N = 16
SPARSITY_MODES = ("latent", "module", "random")
REGIMES = ("independent", "symmetric", "antisymmetric", "both")

# Exact-algebra tolerance.  ``make_system`` explicitly re-symmetrises S and
# antisymmetrises A as its last step, so the residual is not "small", it is
# float64 round-off of a single add/subtract -- i.e. ~1e-17 in practice.
EXACT = 1e-12


def _cfg(**kw) -> SystemConfig:
    base = dict(n_interventions=N, latent_dim=3, n_factors=4, sparsity=0.3,
                n_modules=5, noise_std=0.0, seed=7)
    base.update(kw)
    return SystemConfig(**base)


# --------------------------------------------------------------------------
# 1. Exact algebraic invariants of the ground truth
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
@pytest.mark.parametrize("regime", REGIMES)
def test_ground_truth_S_is_symmetric_and_A_antisymmetric(sparsity_mode, regime):
    s = make_system(_cfg(sparsity_mode=sparsity_mode, regime=regime))

    sym_res = float(np.abs(s.S - s.S.transpose(1, 0, 2)).max())
    anti_res = float(np.abs(s.A + s.A.transpose(1, 0, 2)).max())

    assert sym_res < EXACT, (
        f"[{sparsity_mode}/{regime}] ground-truth S is not symmetric: "
        f"max|S_ij - S_ji| = {sym_res:.3e} (limit {EXACT:.0e}). The algebra "
        f"model imposes S(i,j)=S(j,i) by construction, so an asymmetric truth "
        f"would make that constraint a *misspecification* rather than the "
        f"correct inductive bias the study claims to test.")
    assert anti_res < EXACT, (
        f"[{sparsity_mode}/{regime}] ground-truth A is not antisymmetric: "
        f"max|A_ij + A_ji| = {anti_res:.3e} (limit {EXACT:.0e}).")


@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
@pytest.mark.parametrize("regime", REGIMES)
def test_A_has_zero_diagonal(sparsity_mode, regime):
    s = make_system(_cfg(sparsity_mode=sparsity_mode, regime=regime))
    diag = s.A[np.arange(s.n), np.arange(s.n)]
    worst = float(np.abs(diag).max())
    assert worst < EXACT, (
        f"[{sparsity_mode}/{regime}] A_ii should be identically zero "
        f"(it is forced by A = -A^T), got max|A_ii| = {worst:.3e}.")


# --------------------------------------------------------------------------
# 2. Each regime must zero exactly the right things
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
@pytest.mark.parametrize("regime,S_zero,A_zero,mask_zero", [
    ("independent",   True,  True,  True),
    ("symmetric",     False, True,  False),
    ("antisymmetric", True,  False, False),
    ("both",          False, False, False),
])
def test_regime_zeroes_the_right_components(sparsity_mode, regime,
                                            S_zero, A_zero, mask_zero):
    s = make_system(_cfg(sparsity_mode=sparsity_mode, regime=regime))
    S_rms = float(np.sqrt((s.S ** 2).mean()))
    A_rms = float(np.sqrt((s.A ** 2).mean()))
    n_mask = int(s.mask.sum())

    if S_zero:
        assert S_rms == 0.0, (
            f"regime='{regime}' must have S identically zero, got "
            f"rms(S)={S_rms:.3e}. Runs in this regime are used as the "
            f"'no symmetric interaction' control; a nonzero S makes the "
            f"control meaningless.")
    else:
        assert S_rms > 1e-6, (
            f"regime='{regime}' must have a nonzero S, got rms(S)={S_rms:.3e}. "
            f"With no signal the regime silently degenerates to 'independent' "
            f"and any family would score the same.")

    if A_zero:
        assert A_rms == 0.0, (
            f"regime='{regime}' must have A identically zero, got "
            f"rms(A)={A_rms:.3e}.")
    else:
        assert A_rms > 1e-6, (
            f"regime='{regime}' must have a nonzero A, got rms(A)={A_rms:.3e}.")

    if mask_zero:
        assert n_mask == 0, (
            f"regime='{regime}' has no interactions, so the interaction mask "
            f"must be all-False; {n_mask} entries are True. Topology AUROC is "
            f"scored against this mask and would be reading noise.")
    else:
        assert n_mask > 0, (
            f"regime='{regime}' must have at least one interacting pair; the "
            f"mask is all-False.")


# --------------------------------------------------------------------------
# 3. THE control: generated data must satisfy the defining equations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
@pytest.mark.parametrize("regime", REGIMES)
def test_forward_reverse_consistency_of_generated_rows(sparsity_mode, regime):
    """z(i->j) + z(j->i) == 2 z({i,j}),  z(i->j) - z(j->i) == 2 A_ij,
    and z({i,j}) - z(i) - z(j) == S_ij, read off the actual observation table.

    This is the check that the *rows the models are trained on* encode the
    algebra, not merely that the S/A arrays do.  ``noise_std=0`` so ``z`` and the
    equations are directly comparable.
    """
    s = make_system(_cfg(sparsity_mode=sparsity_mode, regime=regime,
                         noise_std=0.0))
    pairs = s.all_pairs()
    table = generate_observations(s, pairs, include_singles=True)

    # Index the rows by (kind, i, j) instead of relying on row ordering.
    single = {int(a): z for a, k, z in zip(table.i, table.kind, table.z)
              if k == SINGLE}
    sim = {(int(a), int(b)): z
           for a, b, k, z in zip(table.i, table.j, table.kind, table.z)
           if k == SIM}
    ordered = {(int(a), int(b)): z
               for a, b, k, z in zip(table.i, table.j, table.kind, table.z)
               if k == ORD}

    assert len(single) == s.n, f"expected {s.n} SINGLE rows, got {len(single)}"
    assert len(sim) == len(pairs), (
        f"expected {len(pairs)} SIM rows, got {len(sim)}")
    assert len(ordered) == 2 * len(pairs), (
        f"expected {2 * len(pairs)} ORD rows (both orders of every pair), got "
        f"{len(ordered)} -- if only one order is emitted, the antisymmetric "
        f"component is unobservable and the benchmark cannot test it.")

    e_sum = e_diff = e_sim = 0.0
    for a, b in pairs:
        a, b = int(a), int(b)
        f, r = ordered[(a, b)], ordered[(b, a)]
        e_sum = max(e_sum, float(np.abs(f + r - 2.0 * sim[(a, b)]).max()))
        e_diff = max(e_diff, float(np.abs(f - r - 2.0 * s.A[a, b]).max()))
        e_sim = max(e_sim, float(
            np.abs(sim[(a, b)] - single[a] - single[b] - s.S[a, b]).max()))

    ctx = f"[{sparsity_mode}/{regime}]"
    assert e_sum < EXACT, (
        f"{ctx} z(i->j) + z(j->i) != 2 z(i,j): max residual {e_sum:.3e}. The "
        f"two ordered outcomes must average to the simultaneous one -- that is "
        f"the whole content of 'the ordered rows differ from the simultaneous "
        f"one by +/- A'.")
    assert e_diff < EXACT, (
        f"{ctx} z(i->j) - z(j->i) != 2 A_ij: max residual {e_diff:.3e}. The "
        f"order effect the models are asked to predict is not the ground-truth "
        f"A, so 'A recovery' would be scored against the wrong quantity.")
    assert e_sim < EXACT, (
        f"{ctx} z(i,j) - z(i) - z(j) != S_ij: max residual {e_sim:.3e}.")


# --------------------------------------------------------------------------
# 4. Sparsity: the realised topology density must track the request
# --------------------------------------------------------------------------
# ``latent`` thresholds a quantile so it hits the target essentially exactly.
# ``module`` can only realise densities on a coarse grid of module-pair counts
# and ``random`` is a binomial draw over ~n^2/2 pairs, so both wobble
# seed-to-seed.  Tolerances measured over seeds 0-4 at n=24: worst single-seed
# deviation 0.069 (module), worst 5-seed mean deviation 0.025.
DENSITY_TOL_PER_SEED = {"latent": 0.02, "module": 0.12, "random": 0.12}
DENSITY_TOL_MEAN = {"latent": 0.01, "module": 0.05, "random": 0.05}


@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
@pytest.mark.parametrize("sparsity", [0.15, 0.25, 0.40])
def test_mask_density_matches_requested_sparsity(sparsity_mode, sparsity):
    n = 24
    iu = np.triu_indices(n, k=1)
    densities = []
    for seed in range(5):
        s = make_system(_cfg(n_interventions=n, sparsity=sparsity,
                             sparsity_mode=sparsity_mode, n_modules=6,
                             regime="both", seed=seed))
        d = float(s.mask[iu].mean())
        densities.append(d)
        assert abs(d - sparsity) <= DENSITY_TOL_PER_SEED[sparsity_mode], (
            f"[{sparsity_mode}] seed={seed}: requested sparsity={sparsity}, "
            f"realised interacting-pair density={d:.3f} "
            f"(tolerance {DENSITY_TOL_PER_SEED[sparsity_mode]}). The reported "
            f"'sparsity' of a condition would not describe the data.")
    mean = float(np.mean(densities))
    assert abs(mean - sparsity) <= DENSITY_TOL_MEAN[sparsity_mode], (
        f"[{sparsity_mode}] mean density over 5 seeds is {mean:.3f} for "
        f"requested sparsity={sparsity} (per-seed: "
        f"{np.round(densities, 3).tolist()}); the estimator is biased, not "
        f"merely noisy.")


@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
def test_mask_is_symmetric_with_zero_diagonal(sparsity_mode):
    s = make_system(_cfg(sparsity_mode=sparsity_mode, regime="both"))
    assert np.array_equal(s.mask, s.mask.T), (
        f"[{sparsity_mode}] the interaction mask is not symmetric; 'pair {{i,j}} "
        f"interacts' cannot depend on the order it is written in.")
    assert not s.mask[np.arange(s.n), np.arange(s.n)].any(), (
        f"[{sparsity_mode}] the mask has True entries on the diagonal; a pair "
        f"of an intervention with itself is not a pair.")


# --------------------------------------------------------------------------
# 5. Determinism in the seed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
def test_generator_is_deterministic_in_its_seed(sparsity_mode):
    cfg = _cfg(sparsity_mode=sparsity_mode, regime="both", seed=11)
    a, b = make_system(cfg), make_system(cfg)
    for name in ("U", "v", "S", "A", "mask", "gate"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), (
            f"[{sparsity_mode}] make_system is not deterministic: '{name}' "
            f"differs between two calls with an identical config. Every "
            f"paired-by-seed comparison in the study assumes both families see "
            f"the same system.")


@pytest.mark.parametrize("sparsity_mode", SPARSITY_MODES)
def test_different_seeds_give_different_systems(sparsity_mode):
    a = make_system(_cfg(sparsity_mode=sparsity_mode, regime="both", seed=0))
    b = make_system(_cfg(sparsity_mode=sparsity_mode, regime="both", seed=1))
    assert not np.allclose(a.v, b.v), (
        f"[{sparsity_mode}] seeds 0 and 1 produce the same v; the seed is being "
        f"ignored and the 'across seeds' error bars would be one system "
        f"measured five times.")
    assert not np.allclose(a.S, b.S), (
        f"[{sparsity_mode}] seeds 0 and 1 produce the same S.")


def test_observations_are_deterministic_given_the_system_seed():
    s = make_system(_cfg(regime="both", noise_std=0.1))
    pairs = s.all_pairs()[:20]
    t1 = generate_observations(s, pairs)
    t2 = generate_observations(s, pairs)
    assert np.array_equal(t1.y, t2.y), (
        "generate_observations with no explicit rng must derive its noise from "
        "the system seed; two calls disagreed.")


# --------------------------------------------------------------------------
# 6. Noise
# --------------------------------------------------------------------------
@pytest.mark.parametrize("noise_std", [0.0, 0.05, 0.2])
def test_noise_has_approximately_the_requested_std(noise_std):
    s = make_system(_cfg(n_interventions=24, regime="both",
                         noise_std=noise_std))
    table = generate_observations(s, s.all_pairs(), include_singles=True)
    resid = table.y - table.z
    got = float(resid.std())

    if noise_std == 0.0:
        assert got == 0.0, (
            f"noise_std=0 must give y == z exactly; got residual std {got:.3e}. "
            f"A noiseless condition that is not noiseless makes the 'easy "
            f"debug system' tests unable to distinguish an optimisation bug "
            f"from an irreducible floor.")
        return

    n = resid.size
    # Sampling error of an std estimate from n independent draws is
    # ~std/sqrt(2n); n is >= 5000 here, so 10% is ~15 sigma of slack and the
    # test only fires on a real scale error (e.g. variance-vs-std confusion,
    # which would show up as a factor of ~4).
    assert abs(got - noise_std) < 0.1 * noise_std, (
        f"requested noise_std={noise_std}, realised std(y - z)={got:.4f} over "
        f"{n} values. ``noise_floor_mse`` is reported as noise_std**2 on every "
        f"run and is the reference against which test MSE is judged.")
    assert abs(float(resid.mean())) < 5.0 * noise_std / np.sqrt(n), (
        f"the observation noise has mean {float(resid.mean()):.4f}, which is "
        f"not consistent with zero -- a bias would be absorbed into the "
        f"learned first-order effects.")


def test_tanh_observation_map_is_applied():
    s = make_system(_cfg(regime="both", noise_std=0.0,
                         observation_map="tanh", obs_gain=1.0))
    table = generate_observations(s, s.all_pairs()[:30], include_singles=True)
    assert np.allclose(table.y, np.tanh(table.z)), (
        "observation_map='tanh' did not squash the latents; the "
        "non-identifiable control condition would silently be identical to the "
        "identity one.")
