"""A run must be a function of its config, and only of its config.

Two separate properties, both load-bearing:

* **Bit-identical replay.**  ``run_experiment(cfg)`` twice must give the same
  numbers to the last bit.  Anything less means an unseeded source of randomness
  is in the loop, and then a family-vs-family difference of a few percent cannot
  be distinguished from run-to-run jitter.

* **The seed is actually used.**  A config whose ``seed`` is silently ignored
  would produce five identical "independent" runs, and the error bars in the
  README would be measuring nothing.  The bit-identity test above passes
  trivially in that world, so it must be paired with this one.

* **``seeded()`` pairs the families.**  The headline comparison is paired by
  seed: family A and family B at seed s must see the *same* generated system and
  the *same* train/val/test split, or the comparison is confounded by which
  system each happened to draw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from intervention_algebra.experiment import (
    RunConfig, _resolve_model_config, run_experiment, seeded)
from intervention_algebra.generator import (
    SystemConfig, generate_observations, make_system)
from intervention_algebra.models import ModelConfig
from intervention_algebra.splits import SplitConfig, make_pair_split
from intervention_algebra.train import TrainConfig

FAMILIES = ("additive", "unconstrained", "shared_pair", "algebra")

# Keys that must replay bit-for-bit.  These are the four the study reports or
# selects on; if they are stable, so is everything they are derived from.
REPLAY_KEYS = ("test_mse", "test_S_pearson", "best_val_mse", "best_epoch")


def _cfg(family: str = "algebra", seed: int = 0, max_epochs: int = 80) -> RunConfig:
    return seeded(RunConfig(
        family=family,
        system=SystemConfig(n_interventions=24, latent_dim=2, n_factors=3,
                            regime="both", sparsity=0.5, noise_std=0.05,
                            s_scale=1.0, a_scale=1.0),
        split=SplitConfig(pair_coverage=0.4, eligibility_coverage=0.2),
        model=ModelConfig(emb_dim=4, pair_hidden=32),
        train=TrainConfig(lr=5e-3, max_epochs=max_epochs, patience=max_epochs),
    ), seed)


def _same(a, b) -> bool:
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


# --------------------------------------------------------------------------
# 1. Bit-identical replay
# --------------------------------------------------------------------------
def test_running_the_same_config_twice_is_bit_identical():
    cfg = _cfg()
    first, second = run_experiment(cfg), run_experiment(cfg)
    for key in REPLAY_KEYS:
        assert _same(first[key], second[key]), (
            f"'{key}' is not reproducible: {first[key]!r} on the first run vs "
            f"{second[key]!r} on the second, from an identical RunConfig. Some "
            f"randomness is not derived from the config, so a family-vs-family "
            f"difference cannot be separated from run-to-run jitter. "
            f"(Full first run: "
            f"{ {k: first[k] for k in REPLAY_KEYS} })")


@pytest.mark.parametrize("family", FAMILIES)
def test_replay_is_bit_identical_for_every_family(family):
    cfg = _cfg(family=family, max_epochs=60)
    first, second = run_experiment(cfg), run_experiment(cfg)
    diffs = {k: (first[k], second[k]) for k in REPLAY_KEYS
             if not _same(first[k], second[k])}
    assert not diffs, (
        f"[{family}] non-reproducible keys (first, second): {diffs}")


# --------------------------------------------------------------------------
# 2. The seed is not ignored
# --------------------------------------------------------------------------
def test_different_seeds_give_different_results():
    a, b = run_experiment(_cfg(seed=0)), run_experiment(_cfg(seed=1))
    assert a["test_mse"] != b["test_mse"], (
        f"seeds 0 and 1 produced exactly the same test_mse "
        f"({a['test_mse']!r}). Either the seed is ignored or every seed draws "
        f"the same system; the across-seed spread reported as an error bar "
        f"would then be one measurement repeated.")
    # ...and the difference must be a real one, not the last bit.
    rel = abs(a["test_mse"] - b["test_mse"]) / max(a["test_mse"], b["test_mse"])
    assert rel > 1e-6, (
        f"seeds 0 and 1 differ by only {rel:.3e} in relative test_mse "
        f"({a['test_mse']:.8f} vs {b['test_mse']:.8f}); the seed is barely "
        f"affecting the run.")


def test_the_split_seed_is_not_ignored():
    system = make_system(SystemConfig(n_interventions=24, latent_dim=2,
                                      n_factors=3, seed=0))
    a = make_pair_split(system, SplitConfig(pair_coverage=0.4,
                                            eligibility_coverage=0.2, seed=1))
    b = make_pair_split(system, SplitConfig(pair_coverage=0.4,
                                            eligibility_coverage=0.2, seed=2))
    assert not np.array_equal(a.test_pairs, b.test_pairs), (
        "two different split seeds produced identical test pairs; the "
        "'different split per seed' part of the paired design does nothing.")


def test_the_model_seed_is_not_ignored():
    a = run_experiment(_cfg(seed=0))
    # Same system and split, different model init only.
    base = _cfg(seed=0)
    from dataclasses import replace
    b_cfg = replace(base, model=replace(base.model, seed=base.model.seed + 1))
    b = run_experiment(b_cfg)
    assert a["test_mse"] != b["test_mse"], (
        f"changing only ModelConfig.seed left test_mse unchanged at "
        f"{a['test_mse']!r}; the model initialisation seed is not reaching the "
        f"parameters.")


# --------------------------------------------------------------------------
# 3. seeded() makes the family comparison paired
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family_a,family_b", [
    ("algebra", "unconstrained"),
    ("algebra", "additive"),
    ("shared_pair", "unconstrained"),
])
def test_seeded_gives_the_same_system_and_split_to_different_families(
        family_a, family_b):
    seed = 7
    cfg_a = _cfg(family=family_a, seed=seed)
    cfg_b = _cfg(family=family_b, seed=seed)

    assert cfg_a.system == cfg_b.system, (
        f"seeded() produced different SystemConfigs for '{family_a}' and "
        f"'{family_b}' at seed {seed}: {cfg_a.system} vs {cfg_b.system}.")
    assert cfg_a.split == cfg_b.split, (
        f"seeded() produced different SplitConfigs for '{family_a}' and "
        f"'{family_b}' at seed {seed}: {cfg_a.split} vs {cfg_b.split}.")

    sys_a, sys_b = make_system(cfg_a.system), make_system(cfg_b.system)
    for name in ("U", "v", "S", "A", "mask", "gate"):
        assert np.array_equal(getattr(sys_a, name), getattr(sys_b, name)), (
            f"the generated system differs between '{family_a}' and "
            f"'{family_b}' at seed {seed} (field '{name}'). The paired "
            f"comparison assumes both families are scored on the same problem; "
            f"otherwise a difference in means could be a difference in which "
            f"systems each family drew.")

    split_a = make_pair_split(sys_a, cfg_a.split)
    split_b = make_pair_split(sys_b, cfg_b.split)
    for name in ("train_pairs", "val_pairs", "test_pairs"):
        assert np.array_equal(getattr(split_a, name), getattr(split_b, name)), (
            f"the '{name}' split differs between '{family_a}' and '{family_b}' "
            f"at seed {seed}.")

    # And the observation rows -- including the realised noise draw.
    tbl_a = generate_observations(sys_a, split_a.train_pairs,
                                  rng=np.random.default_rng(
                                      cfg_a.system.seed + 77_000))
    tbl_b = generate_observations(sys_b, split_b.train_pairs,
                                  rng=np.random.default_rng(
                                      cfg_b.system.seed + 77_000))
    assert np.array_equal(tbl_a.y, tbl_b.y), (
        f"'{family_a}' and '{family_b}' at seed {seed} see different noise "
        f"realisations of the same system; the pairing is incomplete.")


def test_seeded_derives_every_subseed_from_the_run_seed():
    base = RunConfig()
    for seed in (0, 3, 41):
        cfg = seeded(base, seed)
        assert cfg.system.seed == seed
        assert cfg.split.seed == seed + 1_000
        assert cfg.model.seed == seed + 2_000
        assert cfg.train.seed == seed + 3_000
    a, b = seeded(base, 0), seeded(base, 1)
    assert a.split.seed != b.split.seed and a.model.seed != b.model.seed, (
        "distinct run seeds must produce distinct sub-seeds, otherwise part of "
        "the pipeline is effectively fixed across 'independent' runs.")


def test_capacity_matching_does_not_change_the_data_seen():
    """The pair-hidden search must touch the model only. If it perturbed the RNG
    stream feeding the generator or the split, the capacity control would be
    confounded with a different dataset."""
    cfg_matched = _cfg(family="unconstrained")
    from dataclasses import replace
    cfg_unmatched = replace(cfg_matched, capacity_match=False)

    m1 = _resolve_model_config(cfg_matched)
    m2 = _resolve_model_config(cfg_unmatched)
    assert m1.pair_hidden != m2.pair_hidden, (
        f"capacity_match had no effect: pair_hidden={m1.pair_hidden} either way, "
        f"so the 'unmatched capacity' control is a duplicate condition.")

    r1, r2 = run_experiment(cfg_matched), run_experiment(cfg_unmatched)
    for key in ("n_train_pairs", "n_val_pairs", "n_test_pairs"):
        assert r1[key] == r2[key], (
            f"toggling capacity_match changed '{key}' ({r1[key]} vs {r2[key]}); "
            f"it must change the model only.")
    assert r1["n_pair_params"] != r2["n_pair_params"], (
        f"capacity_match left the pair parameter count unchanged at "
        f"{r1['n_pair_params']}.")
