"""End-to-end single experimental run: generate -> split -> train -> evaluate.

One call to :func:`run_experiment` is the atomic unit of the Phase 1 study.  It
is fully determined by its :class:`RunConfig`, and re-running the same config
reproduces the same numbers bit-for-bit on CPU (asserted in
tests/test_reproducibility.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace

import numpy as np
import torch

from .generator import (SystemConfig, generate_observations, make_system)
from .metrics import (true_v_additive_mse, interaction_variance_share,
                      invariance_diagnostics, prediction_metrics,
                      structure_metrics)
from .models import Family, ModelConfig, build_model, match_pair_capacity
from .splits import (SplitConfig, assert_all_interventions_seen,
                     assert_no_pair_leakage, make_pair_split)
from .train import TrainConfig, train_model


@dataclass(frozen=True)
class RunConfig:
    family: Family = "algebra"
    system: SystemConfig = field(default_factory=SystemConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    capacity_match: bool = True   # match the unconstrained pair net to the algebra one
    tag: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def seeded(base: RunConfig, seed: int) -> RunConfig:
    """Return the config with every sub-seed derived from a single run seed."""
    return replace(
        base,
        system=replace(base.system, seed=seed),
        split=replace(base.split, seed=seed + 1_000),
        model=replace(base.model, seed=seed + 2_000),
        train=replace(base.train, seed=seed + 3_000),
    )


def _resolve_model_config(cfg: RunConfig) -> ModelConfig:
    mcfg = replace(
        cfg.model,
        n_interventions=cfg.system.n_interventions,
        out_dim=cfg.system.latent_dim,
        observation_head=(cfg.system.observation_map != "identity"),
    )
    if cfg.capacity_match and cfg.family in ("unconstrained", "shared_pair"):
        h = match_pair_capacity(cfg.family, mcfg, reference_family="algebra")
        mcfg = replace(mcfg, pair_hidden=h)
    return mcfg


def run_experiment(cfg: RunConfig, verbose: bool = False) -> dict:
    torch.manual_seed(cfg.model.seed)

    system = make_system(cfg.system)
    split = make_pair_split(system, cfg.split)

    # One independent generator per table. Threading a single rng through
    # train -> val -> test would make the number of *training* rows determine how
    # many draws are consumed before the test noise, so the same test pair would
    # get a different observation at every coverage level -- silently defeating
    # the fixed common test set that the coverage curve depends on.
    train_tbl = generate_observations(
        system, split.train_pairs, include_singles=True,
        rng=np.random.default_rng(cfg.system.seed + 77_000))
    val_tbl = generate_observations(
        system, split.val_pairs, include_singles=False,
        rng=np.random.default_rng(cfg.system.seed + 78_000))
    test_tbl = generate_observations(
        system, split.test_pairs, include_singles=False,
        rng=np.random.default_rng(cfg.system.seed + 79_000))

    # --- leakage guards run on every single experiment, not only in tests ---
    held_out = np.concatenate([split.val_pairs, split.test_pairs]) \
        if len(split.val_pairs) or len(split.test_pairs) else np.zeros((0, 2), int)
    assert_no_pair_leakage(train_tbl, held_out, system.n)
    assert_all_interventions_seen(train_tbl, system.n)

    mcfg = _resolve_model_config(cfg)

    # Multiple initialisations, keeping the one that best fits the TRAINING
    # rows. The antisymmetric parameterisation has an exact degenerate optimum
    # (any F_A symmetric under swapping its two argument slots gives A == 0
    # identically, and the antisymmetric part of the gradient vanishes there,
    # so the basin is self-sustaining). Adam fell into it on 3/25 algebra runs,
    # which produced training MSE ~0.40 against a peer median of ~0.04 -- runs
    # that never fit the data they were shown, and so cannot speak to
    # generalisation. Selection uses training loss only, never validation or
    # test, and every family gets the identical restart budget.
    best_model, best_fit, restart_train = None, None, []
    for r in range(max(1, cfg.train.n_restarts)):
        mcfg_r = replace(mcfg, seed=mcfg.seed + 10_000 * r)
        tcfg_r = replace(cfg.train, seed=cfg.train.seed + 10_000 * r)
        model_r = build_model(cfg.family, mcfg_r)
        fit_r = train_model(model_r, train_tbl, val_tbl, tcfg_r)
        restart_train.append(fit_r["final_train_mse"])
        if best_fit is None or fit_r["final_train_mse"] < best_fit["final_train_mse"]:
            best_model, best_fit = model_r, fit_r
    model, fit = best_model, best_fit

    identifiable = cfg.system.observation_map == "identity"
    metrics: dict = {}
    metrics.update(prediction_metrics(model, test_tbl, prefix="test_"))
    metrics.update(prediction_metrics(model, train_tbl, prefix="train_"))
    metrics.update(structure_metrics(model, system, split.test_pairs,
                                     identifiable=identifiable, prefix="test_"))
    metrics.update(invariance_diagnostics(model, split.test_pairs))

    # Reference scale only -- see metrics.true_v_additive_mse for why this is
    # NOT an oracle and why its skill score must not be read as evidence of
    # interaction learning. The headline baseline is the trained additive family.
    # Same metrics at the final epoch: the control for checkpoint selection.
    _best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(fit["final_state"])
    model.eval()
    metrics.update(prediction_metrics(model, test_tbl, prefix="final_test_"))
    metrics.update(structure_metrics(model, system, split.test_pairs,
                                     identifiable=identifiable, prefix="final_test_"))
    model.load_state_dict(_best_state)
    model.eval()

    metrics["test_true_v_additive_mse"] = true_v_additive_mse(system, test_tbl)
    metrics["noise_floor_mse"] = float(cfg.system.noise_std ** 2)
    metrics["interaction_variance_share"] = interaction_variance_share(system, test_tbl)

    result = {
        "family": cfg.family,
        "tag": cfg.tag,
        "regime": cfg.system.regime,
        "sparsity_mode": cfg.system.sparsity_mode,
        "observation_map": cfg.system.observation_map,
        "pair_coverage": cfg.split.pair_coverage,
        "seed": cfg.system.seed,
        "n_params": model.n_params(),
        "n_pair_params": model.n_pair_params(),
        "pair_hidden": mcfg.pair_hidden,
        "identifiable_latents": identifiable,
        **split.summary(),
        **metrics,
        "n_restarts": int(max(1, cfg.train.n_restarts)),
        "restart_train_mses": restart_train,
        "restart_train_mse_worst": float(max(restart_train)),
        "best_val_mse": fit["best_val_mse"],
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "final_train_mse": fit["final_train_mse"],
        "config": cfg.to_dict(),
    }
    if verbose:
        print(f"[{cfg.family:13s}] cov={cfg.split.pair_coverage:.2f} "
              f"seed={cfg.system.seed} test_mse={metrics['test_mse']:.4f} "
              f"S_r={metrics.get('test_S_pearson', float('nan')):.3f} "
              f"A_r={metrics.get('test_A_pearson', float('nan')):.3f}")
    return result
