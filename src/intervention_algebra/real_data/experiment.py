"""One Phase 2 condition: (screen, coverage, family, split seed, init seed).

A "condition" is everything that identifies a run except the seeds, matching the
Phase 1 convention so the two phases can be talked about in the same language.
Each call fits one model on one nested coverage split and returns a flat dict
that is appended to a JSONL -- no analysis happens here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import koplev
from .evaluate import (COLLAPSE_RATIO, antisymmetry_diagnostics, directional_frame,
                       directional_metrics, ordered_metrics)
from .models import RealModelConfig, build_model, match_pair_hidden
from .splits import assert_no_pair_leakage, connectivity_report, make_coverage_splits
from .train import TrainConfig, predict, to_tensors, train_model


@dataclass(frozen=True)
class Phase2Config:
    screen: str = "A375"
    coverage: float = 0.10
    family: str = "structured"
    split_seed: int = 0
    init_seed: int = 0
    coverages: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70)
    emb_dim: int = 16
    pair_hidden: int = 48
    n_hidden_layers: int = 2
    min_train_degree: int = 3
    val_fraction: float = 0.15
    train: TrainConfig = TrainConfig()
    #: Destroys schedule direction in the *training* rows while leaving the
    #: unordered pair and its two values intact. Evaluation is untouched. A
    #: model that still predicts direction after this is reading something other
    #: than the schedule.
    shuffle_train_direction: bool = False
    tag: str = "main"


def _shuffled_direction(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Swap ``y`` between the two orderings of a pair, independently per pair.

    Preserves every marginal the unordered pair supports -- the multiset of the
    pair's two values, hence its symmetric part -- and randomises only which
    ordering each value is attached to.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    y = out["y"].to_numpy().copy()
    pos = {(a, b): k for k, (a, b) in enumerate(zip(out["i"], out["j"]))}
    # sorted(), not set iteration order: the RNG is consumed one draw per pair,
    # so the order pairs are visited in decides which pairs get swapped. Set
    # order happens to be stable for int tuples, but relying on that would make
    # the control's reproducibility an accident of CPython's hash table.
    pairs = sorted(set(out["pair"]))
    n_swapped = 0
    for (a, b) in pairs:
        ka, kb = pos.get((a, b)), pos.get((b, a))
        if ka is None or kb is None:
            raise AssertionError(
                f"pair ({a}, {b}) is missing a direction in the frame being "
                f"shuffled; the control would silently leave it unshuffled")
        if rng.random() < 0.5:
            y[ka], y[kb] = y[kb], y[ka]
            n_swapped += 1
    out["y"] = y
    # The symmetric part of every pair is untouched by construction (the two
    # values are permuted, not altered), so only the direction is destroyed.
    out.attrs["n_pairs_swapped"] = n_swapped
    out.attrs["n_pairs"] = len(pairs)
    return out


def run_condition(cfg: Phase2Config, raw_dir: Path = koplev.DEFAULT_RAW_DIR,
                  screen: koplev.Screen | None = None) -> dict:
    screen = screen or koplev.load_screen(cfg.screen, raw_dir)
    frame = screen.frame

    splits = make_coverage_splits(
        frame, screen.n_drugs, cfg.coverages, split_seed=cfg.split_seed,
        val_fraction=cfg.val_fraction, min_train_degree=cfg.min_train_degree)
    if cfg.coverage not in splits:
        raise ValueError(f"coverage {cfg.coverage} not in grid {cfg.coverages}")
    split = splits[cfg.coverage]
    assert_no_pair_leakage(split, frame)

    train_rows = split.rows(frame, "train")
    val_rows = split.rows(frame, "val")
    test_rows = split.rows(frame, "test")
    if cfg.shuffle_train_direction:
        train_rows = _shuffled_direction(train_rows, seed=cfg.split_seed + 7919)
        val_rows = _shuffled_direction(val_rows, seed=cfg.split_seed + 104729)

    # Standardisation uses training rows only.
    y_mean = float(train_rows["y"].mean())
    y_std = float(train_rows["y"].std())

    mcfg = RealModelConfig(
        n_drugs=screen.n_drugs, emb_dim=cfg.emb_dim,
        pair_hidden=cfg.pair_hidden, n_hidden_layers=cfg.n_hidden_layers,
        seed=cfg.init_seed)

    def build(seed: int):
        return build_model(cfg.family, replace(mcfg, seed=seed))

    fit = train_model(build, to_tensors(train_rows, y_mean, y_std),
                      to_tensors(val_rows, y_mean, y_std), cfg.train,
                      seed=cfg.init_seed)

    pred = predict(fit.model, test_rows, y_mean, y_std)
    d = directional_frame(test_rows, pred)
    noise = koplev.measurement_noise_sd(raw_dir)
    threshold = noise["threshold_2sd_D"][cfg.screen]

    # Collapse is measured on the pair head alone, on the TEST pairs, against
    # the head's own symmetric output -- a ratio, so it does not depend on the
    # arbitrary scale of the standardised target.
    import torch as _torch
    _i = _torch.as_tensor(test_rows["i"].to_numpy(), dtype=_torch.long)
    _j = _torch.as_tensor(test_rows["j"].to_numpy(), dtype=_torch.long)
    with _torch.no_grad():
        head_a = fit.model.pair_head_A(_i, _j)
        pair = fit.model.pair_term(_i, _j)
        head_s = 0.5 * (pair + fit.model.pair_term(_j, _i))
        first_a = fit.model.first_order_A(_i, _j)
    head_a_rms = float(head_a.pow(2).mean().sqrt())
    head_s_rms = float(head_s.pow(2).mean().sqrt())
    first_a_rms = float(first_a.pow(2).mean().sqrt())

    om = ordered_metrics(test_rows["y"].to_numpy(), pred)
    dm = directional_metrics(d, threshold)
    ad = antisymmetry_diagnostics(pred, test_rows, d)

    # Baseline every error against predicting the training mean, so the numbers
    # are readable without holding the screen's variance in your head.
    const = ordered_metrics(test_rows["y"].to_numpy(),
                            np.full(len(test_rows), y_mean))

    row = {
        "tag": cfg.tag, "screen": cfg.screen, "coverage": cfg.coverage,
        "family": cfg.family, "split_seed": cfg.split_seed,
        "init_seed": cfg.init_seed,
        "shuffle_train_direction": cfg.shuffle_train_direction,
        "n_drugs": screen.n_drugs,
        "n_train_rows": len(train_rows), "n_val_rows": len(val_rows),
        "n_test_rows": len(test_rows),
        "n_params": fit.model.n_params(),
        "n_pair_params": fit.model.n_pair_params(),
        "n_first_order_params": fit.model.n_first_order_params(),
        "pair_hidden": fit.model.cfg.pair_hidden,
        "pair_evals_per_row": fit.model.pair_evals_per_row,
        "train_loss": fit.train_loss, "val_loss": fit.val_loss,
        "epochs_run": fit.epochs_run,
        "train_head_A_rms_std_units": fit.train_head_A_rms,
        # The three antisymmetric scales, kept apart on purpose. ``head_A`` is
        # the only one that can detect the degenerate basin; ``first_order_A``
        # is the per-drug tendency every family gets for free and which
        # dominates on this screen.
        "test_head_A_rms_std_units": head_a_rms,
        "test_head_sym_rms_std_units": head_s_rms,
        "test_first_order_A_rms_std_units": first_a_rms,
        "test_head_A_over_sym": head_a_rms / max(head_s_rms, 1e-12),
        "restarts": fit.restarts,
        "restart_train_loss_worst": max(r["train_loss"] for r in fit.restarts),
        "restart_train_loss_ratio": (max(r["train_loss"] for r in fit.restarts)
                                     / max(min(r["train_loss"] for r in fit.restarts),
                                           1e-12)),
        "n_restarts_collapsed": sum(
            1 for r in fit.restarts if r["train_head_A_rms"] < 1e-6),
        "y_mean_train": y_mean, "y_std_train": y_std,
        "ordering_threshold_source": "2 * sqrt(2) * median posterior synergy_sd",
        "const_baseline_mse": const["mse"], "const_baseline_mae": const["mae"],
    }
    row.update({f"test_{k}": v for k, v in om.items()})
    row.update({f"test_{k}": v for k, v in dm.items()})
    row.update({f"test_{k}": v for k, v in ad.items()})
    # A family whose pair term is symmetric by construction has no head to
    # collapse; flagging it would bury the one number this diagnostic exists to
    # surface under three that are structurally pinned to zero.
    row["head_can_be_antisymmetric"] = fit.model.family in ("unrestricted",
                                                            "structured")
    row["collapsed"] = bool(row["head_can_be_antisymmetric"]
                            and row["test_head_A_over_sym"] < COLLAPSE_RATIO)
    row.update({f"split_{k}": v
                for k, v in connectivity_report(split, screen.n_drugs).items()})
    row["config"] = json.loads(json.dumps(asdict(cfg), default=str))
    return row
