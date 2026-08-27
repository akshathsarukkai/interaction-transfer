"""Pair-level compositional splits and leakage checks.

The evaluation of interest is *seen interventions, unseen pair*:

  * every intervention appears in training (all N single-intervention rows are
    always in the training set -- this also pins the gauge, see
    docs/identifiability.md);
  * an unordered pair {i, j} is assigned **as a whole** to train, val or test.

Assigning whole pairs is essential.  If the simultaneous row of {i,j} were in
train while the ordered row i->j were in test, then
``A_ij = z(i->j) - z({i,j})`` would be a one-line subtraction and "prediction"
would be trivial reconstruction.  Likewise if i->j were in train and j->i in
test, ``z(j->i) = 2 z({i,j}) - z(i->j)`` recovers the target exactly.  Both
failure modes are eliminated by construction here and are asserted in
``assert_no_pair_leakage`` and in tests/test_splits_leakage.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import SINGLE, ObservationTable, SyntheticSystem, pair_index_map


@dataclass(frozen=True)
class SplitConfig:
    """Fixed evaluation pool + *nested* training sets.

    For a given seed the evaluation pool is chosen once and never depends on
    ``pair_coverage``; the training pairs at coverage 0.05 are a subset of those
    at 0.10, and so on.  Without this, every point of the pair-coverage curve
    would be measured on a different test set and the curve would be
    uninterpretable.

    ``eligibility_coverage`` fixes the "both endpoints are well connected in the
    training graph" filter at the *sparsest* coverage in the sweep.  Because
    training sets are nested, degrees only grow with coverage, so a pair that is
    eligible at the sparsest setting is eligible everywhere -- giving one common
    test set across the whole curve.
    """

    pair_coverage: float = 0.2      # fraction of *all* unordered pairs used for training
                                    # (capped by 1 - eval_pool_frac)
    eval_pool_frac: float = 0.25    # fraction reserved for val+test, coverage-independent
    val_frac: float = 0.2           # share of the eligible eval pool used for validation
    min_train_degree: int = 2       # training-graph degree required of each eval endpoint
    eligibility_coverage: float | None = 0.05   # None -> use pair_coverage
    max_eval_pairs: int = 400
    min_eval_pairs: int = 30    # refuse to build a split too small to score
    seed: int = 0


@dataclass
class PairSplit:
    train_pairs: np.ndarray         # (T, 2), i < j
    val_pairs: np.ndarray
    test_pairs: np.ndarray
    n_eligible: int                 # eval-eligible pairs before capping
    degrees: np.ndarray             # training degree of each intervention

    def summary(self) -> dict:
        return {
            "n_train_pairs": int(len(self.train_pairs)),
            "n_val_pairs": int(len(self.val_pairs)),
            "n_test_pairs": int(len(self.test_pairs)),
            "n_eligible_pairs": int(self.n_eligible),
            "min_train_degree": int(self.degrees.min()),
            "mean_train_degree": float(self.degrees.mean()),
            "n_isolated_interventions": int((self.degrees == 0).sum()),
        }


def _degrees(n: int, pairs: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=int)
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    return deg


def make_pair_split(system: SyntheticSystem, cfg: SplitConfig) -> PairSplit:
    rng = np.random.default_rng(cfg.seed)
    pairs = system.all_pairs()
    n_pairs = len(pairs)
    perm = rng.permutation(n_pairs)

    n_eval_pool = int(round(cfg.eval_pool_frac * n_pairs))
    eval_pool = pairs[perm[:n_eval_pool]]
    trainable = pairs[perm[n_eval_pool:]]

    if cfg.pair_coverage > 1.0 - cfg.eval_pool_frac + 1e-9:
        raise ValueError(
            f"pair_coverage={cfg.pair_coverage} exceeds the trainable pool "
            f"({1 - cfg.eval_pool_frac:.2f} of all pairs)")

    n_train = min(len(trainable), int(round(cfg.pair_coverage * n_pairs)))
    train_pairs = trainable[:n_train]           # nested across coverage levels

    # Eligibility is evaluated at a *fixed* coverage so the test set is common
    # to the whole sweep (training sets are nested => degrees are monotone).
    elig_cov = cfg.eligibility_coverage if cfg.eligibility_coverage is not None \
        else cfg.pair_coverage
    n_elig_train = min(len(trainable), int(round(elig_cov * n_pairs)))
    elig_deg = _degrees(system.n, trainable[:n_elig_train])

    keep = (elig_deg[eval_pool[:, 0]] >= cfg.min_train_degree) & (
        elig_deg[eval_pool[:, 1]] >= cfg.min_train_degree)
    eligible = eval_pool[keep]
    n_eligible = len(eligible)
    eligible = eligible[: cfg.max_eval_pairs]   # deterministic: pool already shuffled

    # A split that yields a handful of eval pairs silently produces correlations
    # computed off almost no data, which look like results but are noise. Fail
    # loudly instead: it means the system is too small or the degree filter too
    # strict for the requested coverage.
    if n_eligible < cfg.min_eval_pairs:
        raise ValueError(
            f"only {n_eligible} eligible eval pairs (need >= {cfg.min_eval_pairs}); "
            f"increase n_interventions, raise eligibility_coverage, or lower "
            f"min_train_degree")

    n_val = max(1, int(round(cfg.val_frac * len(eligible)))) if len(eligible) else 0
    val_pairs, test_pairs = eligible[:n_val], eligible[n_val:]

    return PairSplit(train_pairs=train_pairs, val_pairs=val_pairs,
                     test_pairs=test_pairs, n_eligible=n_eligible,
                     degrees=_degrees(system.n, train_pairs))


def assert_no_pair_leakage(train: ObservationTable, held_out_pairs: np.ndarray,
                           n_interventions: int) -> None:
    """Raise if any held-out pair appears in the training rows, in either order.

    Checks both the recorded ``pair_key`` and the raw (i, j) indices, so a bug
    in ``pair_key`` bookkeeping cannot hide a leak.
    """
    if len(held_out_pairs) == 0:
        return
    pmap = pair_index_map(n_interventions)
    held_keys = {pmap[(int(a), int(b))] for a, b in held_out_pairs}

    non_single = train.kind != SINGLE
    train_keys = set(train.pair_key[non_single].tolist())
    overlap = held_keys & train_keys
    if overlap:
        raise AssertionError(
            f"pair leakage: {len(overlap)} held-out pairs present in training rows"
        )

    raw = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(train.i[non_single], train.j[non_single])
    }
    raw_keys = {pmap[p] for p in raw}
    overlap_raw = held_keys & raw_keys
    if overlap_raw:
        raise AssertionError(
            f"pair leakage (raw indices): {len(overlap_raw)} held-out pairs in training"
        )

    if train.pair_key[non_single].size and set(train.pair_key[non_single]) != raw_keys:
        raise AssertionError("pair_key bookkeeping disagrees with raw (i, j) indices")


def assert_all_interventions_seen(train: ObservationTable, n_interventions: int) -> None:
    seen = set(train.i.tolist()) | set(train.j.tolist())
    missing = set(range(n_interventions)) - seen
    if missing:
        raise AssertionError(f"{len(missing)} interventions never appear in training")
