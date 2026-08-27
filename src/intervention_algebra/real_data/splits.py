"""Held-out *unordered pair* splits over a real ordered screen.

The evaluation this whole phase rests on is

    seen drugs, entirely unseen drug pair

so the split has exactly one job and one way to be wrong. For a test pair
``{i, j}``:

* both ``i -> j`` and ``j -> i`` are withheld -- withholding one direction would
  hand the model the other, and the directional target ``D = y_ij - y_ji`` would
  then be half-observed;
* ``i`` and ``j`` each appear in other *training* pairs, so this stays a
  compositional-generalisation task and does not silently degrade into
  cold-start prediction for an unseen drug;
* nothing derived from ``{i, j}`` -- in particular no schedule-difference value
  -- is present in training. The screen contains no such derived column, and
  :func:`assert_no_pair_leakage` checks the ordered rows directly.

Coverage nesting
----------------
A single permutation of the unordered pairs is drawn per ``split_seed``; the
training pool at coverage ``c`` is its first ``round(c * n_pairs)`` entries. That
*pool* (train + validation) is therefore nested across the coverage grid; the
training set proper is not, because validation takes a growing prefix of the
pool, so some pairs that train at a low coverage become validation pairs at a
higher one. The property the coverage curve actually needs is the weaker one,
and it holds: the evaluation pool --
pairs excluded at the highest coverage, further restricted to those eligible at
the *lowest* coverage -- is identical at every coverage. A coverage curve built
this way varies only the amount of training data; it does not also swap the test
set underneath the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Pair = tuple[int, int]


def canonical_pair(i: int, j: int) -> Pair:
    """Unordered pair id. The only place pair identity is defined."""
    if i == j:
        raise ValueError(f"self-pair ({i}, {j}) has no unordered identity here")
    return (int(min(i, j)), int(max(i, j)))


@dataclass(frozen=True)
class CoverageSplit:
    coverage: float
    split_seed: int
    train_pairs: tuple[Pair, ...]
    val_pairs: tuple[Pair, ...]
    test_pairs: tuple[Pair, ...]
    train_degree: np.ndarray = field(repr=False)
    n_eligible_dropped: int = 0

    def rows(self, frame: pd.DataFrame, which: str) -> pd.DataFrame:
        want = {"train": self.train_pairs, "val": self.val_pairs,
                "test": self.test_pairs}[which]
        keep = set(want)
        mask = [p in keep for p in frame["pair"]]
        return frame[np.asarray(mask)].reset_index(drop=True)


def all_pairs(frame: pd.DataFrame) -> list[Pair]:
    return sorted({canonical_pair(*p) for p in frame["pair"]})


def degrees(n_drugs: int, pairs) -> np.ndarray:
    deg = np.zeros(n_drugs, dtype=int)
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    return deg


def make_coverage_splits(
    frame: pd.DataFrame,
    n_drugs: int,
    coverages: tuple[float, ...],
    split_seed: int,
    val_fraction: float = 0.15,
    min_train_degree: int = 3,
    min_eligible_test_pairs: int = 50,
) -> dict[float, CoverageSplit]:
    """Build the nested coverage family for one ``split_seed``.

    ``min_train_degree`` is the connectivity floor a test pair's endpoints must
    clear. It is enforced using the *sparsest* training set in the grid, so the
    surviving evaluation pool is common to every coverage. Eligibility depends
    only on the pair graph -- never on ``y`` -- so it cannot select easy pairs.
    """
    if not coverages:
        raise ValueError("need at least one coverage")
    if not 0.0 < min(coverages) <= max(coverages) < 1.0:
        raise ValueError(f"coverages must lie in (0, 1): {coverages}")

    pairs = all_pairs(frame)
    n = len(pairs)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    ranked = [pairs[k] for k in order]

    cov = tuple(sorted(coverages))
    n_hi = int(round(max(cov) * n))
    n_lo = int(round(min(cov) * n))
    if n_hi >= n:
        raise ValueError("highest coverage leaves no evaluation pool")

    # Evaluation pool: never in training at any coverage, and connected enough
    # at the sparsest coverage.
    #
    # The degree is computed over the sparsest coverage's *training* pairs with
    # its validation pairs already removed -- not over the whole pool. Using the
    # pool would state a floor the split does not actually deliver: at coverage
    # 0.05 validation takes 15% of the pool, and an endpoint with three pool
    # pairs can end up with one training pair, so a "min degree 3" guarantee
    # would be off by a factor of three exactly where the data is thinnest.
    candidate_test = ranked[n_hi:]
    lo_pool = ranked[:n_lo]
    lo_n_val = min(max(4, int(round(val_fraction * n_lo))), n_lo // 2)
    deg_lo = degrees(n_drugs, lo_pool[lo_n_val:])
    eligible = [p for p in candidate_test
                if deg_lo[p[0]] >= min_train_degree and deg_lo[p[1]] >= min_train_degree]
    dropped = len(candidate_test) - len(eligible)
    if len(eligible) < min_eligible_test_pairs:
        raise ValueError(
            f"only {len(eligible)} eligible test pairs at min_train_degree="
            f"{min_train_degree}; the coverage grid is too aggressive for this "
            f"graph. Refusing rather than reporting a metric over a pool too "
            f"small to mean anything.")

    out: dict[float, CoverageSplit] = {}
    for c in cov:
        k = int(round(c * n))
        pool = ranked[:k]
        # Validation pairs are carved out of the training pool at pair level, so
        # early stopping never sees a test pair. Clamped to at most half the
        # pool: an unclamped floor of 10 swallows the entire training set at the
        # sparsest coverage on a small graph, leaving no training pairs and a
        # confusing downstream failure about unseen endpoints.
        n_val = min(max(4, int(round(val_fraction * k))), k // 2)
        if k - n_val < 4:
            raise ValueError(
                f"coverage {c} leaves {k - n_val} training pairs out of {n} -- "
                f"too few to fit anything")
        val = tuple(pool[:n_val])
        train = tuple(pool[n_val:])
        out[c] = CoverageSplit(
            coverage=c, split_seed=split_seed,
            train_pairs=train, val_pairs=val, test_pairs=tuple(eligible),
            train_degree=degrees(n_drugs, train), n_eligible_dropped=dropped)
    return out


def assert_no_pair_leakage(split: CoverageSplit, frame: pd.DataFrame) -> None:
    """Fail loudly on any way the split could have gone wrong. Run every time."""
    tr, va, te = set(split.train_pairs), set(split.val_pairs), set(split.test_pairs)
    if tr & te:
        raise AssertionError(f"{len(tr & te)} pairs in both train and test")
    if va & te:
        raise AssertionError(f"{len(va & te)} pairs in both val and test")
    if tr & va:
        raise AssertionError(f"{len(tr & va)} pairs in both train and val")

    train_rows = split.rows(frame, "train")
    test_rows = split.rows(frame, "test")

    # Both directions of every test pair must be absent from training...
    seen = set(zip(train_rows["i"], train_rows["j"]))
    for a, b in te:
        if (a, b) in seen or (b, a) in seen:
            raise AssertionError(f"ordered row for held-out pair ({a}, {b}) is in train")
    # ...and present in test, or the directional target is half-missing.
    have = set(zip(test_rows["i"], test_rows["j"]))
    incomplete = [p for p in te if (p[0], p[1]) not in have or (p[1], p[0]) not in have]
    if incomplete:
        raise AssertionError(
            f"{len(incomplete)} test pairs lack one direction, e.g. {incomplete[:3]}")

    # Every drug in a test pair must be trained on elsewhere.
    trained_drugs = set(train_rows["i"]) | set(train_rows["j"])
    orphans = {d for p in te for d in p} - trained_drugs
    if orphans:
        raise AssertionError(
            f"{len(orphans)} test-pair endpoints never appear in training: "
            f"{sorted(orphans)[:5]} -- this is a cold-start task, not the "
            f"unseen-pair task")


def connectivity_report(split: CoverageSplit, n_drugs: int) -> dict:
    deg = split.train_degree
    endpoints = sorted({d for p in split.test_pairs for d in p})
    return {
        "coverage": split.coverage,
        "split_seed": split.split_seed,
        "n_train_pairs": len(split.train_pairs),
        "n_val_pairs": len(split.val_pairs),
        "n_test_pairs": len(split.test_pairs),
        "n_test_pairs_dropped_for_connectivity": split.n_eligible_dropped,
        "train_degree_min": int(deg.min()),
        "train_degree_median": float(np.median(deg)),
        "train_degree_max": int(deg.max()),
        "n_isolated_drugs": int((deg == 0).sum()),
        "test_endpoint_degree_min": int(deg[endpoints].min()) if endpoints else 0,
        "test_endpoint_degree_median": float(np.median(deg[endpoints])) if endpoints else 0.0,
    }
