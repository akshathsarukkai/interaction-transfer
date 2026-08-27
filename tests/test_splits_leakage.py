"""Leakage is the failure mode that would invalidate the entire study.

The claim is "the model predicts the outcome of an intervention pair it has
never seen".  Three rows exist per unordered pair -- simultaneous {i,j},
ordered i->j, ordered j->i -- and they are algebraically redundant:

    A_ij      = z(i->j) - z({i,j})
    z(j->i)   = 2 z({i,j}) - z(i->j)

So if *any one* of a pair's three rows is in training while another is in the
test set, the "prediction" is a subtraction, not a generalisation.  The split
must therefore move whole pairs, never rows.

These tests check the property on the observation tables themselves.  Calling
``assert_no_pair_leakage`` and observing that it does not raise proves nothing
unless the guard is also shown to be capable of raising -- so that is tested
too.
"""

from __future__ import annotations

import numpy as np
import pytest

from intervention_algebra.generator import (
    ORD, SIM, SINGLE, ObservationTable, SystemConfig, generate_observations,
    make_system, pair_index_map)
from intervention_algebra.splits import (
    SplitConfig, assert_all_interventions_seen, assert_no_pair_leakage,
    make_pair_split)

N = 24
COVERAGE_GRID = (0.1, 0.2, 0.4, 0.7)
# ``eligibility_coverage`` is fixed below the whole grid so the eval pool -- and
# hence the test set -- is common to every coverage (see SplitConfig docstring).
ELIG = 0.2


def _system(seed: int = 0):
    return make_system(SystemConfig(n_interventions=N, latent_dim=2,
                                    n_factors=3, regime="both", noise_std=0.05,
                                    seed=seed))


def _split(system, coverage: float, seed: int = 1000):
    return make_pair_split(system, SplitConfig(pair_coverage=coverage,
                                               eligibility_coverage=ELIG,
                                               seed=seed))


def _tables(system, split):
    train = generate_observations(system, split.train_pairs, include_singles=True)
    val = generate_observations(system, split.val_pairs, include_singles=False)
    test = generate_observations(system, split.test_pairs, include_singles=False)
    return train, val, test


def _canonical_pairs_in(table: ObservationTable) -> set[tuple[int, int]]:
    """Unordered pairs actually touched by the non-single rows of a table."""
    m = table.kind != SINGLE
    return {(min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(table.i[m], table.j[m])}


def _as_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in pairs}


# --------------------------------------------------------------------------
# 1. The property itself, read off the tables
# --------------------------------------------------------------------------
@pytest.mark.parametrize("coverage", COVERAGE_GRID)
def test_no_held_out_pair_appears_in_training_in_either_order(coverage):
    system = _system()
    split = _split(system, coverage)
    train, _, _ = _tables(system, split)

    held_out = _as_set(split.val_pairs) | _as_set(split.test_pairs)
    in_train = _canonical_pairs_in(train)
    leaked = held_out & in_train

    assert not leaked, (
        f"coverage={coverage}: {len(leaked)} held-out pairs appear among the "
        f"training rows (first few: {sorted(leaked)[:5]}). Checked directly on "
        f"the (i, j) indices of the training table, canonicalised so that a "
        f"pair leaked in reverse order (j, i) is still caught. Any such pair "
        f"makes its test row memorisable rather than predictable.")


@pytest.mark.parametrize("coverage", COVERAGE_GRID)
def test_train_val_test_pair_sets_are_pairwise_disjoint(coverage):
    split = _split(_system(), coverage)
    tr, va, te = (_as_set(split.train_pairs), _as_set(split.val_pairs),
                  _as_set(split.test_pairs))
    assert len(tr) == len(split.train_pairs), "duplicate pairs inside train"
    assert len(va) == len(split.val_pairs), "duplicate pairs inside val"
    assert len(te) == len(split.test_pairs), "duplicate pairs inside test"
    for a, b, na, nb in ((tr, va, "train", "val"), (tr, te, "train", "test"),
                         (va, te, "val", "test")):
        overlap = a & b
        assert not overlap, (
            f"coverage={coverage}: {len(overlap)} pairs shared between {na} and "
            f"{nb} (e.g. {sorted(overlap)[:5]}). Model selection on val and "
            f"reporting on test both become circular.")


@pytest.mark.parametrize("coverage", COVERAGE_GRID)
def test_all_three_rows_of_a_pair_live_in_the_same_split(coverage):
    """No pair may be partially exposed.

    This is the check that rules out reconstructing ``A_ij`` by subtracting a
    training row from a test row: for every pair, the SIM row and *both* ORD
    rows must be in exactly one of the three tables.
    """
    system = _system()
    split = _split(system, coverage)
    train, val, test = _tables(system, split)

    owner: dict[tuple[int, int], str] = {}
    for name, table in (("train", train), ("val", val), ("test", test)):
        by_pair: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        m = table.kind != SINGLE
        for a, b, k in zip(table.i[m], table.j[m], table.kind[m]):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            by_pair.setdefault(key, []).append((int(a), int(b), int(k)))

        for key, rows in by_pair.items():
            kinds = sorted(k for *_, k in rows)
            assert kinds == [SIM, ORD, ORD], (
                f"pair {key} in '{name}' has row kinds {kinds}, expected one "
                f"SIM and two ORD rows. A pair whose rows are split across "
                f"tables is reconstructable: "
                f"A_ij = z(i->j) - z({{i,j}}).")
            ord_dirs = sorted((a, b) for a, b, k in rows if k == ORD)
            assert ord_dirs == [(key[0], key[1]), (key[1], key[0])], (
                f"pair {key} in '{name}' has ordered rows {ord_dirs}; both "
                f"directions i->j and j->i must be present, otherwise "
                f"z(j->i) = 2 z({{i,j}}) - z(i->j) recovers the missing one.")

            prev = owner.setdefault(key, name)
            assert prev == name, (
                f"pair {key} appears in both '{prev}' and '{name}'.")


@pytest.mark.parametrize("coverage", COVERAGE_GRID)
def test_every_single_intervention_row_is_in_training(coverage):
    system = _system()
    split = _split(system, coverage)
    train, val, test = _tables(system, split)

    singles = train.i[train.kind == SINGLE]
    assert sorted(singles.tolist()) == list(range(N)), (
        f"coverage={coverage}: training contains SINGLE rows for "
        f"{sorted(set(singles.tolist()))} but the system has {N} "
        f"interventions. All N singles must be in training -- they are what "
        f"pins the v/S gauge (docs/identifiability.md) and what makes the "
        f"evaluation 'seen interventions, unseen pair' rather than "
        f"'unseen intervention'.")
    for name, table in (("val", val), ("test", test)):
        assert not (table.kind == SINGLE).any(), (
            f"the {name} table contains SINGLE rows; the held-out quantity is "
            f"supposed to be the *pair*, so scoring singles there inflates the "
            f"result with rows whose interventions were trained on directly.")

    # ...and the library's own guard agrees.
    assert_all_interventions_seen(train, N)


# --------------------------------------------------------------------------
# 2. Nesting and a common test set across the coverage curve
# --------------------------------------------------------------------------
def test_training_sets_are_strictly_nested_and_test_set_is_common():
    system = _system()
    splits = {c: _split(system, c) for c in COVERAGE_GRID}

    ref = splits[COVERAGE_GRID[0]]
    for c, sp in splits.items():
        assert np.array_equal(sp.test_pairs, ref.test_pairs), (
            f"the test set at coverage={c} differs from the one at "
            f"coverage={COVERAGE_GRID[0]} "
            f"({len(sp.test_pairs)} vs {len(ref.test_pairs)} pairs). Each point "
            f"of the pair-coverage curve would then be measured on a different "
            f"problem and the curve could not be read as an effect of coverage.")
        assert np.array_equal(sp.val_pairs, ref.val_pairs), (
            f"the validation set at coverage={c} differs from the reference.")

    prev_c, prev = COVERAGE_GRID[0], _as_set(splits[COVERAGE_GRID[0]].train_pairs)
    for c in COVERAGE_GRID[1:]:
        cur = _as_set(splits[c].train_pairs)
        assert prev < cur, (
            f"training pairs at coverage={prev_c} ({len(prev)}) are not a "
            f"*proper* subset of those at coverage={c} ({len(cur)}); "
            f"{len(prev - cur)} pairs were dropped when coverage increased. "
            f"Without nesting, a non-monotone point on the coverage curve could "
            f"be caused by a lucky/unlucky resample rather than by data volume.")
        prev_c, prev = c, cur


def test_test_set_is_common_across_coverage_for_several_seeds():
    for seed in (0, 1, 2):
        system = _system(seed=seed)
        splits = [_split(system, c, seed=1000 + seed) for c in COVERAGE_GRID]
        for sp in splits[1:]:
            assert np.array_equal(sp.test_pairs, splits[0].test_pairs), (
                f"seed={seed}: the test set is not coverage-invariant.")


# --------------------------------------------------------------------------
# 3. The guard must be able to fail
# --------------------------------------------------------------------------
def test_leakage_guard_raises_on_a_deliberately_leaked_split():
    """A guard that cannot fail is worthless: hand it a real leak."""
    system = _system()
    split = _split(system, 0.2)
    held_out = np.concatenate([split.val_pairs, split.test_pairs])

    # Sanity: the honest split passes.
    assert_no_pair_leakage(
        generate_observations(system, split.train_pairs, include_singles=True),
        held_out, N)

    leaked_pair = split.test_pairs[0:1]
    dirty_pairs = np.concatenate([split.train_pairs, leaked_pair])
    dirty = generate_observations(system, dirty_pairs, include_singles=True)

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_pair_leakage(dirty, held_out, N)


def test_leakage_guard_catches_a_leak_that_pair_key_bookkeeping_hides():
    """The ``pair_key`` column must not be the only thing standing between a
    leaked pair and a passing check: corrupt it and the raw (i, j) indices must
    still betray the leak."""
    system = _system()
    split = _split(system, 0.2)
    held_out = np.concatenate([split.val_pairs, split.test_pairs])
    pmap = pair_index_map(N)

    leaked_pair = split.test_pairs[0:1]
    dirty_pairs = np.concatenate([split.train_pairs, leaked_pair])
    dirty = generate_observations(system, dirty_pairs, include_singles=True)

    leaked_key = pmap[(int(leaked_pair[0, 0]), int(leaked_pair[0, 1]))]
    innocent_key = pmap[(int(split.train_pairs[0, 0]),
                         int(split.train_pairs[0, 1]))]
    forged = dirty.pair_key.copy()
    forged[forged == leaked_key] = innocent_key
    dirty = ObservationTable(dirty.i, dirty.j, dirty.kind, dirty.y, dirty.z,
                             forged)

    assert leaked_key not in set(dirty.pair_key.tolist()), (
        "test setup failed: the leaked pair_key should have been overwritten")
    with pytest.raises(AssertionError):
        assert_no_pair_leakage(dirty, held_out, N)


def test_leakage_guard_catches_a_reverse_order_leak():
    """Only the reversed ordered row of a held-out pair is present in training.

    ``z(j->i) = 2 z({i,j}) - z(i->j)`` means one direction is enough.
    """
    system = _system()
    split = _split(system, 0.2)
    held_out = np.concatenate([split.val_pairs, split.test_pairs])
    a, b = int(split.test_pairs[0, 0]), int(split.test_pairs[0, 1])

    clean = generate_observations(system, split.train_pairs, include_singles=True)
    pmap = pair_index_map(N)
    key = pmap[(a, b)]
    dirty = ObservationTable(
        i=np.concatenate([clean.i, [b]]),
        j=np.concatenate([clean.j, [a]]),
        kind=np.concatenate([clean.kind, [ORD]]),
        y=np.concatenate([clean.y, system.latent_ordered(np.array([b]),
                                                         np.array([a]))]),
        z=np.concatenate([clean.z, system.latent_ordered(np.array([b]),
                                                         np.array([a]))]),
        pair_key=np.concatenate([clean.pair_key, [key]]),
    )
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_pair_leakage(dirty, held_out, N)


def test_assert_all_interventions_seen_raises_when_one_is_missing():
    system = _system()
    split = _split(system, 0.2)
    train = generate_observations(system, split.train_pairs, include_singles=True)
    keep = (train.i != 0) & (train.j != 0)
    with pytest.raises(AssertionError, match="never appear"):
        assert_all_interventions_seen(train.subset(keep), N)


# --------------------------------------------------------------------------
# 4. Refusing impossible requests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("coverage", [0.76, 0.9, 1.0])
def test_pair_coverage_above_the_trainable_pool_raises(coverage):
    system = _system()
    cfg = SplitConfig(pair_coverage=coverage, eval_pool_frac=0.25,
                      eligibility_coverage=ELIG, seed=1000)
    with pytest.raises(ValueError, match="exceeds the trainable pool"):
        make_pair_split(system, cfg)


def test_coverage_exactly_at_the_pool_boundary_is_allowed():
    """0.75 with eval_pool_frac=0.25 is the largest legal request; it must not
    be rejected by an off-by-epsilon in the guard."""
    split = make_pair_split(_system(), SplitConfig(
        pair_coverage=0.75, eval_pool_frac=0.25, eligibility_coverage=ELIG,
        seed=1000))
    n_pairs = N * (N - 1) // 2
    assert len(split.train_pairs) == int(round(0.75 * n_pairs)), (
        f"expected {int(round(0.75 * n_pairs))} training pairs at the boundary "
        f"coverage, got {len(split.train_pairs)}")


def test_split_refuses_to_build_an_unscoreable_eval_set():
    """Too few eligible eval pairs must fail loudly, not silently produce
    correlations computed off a handful of points."""
    system = make_system(SystemConfig(n_interventions=12, latent_dim=2,
                                      n_factors=2, seed=0))
    with pytest.raises(ValueError, match="eligible eval pairs"):
        make_pair_split(system, SplitConfig(pair_coverage=0.2,
                                            eligibility_coverage=0.05,
                                            seed=1000))
