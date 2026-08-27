"""Entity-level splits: the unit held out is a drug, not a pair.

What changes from Phase 2R, and why it is not a small change
------------------------------------------------------------
Phase 2R's :class:`~intervention_algebra.real_data.splits.CoverageSplit` holds
out unordered *pairs* and then explicitly *requires* that both endpoints of
every test pair appear somewhere in training -- ``assert_no_pair_leakage``
raises on "orphans". That requirement is the definition of transductive pair
completion, and it forbids exactly what Phase 3 is for. So the Phase 2R split
machinery is not adapted here; it is replaced, and the guard is inverted.

The partition
-------------
Given a set ``T`` of test drugs and a disjoint set ``V`` of validation drugs,
every one of the ``C(n, 2)`` unordered pairs lands in exactly one bucket,
decided only by which drugs it touches:

===================================  ==========
both endpoints in ``T``              ``test_e2``
exactly one endpoint in ``T``        ``test_e1``
no endpoint in ``T``, one in ``V``   ``val``
no endpoint in ``T`` or ``V``        ``train``
===================================  ==========

The four buckets are disjoint and exhaust the pair set -- :func:`assert_partition`
checks it by counting, not by trusting the construction. With ``n = 100``,
``|T| = 10``, ``|V| = 10`` that is 3,160 train / 845 validation / 900 E1 / 45 E2.

Why validation must also be entity-OOD
--------------------------------------
The tempting shortcut is to tune on held-out *pairs* among the training drugs
and then evaluate on held-out drugs. It is wrong for a reason that has nothing to
do with contamination: the validation set would be measuring a different task
from the test set, so the hyperparameter chosen would be the best one for
transductive completion, and the entity-OOD number would report a model that was
never selected for entity-OOD. Validation here therefore holds out its own
drugs, and a validation pair always has an endpoint the model never trained on.

E1 and E2 are never pooled
--------------------------
They are different questions -- "a new drug meeting a known one" and "two new
drugs meeting each other" -- and they have wildly different sample sizes (900 vs
45 per fold). Pooling would let E1 dominate an average that gets described as
"unseen drugs", and would hide the harder result inside the easier one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Pair = tuple[int, int]

#: The four buckets every unordered pair lands in, in reporting order. One
#: definition: :meth:`DrugFold.pairs` dispatches on it, the partition check
#: iterates it, and ``test_docs_integrity`` reads it so that "test_e1" in a
#: document is understood as a bucket name rather than a missing pytest function.
BUCKETS: tuple[str, ...] = ("train", "val", "test_e1", "test_e2")


@dataclass(frozen=True)
class DrugFold:
    """One entity-level fold. Drug sets are the primitive; pairs are derived."""

    partition: int
    fold: int
    test_drugs: tuple[int, ...]
    val_drugs: tuple[int, ...]
    train_drugs: tuple[int, ...]
    train_pairs: tuple[Pair, ...]
    val_pairs: tuple[Pair, ...]
    test_e1_pairs: tuple[Pair, ...]
    test_e2_pairs: tuple[Pair, ...]
    #: Fraction of the *eligible* training pairs actually kept. 1.0 in the
    #: primary setting; the coverage sweep varies it among training entities
    #: only, never touching which drugs are held out.
    coverage: float = 1.0
    n_eligible_train_pairs: int = 0

    @property
    def key(self) -> str:
        return f"p{self.partition}f{self.fold}"

    def pairs(self, which: str) -> tuple[Pair, ...]:
        return dict(zip(BUCKETS, (self.train_pairs, self.val_pairs,
                                  self.test_e1_pairs, self.test_e2_pairs)))[which]

    def rows(self, frame: pd.DataFrame, which: str) -> pd.DataFrame:
        """Rows of a canonical ``i < j`` pair frame belonging to one bucket."""
        want = set(self.pairs(which))
        keep = [(int(a), int(b)) in want for a, b in zip(frame["i"], frame["j"])]
        return frame.loc[np.asarray(keep)].reset_index(drop=True)


def _bucket(a: int, b: int, test: frozenset[int], val: frozenset[int]) -> str:
    in_t = (a in test) + (b in test)
    if in_t == 2:
        return "test_e2"
    if in_t == 1:
        return "test_e1"
    if a in val or b in val:
        return "val"
    return "train"


def make_drug_folds(n_drugs: int, n_partitions: int = 3, n_test: int = 10,
                    n_val: int = 10, seed: int = 20260825,
                    coverage: float = 1.0) -> list[DrugFold]:
    """Repeated disjoint entity partitions.

    Each *partition* is one shuffle of the drug list cut into ``n_drugs //
    n_test`` folds, so within a partition every drug is held out exactly once and
    the test groups are disjoint. ``n_partitions`` independent shuffles give each
    drug ``n_partitions`` independent turns as a test entity, which is what makes
    a per-drug summary possible without any drug's number resting on a single
    fold.

    Validation drugs are drawn from the non-test drugs of that fold, using the
    fold's own deterministic stream -- so ``V`` differs between folds and no
    single unlucky validation set can drive the whole partition.

    Everything here is a function of ``seed`` and the arguments; no outcome is
    read, and this function does not even take the screen.
    """
    if n_drugs % n_test:
        raise ValueError(f"{n_drugs} drugs is not a whole number of {n_test}-drug folds")
    all_pairs = list(itertools.combinations(range(n_drugs), 2))
    folds: list[DrugFold] = []
    for p in range(n_partitions):
        rng = np.random.default_rng([seed, p])
        order = rng.permutation(n_drugs)
        for f in range(n_drugs // n_test):
            test = frozenset(int(d) for d in order[f * n_test:(f + 1) * n_test])
            rest = np.array([d for d in order if int(d) not in test])
            vrng = np.random.default_rng([seed, p, f])
            val = frozenset(int(d) for d in vrng.permutation(rest)[:n_val])
            buckets: dict[str, list[Pair]] = {
                "train": [], "val": [], "test_e1": [], "test_e2": []}
            for a, b in all_pairs:
                buckets[_bucket(a, b, test, val)].append((a, b))
            eligible = len(buckets["train"])
            if coverage < 1.0:
                crng = np.random.default_rng([seed, p, f, 1])
                keep = crng.permutation(eligible)[:max(1, round(coverage * eligible))]
                buckets["train"] = [buckets["train"][k] for k in sorted(keep)]
            folds.append(DrugFold(
                partition=p, fold=f,
                test_drugs=tuple(sorted(test)), val_drugs=tuple(sorted(val)),
                train_drugs=tuple(sorted(set(range(n_drugs)) - test - val)),
                train_pairs=tuple(buckets["train"]), val_pairs=tuple(buckets["val"]),
                test_e1_pairs=tuple(buckets["test_e1"]),
                test_e2_pairs=tuple(buckets["test_e2"]),
                coverage=coverage, n_eligible_train_pairs=eligible))
    return folds


def assert_partition(fold: DrugFold, n_drugs: int) -> None:
    """The four buckets are disjoint and exhaust ``C(n, 2)``.

    Checked by counting rather than inferred from the construction, so a future
    change to :func:`make_drug_folds` that starts dropping pairs is caught here
    rather than showing up as a quietly smaller training set.
    """
    seen: set[Pair] = set()
    total = 0
    for which in BUCKETS:
        ps = set(fold.pairs(which))
        if len(ps) != len(fold.pairs(which)):
            raise AssertionError(f"{fold.key}: duplicate pairs in {which}")
        if seen & ps:
            raise AssertionError(f"{fold.key}: {which} overlaps an earlier bucket")
        seen |= ps
        total += len(ps)
    # At coverage < 1.0 the training bucket is deliberately thinned, so the
    # buckets no longer exhaust the pair set -- but the arithmetic is still
    # checkable, and an earlier version simply skipped the check there, which
    # disabled it in exactly the setting where pairs are being dropped on
    # purpose and a bug would look like the intended behaviour.
    expected = n_drugs * (n_drugs - 1) // 2
    dropped = fold.n_eligible_train_pairs - len(fold.train_pairs)
    # At full coverage nothing may be dropped at all. Without this the
    # bookkeeping absorbs a genuinely lost training pair as though it had been
    # thinned on purpose -- which is how tightening the sparse-coverage case
    # silently loosened the full-coverage one.
    if fold.coverage == 1.0 and dropped:
        raise AssertionError(
            f"{fold.key}: coverage is 1.0 but {dropped} eligible training pairs "
            f"are missing")
    if total + dropped != expected:
        raise AssertionError(
            f"{fold.key}: buckets hold {total} pairs and {dropped} were dropped by "
            f"coverage, which is {total + dropped}, expected {expected}")


def assert_no_drug_leakage(fold: DrugFold, n_drugs: int) -> None:
    """The guard the whole phase rests on: a test drug is absent from everything.

    Deliberately stronger than a pair-level check. A pair-level guard would pass
    a split in which drug ``k`` is "held out" in pair ``{k, 3}`` while ``{k, 7}``
    sits in training -- which is Phase 2R's setting, not this one. What is
    asserted here is at the level of *drugs*:

    * the three drug sets are disjoint and cover every drug;
    * **no training pair touches a test or validation drug, in either position**;
    * no validation pair touches a test drug;
    * every E1 pair has exactly one test endpoint, every E2 pair has two;
    * no test drug appears in any training pair -- restated as a set membership
      over the flattened training endpoints, so a bug in the bucket logic and a
      bug in this check would have to agree to pass.

    Order matters. The drug-membership checks run **first** and the pair-set
    arithmetic second. A planted test-drug pair violates both -- it puts a drug
    where it does not belong *and* duplicates a pair across buckets -- and if the
    arithmetic ran first the failure would be reported as "test_e1 overlaps an
    earlier bucket", which is true, unhelpful, and hides the leak behind a
    bookkeeping complaint. :func:`assert_partition` still runs, because a guard
    that only inspects the buckets it is given cannot notice pairs that fell out
    of all of them.
    """
    test, val, train = set(fold.test_drugs), set(fold.val_drugs), set(fold.train_drugs)
    if test & val or test & train or val & train:
        raise AssertionError(f"{fold.key}: drug sets overlap")
    if test | val | train != set(range(n_drugs)):
        raise AssertionError(f"{fold.key}: drug sets do not cover all {n_drugs} drugs")

    train_endpoints = {d for pair in fold.train_pairs for d in pair}
    if train_endpoints & test:
        raise AssertionError(
            f"{fold.key}: {sorted(train_endpoints & test)} are test drugs and appear "
            f"in a training pair")
    if train_endpoints & val:
        raise AssertionError(
            f"{fold.key}: {sorted(train_endpoints & val)} are validation drugs and "
            f"appear in a training pair")
    val_endpoints = {d for pair in fold.val_pairs for d in pair}
    if val_endpoints & test:
        raise AssertionError(
            f"{fold.key}: {sorted(val_endpoints & test)} are test drugs and appear "
            f"in a validation pair")
    # Validation must be entity-OOD too, and until this check existed nothing
    # asserted it: the guard only forbade a *test* drug in a validation pair, so
    # a validation bucket padded with pairs between two training drugs passed.
    # The remaining checks make that dilution rather than contamination -- the
    # counting forces every val-touching pair into val -- but an invariant the
    # documents call load-bearing should be asserted, not inferred.
    for a, b in fold.val_pairs:
        if a not in val and b not in val:
            raise AssertionError(
                f"{fold.key}: validation pair {(a, b)} has no validation endpoint")
    for a, b in fold.test_e1_pairs:
        if (a in test) + (b in test) != 1:
            raise AssertionError(f"{fold.key}: E1 pair {(a, b)} does not have exactly "
                                 f"one unseen endpoint")
    for a, b in fold.test_e2_pairs:
        if not (a in test and b in test):
            raise AssertionError(f"{fold.key}: E2 pair {(a, b)} is not both-unseen")
    assert_partition(fold, n_drugs)


def fold_summary(fold: DrugFold) -> dict:
    return {
        "partition": fold.partition, "fold": fold.fold, "fold_key": fold.key,
        "coverage": fold.coverage,
        "n_test_drugs": len(fold.test_drugs), "n_val_drugs": len(fold.val_drugs),
        "n_train_drugs": len(fold.train_drugs),
        "n_train_pairs": len(fold.train_pairs),
        "n_eligible_train_pairs": fold.n_eligible_train_pairs,
        "n_val_pairs": len(fold.val_pairs),
        "n_test_e1_pairs": len(fold.test_e1_pairs),
        "n_test_e2_pairs": len(fold.test_e2_pairs),
        "test_drugs": ",".join(str(d) for d in fold.test_drugs),
    }
