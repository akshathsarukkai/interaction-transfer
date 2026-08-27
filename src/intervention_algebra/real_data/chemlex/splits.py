"""Bipartite entity-level splits: the unit held out is a reactant, not a pair.

What changes from Phase 3, and why it is not a small change
-----------------------------------------------------------
Phase 3 held out drugs from a screen in which every entity plays the same role,
so one held-out set and one bucket rule sufficed. Here there are **two entity
types**. An acid and an amine are not interchangeable: the reaction is
``R-COOH + H2N-R'``, the roles are chemically distinct, and no permutation of the
data turns one into the other. So there are two independent partitions -- acids
into train/val/test, amines into train/val/test -- and a row's bucket is decided
by the *pair* of role memberships.

That gives nine buckets, and naming all nine is the point. Phase 3's adversarial
review found that its E1 bucket silently contained two different regimes: a test
drug meeting a *training* drug, and a test drug meeting a *validation* drug --
and a validation drug appears in no training pair either, so those rows were
quietly both-unseen. Every document asserted E1 was homogeneous. Here the
distinction is in the type system from the start.

===========================  ================  ==============================
acid role                    amine role        bucket
===========================  ================  ==============================
train                        train             ``train``
val                          train             ``val_e1a``
train                        val               ``val_e1n``
val                          val               ``val_e2``
test                         train             ``test_e1a``   *unseen acid*
train                        test              ``test_e1n``   *unseen amine*
test                         test              ``test_e2``    *both unseen*
test                         val               ``test_e2_mixed``
val                          test              ``test_e2_mixed``
===========================  ================  ==============================

The three primary readouts are ``test_e1a``, ``test_e1n`` and ``test_e2``. They
are never pooled: "a new acid meeting a known amine", "a new amine meeting a
known acid" and "two new reactants meeting each other" are three different
questions with different sample sizes, and it is chemically plausible that
unseen-amine generalisation is easier than unseen-acid generalisation or the
reverse. Averaging them would hide exactly the asymmetry worth looking for.

``test_e2_mixed`` is the honest name for a real regime that is neither. Its rows
have *both* endpoints absent from every training row, so they answer E2's
question, and there are about twice as many of them as there are strict E2 rows.
But one endpoint is a validation entity, so hyperparameter selection saw rows
containing it. That is a weaker contamination than training on it and a stronger
one than nothing, so the bucket is reported separately and never promoted into a
primary readout.

Why validation must also be entity-OOD
--------------------------------------
The same reason as Phase 3, restated for two roles. Tuning on held-out *pairs*
among training entities and then reporting entity-OOD test numbers selects the
hyperparameter that is best for transductive completion and reports it as though
it had been selected for extrapolation. Here validation holds out its own acids
and its own amines, so a validation row always has at least one endpoint the
model never trained on -- and ``val_e1a`` / ``val_e1n`` / ``val_e2`` mirror the
three test regimes, so selection can be done on the regime being tested.

What is held out is a *group*, not an entity
--------------------------------------------
An entity split is only as strong as its notion of "the same molecule". Two
failures on this screen would defeat a naive one, and both were found by looking
rather than assumed:

* **Stereoisomers.** 11 of the 272 acids and 4 of the 230 amines share a
  constitution with another entry -- the deposit carries, for instance, both the
  ring-unspecified and the (1r,4r) form of 4-isopropylcyclohexanecarbonyl-L-Phe.
  Holding out one while training on the other is not an unseen reactant in any
  chemically meaningful sense. (The authors' own ``Both_Unseen`` split leaks 4
  rows under this criterion.)
* **Feature twins.** Three perfluoroalkanoic acids -- C9, C11 and C12, molecular
  weights 464, 564 and 614 -- have **byte-identical 2048-bit ECFP4 vectors**,
  because at radius 2 every environment in a perfluoro chain repeats. They are
  different molecules that the primary representation cannot tell apart, so a
  held-out one has an exact twin in training and its "extrapolation" is a lookup.
  The degeneracy is not a fingerprint-parameter accident: it survives radius 3
  and it survives ``includeChirality``. Only count fingerprints or radius 6
  separate them.
* **Charge states.** 8-anilino-1-naphthalenesulfonic acid appears twice in the
  amine column, once as the free acid (48 rows) and once as the anion (44 rows).
  They are one compound in two protonation states, with different fingerprints.

:func:`split_groups` therefore merges entities of the same role into groups by
the transitive closure of three relations -- *same stereo-stripped constitution*,
*same neutralised stereo-stripped constitution*, *same canonical tautomer of the
stereo-stripped structure*, or *identical primary fingerprint* -- and folds are
cut over **groups**. 272 acids collapse to 257 groups and 230 amines to 225. 272 acids collapse to 259
groups and 230 amines to 225. Entities stay distinct everywhere else -- in the
features, in the per-entity statistics, in the similarity strata -- because they
are distinct rows of the deposit; the grouping constrains only which side of the
split they may land on.

* **Tautomers.** Two acids are Fmoc-Lys(Dde)-OH drawn as the imine and as the
  enaminone -- same formula C31H36N2O6, different InChI skeletons, so standard
  InChI does not equate them either -- and two are valsartan with the tetrazole
  drawn 1H and 2H. Both pairs landed on opposite sides of a fold. A tautomer is
  not an unseen molecule.

The tautomer relation is applied to the **already stereo-stripped** structure,
which is what makes it usable. RDKit's ``TautomerEnumerator.Canonicalize``
silently discards stereochemistry, and on the raw structures 7 of the 9 acid
merges it produces are stereo-flattening artefacts wearing a tautomer's name --
it does the first relation's job badly while pretending to do a fourth. Applied
after stereo has already been removed, that objection is empty: stereo is gone
in that relation by construction, and what is left is the tautomer perception
alone.

Plain salt-stripping is deliberately *not* a relation: it is a provable no-op
here, since not one of the 272 acid or 231 amine strings contains a ``.``.

Note what this is *not*. It is not congener clustering, and it does not hold out
chemical families. Two entities that are merely similar stay separate, and
whether the pair term still works for a held-out reactant with no close analogue
is measured by the similarity stratification, not decided by the split.

Conditions are never held out on purpose
----------------------------------------
No condition is *deliberately* held out: Phase 4 asks about entity
extrapolation, not condition extrapolation, and holding out a condition would
confound the two. The condition column is a covariate on both sides of every
comparison.

But "every condition appears in training" is **not** an invariant, and an
earlier version of this docstring asserted that it was. Two conditions are
vanishingly rare -- BOP/DIPEA has 5 rows in the whole deposit and PyBrOP/DIPEA
has 7 -- so a fold that holds out their handful of entities leaves the level
with no training rows at all. It happens on the pooled screen.

The consequence is bounded rather than absent. A condition with no training
rows keeps its zero-initialised intercept, which is the sensible prior for a
level nothing was observed at, and it affects the baseline and the pair model
identically because they share that term. What would matter is if it affected
many rows, and
``test_how_many_conditions_have_no_training_rows_is_measured_not_assumed``
measures that instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The nine buckets, in reporting order. One definition: :meth:`Fold.mask`
#: dispatches on it, :func:`assert_partition` iterates it, and the document
#: checker reads it so that ``test_e1a`` in a document is understood as a bucket
#: name rather than a missing pytest function.
BUCKETS: tuple[str, ...] = (
    "train", "val_e1a", "val_e1n", "val_e2",
    "test_e1a", "test_e1n", "test_e2", "test_e2_mixed",
)

#: Buckets used to fit. Exactly one, and it is named so no other module can
#: disagree about it.
FIT_BUCKETS: tuple[str, ...] = ("train",)
#: Buckets hyperparameter selection may look at. Nothing else, ever.
SELECT_BUCKETS: tuple[str, ...] = ("val_e1a", "val_e1n", "val_e2")
#: The three primary readouts.
PRIMARY_BUCKETS: tuple[str, ...] = ("test_e1a", "test_e1n", "test_e2")

#: Role membership codes. Integers rather than strings because they index the
#: bucket table below and end up in a hot loop over ~12,000 rows.
TRAIN, VAL, TEST = 0, 1, 2

# bucket[acid_role][amine_role]
_TABLE: tuple[tuple[str, ...], ...] = (
    ("train",         "val_e1n", "test_e1n"),       # acid TRAIN
    ("val_e1a",       "val_e2",  "test_e2_mixed"),  # acid VAL
    ("test_e1a", "test_e2_mixed", "test_e2"),       # acid TEST
)


def bucket_of(acid_role: int, amine_role: int) -> str:
    return _TABLE[acid_role][amine_role]


@dataclass(frozen=True)
class Fold:
    """One bipartite entity fold. Entity sets are the primitive; rows derive."""

    partition: int
    fold: int
    #: Role code per acid index and per amine index, length ``n_acids`` /
    #: ``n_amines``. These are the fold: everything else is derived from them.
    acid_role: np.ndarray
    amine_role: np.ndarray
    #: Bucket name per row of the screen frame, length ``n_rows``.
    row_bucket: np.ndarray

    @property
    def key(self) -> str:
        return f"p{self.partition}f{self.fold}"

    def entities(self, role_codes: np.ndarray, code: int) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(role_codes == code))

    @property
    def test_acids(self) -> tuple[int, ...]:
        return self.entities(self.acid_role, TEST)

    @property
    def val_acids(self) -> tuple[int, ...]:
        return self.entities(self.acid_role, VAL)

    @property
    def train_acids(self) -> tuple[int, ...]:
        return self.entities(self.acid_role, TRAIN)

    @property
    def test_amines(self) -> tuple[int, ...]:
        return self.entities(self.amine_role, TEST)

    @property
    def val_amines(self) -> tuple[int, ...]:
        return self.entities(self.amine_role, VAL)

    @property
    def train_amines(self) -> tuple[int, ...]:
        return self.entities(self.amine_role, TRAIN)

    def mask(self, which: str | tuple[str, ...]) -> np.ndarray:
        names = (which,) if isinstance(which, str) else tuple(which)
        unknown = set(names) - set(BUCKETS)
        if unknown:
            raise KeyError(f"unknown bucket(s) {sorted(unknown)}; "
                           f"expected a subset of {BUCKETS}")
        return np.isin(self.row_bucket, names)

    def rows(self, frame: pd.DataFrame, which: str | tuple[str, ...]) -> pd.DataFrame:
        return frame.loc[self.mask(which)].reset_index(drop=True)

    def counts(self) -> dict[str, int]:
        return {b: int((self.row_bucket == b).sum()) for b in BUCKETS}


def _assign_roles(n: int, k: int, fold: int, rng: np.random.Generator,
                  degree: np.ndarray | None) -> np.ndarray:
    """Cut ``n`` entities into ``k`` groups, then read off one fold's roles.

    Group ``fold`` is test and group ``(fold + 1) % k`` is validation, so within
    a partition every entity is held out for test exactly once and for
    validation exactly once, and the two never coincide (``k >= 2``).

    With ``degree`` supplied the cut is stratified: entities are ordered by
    degree, taken in consecutive blocks of ``k``, and one member of each block
    goes to each group. Uniform assignment leaves the fold-to-fold spread of
    test-row counts much wider -- on this screen the worst fold of a random
    8-way cut has ~720 unseen-acid rows against a mean of ~1,094, and stratified
    lifts that floor to ~937 -- and a fold with few test rows is a fold whose
    per-entity estimate is noise. Stratification uses only the observation
    *count*, which is fixed before any outcome is read.
    """
    if k < 2:
        raise ValueError("k must be at least 2 so test and validation differ")
    group = np.empty(n, dtype=np.int64)
    if degree is None:
        group[rng.permutation(n)] = np.arange(n) % k
    else:
        order = np.argsort(-np.asarray(degree), kind="stable")
        for start in range(0, n, k):
            block = order[start:start + k]
            group[block] = rng.permutation(len(block))
    role = np.full(n, TRAIN, dtype=np.int64)
    role[group == fold] = TEST
    role[group == (fold + 1) % k] = VAL
    return role


def make_folds(acid_index: np.ndarray, amine_index: np.ndarray,
               n_acids: int, n_amines: int, k: int = 5,
               n_partitions: int = 3, seed: int = 20260826,
               stratify: bool = True,
               acid_group: np.ndarray | None = None,
               amine_group: np.ndarray | None = None) -> list[Fold]:
    """``k``-way bipartite entity folds, repeated over ``n_partitions`` shuffles.

    Each partition is one independent cut of the acid groups and one of the
    amine groups, so across partitions every entity gets ``n_partitions``
    independent turns as a test entity. That is what makes a per-entity summary
    possible without any entity's number resting on a single fold.

    ``acid_group`` / ``amine_group`` come from :func:`split_groups`. Roles are
    assigned to *groups* and then broadcast to their members, so stereoisomers
    and feature twins can never be split across the boundary. Omitting them
    treats every entity as its own group, which is what the tiny CI fixture
    does; the authoritative sweep always passes them.

    Everything here is a function of ``seed``, ``k`` and the *observed row
    counts*. No outcome is read; this function does not even take the target.
    """
    acid_index = np.asarray(acid_index)
    amine_index = np.asarray(amine_index)
    if len(acid_index) != len(amine_index):
        raise ValueError("acid_index and amine_index must be row-aligned")
    if acid_group is None:
        acid_group = np.arange(n_acids)
    if amine_group is None:
        amine_group = np.arange(n_amines)
    acid_group = np.asarray(acid_group, dtype=np.int64)
    amine_group = np.asarray(amine_group, dtype=np.int64)
    if len(acid_group) != n_acids or len(amine_group) != n_amines:
        raise ValueError("group vectors must be entity-index ordered")

    n_ga = int(acid_group.max()) + 1
    n_gn = int(amine_group.max()) + 1
    # Stratify on the group's *total* row count, not an entity's: it is the
    # group that is assigned, and two feature twins held out together remove
    # both their row counts from training at once.
    deg_a = (np.bincount(acid_group[acid_index], minlength=n_ga)
             if stratify else None)
    deg_n = (np.bincount(amine_group[amine_index], minlength=n_gn)
             if stratify else None)

    folds: list[Fold] = []
    for p in range(n_partitions):
        for f in range(k):
            ra = _assign_roles(n_ga, k, f,
                               np.random.default_rng([seed, p, 0]), deg_a)[acid_group]
            rn = _assign_roles(n_gn, k, f,
                               np.random.default_rng([seed, p, 1]), deg_n)[amine_group]
            row_bucket = np.array(
                [_TABLE[ra[a]][rn[n]] for a, n in zip(acid_index, amine_index)],
                dtype=object)
            folds.append(Fold(partition=p, fold=f, acid_role=ra, amine_role=rn,
                              row_bucket=row_bucket))
    return folds


def assert_partition(fold: Fold, n_rows: int) -> None:
    """The nine buckets are disjoint and exhaust every measured row.

    Checked by counting rather than inferred from the construction. A future
    change to :func:`make_folds` that starts dropping rows must surface here and
    not as a quietly smaller training set. There is no coverage knob to excuse a
    shortfall: every measured row belongs to exactly one bucket, always.
    """
    if len(fold.row_bucket) != n_rows:
        raise AssertionError(
            f"{fold.key}: row_bucket has {len(fold.row_bucket)} entries for "
            f"{n_rows} rows")
    total = 0
    seen = np.zeros(n_rows, dtype=bool)
    for b in BUCKETS:
        m = fold.row_bucket == b
        if (seen & m).any():
            raise AssertionError(f"{fold.key}: bucket {b} overlaps an earlier one")
        seen |= m
        total += int(m.sum())
    if total != n_rows:
        raise AssertionError(
            f"{fold.key}: buckets hold {total} rows, expected {n_rows}")
    if not seen.all():
        stray = sorted(set(fold.row_bucket[~seen]))
        raise AssertionError(
            f"{fold.key}: {int((~seen).sum())} rows are in no bucket; "
            f"unrecognised labels {stray}")


def assert_no_entity_leakage(fold: Fold, acid_index: np.ndarray,
                             amine_index: np.ndarray, n_rows: int) -> None:
    """The guard the whole phase rests on, stated over *entities*.

    Deliberately stronger than a row-level or pair-level check. A pair-level
    guard would pass a split in which acid ``k`` is "held out" in one row while
    another row containing ``k`` sits in training -- which is transductive
    completion, not extrapolation.

    What is asserted:

    * the three role codes partition each entity list;
    * **no training row touches a test or validation entity in either role** --
      restated as a set membership over the flattened training endpoints, so a
      bug in the bucket table and a bug in this check would have to agree;
    * no selection row touches a test entity;
    * every bucket's rows really have the role pattern its name claims;
    * and only then, the counting in :func:`assert_partition`.

    Order matters. The membership checks run **first**. A planted test entity
    violates both the membership rule and the bucket arithmetic, and if the
    arithmetic ran first the failure would be reported as a bookkeeping
    complaint, which is true, unhelpful, and hides the leak.
    """
    for name, role, n in (("acid", fold.acid_role, len(fold.acid_role)),
                          ("amine", fold.amine_role, len(fold.amine_role))):
        if set(np.unique(role)) - {TRAIN, VAL, TEST}:
            raise AssertionError(f"{fold.key}: {name} roles hold an unknown code")
        if len(role) != n:
            raise AssertionError(f"{fold.key}: {name} role vector is the wrong length")

    acid_index = np.asarray(acid_index)
    amine_index = np.asarray(amine_index)

    tr = fold.mask("train")
    for name, idx, role in (("acid", acid_index, fold.acid_role),
                            ("amine", amine_index, fold.amine_role)):
        touched = set(int(i) for i in np.unique(idx[tr])) if tr.any() else set()
        bad_test = sorted(touched & set(fold.entities(role, TEST)))
        if bad_test:
            raise AssertionError(
                f"{fold.key}: {name}s {bad_test} are held out for test and appear "
                f"in a training row")
        bad_val = sorted(touched & set(fold.entities(role, VAL)))
        if bad_val:
            raise AssertionError(
                f"{fold.key}: {name}s {bad_val} are validation entities and appear "
                f"in a training row")

    sel = fold.mask(SELECT_BUCKETS)
    for name, idx, role in (("acid", acid_index, fold.acid_role),
                            ("amine", amine_index, fold.amine_role)):
        touched = set(int(i) for i in np.unique(idx[sel])) if sel.any() else set()
        bad = sorted(touched & set(fold.entities(role, TEST)))
        if bad:
            raise AssertionError(
                f"{fold.key}: {name}s {bad} are test entities and appear in a "
                f"selection row")

    # Every bucket really has the role pattern its name claims. Without this the
    # table could be transposed -- unseen-acid rows reported as unseen-amine --
    # and every count would still add up.
    for b in BUCKETS:
        m = fold.row_bucket == b
        if not m.any():
            continue
        want = {(ra, rn) for ra in range(3) for rn in range(3)
                if _TABLE[ra][rn] == b}
        got = set(zip(fold.acid_role[acid_index[m]].tolist(),
                      fold.amine_role[amine_index[m]].tolist()))
        if got - want:
            raise AssertionError(
                f"{fold.key}: bucket {b} contains rows with role patterns "
                f"{sorted(got - want)}, expected {sorted(want)}")

    assert_partition(fold, n_rows)


def fold_summary(fold: Fold) -> dict:
    out = {"partition": fold.partition, "fold": fold.fold, "fold_key": fold.key,
           "n_test_acids": len(fold.test_acids),
           "n_test_amines": len(fold.test_amines),
           "n_val_acids": len(fold.val_acids),
           "n_val_amines": len(fold.val_amines),
           "n_train_acids": len(fold.train_acids),
           "n_train_amines": len(fold.train_amines)}
    out.update({f"n_{b}": v for b, v in fold.counts().items()})
    return out


def split_groups(smiles: tuple[str, ...], features: np.ndarray | None = None
                 ) -> np.ndarray:
    """Group ids over the entities of one role. Outcome-independent, by construction.

    Two entities land in the same group when they share a stereo-stripped
    canonical SMILES, a neutralised one, a canonical tautomer of one, or a
    byte-identical primary feature vector. The relation is closed transitively,
    so a chain of near-identities cannot leave two members of it on opposite
    sides of a split.

    ``features`` is the *primary* representation. Passing a control
    representation here would define the groups differently for the control run
    than for the real one, which would make the two incomparable -- so the sweep
    always builds groups from the real fingerprints and reuses them.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    n = len(smiles)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    from rdkit.Chem.MolStandardize import rdMolStandardize
    uncharger = rdMolStandardize.Uncharger()
    enumerator = rdMolStandardize.TautomerEnumerator()

    by_flat: dict[str, int] = {}
    by_neutral: dict[str, int] = {}
    by_tautomer: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit cannot parse {smi!r}")
        flat = Chem.MolToSmiles(mol, isomericSmiles=False)
        if flat in by_flat:
            union(by_flat[flat], i)
        else:
            by_flat[flat] = i
        neutral = Chem.MolToSmiles(uncharger.uncharge(Chem.Mol(mol)),
                                   isomericSmiles=False)
        if neutral in by_neutral:
            union(by_neutral[neutral], i)
        else:
            by_neutral[neutral] = i
        taut = Chem.MolToSmiles(
            enumerator.Canonicalize(Chem.MolFromSmiles(flat)),
            isomericSmiles=False)
        if taut in by_tautomer:
            union(by_tautomer[taut], i)
        else:
            by_tautomer[taut] = i

    if features is not None:
        if len(features) != n:
            raise ValueError("features must be entity-index ordered")
        by_fp: dict[bytes, int] = {}
        for i, row in enumerate(np.ascontiguousarray(features)):
            key = row.tobytes()
            if key in by_fp:
                union(by_fp[key], i)
            else:
                by_fp[key] = i

    roots = np.array([find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group.astype(np.int64)


def group_report(smiles: tuple[str, ...], group: np.ndarray, role: str) -> dict:
    sizes = np.bincount(group)
    merged = [sorted(int(i) for i in np.flatnonzero(group == g))
              for g in np.flatnonzero(sizes > 1)]
    return {"role": role, "n_entities": len(smiles), "n_groups": int(sizes.size),
            "n_groups_with_more_than_one": int((sizes > 1).sum()),
            "n_entities_merged": int(sizes[sizes > 1].sum()),
            "largest_group": int(sizes.max()),
            "merged_groups": [[smiles[i] for i in g] for g in merged]}


@dataclass(frozen=True)
class PairFold:
    """A **transductive** fold: the held-out unit is a pair, not a reactant.

    Every test pair has both endpoints present in the training rows, so a model
    with free per-entity embeddings can estimate both and complete the matrix.
    That is a different question from the rest of this phase and its results are
    never placed in an entity-OOD table -- it exists to answer the prior
    question, "is there any acid-amine interaction structure here to find?", and
    if the answer is no then an inductive failure says nothing.

    Pairs with an endpoint that survives nowhere in training are dropped rather
    than scored, and the count is carried in :attr:`n_dropped` so that a fold
    which silently lost half its test set is visible.
    """

    partition: int
    fold: int
    row_bucket: np.ndarray
    n_dropped: int

    @property
    def key(self) -> str:
        return f"t{self.partition}f{self.fold}"

    def mask(self, which: str | tuple[str, ...]) -> np.ndarray:
        names = (which,) if isinstance(which, str) else tuple(which)
        return np.isin(self.row_bucket, names)

    def counts(self) -> dict[str, int]:
        return {b: int((self.row_bucket == b).sum())
                for b in ("train", "val", "test", "dropped")}


def make_pair_folds(acid_index: np.ndarray, amine_index: np.ndarray,
                    k: int = 5, n_partitions: int = 1,
                    seed: int = 20260904) -> list[PairFold]:
    """``k``-way folds over unique (acid, amine) pairs, for the transductive ceiling.

    Whole pairs move together, so no row of a test pair can appear in training --
    without that the "ceiling" would be memorisation of a replicate rather than
    matrix completion. Entities are deliberately *not* held out.
    """
    acid_index = np.asarray(acid_index)
    amine_index = np.asarray(amine_index)
    pairs, inverse = np.unique(np.stack([acid_index, amine_index], 1),
                               axis=0, return_inverse=True)
    n_pairs = len(pairs)
    folds: list[PairFold] = []
    for p in range(n_partitions):
        rng = np.random.default_rng([seed, p, 99])
        group = np.empty(n_pairs, dtype=np.int64)
        group[rng.permutation(n_pairs)] = np.arange(n_pairs) % k
        for f in range(k):
            role = np.where(group[inverse] == f, "test",
                            np.where(group[inverse] == (f + 1) % k, "val",
                                     "train"))
            tr = role == "train"
            seen_a = set(acid_index[tr].tolist())
            seen_n = set(amine_index[tr].tolist())
            orphan = ~(np.isin(acid_index, list(seen_a))
                       & np.isin(amine_index, list(seen_n)))
            role = np.where(orphan & (role != "train"), "dropped", role)
            folds.append(PairFold(partition=p, fold=f, row_bucket=role,
                                  n_dropped=int((role == "dropped").sum())))
    return folds


def assert_transductive(fold: PairFold, acid_index: np.ndarray,
                        amine_index: np.ndarray) -> None:
    """Every scored row's endpoints appear in training, and no pair straddles.

    The inverse of :func:`assert_no_entity_leakage`, asserted just as hard. A
    "ceiling" whose test pairs are actually entity-OOD would be measuring the
    inductive question again and reporting it as the transductive one.
    """
    tr = fold.mask("train")
    seen_a = set(np.asarray(acid_index)[tr].tolist())
    seen_n = set(np.asarray(amine_index)[tr].tolist())
    for bucket in ("val", "test"):
        m = fold.mask(bucket)
        if not m.any():
            continue
        bad_a = sorted(set(np.asarray(acid_index)[m].tolist()) - seen_a)
        bad_n = sorted(set(np.asarray(amine_index)[m].tolist()) - seen_n)
        if bad_a or bad_n:
            raise AssertionError(
                f"{fold.key}: {bucket} contains untrained acids {bad_a[:5]} or "
                f"amines {bad_n[:5]}; this is not a transductive fold")
    train_pairs = set(zip(np.asarray(acid_index)[tr].tolist(),
                          np.asarray(amine_index)[tr].tolist()))
    for bucket in ("val", "test"):
        m = fold.mask(bucket)
        if not m.any():
            continue
        overlap = train_pairs & set(zip(np.asarray(acid_index)[m].tolist(),
                                        np.asarray(amine_index)[m].tolist()))
        if overlap:
            raise AssertionError(
                f"{fold.key}: {len(overlap)} pairs appear in both train and "
                f"{bucket}")
