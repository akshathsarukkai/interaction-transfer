"""Phase 4 invariants: the guards the ChemLex entity-OOD result rests on.

Organised by what a failure would mean rather than by module. The expensive
tests are the ones that plant a defect and assert it is caught -- a guard nobody
has watched fail is a guard nobody knows works, and three of the four defects
Phase 3's audit found in its own machinery were in code that had no test.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from intervention_algebra.real_data.chemlex import dataset as ds
from intervention_algebra.real_data.chemlex import experiment as ex
from intervention_algebra.real_data.chemlex import features as ft
from intervention_algebra.real_data.chemlex import models as md
from intervention_algebra.real_data.chemlex import splits as sp
from intervention_algebra.real_data.chemlex import sweep as sw
from intervention_algebra.real_data.chemlex import train as tr
from intervention_algebra.real_data.chemlex.acquire import (CURRENT, VERSIONS,
                                                            raw_path, verify_raw)
from intervention_algebra.real_data.chemlex.evaluate import (additive_projection,
                                                             incremental,
                                                             paired_summary,
                                                             per_entity_incremental)

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "chemlex_tiny"
RESULTS = REPO / "results" / "phase4_chemlex"

#: The Zenodo file is 550 kB of CC BY-NC data, is gitignored, and is fetched by
#: `scripts/download_chemlex.py`. Tests that genuinely need it skip without it: a
#: green build must never require a third-party host to be up, and "the data is
#: missing" must not be reported as "the code is broken".
DEPOSIT = raw_path()

needs_deposit = pytest.mark.skipif(
    not DEPOSIT.exists(),
    reason=f"{DEPOSIT} absent; run scripts/download_chemlex.py")
needs_results = pytest.mark.skipif(
    not (RESULTS / "primary.jsonl").exists(),
    reason="no committed Phase 4 results yet")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_excel(FIXTURE / "Chemlex_Acidamine_Wetlab_Data.xlsx",
                         sheet_name=0)


@pytest.fixture(scope="module")
def screen(raw) -> ds.Screen:
    return ds.load_screen("all", raw=raw)


@pytest.fixture(scope="module")
def prepared(raw) -> ex.Prepared:
    ex._CACHE.clear()
    return ex.prepare(ex.Spec(block="primary", screen="all"), raw=raw)


@pytest.fixture(scope="module")
def folds(prepared) -> list[sp.Fold]:
    f = prepared.screen.frame
    return sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                         len(prepared.screen.acids), len(prepared.screen.amines),
                         k=3, n_partitions=2, seed=7,
                         acid_group=prepared.acid_group,
                         amine_group=prepared.amine_group)


# --------------------------------------------------------------------------
# The split. Everything else is downstream of these.
# --------------------------------------------------------------------------

def test_no_test_entity_of_either_role_appears_in_training(prepared, folds):
    f = prepared.screen.frame
    a, n = f["acid"].to_numpy(), f["amine"].to_numpy()
    for fold in folds:
        m = fold.mask("train")
        assert not (set(a[m].tolist()) & set(fold.test_acids)), fold.key
        assert not (set(n[m].tolist()) & set(fold.test_amines)), fold.key


def test_validation_entities_are_absent_from_training(prepared, folds):
    """Validation must be entity-OOD too, not merely test-free.

    Until Phase 3's audit, its guard only forbade a *test* drug in a validation
    pair, so a validation bucket padded with pairs between two training drugs
    passed. Here the invariant is asserted directly.
    """
    f = prepared.screen.frame
    a, n = f["acid"].to_numpy(), f["amine"].to_numpy()
    for fold in folds:
        m = fold.mask("train")
        assert not (set(a[m].tolist()) & set(fold.val_acids)), fold.key
        assert not (set(n[m].tolist()) & set(fold.val_amines)), fold.key


def test_selection_never_sees_a_test_entity(prepared, folds):
    f = prepared.screen.frame
    a, n = f["acid"].to_numpy(), f["amine"].to_numpy()
    for fold in folds:
        m = fold.mask(sp.SELECT_BUCKETS)
        assert not (set(a[m].tolist()) & set(fold.test_acids)), fold.key
        assert not (set(n[m].tolist()) & set(fold.test_amines)), fold.key


def test_every_row_is_in_exactly_one_bucket_and_the_counts_sum(prepared, folds):
    n_rows = len(prepared.screen.frame)
    for fold in folds:
        counts = fold.counts()
        assert sum(counts.values()) == n_rows, fold.key
        seen = np.zeros(n_rows, dtype=int)
        for b in sp.BUCKETS:
            seen += fold.mask(b).astype(int)
        assert (seen == 1).all(), fold.key


def test_the_three_test_regimes_are_disjoint_and_correctly_typed(prepared, folds):
    """E1-A, E1-N and E2 must be distinct, and each must be what its name says."""
    f = prepared.screen.frame
    a, n = f["acid"].to_numpy(), f["amine"].to_numpy()
    for fold in folds:
        ma, mn, m2 = (fold.mask("test_e1a"), fold.mask("test_e1n"),
                      fold.mask("test_e2"))
        assert not (ma & mn).any() and not (ma & m2).any() and not (mn & m2).any()
        ta, tn = set(fold.test_acids), set(fold.test_amines)
        tra, trn = set(fold.train_acids), set(fold.train_amines)
        assert set(a[ma].tolist()) <= ta and set(n[ma].tolist()) <= trn
        assert set(a[mn].tolist()) <= tra and set(n[mn].tolist()) <= tn
        assert set(a[m2].tolist()) <= ta and set(n[m2].tolist()) <= tn


def test_no_unmeasured_combination_is_ever_fabricated(prepared, folds):
    """Every row of every bucket is a row of the deposit.

    Never label an unmeasured acid-amine combination as a failed reaction: 87 %
    of the acid x amine grid was never run, and inventing negatives there would
    manufacture exactly the structure the phase is testing for.
    """
    f = prepared.screen.frame
    observed = set(zip(f["acid"].tolist(), f["amine"].tolist(),
                       f["cond"].tolist()))
    for fold in folds:
        for b in sp.BUCKETS:
            sub = f.loc[fold.mask(b)]
            got = set(zip(sub["acid"].tolist(), sub["amine"].tolist(),
                          sub["cond"].tolist()))
            assert got <= observed, f"{fold.key}/{b} invented a combination"


def test_a_planted_test_entity_is_caught_by_the_leakage_guard(prepared, folds):
    """Mutation test: corrupt the bucket table and the guard must raise.

    Deliberately a *bucket* mutation rather than a role mutation -- a role
    mutation would also break the counting, and the failure would be reported as
    a bookkeeping complaint instead of a leak.
    """
    f = prepared.screen.frame
    a, n = f["acid"].to_numpy(), f["amine"].to_numpy()
    fold = folds[0]
    sp.assert_no_entity_leakage(fold, a, n, len(f))          # clean

    bad = fold.row_bucket.copy()
    victims = np.flatnonzero(fold.mask("test_e1a"))
    assert victims.size, "the fixture produced no unseen-acid rows to plant with"
    bad[victims[0]] = "train"
    planted = sp.Fold(fold.partition, fold.fold, fold.acid_role,
                      fold.amine_role, bad)
    with pytest.raises(AssertionError, match="test and appear in a training row"):
        sp.assert_no_entity_leakage(planted, a, n, len(f))


def test_a_lost_row_is_caught_by_the_partition_check(prepared, folds):
    f = prepared.screen.frame
    fold = folds[0]
    bad = fold.row_bucket.copy()
    bad[0] = "nowhere"
    planted = sp.Fold(fold.partition, fold.fold, fold.acid_role,
                      fold.amine_role, bad)
    with pytest.raises(AssertionError):
        sp.assert_partition(planted, len(f))


def test_a_transposed_bucket_table_is_caught(prepared, folds):
    """Swap the two E1 buckets and the role-pattern check must notice.

    Without this, unseen-acid rows reported as unseen-amine rows would still add
    up perfectly and every count in every table would look right.
    """
    f = prepared.screen.frame
    fold = folds[0]
    bad = fold.row_bucket.copy()
    swap = {"test_e1a": "test_e1n", "test_e1n": "test_e1a"}
    bad = np.array([swap.get(b, b) for b in bad], dtype=object)
    planted = sp.Fold(fold.partition, fold.fold, fold.acid_role,
                      fold.amine_role, bad)
    with pytest.raises(AssertionError):
        sp.assert_no_entity_leakage(planted, f["acid"].to_numpy(),
                                    f["amine"].to_numpy(), len(f))


def test_split_groups_keep_stereoisomers_and_feature_twins_together(prepared):
    """The fixture plants one stereo pair per role; both must be merged."""
    for smiles, group, fp in ((prepared.screen.acids, prepared.acid_group,
                               prepared.acid_fp),
                              (prepared.screen.amines, prepared.amine_group,
                               prepared.amine_fp)):
        assert group.max() + 1 < len(smiles), "nothing was merged at all"
        for g in range(group.max() + 1):
            members = np.flatnonzero(group == g)
            if members.size < 2:
                continue
            from rdkit import Chem
            flats = {Chem.MolToSmiles(Chem.MolFromSmiles(smiles[i]),
                                      isomericSmiles=False) for i in members}
            same_fp = len({fp.x[i].tobytes() for i in members}) == 1
            assert len(flats) == 1 or same_fp, (
                f"group {g} merged entities that are neither the same "
                f"constitution nor feature twins")


def test_a_stereoisomer_never_straddles_a_fold(prepared, folds):
    """The reason split groups exist, asserted end to end."""
    for fold in folds:
        for group, test_attr, train_attr in (
                (prepared.acid_group, "test_acids", "train_acids"),
                (prepared.amine_group, "test_amines", "train_amines")):
            test_groups = {int(group[i]) for i in getattr(fold, test_attr)}
            train_groups = {int(group[i]) for i in getattr(fold, train_attr)}
            assert not (test_groups & train_groups), fold.key


def test_folds_are_deterministic_from_the_seed(prepared):
    f = prepared.screen.frame
    kw = dict(n_acids=len(prepared.screen.acids),
              n_amines=len(prepared.screen.amines), k=3, n_partitions=1,
              acid_group=prepared.acid_group, amine_group=prepared.amine_group)
    a, b = (sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                          seed=11, **kw) for _ in range(2))
    for x, y in zip(a, b):
        assert (x.row_bucket == y.row_bucket).all()
        assert (x.acid_role == y.acid_role).all()
    other = sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                          seed=12, **kw)
    assert any((x.row_bucket != y.row_bucket).any() for x, y in zip(a, other))


def test_fold_construction_never_reads_the_outcome(prepared):
    """Scramble every conversion and the folds must not move.

    The strongest available statement of "no outcome-based fold selection":
    the buckets are a function of the entity lists and the observed row counts
    and of nothing else.
    """
    f = prepared.screen.frame
    kw = dict(n_acids=len(prepared.screen.acids),
              n_amines=len(prepared.screen.amines), k=3, n_partitions=1, seed=3,
              acid_group=prepared.acid_group, amine_group=prepared.amine_group)
    before = sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(), **kw)
    rng = np.random.default_rng(0)
    g = f.copy()
    g["y"] = rng.permutation(g["y"].to_numpy())
    after = sp.make_folds(g["acid"].to_numpy(), g["amine"].to_numpy(), **kw)
    for x, y in zip(before, after):
        assert (x.row_bucket == y.row_bucket).all()


# --------------------------------------------------------------------------
# The transductive ceiling is transductive, and labelled as such.
# --------------------------------------------------------------------------

def test_pair_folds_are_transductive_and_hold_out_whole_pairs(prepared):
    f = prepared.screen.frame
    pfolds = sp.make_pair_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                                k=3, n_partitions=1, seed=5)
    for fold in pfolds:
        sp.assert_transductive(fold, f["acid"].to_numpy(), f["amine"].to_numpy())
        assert fold.mask("test").sum() > 0


def test_a_pair_that_straddles_train_and_test_is_caught():
    """Mutation test on a hand-built fold, not on the fixture's geometry.

    The guard is about one pair appearing on both sides, so the mutation has to
    put a *second copy* of a training pair into test. Reaching into the fixture
    for a pair that happens to carry two rows in the right bucket would make the
    test depend on the fixture's sampling; constructing the arrays makes it
    depend on the guard.
    """
    #        rows:  (0,0) (0,0) (0,1) (1,0) (1,1) (1,1) (2,0) (2,1)
    a = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    n = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    B = lambda *xs: np.array(xs, dtype=object)

    clean = sp.PairFold(0, 0, B("train", "train", "test", "train", "test",
                                "test", "train", "train"), 0)
    sp.assert_transductive(clean, a, n)

    # Pair (1,1) is now in train (row 4) and still in test (row 5).
    straddle = sp.PairFold(0, 0, B("train", "train", "test", "train", "train",
                                   "test", "train", "train"), 0)
    with pytest.raises(AssertionError, match="appear in both train and"):
        sp.assert_transductive(straddle, a, n)

    # Acid 2 appears in no training row: this fold is entity-OOD wearing the
    # ceiling's name, which is exactly what the guard exists to refuse.
    entity_ood = sp.PairFold(0, 0, B("train", "train", "test", "train", "test",
                                     "test", "test", "test"), 0)
    with pytest.raises(AssertionError, match="not a transductive fold"):
        sp.assert_transductive(entity_ood, a, n)


# --------------------------------------------------------------------------
# Features and controls.
# --------------------------------------------------------------------------

def test_fingerprints_are_a_function_of_structure_and_nothing_else(screen):
    """The outcome cannot reach the representation, asserted by construction.

    Scramble every conversion, refeaturise, and the matrices must be identical.
    A fingerprint is computed by RDKit from a SMILES string with no path to a
    label, and that is the main reason the primary representation is a
    fingerprint rather than a learned embedding from a corpus nobody here can
    audit.
    """
    before = ft.fingerprints(screen.acids, "acid").x
    scrambled = screen.frame.copy()
    scrambled["y"] = np.random.default_rng(0).permutation(scrambled["y"])
    after = ft.fingerprints(tuple(sorted(scrambled["acid_smiles"].unique())),
                            "acid").x
    assert np.array_equal(before, after)


def test_shuffling_destroys_entity_correspondence_but_keeps_the_distribution(screen):
    base = ft.fingerprints(screen.acids, "acid")
    sh = ft.shuffled(base, seed=3)
    assert not np.array_equal(base.x, sh.x), "the shuffle was a no-op"
    assert sorted(base.bits_set.tolist()) == sorted(sh.bits_set.tolist())
    rows_before = sorted(r.tobytes() for r in base.x)
    rows_after = sorted(r.tobytes() for r in sh.x)
    assert rows_before == rows_after, "the shuffle changed the feature set"


def test_shuffled_both_permutes_the_two_roles_independently(screen):
    fa = ft.fingerprints(screen.acids, "acid")
    fn = ft.fingerprints(screen.amines, "amine")
    rep = ft.build_representation("shuffled_both", fa, fn, seed=5)
    assert not np.array_equal(rep.acid.x, fa.x)
    assert not np.array_equal(rep.amine.x, fn.x)
    only_acid = ft.build_representation("shuffled_acid", fa, fn, seed=5)
    assert np.array_equal(only_acid.amine.x, fn.x)


def test_random_features_are_fixed_before_any_split_exists(screen):
    """Drawn from a seed, not from a fold. Two draws must agree exactly."""
    fa = ft.fingerprints(screen.acids, "acid")
    a = ft.random_like(fa, seed=9).x
    b = ft.random_like(fa, seed=9).x
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ft.random_like(fa, seed=10).x)


def test_control_representations_do_not_change_the_split_groups(prepared, raw):
    """Groups come from the real fingerprints for every representation.

    Defining them from a control would give the control a different fold
    geometry, and the control would then differ from the real run in two ways
    instead of one.
    """
    spec = ex.Spec(block="control", screen="all", representation="shuffled_both")
    other = ex.prepare(spec, raw=raw)
    assert np.array_equal(other.acid_group, prepared.acid_group)
    assert np.array_equal(other.amine_group, prepared.amine_group)


def test_blinding_uses_the_training_marginal_and_never_a_zero_vector(prepared,
                                                                     folds):
    """Phase 3 used zeros here and it manufactured a result.

    A zero row asserts "this molecule has no substructures at all", a point no
    real molecule occupies, and against it a random-feature control containing
    no chemistry scored a significant effect. The replacement must be the mean
    over the training entities of that role.
    """
    fold = folds[0]
    base = prepared.acid_fp
    blinded = ft.blind_features(base, np.array(fold.train_acids),
                                np.array(fold.test_acids))
    marginal = base.x[np.array(fold.train_acids)].mean(axis=0)
    for i in fold.test_acids:
        assert np.allclose(blinded.x[i], marginal)
        assert blinded.x[i].sum() > 0, "the blind row is a zero vector"
    for i in fold.train_acids:
        assert np.array_equal(blinded.x[i], base.x[i]), "training rows moved"


def test_blinding_refuses_to_blind_an_entity_it_also_averages(prepared, folds):
    fold = folds[0]
    overlapping = np.array(fold.train_acids[:1] + fold.test_acids[:1])
    with pytest.raises(ValueError, match="both trained on and blinded"):
        ft.blind_features(prepared.acid_fp, np.array(fold.train_acids),
                          overlapping)


def test_similarity_refuses_a_self_match(prepared, folds):
    fold = folds[0]
    with pytest.raises(ValueError, match="both the query and the reference"):
        ft.max_similarity_to(prepared.acid_fp, np.array(fold.test_acids),
                             np.array(fold.test_acids))


def test_similarity_strata_are_outcome_independent(prepared, folds):
    """Scramble the outcome; every held-out entity's similarity is unchanged."""
    fold = folds[0]
    a = ft.max_similarity_to(prepared.acid_fp, np.array(fold.test_acids),
                             np.array(fold.train_acids))
    prepared.screen.frame["y"] = np.random.default_rng(1).permutation(
        prepared.screen.frame["y"].to_numpy())
    b = ft.max_similarity_to(prepared.acid_fp, np.array(fold.test_acids),
                             np.array(fold.train_acids))
    assert np.allclose(a, b)


# --------------------------------------------------------------------------
# The models: nested, alive, and identical apart from the term under test.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_cfg() -> md.ModelConfig:
    rng = np.random.default_rng(0)
    return md.ModelConfig(
        n_acids=9, n_amines=7, n_conditions=3,
        x_acid=(rng.random((9, 32)) < 0.3).astype(np.float32),
        x_amine=(rng.random((7, 32)) < 0.3).astype(np.float32), rank=4)


def _idx(cfg):
    g = torch.Generator().manual_seed(0)
    return (torch.randint(0, cfg.n_acids, (40,), generator=g),
            torch.randint(0, cfg.n_amines, (40,), generator=g),
            torch.randint(0, cfg.n_conditions, (40,), generator=g))


@pytest.mark.parametrize("simpler,richer", [
    ("additive", "lowrank"),
    ("additive", "flexible"),
    ("additive", "condition_expanded"),
    ("condition_expanded", "condition_expanded_pair"),
    ("transductive_additive", "transductive"),
])
def test_each_rung_is_exactly_its_simpler_form_at_initialisation(tiny_cfg,
                                                                 simpler, richer):
    a, n, c = _idx(tiny_cfg)
    with torch.no_grad():
        s = md.build(simpler, tiny_cfg)(a, n, c)
        r = md.build(richer, tiny_cfg)(a, n, c)
    assert torch.equal(s, r), f"{richer} is not {simpler} at init"


@pytest.mark.parametrize("name", sorted(md.BUILDERS))
def test_every_interaction_term_has_a_live_gradient_at_init(tiny_cfg, name):
    """The trap Phase 2R fell into: every partial derivative zero at init.

    If the encoder and the bilinear form both start at zero, the term stays dead
    for the whole run and reports exactly 0.0 incremental skill -- which is
    indistinguishable in the results table from the finding under test.
    """
    a, n, c = _idx(tiny_cfg)
    model = md.build(name, tiny_cfg)
    torch.nn.functional.mse_loss(model(a, n, c),
                                 torch.randn(40, generator=torch.Generator().manual_seed(1))
                                 ).backward()
    for key, tensor in model.interaction_tensors().items():
        assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0, \
            f"{name}.{key} is dead at initialisation"


def test_per_entity_and_gathered_heads_agree(tiny_cfg):
    model = md.build("additive", tiny_cfg)
    a, n, c = _idx(tiny_cfg)
    with torch.no_grad():
        gathered = model.gather(model.fa, model.XA, a)
        direct = model.fa(model.XA[a])
    assert torch.allclose(gathered, direct, atol=1e-6)


def test_the_transductive_baseline_has_no_dead_weights(tiny_cfg):
    """It must not inherit embeddings it never uses.

    Dead weights would be shrunk by weight decay while still counting in the
    parameter totals the capacity comparison reads.
    """
    model = md.build("transductive_additive", tiny_cfg)
    a, n, c = _idx(tiny_cfg)
    torch.nn.functional.mse_loss(model(a, n, c), torch.zeros(40) + 0.5).backward()
    dead = [k for k, p in model.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, f"unreachable parameters: {dead}"


def test_condition_features_are_identical_across_compared_models(tiny_cfg):
    """Both sides of a contrast see the same condition encoding.

    Not decoration: if the pair model saw a richer condition representation, its
    advantage would be a condition effect wearing an interaction's name.
    """
    a, n, c = _idx(tiny_cfg)
    base, pair = md.build("additive", tiny_cfg), md.build("lowrank", tiny_cfg)
    assert base.fc.weight.shape == pair.fc.weight.shape
    with torch.no_grad():
        base.fc.weight.copy_(torch.arange(tiny_cfg.n_conditions,
                                          dtype=torch.float32).unsqueeze(1))
        pair.fc.weight.copy_(base.fc.weight)
        assert torch.equal(base.fc(c), pair.fc(c))


# --------------------------------------------------------------------------
# The metric.
# --------------------------------------------------------------------------

def test_incremental_skill_refuses_unpaired_inputs():
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="paired predictions"):
        incremental(y, np.zeros(10), np.zeros(9))


def test_incremental_skill_is_zero_when_the_two_models_agree():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=50), rng.normal(size=50)
    assert incremental(y, p, p) == pytest.approx(0.0)


def test_incremental_skill_is_computed_from_predictions_not_from_two_skills():
    """A ratio of MSEs, not a difference of R2s. They are not the same number."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    pb = y * 0.5 + rng.normal(scale=0.5, size=200)
    pp = y * 0.7 + rng.normal(scale=0.4, size=200)
    got = incremental(y, pb, pp)
    want = 1 - ((pp - y) ** 2).mean() / ((pb - y) ** 2).mean()
    assert got == pytest.approx(want)


def test_per_entity_statistics_do_not_treat_rows_as_replicates():
    """One number per entity, however many rows that entity carries."""
    frame = pd.DataFrame({"acid": [0] * 50 + [1] * 3})
    rng = np.random.default_rng(2)
    y = rng.normal(size=53)
    table = per_entity_incremental(frame, "acid", y, y + 0.3, y + 0.2)
    assert len(table) == 2
    assert set(table["n_rows"]) == {50, 3}
    summary = paired_summary(table["incremental"].to_numpy())
    assert summary["n"] == 2, "the summary counted rows, not entities"


def test_the_projection_diagnostic_uses_no_outcome():
    """Scramble y; the non-additive fraction must not move."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"acid": rng.integers(0, 5, 200),
                          "amine": rng.integers(0, 4, 200),
                          "cond": rng.integers(0, 2, 200)})
    pred = rng.normal(size=200)
    y, base = rng.normal(size=200), rng.normal(size=200)
    a = additive_projection(frame, pred, y, base)["nonadditive_fraction"]
    b = additive_projection(frame, pred, rng.permutation(y),
                            base)["nonadditive_fraction"]
    assert a == pytest.approx(b)


def test_a_purely_additive_prediction_projects_with_no_residual():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"acid": rng.integers(0, 5, 300),
                          "amine": rng.integers(0, 4, 300),
                          "cond": rng.integers(0, 2, 300)})
    a, n, c = rng.normal(size=5), rng.normal(size=4), rng.normal(size=2)
    pred = a[frame.acid] + n[frame.amine] + c[frame.cond]
    out = additive_projection(frame, pred, rng.normal(size=300),
                              rng.normal(size=300))
    assert out["nonadditive_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert out["gain_in_nonadditive"] == pytest.approx(0.0, abs=1e-12)


def test_an_interaction_leaves_a_residual_the_projection_cannot_absorb():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"acid": rng.integers(0, 6, 400),
                          "amine": rng.integers(0, 5, 400),
                          "cond": rng.integers(0, 2, 400)})
    inter = rng.normal(size=(6, 5))
    pred = inter[frame.acid, frame.amine]
    out = additive_projection(frame, pred, rng.normal(size=400),
                              rng.normal(size=400))
    assert out["nonadditive_fraction"] > 0.1


# --------------------------------------------------------------------------
# Training.
# --------------------------------------------------------------------------

def test_target_scaling_uses_training_rows_only(prepared, folds):
    """Mutate a test row's outcome; the fitted scale must not move."""
    f = prepared.screen.frame
    fold = folds[0]
    y = f["y"].to_numpy().copy()
    before = tr.fit_scaling(y[fold.mask("train")])
    y[np.flatnonzero(fold.mask("test_e1a"))[0]] = 1e6
    after = tr.fit_scaling(y[fold.mask("train")])
    assert before == after


def test_a_constant_training_target_fails_loudly():
    with pytest.raises(ValueError, match="zero variance"):
        tr.fit_scaling(np.full(10, 0.3))


def test_the_grid_routes_optimiser_and_architecture_settings_separately():
    grid = ex._grid_for("lowrank")
    assert len(grid) == len(ex.WEIGHT_DECAY_GRID) * len(ex.RANK_GRID)
    assert all({"rank", "cond_rank", "weight_decay"} == set(g) for g in grid)
    flat = ex._grid_for("additive")
    assert len(flat) == len(ex.WEIGHT_DECAY_GRID), \
        "the additive rung searched a rank it does not have"


def test_weight_decay_is_searched_for_the_baseline_too():
    """Searching it for only the pair model would be an asymmetry in the ratio."""
    assert {g["weight_decay"] for g in ex._grid_for("additive")} == \
           set(ex.WEIGHT_DECAY_GRID)


# --------------------------------------------------------------------------
# Dataset semantics.
# --------------------------------------------------------------------------

def test_an_unknown_reagent_component_raises_rather_than_being_swept_up(raw):
    bad = raw.copy()
    bad.loc[0, "Reagents"] = bad.loc[0, "Reagents"] + ".CCO"
    with pytest.raises(ValueError, match="unrecognised reagent component"):
        ds.decode_conditions(bad["Reagents"])


def test_substrate_salt_prefixes_are_stripped_from_the_condition(raw, screen):
    """A counterion of the amine is not a reaction condition.

    Left in, it would put substrate identity into the condition channel -- and in
    the real deposit its naive marginal reads as a 22-point reagent effect that
    is entirely a panel of easy substrates.
    """
    assert screen.notes["n_salt_annotated_rows"] > 0, "the fixture plants none"
    assert screen.n_conditions < raw["Reagents"].nunique()
    for c in screen.conditions:
        assert not c.startswith("Cl.") and not c.startswith("O=S")


def test_the_two_hatu_depictions_collapse_to_one_reagent(raw):
    decoded = ds.decode_conditions(raw["Reagents"])
    hatu = decoded[decoded["reagent"] == "HATU"]
    assert len(hatu) > 1, "the fixture plants only one HATU string"
    assert hatu["protocol"].nunique() > 1
    assert ds.load_screen("all", "chemistry", raw=raw).n_conditions < \
           ds.load_screen("all", "protocol", raw=raw).n_conditions


def test_entities_are_keyed_on_canonical_structure_not_on_the_raw_string(raw,
                                                                        screen):
    """The fixture writes one amine two ways; they must be one entity."""
    assert screen.notes["amines_merged_by_canonicalisation"] >= 1
    assert screen.n_amines < raw["Amine"].nunique()


def test_every_acid_has_one_carboxyl_and_every_amine_an_nh(screen):
    check = ds.role_check(screen)
    assert check["acids_failing_role"] == []
    assert check["amines_failing_role"] == []
    assert set(check["acid_cooh_counts"]) == {1}


def test_the_feasibility_threshold_is_the_authors_rule_and_is_not_re_tuned():
    assert ds.FEASIBLE_AT == 20.0


def test_the_endpoint_columns_agree_with_each_other(screen):
    f = screen.frame
    assert np.allclose(f["y"] * ds.YIELD_SCALE, f["conversion"])
    assert (f["feasible"].to_numpy()
            == (f["conversion"] >= ds.FEASIBLE_AT).astype(int).to_numpy()).all()


def test_replicate_noise_is_estimated_from_repeated_cells(screen):
    noise = ds.replicate_noise(screen)
    assert noise["n_cells_repeated"] >= 1


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------

def test_every_recorded_zenodo_version_has_a_digest_and_a_url():
    assert len(VERSIONS) == 3
    for key, v in VERSIONS.items():
        assert len(v.sha256) == 64 and v.size > 0
        assert v.url.startswith("https://zenodo.org/api/records/")
    assert CURRENT.record == "17596563"


@needs_deposit
def test_the_file_on_disk_is_the_recorded_record():
    got = verify_raw()
    assert got["matches_record"], (
        f"{got['path']} has sha256 {got['sha256']}, not {CURRENT.sha256}")


@needs_deposit
def test_the_deposit_reproduces_the_papers_own_counts():
    """272 acids, 231 amines, 6 reagents, 2 bases, 1 solvent, 11,669 rows.

    Re-derived on every run rather than trusted: the reagent and base counts are
    only right after the salt prefixes are stripped and the two HATU depictions
    are merged, so this is simultaneously a check on the deposit and on the
    decoding.
    """
    a = ds.audit()
    assert a["n_rows"] == 11669
    assert a["n_reagents"] == 6 and a["n_bases"] == 2 and a["n_solvents"] == 1
    assert a["screens"]["all"]["n_acids"] == 272
    assert a["screens"]["all"]["n_raw_amine_smiles"] == 231
    assert a["screens"]["all"]["n_canonical_amines"] == 230


# --------------------------------------------------------------------------
# The sweep's own bookkeeping.
# --------------------------------------------------------------------------

def test_the_development_and_authoritative_seeds_differ():
    """The registration rests on this. A pilot was run at the development seed."""
    assert ex.DEV_SEED != ex.AUTH_SEED
    assert ex.Spec(block="primary").seed == ex.AUTH_SEED


def test_every_advertised_part_builds_specs():
    counts = sw.part_counts()
    for part in sw.PART_GRIDS:
        assert counts[part] > 0, part
    assert counts["all"] == sum(counts[p] for p in sw.ALL_PARTS)


def test_the_runner_exits_nonzero_when_a_condition_fails(tmp_path, monkeypatch):
    """Phase 3's runner returned 0 after printing '6 failed'.

    That is exactly how a CI step goes green on a sweep in which nothing worked.
    Both directions are pinned.
    """
    import scripts.run_phase4_chemlex as runner

    monkeypatch.setattr(runner, "part_jobs",
                        lambda part: [("smoke", [ex.Spec(block="primary")])])
    monkeypatch.setattr(runner, "run_sweep",
                        lambda specs, out, **kw: [{"key": "k", "error": "boom"}])
    monkeypatch.setattr("sys.argv",
                        ["run", "--part", "smoke", "--outdir", str(tmp_path)])
    assert runner.main() == 1

    monkeypatch.setattr(runner, "run_sweep",
                        lambda specs, out, **kw: [{"key": "k"}])
    assert runner.main() == 0


def test_control_specs_carry_their_representation_into_the_key():
    for spec in sw.control_grid():
        assert spec.representation in spec.key
        assert spec.representation != "ecfp4"


def test_the_positive_control_is_run_at_more_than_one_planted_size():
    """A pipeline that finds a huge planted effect has bounded nothing."""
    scales = {s.synthetic_scale for s in sw.positive_grid()}
    assert len(scales) >= 3


def test_the_synthetic_target_contains_the_interaction_it_claims(prepared):
    y, notes = ex.synthetic_target(prepared, rank=3, scale=1.0, seed=1)
    assert notes["planted_rank"] == 3
    assert 0.05 < notes["interaction_sd_fraction"] < 0.95
    assert len(y) == len(prepared.screen.frame)
    again, _ = ex.synthetic_target(prepared, rank=3, scale=1.0, seed=1)
    assert np.allclose(y, again), "the planted target is not reproducible"


# --------------------------------------------------------------------------
# End to end on the fixture, and the committed results.
# --------------------------------------------------------------------------

def test_one_condition_runs_end_to_end_and_reports_every_regime(raw):
    ex._CACHE.clear()
    spec = ex.Spec(block="primary", screen="all", endpoint="yield",
                   partition=0, fold=0, k=3, n_partitions=1,
                   ladder=("additive", "lowrank"), max_epochs=30, n_restarts=1)
    row = ex.run(spec, raw=raw)
    assert "error" not in row
    for bucket in ("test_e1a", "test_e1n"):
        assert np.isfinite(row[f"{bucket}_additive_mse"])
        assert np.isfinite(row[f"{bucket}_primary_incremental"])
        assert np.isfinite(row[f"{bucket}_primary_incremental_blind"])
        assert np.isfinite(row[f"{bucket}_primary_proj_nonadditive_fraction"])
    assert row["per_entity"], "no per-entity records were emitted"
    assert {r["role"] for r in row["per_entity"]} == {"acid", "amine"}


def test_the_baseline_and_the_pair_model_score_the_same_rows(raw):
    """A contrast computed on two different row sets is not a contrast."""
    ex._CACHE.clear()
    spec = ex.Spec(block="primary", screen="all", partition=0, fold=0, k=3,
                   n_partitions=1, ladder=("additive", "lowrank"),
                   max_epochs=20, n_restarts=1)
    row = ex.run(spec, raw=raw)
    for bucket in ("test_e1a", "test_e1n", "test_e2"):
        key = f"{bucket}_additive_n"
        if key in row:
            assert row[key] == row[f"{bucket}_lowrank_n"]


def test_the_binary_endpoint_produces_probabilities(raw):
    ex._CACHE.clear()
    spec = ex.Spec(block="primary", screen="all", endpoint="feasible",
                   partition=0, fold=0, k=3, n_partitions=1,
                   ladder=("additive", "lowrank"), max_epochs=20, n_restarts=1)
    row = ex.run(spec, raw=raw)
    for bucket in ("test_e1a", "test_e1n"):
        brier = row.get(f"{bucket}_additive_brier")
        if brier is not None and np.isfinite(brier):
            assert 0.0 <= brier <= 1.0


@needs_results
def test_the_committed_results_have_no_failed_conditions():
    for path in sorted(RESULTS.glob("*.jsonl")):
        if path.name == "smoke.jsonl":
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        failed = [r for r in rows if "error" in r]
        assert not failed, f"{path.name}: {len(failed)} failed conditions"


@needs_results
def test_the_committed_results_use_the_authoritative_seed():
    rows = [json.loads(l) for l in
            (RESULTS / "primary.jsonl").read_text().splitlines() if l.strip()]
    assert rows and all(r.get("seed") == ex.AUTH_SEED for r in rows)


@needs_results
def test_no_phase_4_row_can_be_mistaken_for_an_earlier_phase(raw):
    """Every Phase 4 row names its block, screen and representation.

    Phase 3's results went unindexed because a basename collided in the results
    index; the cheap half of the fix is making every row self-identifying.
    """
    for path in sorted(RESULTS.glob("*.jsonl")):
        if path.name == "smoke.jsonl":
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            assert r.get("block") and r.get("screen")
            assert r.get("representation")
            assert "key" in r and r["key"].startswith(r["block"])


# --------------------------------------------------------------------------
# The decision rule. Phase 3's audit found four defects in its own version of
# this, none of which had a test; three of them could have changed a verdict.
# --------------------------------------------------------------------------

from intervention_algebra.real_data.chemlex import report as rp  # noqa: E402


def _synthetic_blocks(n_folds: int = 12, incremental: float = 0.05,
                      control: float = 0.0, positive_real: float = 0.20,
                      positive_shuffled: float = 0.0,
                      blind_drop: float = 0.04, projection_gain: float = 0.03,
                      low_stratum: float = 0.04, error: str | None = None):
    """Result frames shaped like a real sweep, with every number dialled in."""
    rng = np.random.default_rng(0)

    def prim():
        rows = []
        for screen in ("hatu", "all"):
            for p in range(3):
                for f in range(4):
                    r = {"key": f"primary/{screen}/p{p}f{f}", "block": "primary",
                         "screen": screen, "encoding": "chemistry",
                         "endpoint": "yield", "representation": "ecfp4",
                         "partition": p, "fold": f, "tag": "",
                         "counts": {b: 100 for b in sp.BUCKETS},
                         "n_conditions": 1 if screen == "hatu" else 7,
                         "n_rows": 8454 if screen == "hatu" else 11669,
                         "n_acids": 272, "n_amines": 230, "seed": ex.AUTH_SEED}
                    if error:
                        r["error"] = error
                    for regime in ("test_e1a", "test_e1n", "test_e2"):
                        for model in ("additive", "lowrank", "condition_expanded",
                                      "condition_expanded_pair", "flexible",
                                      "condition_only"):
                            r[f"{regime}_{model}_mse"] = 0.05
                            r[f"{regime}_{model}_r2"] = 0.4
                            r[f"{regime}_{model}_pearson"] = 0.6
                            r[f"{regime}_{model}_spearman"] = 0.6
                        for cname in ("primary", "robust", "flexible"):
                            v = incremental + rng.normal(scale=0.004)
                            r[f"{regime}_{cname}_incremental"] = v
                            r[f"{regime}_{cname}_incremental_blind"] = v - blind_drop
                            r[f"{regime}_{cname}_blind_drop"] = blind_drop
                            r[f"{regime}_{cname}_proj_incremental"] = v
                            r[f"{regime}_{cname}_proj_incremental_projected"] = \
                                v - projection_gain
                            r[f"{regime}_{cname}_proj_gain_in_nonadditive"] = \
                                projection_gain + rng.normal(scale=0.002)
                            r[f"{regime}_{cname}_proj_nonadditive_fraction"] = 0.05
                            r[f"{regime}_{cname}_proj_corr_nonadditive_with_base_error"] = 0.2
                    rows.append(r)
        return pd.DataFrame(rows)

    def ctrl():
        rows = []
        for rep in ("shuffled_acid", "shuffled_amine", "shuffled_both", "random"):
            for f in range(4):
                r = {"key": f"control/{rep}/f{f}", "block": "control",
                     "screen": "all", "endpoint": "yield", "representation": rep,
                     "partition": 0, "fold": f, "tag": rep}
                for regime in ("test_e1a", "test_e1n", "test_e2"):
                    r[f"{regime}_primary_incremental"] = control + rng.normal(scale=0.002)
                    r[f"{regime}_primary_incremental_blind"] = control
                rows.append(r)
        return pd.DataFrame(rows)

    def pos():
        rows = []
        for scale in (0.25, 0.5, 1.0):
            for rep, val in (("ecfp4", positive_real),
                             ("shuffled_both", positive_shuffled)):
                for f in range(4):
                    r = {"key": f"positive/{scale}/{rep}/f{f}", "block": "positive",
                         "screen": "all", "endpoint": "yield",
                         "representation": rep, "partition": 0, "fold": f,
                         "tag": f"scale{scale:g}_{rep}",
                         "synth_planted_scale": scale,
                         "synth_interaction_sd_fraction": 0.2 * scale}
                    for regime in ("test_e1a", "test_e1n", "test_e2"):
                        r[f"{regime}_primary_incremental"] = val * (scale / 1.0)
                    rows.append(r)
        return pd.DataFrame(rows)

    def per_entity():
        rows = []
        for screen in ("hatu", "all"):
            for bucket, role in (("test_e1a", "acid"), ("test_e1n", "amine")):
                for e in range(N_SYNTH_ENTITIES):
                    sim = 0.1 + 0.8 * (e / N_SYNTH_ENTITIES)
                    stratum_low = sim < rp.SIM_CUTS[role][0]
                    v = (low_stratum if stratum_low else incremental)
                    rows.append({
                        "block": "primary", "screen": screen, "bucket": bucket,
                        "role": role, "entity": e, "contrast": "primary",
                        "endpoint": "yield", "usable": True, "n_rows": 20,
                        "incremental": v + rng.normal(scale=0.004),
                        "incremental_blind": v - blind_drop,
                        "max_similarity_to_train": sim,
                        "fold_key": "p0f0", "smiles": "CC(=O)O"})
        return pd.DataFrame(rows)

    return prim(), ctrl(), pos(), pd.DataFrame(), per_entity()


N_SYNTH_ENTITIES = 60


@pytest.fixture(scope="module")
def tiny_screens():
    """A stub screen with enough entities to exercise the statistics.

    Sixty linear carboxylic acids and sixty linear primary amines, generated
    rather than taken from the fixture: the decision rule's tests are about the
    rule, and the 14-acid fixture would leave a similarity stratum with five
    members and a bootstrap that says nothing. The molecules are real, so
    `congener_families` runs real RDKit rather than a mock.
    """
    acids = tuple("C" * k + "C(=O)O" for k in range(1, N_SYNTH_ENTITIES + 1))
    amines = tuple("C" * k + "CN" for k in range(1, N_SYNTH_ENTITIES + 1))
    frame = pd.DataFrame({"acid": [0], "amine": [0], "cond": [0], "y": [0.5],
                          "feasible": [1], "conversion": [50.0],
                          "acid_smiles": [acids[0]], "amine_smiles": [amines[0]],
                          "protocol": ["p"], "reagent": ["HATU"],
                          "base": ["DIPEA"], "raw_acid": [acids[0]],
                          "raw_amine": [amines[0]]})
    return {"all": ds.Screen(name="all", encoding="chemistry", frame=frame,
                             acids=acids, amines=amines, conditions=("c",),
                             condition_names=("c",), n_raw_rows=1, notes={})}


def test_a_clean_positive_run_is_classified_as_transfer(tiny_screens):
    prim, ctrl, pos, trans, pe = _synthetic_blocks()
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["invalidating_reasons"] == []
    assert v["verdict"] in ("BROAD CHEMICAL ENTITY TRANSFER",
                            "ANALOGUE-ONLY CHEMICAL TRANSFER")


def test_a_leaking_condition_is_a_zero_tolerance_gate(tiny_screens):
    """Not routed into the 10 % failure-fraction gate.

    Phase 3's rule omitted the leakage gate entirely, so a leaking fold would
    have been absorbed by a tolerance nobody would register for leakage.
    """
    prim, ctrl, pos, trans, pe = _synthetic_blocks(
        error="AssertionError: p0f0: acids [3] are test and appear in a training row")
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] == "INCONCLUSIVE"
    assert any("tolerance for this is zero" in r for r in v["invalidating_reasons"])
    assert v["leaking_conditions"]


def test_a_leaking_control_invalidates(tiny_screens):
    prim, ctrl, pos, trans, pe = _synthetic_blocks(control=0.09)
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] == "INCONCLUSIVE"
    assert any("so something is leaking" in r for r in v["invalidating_reasons"])


def test_a_dead_positive_control_invalidates(tiny_screens):
    """A pipeline that cannot find a planted interaction cannot report there is none."""
    prim, ctrl, pos, trans, pe = _synthetic_blocks(positive_real=0.01)
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] == "INCONCLUSIVE"
    assert any("positive control recovers only" in r
               for r in v["invalidating_reasons"])


def test_a_positive_control_that_survives_shuffling_invalidates(tiny_screens):
    prim, ctrl, pos, trans, pe = _synthetic_blocks(positive_shuffled=0.19)
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] == "INCONCLUSIVE"
    assert any("falls only" in r for r in v["invalidating_reasons"])


def test_missing_controls_or_positive_control_invalidate(tiny_screens):
    """Absence is not a neutral omission."""
    prim, ctrl, pos, trans, pe = _synthetic_blocks()
    empty = pd.DataFrame(columns=list(rp._INDEX_COLUMNS))
    v = rp.verdict(prim, empty, pos, trans, pe, tiny_screens)
    assert v["verdict"] == "INCONCLUSIVE"
    v = rp.verdict(prim, ctrl, empty, trans, pe, tiny_screens)
    assert v["verdict"] == "INCONCLUSIVE"


def test_no_pair_advantage_falls_through_to_substrate_only(tiny_screens):
    prim, ctrl, pos, trans, pe = _synthetic_blocks(
        incremental=0.0, blind_drop=0.0, projection_gain=0.0, low_stratum=0.0)
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] in ("SUBSTRATE/CONDITION-ONLY TRANSFER",
                            "NO REUSABLE PAIR STRUCTURE",
                            "TRANSDUCTIVE-ONLY PAIR STRUCTURE")


def test_a_gain_that_projects_away_entirely_is_not_transfer(tiny_screens):
    """The projection diagnostic must be able to veto."""
    prim, ctrl, pos, trans, pe = _synthetic_blocks(projection_gain=0.0)
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] not in ("BROAD CHEMICAL ENTITY TRANSFER",
                                "ANALOGUE-ONLY CHEMICAL TRANSFER")


def test_a_gain_that_survives_blinding_is_not_transfer(tiny_screens):
    """If the pair advantage does not need the unseen reactant, it is not transfer."""
    prim, ctrl, pos, trans, pe = _synthetic_blocks(blind_drop=0.0)
    v = rp.verdict(prim, ctrl, pos, trans, pe, tiny_screens)
    assert v["verdict"] not in ("BROAD CHEMICAL ENTITY TRANSFER",
                                "ANALOGUE-ONLY CHEMICAL TRANSFER")


def test_criterion_g_needs_significance_not_merely_a_positive_mean(tiny_screens):
    """Phase 3's (g) asked only for a mean above zero and passed on +0.016 with a
    CI spanning it. That bar is not repeated."""
    strong = _synthetic_blocks(low_stratum=0.05)
    weak = _synthetic_blocks(low_stratum=0.0005)
    vs = rp.verdict(*strong[:4], strong[4], tiny_screens)
    vw = rp.verdict(*weak[:4], weak[4], tiny_screens)
    g_strong = any(c.get("g_low_similarity_holds")
                   for c in vs["criteria"].values())
    g_weak = any(c.get("g_low_similarity_holds")
                 for c in vw["criteria"].values())
    assert g_strong and not g_weak


def test_the_verdict_is_one_of_the_registered_classifications(tiny_screens):
    allowed = {"BROAD CHEMICAL ENTITY TRANSFER",
               "ANALOGUE-ONLY CHEMICAL TRANSFER",
               "SUBSTRATE/CONDITION-ONLY TRANSFER",
               "TRANSDUCTIVE-ONLY PAIR STRUCTURE",
               "NO REUSABLE PAIR STRUCTURE", "INCONCLUSIVE"}
    for kw in ({}, {"incremental": 0.0}, {"control": 0.09},
               {"positive_real": 0.0}, {"low_stratum": 0.0}):
        blocks = _synthetic_blocks(**kw)
        v = rp.verdict(*blocks[:4], blocks[4], tiny_screens)
        assert v["verdict"] in allowed


def test_the_per_condition_seed_is_stable_across_processes():
    """`hash()` is salted per interpreter unless PYTHONHASHSEED is set.

    Using it to derive a seed made every worker and every run draw a different
    initialisation for the same condition, so the committed results could not be
    regenerated from the committed code. The conclusions never rested on one
    initialisation; the reproducibility claim did.

    Checked by running it in a *fresh interpreter*, because the whole failure is
    invisible inside a single process.
    """
    import subprocess
    import sys

    code = ("from intervention_algebra.real_data.chemlex.experiment import "
            "Spec, hash_seed; "
            "print(hash_seed(Spec(block='primary', screen='hatu', "
            "endpoint='yield', partition=0, fold=0)))")
    seen = set()
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(REPO),
                             env={"PATH": "/usr/bin:/bin",
                                  "PYTHONPATH": str(REPO / "src")})
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"the seed differs between processes: {seen}"


def test_every_condition_in_the_sweep_gets_its_own_seed():
    """A collision would silently share an initialisation between two conditions."""
    seeds = [ex.hash_seed(s) for _, specs in sw.part_jobs("all") for s in specs]
    assert len(seeds) == len(set(seeds)), "two conditions share a seed"


def test_the_same_condition_run_twice_gives_the_same_row(raw):
    """The claim the seed fix exists to make true, end to end on the fixture."""
    ex._CACHE.clear()
    spec = ex.Spec(block="primary", screen="all", partition=0, fold=0, k=3,
                   n_partitions=1, ladder=("additive", "lowrank"),
                   max_epochs=25, n_restarts=1)
    a = ex.run(spec, raw=raw)
    ex._CACHE.clear()
    b = ex.run(spec, raw=raw)
    for key in ("test_e1a_additive_mse", "test_e1a_lowrank_mse",
                "test_e1a_primary_incremental", "test_e1n_primary_incremental"):
        if key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-12), key


def test_the_control_permutation_does_not_change_between_folds():
    """`features.shuffled` says "permuted once, before any split exists".

    It was not: the representation was built from the model-initialisation seed,
    which keys on partition and fold, so the control fingerprints were
    re-permuted on every fold. The control arm then carried permutation variance
    the real arm did not have -- a second difference from the real run, which is
    exactly what the docstring forbids.
    """
    seeds = {ex.representation_seed(
        ex.Spec(block="control", screen="hatu", representation="shuffled_acid",
                partition=p, fold=f, tag="shuffled_acid"))
        for p in range(3) for f in range(5)}
    assert len(seeds) == 1, f"the control permutation changes per fold: {seeds}"


def test_different_control_representations_get_different_permutations():
    seeds = {r: ex.representation_seed(
        ex.Spec(block="control", screen="hatu", representation=r, tag=r))
        for r in ft.REPRESENTATIONS}
    assert len(set(seeds.values())) == len(seeds), seeds


def test_the_model_seed_still_varies_by_fold():
    """The fix must not make every fold share an initialisation."""
    seeds = {ex.hash_seed(ex.Spec(block="primary", screen="hatu",
                                  partition=p, fold=f))
             for p in range(3) for f in range(5)}
    assert len(seeds) == 15


CRITERIA = ("a_mean_above_floor", "b_both_tests_significant",
            "c_majority_favouring", "d_blind_drop_positive",
            "e_gain_survives_projection", "f_robust_contrast_holds",
            "g_low_similarity_holds")


def _criteria(**overrides):
    base = {c: True for c in CRITERIA}
    base.update(overrides)
    return {f"{s}/{r}": dict(base, n_entities=200, mean=0.05)
            for s in ("hatu", "all") for r in ("E1-A", "E1-N", "E2")}


def _dummy_primary():
    return pd.DataFrame({"block": ["primary"], "endpoint": ["yield"],
                         "test_e1a_additive_r2": [0.5]})


@pytest.mark.parametrize("criterion", CRITERIA)
def test_every_registered_criterion_can_change_the_verdict(criterion):
    """No criterion may be decorative.

    The first implementation of `_classify` never read (f) or (g): forcing
    either to True or False in every cell left the output unchanged, so two of
    the seven registered criteria decided nothing while the document reported
    them in a table. This mutation-checks all seven.
    """
    on = rp._classify({}, [], _criteria(), True, _dummy_primary())
    off = rp._classify({}, [], _criteria(**{criterion: False}), True,
                       _dummy_primary())
    assert on != off, (
        f"forcing {criterion}=False changed nothing; the classifier does not "
        f"read it")


def test_all_criteria_satisfied_gives_broad_transfer():
    assert rp._classify({}, [], _criteria(), True,
                        _dummy_primary()) == "BROAD CHEMICAL ENTITY TRANSFER"


def test_g_failing_alone_gives_analogue_only():
    """The registered ANALOGUE-ONLY row: (a)-(f) hold somewhere, (g) fails."""
    c = _criteria()
    c["hatu/E1-N"]["g_low_similarity_holds"] = False
    assert rp._classify({}, [], c, True,
                        _dummy_primary()) == "ANALOGUE-ONLY CHEMICAL TRANSFER"


def test_a_pattern_no_registered_row_describes_is_inconclusive():
    """Screens disagreeing about which criterion fails is not a verdict.

    The registered fallback, and what this phase's committed criteria actually
    produce. An INCONCLUSIVE that names which cell failed which criterion is
    worth more than a classification the table does not license.
    """
    c = _criteria()
    c["hatu/E1-A"]["e_gain_survives_projection"] = False
    c["hatu/E1-N"]["f_robust_contrast_holds"] = False
    c["all/E1-A"]["f_robust_contrast_holds"] = False
    c["all/E1-N"]["f_robust_contrast_holds"] = False
    out = {}
    assert rp._classify(out, [], c, True, _dummy_primary()) == "INCONCLUSIVE"
    assert out["conflict"], "INCONCLUSIVE must say which cell failed what"


def test_no_pair_skill_but_the_additive_model_transfers_is_substrate_only():
    """The registered row order matters and this pins it.

    With no pair skill anywhere, SUBSTRATE/CONDITION-ONLY is checked *before*
    the ceiling rows, because the registration distinguishes "structure predicts
    each reactant's own contribution but not the pair term" from "there is no
    reusable pair structure at all". A classifier that reached the ceiling rows
    first would report the second when the first is true.
    """
    c = _criteria(a_mean_above_floor=False, e_gain_survives_projection=False)
    assert rp._classify({}, [], c, True,
                        _dummy_primary()) == "SUBSTRATE/CONDITION-ONLY TRANSFER"


def test_no_pair_skill_and_no_substrate_transfer_falls_to_the_ceiling_rows():
    c = _criteria(a_mean_above_floor=False, e_gain_survives_projection=False)
    flat = pd.DataFrame({"block": ["primary"], "endpoint": ["yield"],
                         "test_e1a_additive_r2": [-0.1]})
    assert rp._classify({}, [], c, True,
                        flat) == "TRANSDUCTIVE-ONLY PAIR STRUCTURE"
    assert rp._classify({}, [], c, False,
                        flat) == "NO REUSABLE PAIR STRUCTURE"


def test_a_validity_gate_beats_every_criterion():
    assert rp._classify({}, ["a control is leaking"], _criteria(), True,
                        _dummy_primary()) == "INCONCLUSIVE"


def test_the_registered_002_control_ceiling_is_actually_read():
    """It was defined, drawn on a figure, and never evaluated."""
    prim, ctrl, pos, trans, pe = _synthetic_blocks(control=0.03)
    v = rp.verdict(prim, ctrl, pos, trans, pe,
                   {"all": _stub_screen()})
    assert v["soft_threshold_breaches"], (
        "a control at +0.03 breaches the registered +0.02 ceiling and nothing "
        "recorded it")
    assert not v["invalidating_reasons"], (
        "+0.03 is below the +0.05 invalidation threshold and must not invalidate")


def _stub_screen():
    acids = tuple("C" * k + "C(=O)O" for k in range(1, N_SYNTH_ENTITIES + 1))
    amines = tuple("C" * k + "CN" for k in range(1, N_SYNTH_ENTITIES + 1))
    frame = pd.DataFrame({"acid": [0], "amine": [0], "cond": [0], "y": [0.5],
                          "feasible": [1], "conversion": [50.0],
                          "acid_smiles": [acids[0]], "amine_smiles": [amines[0]],
                          "protocol": ["p"], "reagent": ["HATU"],
                          "base": ["DIPEA"], "raw_acid": [acids[0]],
                          "raw_amine": [amines[0]]})
    return ds.Screen(name="all", encoding="chemistry", frame=frame, acids=acids,
                     amines=amines, conditions=("c",), condition_names=("c",),
                     n_raw_rows=1, notes={})


@needs_deposit
def test_tautomers_and_alternative_drawings_share_a_split_group():
    """Two acids were the same compound on opposite sides of a fold.

    Fmoc-Lys(Dde)-OH drawn as the imine and as the enaminone -- same formula,
    *different* standard InChI skeletons, so InChI does not equate them either --
    and valsartan with the tetrazole drawn 1H and 2H. A tautomer is not an unseen
    molecule, and both pairs landed test-vs-train in several authoritative folds.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    screen = ds.load_screen("all")
    fp = ft.fingerprints(screen.acids, "acid")
    group = sp.split_groups(screen.acids, fp.x)
    by_smiles = {s: i for i, s in enumerate(screen.acids)}

    pairs = [
        # Fmoc-Lys(Dde)-OH: imine and enaminone.
        ("CC(=NCCCCC(NC(=O)OCC1c2ccccc2-c2ccccc21)C(=O)O)C1=C(O)CC(C)(C)CC1=O",
         "CC(NCCCC[C@H](NC(=O)OCC1c2ccccc2-c2ccccc21)C(=O)O)=C1C(=O)CC(C)(C)CC1=O"),
        # Valsartan: 1H and 2H tetrazole.
        ("CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(=O)O)C(C)C",
         "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@H](C(=O)O)C(C)C"),
    ]
    for a, b in pairs:
        ia, ib = by_smiles.get(a), by_smiles.get(b)
        assert ia is not None and ib is not None, "the deposit no longer has this pair"
        assert group[ia] == group[ib], (
            f"same compound in different split groups: {group[ia]} vs {group[ib]}")


def test_the_tautomer_relation_runs_after_stereo_is_already_stripped():
    """Which is what makes it usable at all.

    RDKit's TautomerEnumerator discards stereochemistry. Applied to the raw
    structures it would merge stereoisomers under a tautomer's name -- on this
    deposit 7 of the 9 acid merges it produces that way are stereo-flattening
    artefacts. Applied after the stereo-stripped relation has already run, it
    can only add tautomer perception, so a *stereoisomer pair that is not a
    tautomer pair* must still be merged by the stereo relation and not by this
    one -- i.e. removing the tautomer relation must not unmerge it.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")

    # (R)- and unspecified 2-methyl-3-phenylpropanoic acid: stereoisomers, not
    # tautomers. The stereo relation alone must catch them.
    a, b = "O=C(O)C(C)Cc1ccccc1", "O=C(O)[C@@H](C)Cc1ccccc1"
    flat = {Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False)
            for x in (a, b)}
    assert len(flat) == 1, "these are not stereoisomers of one another"
    te = rdMolStandardize.TautomerEnumerator()
    tauts = {Chem.MolToSmiles(te.Canonicalize(Chem.MolFromSmiles(x)))
             for x in (a, b)}
    assert len(tauts) == 1, (
        "TautomerEnumerator on the RAW structures merges these stereoisomers, "
        "which is exactly the artefact the stereo-stripped ordering avoids")


# --------------------------------------------------------------------------
# The guards, run on the ACTUAL authoritative folds.
#
# Every split test above runs on the 14-acid fixture. That is what lets them run
# in CI with no network, and it is not the same thing as checking the folds the
# result was computed on -- while train.py's docstring claimed
# `test_selection_never_sees_a_test_entity` "proves it holds for every
# authoritative fold". These do check them, and skip without the deposit.
# --------------------------------------------------------------------------

@needs_deposit
@pytest.mark.parametrize("screen", ["hatu", "all"])
def test_the_authoritative_folds_pass_every_guard(screen):
    s = ds.load_screen(screen)
    f = s.frame
    fa = ft.fingerprints(s.acids, "acid")
    fn = ft.fingerprints(s.amines, "amine")
    folds = sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                          len(s.acids), len(s.amines), k=ex.K_FOLDS,
                          n_partitions=ex.N_PARTITIONS, seed=ex.AUTH_SEED,
                          acid_group=sp.split_groups(s.acids, fa.x),
                          amine_group=sp.split_groups(s.amines, fn.x))
    assert len(folds) == ex.K_FOLDS * ex.N_PARTITIONS
    a, n = f["acid"].to_numpy(), f["amine"].to_numpy()
    for fold in folds:
        sp.assert_no_entity_leakage(fold, a, n, len(f))


@needs_deposit
@pytest.mark.parametrize("screen", ["hatu", "all"])
def test_every_authoritative_entity_is_a_test_entity_the_registered_number_of_times(screen):
    s = ds.load_screen(screen)
    f = s.frame
    fa = ft.fingerprints(s.acids, "acid")
    fn = ft.fingerprints(s.amines, "amine")
    folds = sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                          len(s.acids), len(s.amines), k=ex.K_FOLDS,
                          n_partitions=ex.N_PARTITIONS, seed=ex.AUTH_SEED,
                          acid_group=sp.split_groups(s.acids, fa.x),
                          amine_group=sp.split_groups(s.amines, fn.x))
    for attr, size in (("test_acids", len(s.acids)),
                       ("test_amines", len(s.amines))):
        turns = np.zeros(size, dtype=int)
        for fold in folds:
            turns[list(getattr(fold, attr))] += 1
        assert set(turns.tolist()) == {ex.N_PARTITIONS}, (
            f"{attr}: entities are test entities {sorted(set(turns.tolist()))} "
            f"times, not exactly {ex.N_PARTITIONS}")


@needs_deposit
def test_how_many_conditions_have_no_training_rows_is_measured_not_assumed():
    """splits.py asserted "Every reaction condition appears in training".

    It is false on the pooled screen: BOP/DIPEA has 5 rows in the whole deposit
    and PyBrOP/DIPEA has 7, so a fold that holds out their handful of entities
    leaves the level with no training rows at all. Nothing guarded it and the
    docstring asserted the opposite.

    The consequence is bounded and this test states it rather than hiding it: a
    condition with no training rows keeps its zero-initialised intercept, which
    is the sensible prior, and it affects the baseline and the pair model
    identically because they share the term. What must not happen is for it to
    affect *many* rows.
    """
    s = ds.load_screen("all")
    f = s.frame
    fa = ft.fingerprints(s.acids, "acid")
    fn = ft.fingerprints(s.amines, "amine")
    folds = sp.make_folds(f["acid"].to_numpy(), f["amine"].to_numpy(),
                          len(s.acids), len(s.amines), k=ex.K_FOLDS,
                          n_partitions=ex.N_PARTITIONS, seed=ex.AUTH_SEED,
                          acid_group=sp.split_groups(s.acids, fa.x),
                          amine_group=sp.split_groups(s.amines, fn.x))
    cond = f["cond"].to_numpy()
    worst_rows = 0
    for fold in folds:
        trained = set(cond[fold.mask("train")].tolist())
        for bucket in sp.PRIMARY_BUCKETS:
            m = fold.mask(bucket)
            if not m.any():
                continue
            affected = int(np.isin(cond[m], list(set(cond[m].tolist()) - trained)).sum())
            worst_rows = max(worst_rows, affected)
    assert worst_rows <= 20, (
        f"{worst_rows} test rows in some fold use a condition with no training "
        f"rows; that is no longer a rounding error")


def test_the_pair_term_measurement_distinguishes_a_live_term_from_a_dead_one():
    """The check that would have caught the flexible comparator.

    Its interaction term is numerically zero at the selected fit -- standard
    deviation 1e-19 to 1e-43 against 0.5 for the low-rank term -- so its
    incremental skill of ~0.000 is an artefact of initialisation and not a
    finding of no benefit. Nothing measured it until an adversarial reviewer
    refitted the models by hand.
    """
    import scripts.measure_pair_terms as mp

    rng = np.random.default_rng(0)
    cfg = md.ModelConfig(
        n_acids=9, n_amines=7, n_conditions=3,
        x_acid=(rng.random((9, 32)) < 0.3).astype(np.float32),
        x_amine=(rng.random((7, 32)) < 0.3).astype(np.float32), rank=4)
    sub = pd.DataFrame({"acid": rng.integers(0, 9, 40),
                        "amine": rng.integers(0, 7, 40),
                        "cond": rng.integers(0, 3, 40)})

    additive = md.build("additive", cfg)
    assert np.isnan(mp.pair_term_sd(additive, sub)), (
        "a rung with no interaction term must report nan, not zero")

    dead = md.build("lowrank", cfg)          # W is zero at initialisation
    assert mp.pair_term_sd(dead, sub) == pytest.approx(0.0, abs=1e-30)

    with torch.no_grad():                     # wake it up
        dead.W.normal_(0.0, 1.0)
    assert mp.pair_term_sd(dead, sub) > 1e-3, (
        "a term with a non-zero bilinear form must measure as alive")
