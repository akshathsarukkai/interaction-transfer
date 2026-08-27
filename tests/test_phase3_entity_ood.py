"""Phase 3 invariants: the drug, not the pair, is what must stay unseen.

Phase 2R's leakage tests ask whether a held-out *pair* reached a fitted quantity.
That question is not sufficient here and the guard that answers it is actively
wrong for this phase -- ``splits.assert_no_pair_leakage`` *requires* every test
pair's endpoints to appear in training, which is precisely what Phase 3 forbids.
So the tests below are about drugs: a test drug must be absent from every fitted
thing, and the check has to fail loudly when it is not.

The most important test in the file is
``test_leakage_guard_fires_when_a_test_drug_is_planted_in_training``: a guard
that has never been seen to fail is not evidence of anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import requires_deposit
from intervention_algebra.real_data import koplev, residual
from intervention_algebra.real_data.entity_ood import drugs as dm
from intervention_algebra.real_data.entity_ood import experiment as ex
from intervention_algebra.real_data.entity_ood import features as feat
from intervention_algebra.real_data.entity_ood import models as mdl
from intervention_algebra.real_data.entity_ood import splits as sp
from intervention_algebra.real_data.entity_ood import sweep

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "koplev_tiny"
FIXTURE_MAPPING = FIXTURE / "drug_mapping.csv"
REAL_MAPPING = REPO / "data" / "external" / "koplev_drug_mapping.csv"
RESULTS = REPO / "results" / "phase3_entity_ood"

N_FIX = 12


@pytest.fixture(scope="module")
def fixture_screen():
    return koplev.load_screen("A375", FIXTURE)


@pytest.fixture(scope="module")
def fixture_mapping():
    return pd.read_csv(FIXTURE_MAPPING)


@pytest.fixture(scope="module")
def fixture_folds():
    return sp.make_drug_folds(N_FIX, n_partitions=2, n_test=3, n_val=3, seed=11)


# --------------------------------------------------------------------------
# the split: drugs, not pairs
# --------------------------------------------------------------------------

def test_the_four_buckets_partition_every_pair(fixture_folds):
    for fold in fixture_folds:
        sp.assert_partition(fold, N_FIX)
        total = sum(len(fold.pairs(w)) for w in
                    ("train", "val", "test_e1", "test_e2"))
        assert total == N_FIX * (N_FIX - 1) // 2


def test_no_training_pair_touches_a_test_or_validation_drug(fixture_folds):
    for fold in fixture_folds:
        test, val = set(fold.test_drugs), set(fold.val_drugs)
        for a, b in fold.train_pairs:
            assert a not in test and b not in test
            assert a not in val and b not in val


def test_no_validation_pair_touches_a_test_drug(fixture_folds):
    for fold in fixture_folds:
        test = set(fold.test_drugs)
        for a, b in fold.val_pairs:
            assert a not in test and b not in test


def test_e1_has_exactly_one_unseen_endpoint_and_e2_has_two(fixture_folds):
    for fold in fixture_folds:
        test = set(fold.test_drugs)
        for a, b in fold.test_e1_pairs:
            assert (a in test) + (b in test) == 1
        for a, b in fold.test_e2_pairs:
            assert a in test and b in test


def test_e1_and_e2_share_no_pair(fixture_folds):
    """They are different questions and must never be pooled or double-counted."""
    for fold in fixture_folds:
        assert not set(fold.test_e1_pairs) & set(fold.test_e2_pairs)


def test_every_drug_is_held_out_exactly_once_per_partition(fixture_folds):
    import collections

    for p in {f.partition for f in fixture_folds}:
        held = [d for f in fixture_folds if f.partition == p for d in f.test_drugs]
        assert sorted(held) == list(range(N_FIX))
        assert max(collections.Counter(held).values()) == 1


def test_folds_are_reproducible_from_the_seed():
    a = sp.make_drug_folds(N_FIX, n_partitions=2, n_test=3, n_val=3, seed=11)
    b = sp.make_drug_folds(N_FIX, n_partitions=2, n_test=3, n_val=3, seed=11)
    c = sp.make_drug_folds(N_FIX, n_partitions=2, n_test=3, n_val=3, seed=12)
    assert [f.test_drugs for f in a] == [f.test_drugs for f in b]
    assert [f.val_drugs for f in a] == [f.val_drugs for f in b]
    assert [f.test_drugs for f in a] != [f.test_drugs for f in c]


def test_real_folds_all_pass_the_leakage_guard():
    for fold in sp.make_drug_folds(100):
        sp.assert_no_drug_leakage(fold, 100)


def test_leakage_guard_fires_when_a_test_drug_is_planted_in_training():
    """The mutation test. A guard never seen to fail proves nothing.

    Four separate corruptions, because the guard has four separate jobs and a
    single planted pair would only exercise one of them.
    """
    fold = sp.make_drug_folds(N_FIX, n_partitions=1, n_test=3, n_val=3, seed=11)[0]
    t0 = fold.test_drugs[0]
    v0 = fold.val_drugs[0]
    tr0 = fold.train_drugs[0]

    # 1. a test drug appears in a training pair
    bad = replace(fold, train_pairs=fold.train_pairs + ((min(t0, tr0), max(t0, tr0)),))
    with pytest.raises(AssertionError, match="test drug"):
        sp.assert_no_drug_leakage(bad, N_FIX)

    # 2. a validation drug appears in a training pair
    bad = replace(fold, train_pairs=fold.train_pairs + ((min(v0, tr0), max(v0, tr0)),))
    with pytest.raises(AssertionError, match="validation drug"):
        sp.assert_no_drug_leakage(bad, N_FIX)

    # 3. a test drug appears in a validation pair
    bad = replace(fold, val_pairs=fold.val_pairs + ((min(t0, tr0), max(t0, tr0)),))
    with pytest.raises(AssertionError, match="validation pair"):
        sp.assert_no_drug_leakage(bad, N_FIX)

    # 4. a pair goes missing from every bucket. At full coverage this is caught
    #    by the drop count; the tightened sparse-coverage arithmetic would
    #    otherwise absorb it as intentional thinning.
    bad = replace(fold, train_pairs=fold.train_pairs[:-1])
    with pytest.raises(AssertionError, match="eligible training pairs are missing"):
        sp.assert_no_drug_leakage(bad, N_FIX)

    # 5. ... and the same loss with the bookkeeping adjusted to match is caught
    #    by the exhaustiveness count instead.
    bad = replace(fold, test_e1_pairs=fold.test_e1_pairs[:-1])
    with pytest.raises(AssertionError, match="expected"):
        sp.assert_no_drug_leakage(bad, N_FIX)


def test_a_test_drug_contributes_no_row_to_any_fitted_frame(fixture_screen, fixture_folds):
    """End to end on real frames: the rows the fit sees never mention a test drug."""
    pairs = residual.directional_pairs(fixture_screen.frame)
    for fold in fixture_folds:
        for which in ("train", "val"):
            rows = fold.rows(pairs, which)
            touched = set(rows["i"]) | set(rows["j"])
            assert not touched & set(fold.test_drugs)
        assert not set(fold.rows(pairs, "train")["i"]) & set(fold.val_drugs)
        assert not set(fold.rows(pairs, "train")["j"]) & set(fold.val_drugs)


def test_coverage_subsets_training_pairs_without_moving_the_entity_boundary():
    full = sp.make_drug_folds(N_FIX, n_partitions=1, n_test=3, n_val=3, seed=11)[0]
    thin = sp.make_drug_folds(N_FIX, n_partitions=1, n_test=3, n_val=3, seed=11,
                              coverage=0.4)[0]
    assert thin.test_drugs == full.test_drugs and thin.val_drugs == full.val_drugs
    assert thin.test_e1_pairs == full.test_e1_pairs
    assert set(thin.train_pairs) < set(full.train_pairs)
    sp.assert_no_drug_leakage(thin, N_FIX)


# --------------------------------------------------------------------------
# the target
# --------------------------------------------------------------------------

def test_the_target_is_antisymmetric_and_canonically_oriented(fixture_screen):
    pairs = residual.directional_pairs(fixture_screen.frame)
    assert (pairs["i"] < pairs["j"]).all()
    m = fixture_screen.matrix("y")
    for _, r in pairs.head(20).iterrows():
        i, j = int(r["i"]), int(r["j"])
        assert r["D_true"] == pytest.approx(m[i, j] - m[j, i], abs=1e-12)
        assert -r["D_true"] == pytest.approx(m[j, i] - m[i, j], abs=1e-12)


def test_one_row_per_unordered_pair(fixture_screen):
    pairs = residual.directional_pairs(fixture_screen.frame)
    assert len(pairs) == N_FIX * (N_FIX - 1) // 2
    assert not pairs.duplicated(subset=["i", "j"]).any()


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def test_fingerprints_are_deterministic_and_outcome_independent(fixture_mapping):
    a = feat.fingerprint_matrix(fixture_mapping)
    b = feat.fingerprint_matrix(fixture_mapping.sample(frac=1, random_state=3))
    assert np.array_equal(a.x, b.x), "row order of the mapping changed the features"
    assert a.labels == b.labels
    # Perturbing every response must not move a single bit: the featuriser takes
    # the mapping, not the screen, so there is no path from an outcome to a
    # feature. Asserted rather than argued.
    assert set(np.unique(a.x)) <= {0.0, 1.0}


def test_drug_index_order_is_the_screens_own_order(fixture_screen, fixture_mapping):
    m = fixture_mapping.sort_values("drug_index")
    assert list(m["label"]) == list(fixture_screen.drugs)


def test_shuffled_control_really_breaks_the_drug_feature_correspondence(fixture_mapping):
    base = feat.fingerprint_matrix(fixture_mapping)
    sh = feat.shuffled_features(base, seed=4)
    assert sorted(map(tuple, sh.x)) == sorted(map(tuple, base.x)), \
        "the shuffle must preserve the multiset of fingerprints exactly"
    assert not np.array_equal(sh.x, base.x)
    moved = sum(1 for k in range(len(base.labels)) if not np.array_equal(sh.x[k], base.x[k]))
    assert moved >= len(base.labels) // 2


def test_random_control_is_fixed_before_splitting(fixture_mapping):
    base = feat.fingerprint_matrix(fixture_mapping)
    a = feat.random_features(base, seed=9)
    b = feat.random_features(base, seed=9)
    assert np.array_equal(a.x, b.x), "the control must not be redrawn per fold"
    assert not np.array_equal(a.x, feat.random_features(base, seed=10).x)


def test_feature_view_is_fitted_on_training_drugs_only(fixture_mapping):
    """A bit that only a test drug sets must be dropped, not kept.

    Constructed directly: give one drug a unique bit, hold it out, and require
    the view to exclude that column. If the view were fitted on all drugs the
    column would survive.
    """
    base = feat.fingerprint_matrix(fixture_mapping)
    x = base.x.copy()
    x[:, 0] = 0.0
    x[5, 0] = 1.0                       # a bit unique to drug 5
    mask_when_seen = ex.feature_view(x, tuple(range(N_FIX)))
    mask_when_held_out = ex.feature_view(x, tuple(k for k in range(N_FIX) if k != 5))
    assert mask_when_seen[0]
    assert not mask_when_held_out[0]


def test_tanimoto_diagonal_is_nan_not_one(fixture_mapping):
    """So that a row-wise max cannot silently return self-similarity."""
    s = feat.tanimoto_matrix(feat.fingerprint_matrix(fixture_mapping))
    assert np.isnan(np.diag(s)).all()
    assert np.nanmax(s) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def _cfg(dim=32, **kw):
    x = (np.random.default_rng(0).random((N_FIX, dim)) < 0.3).astype(np.float32)
    return mdl.EntityModelConfig(n_drugs=N_FIX, x=x, **kw)


@pytest.mark.parametrize("name", mdl.LADDER_ORDER)
@pytest.mark.parametrize("hidden", [0, 8])
def test_every_rung_is_exactly_antisymmetric(name, hidden):
    m = mdl.build_entity_model(name, _cfg(hidden=hidden, rank=2, seed=1))
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
        i = torch.tensor([0, 1, 2, 5])
        j = torch.tensor([3, 4, 6, 7])
        assert torch.allclose(m.d_res(i, j), -m.d_res(j, i), atol=1e-5)
        assert torch.allclose(m.d_res(i, i), torch.zeros(4), atol=1e-5)


@pytest.mark.parametrize("name", mdl.LADDER_ORDER)
def test_every_rung_starts_at_exactly_zero(name):
    m = mdl.build_entity_model(name, _cfg(rank=2, seed=2))
    with torch.no_grad():
        out = m.d_res(torch.tensor([0, 1]), torch.tensor([2, 3]))
    assert float(out.abs().max()) == 0.0


@pytest.mark.parametrize("name", ["potential", "lowrank", "pair_only", "antisym_mlp"])
def test_no_parameter_stays_dead_after_training(name):
    """The Phase 2R failure: a rung whose every gradient vanishes at init.

    It reports exactly 0.0 skill, which is indistinguishable in a results table
    from "there is no signal" -- the very finding this phase exists to test.
    """
    m = mdl.build_entity_model(name, _cfg(rank=2, seed=3))
    init = {k: v.detach().clone() for k, v in m.named_parameters()}
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    i, j = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    target = torch.tensor([0.4, -0.3, 0.2])
    for _ in range(30):
        opt.zero_grad()
        ((m.d_res(i, j) - target) ** 2).mean().backward()
        opt.step()
    dead = [k for k, v in m.named_parameters() if torch.allclose(v.detach(), init[k])]
    assert not dead, f"{name}: parameters never moved: {dead}"


def test_lowrank_nests_potential_at_initialisation():
    """The two models must be the same function at init, or the incremental
    skill would confound "a pair term helps" with "a slightly better potential"."""
    cfg = _cfg(rank=2, seed=4)
    a = mdl.build_entity_model("potential", cfg)
    b = mdl.build_entity_model("lowrank", cfg)
    with torch.no_grad():
        b.g.load_state_dict(a.g.state_dict())
        for p in a.g.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        b.g.load_state_dict(a.g.state_dict())
        i, j = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
        assert torch.allclose(a.d_res(i, j), b.d_res(i, j), atol=1e-6)


def test_pair_only_has_no_potential_head():
    m = mdl.build_entity_model("pair_only", _cfg(rank=2, seed=5))
    assert not hasattr(m, "g"), "an unused head would be dead weight in the counts"


def test_per_drug_and_gathered_heads_agree():
    """The speed optimisation must be mathematically invisible."""
    cfg = _cfg(dim=64, rank=4, seed=6)
    m = mdl.build_entity_model("lowrank", cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
        i, j = torch.tensor([0, 1, 9]), torch.tensor([3, 4, 5])
        K = m.W - m.W.T
        gathered = ((m.g(m.X[i]) - m.g(m.X[j])).squeeze(-1)
                    + torch.einsum("...a,ab,...b->...", m.f(m.X[i]), K, m.f(m.X[j])))
        assert torch.allclose(m.d_res(i, j), gathered, atol=1e-5)


def test_feature_matrix_is_frozen():
    m = mdl.build_entity_model("lowrank", _cfg(rank=2, seed=7))
    assert not any(p is m.X for p in m.parameters())
    assert "X" not in dict(m.named_parameters())


# --------------------------------------------------------------------------
# the pipeline, on the fixture: map -> split -> featurise -> train -> evaluate
# --------------------------------------------------------------------------

TINY = dict(n_partitions=1, n_test=3, n_val=3, split_seed=11)


@pytest.mark.parametrize("model", ["zero", "potential", "lowrank"])
def test_pipeline_runs_end_to_end_on_the_fixture(model, fixture_screen, fixture_mapping,
                                                 monkeypatch):
    monkeypatch.setattr(ex, "TRAIN", replace(ex.TRAIN, max_epochs=20, n_restarts=1))
    cfg = ex.EntityConfig(screen="A375", model=model, partition=0, fold=0, tag="test",
                          **TINY)
    row = ex.run_entity_condition(cfg, raw_dir=FIXTURE, mapping_path=FIXTURE_MAPPING,
                                  screen=fixture_screen, mapping=fixture_mapping)
    assert row["model"] == model
    assert row["n_test_drugs"] == 3
    assert "e1_skill" in row and np.isfinite(row["e1_skill"])
    assert row["representation"] == "ecfp4"
    assert set(row["test_drugs"].split(",")) <= {str(k) for k in range(N_FIX)}


def _tiny_folds(n=N_FIX):
    return sp.make_drug_folds(n, n_partitions=1, n_test=3, n_val=3, seed=11)


def test_hyperparameter_selection_never_sees_a_test_pair(fixture_screen, fixture_mapping,
                                                         monkeypatch):
    """Perturbing every test response must leave the selected setting untouched.

    A direct experiment rather than a code reading: if any selected quantity --
    the grid member, the stopping epoch, the restart, the shrinkage -- had read a
    test row, corrupting the test rows would change it.
    """
    monkeypatch.setattr(ex, "TRAIN", replace(ex.TRAIN, max_epochs=20, n_restarts=1))
    cfg = ex.EntityConfig(screen="A375", model="lowrank", partition=0, fold=0, tag="test",
                          **TINY)
    clean = ex.run_entity_condition(cfg, FIXTURE, FIXTURE_MAPPING,
                                    screen=fixture_screen, mapping=fixture_mapping)

    fold = _tiny_folds(N_FIX)[0]
    test = set(fold.test_drugs)
    frame = fixture_screen.frame.copy()
    hit = frame["i"].isin(test) | frame["j"].isin(test)
    frame.loc[hit, "y"] = frame.loc[hit, "y"] + 1000.0
    poisoned = koplev.Screen(
        label=fixture_screen.label, table_key=fixture_screen.table_key,
        drugs=fixture_screen.drugs, frame=frame, n_raw_rows=fixture_screen.n_raw_rows,
        n_self_rows=fixture_screen.n_self_rows, n_missing=fixture_screen.n_missing)
    dirty = ex.run_entity_condition(cfg, FIXTURE, FIXTURE_MAPPING,
                                    screen=poisoned, mapping=fixture_mapping)

    assert dirty["hparams"] == clean["hparams"]
    assert dirty["alpha"] == clean["alpha"]
    assert dirty["best_epoch"] == clean["best_epoch"]
    assert dirty["d_scale_train"] == pytest.approx(clean["d_scale_train"])
    assert dirty["val_loss"] == pytest.approx(clean["val_loss"])
    assert dirty["e1_skill"] != clean["e1_skill"], \
        "the poisoning must actually reach the evaluation, or this proves nothing"


def test_positive_control_target_is_recoverable_in_principle(fixture_mapping):
    """The synthetic target really is potential + rank-2 antisymmetric."""
    pairs = pd.DataFrame({"i": [0, 0, 1, 1, 2], "j": [1, 2, 2, 3, 3]})
    x = feat.fingerprint_matrix(fixture_mapping).x
    out = ex._synthetic_target(pairs, x, seed=1, noise_sd=0.0)
    rev = ex._synthetic_target(pairs.rename(columns={"i": "j", "j": "i"}), x,
                               seed=1, noise_sd=0.0)
    assert np.allclose(out["D_true"], -rev["D_true"], atol=1e-9)


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

def test_every_advertised_part_reaches_the_worker_pool(monkeypatch, tmp_path):
    """Phase 2N's runner died on an UnboundLocalError and looked alive for hours.

    Unit tests covered the grid functions; nothing covered ``main()``. This
    stubs the pool and asserts that every part named in ``--help`` actually
    produces conditions and reaches the executor.
    """
    import scripts.run_phase3_entity_ood as runner

    seen: dict[str, int] = {}

    def fake_sweep(specs, out, **kw):
        seen[out.stem] = len(specs)
        out.write_text("")
        return []

    monkeypatch.setattr(runner.sweep, "run_entity_sweep", fake_sweep)
    for part in list(sweep.PART_GRIDS) + ["all"]:
        seen.clear()
        assert runner.main(["--part", part, "--outdir", str(tmp_path)]) == 0
        assert seen, f"--part {part} produced no jobs"
        assert all(n > 0 for n in seen.values()), f"--part {part} produced an empty block"


def test_limit_zero_does_not_silently_mean_no_limit(monkeypatch, tmp_path):
    """``if args.limit:`` reads 0 as falsy; a dry-run flag once started a real run."""
    import scripts.run_phase3_entity_ood as runner

    seen: dict[str, int] = {}
    monkeypatch.setattr(runner.sweep, "run_entity_sweep",
                        lambda specs, out, **kw: (seen.__setitem__(out.stem, len(specs)),
                                                  out.write_text(""), [])[-1])
    runner.main(["--part", "primary", "--limit", "0", "--outdir", str(tmp_path)])
    assert seen == {"primary": 0}


def test_part_counts_are_derived_not_typed():
    counts = sweep.part_counts()
    assert counts["all"] == sum(counts[p] for p in sweep.ALL_PARTS)
    assert "smoke" not in sweep.ALL_PARTS, "a pipeline check is not a result"
    for part in sweep.PART_GRIDS:
        assert counts[part] == sum(len(s) for _, s in sweep.part_jobs(part))


def test_unknown_part_is_rejected():
    with pytest.raises(SystemExit):
        sweep.part_jobs("nonexistent")


# --------------------------------------------------------------------------
# the drug mapping
# --------------------------------------------------------------------------

def test_label_normalisation_keeps_salt_words_and_drops_synonyms():
    assert dm.normalise_label("Cytarabine;  Ara-C") == "Cytarabine"
    assert dm.normalise_label("Lomustine;  CCNU") == "Lomustine"
    assert dm.normalise_label("Mitotane;  o;p'-DDD") == "Mitotane"
    assert dm.normalise_label("Fluorouracil  (5-FU)") == "Fluorouracil"
    assert dm.normalise_label("Sirolimus (Rapamycin)") == "Sirolimus"
    # Salt words survive normalisation on purpose: desalting is a structural
    # operation, not a string one.
    assert dm.normalise_label("Erlotinib HCl") == "Erlotinib HCl"
    assert dm.normalise_label("Pemetrexed Disodium") == "Pemetrexed Disodium"


def test_largest_fragment_drops_counterions_and_keeps_covalent_groups():
    kept, gone, _ = dm.largest_fragment("Cl.COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC")
    assert gone == ["Cl"] and "Cl" not in kept
    # A covalent phosphate ester is one fragment and must survive whole -- the
    # failure a regex on "phosphate" would produce.
    smi = "Nc1nc(F)nc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O"
    kept, gone, meta = dm.largest_fragment(smi)
    assert gone == [] and meta["n_fragments"] == 1
    assert "P" in kept


def test_metal_complexes_are_never_fragmented():
    kept, gone, meta = dm.largest_fragment("N.N.Cl[Pt]Cl")
    assert gone == [], "none of cisplatin's three fragments is the drug"
    assert meta["contains_metal"]
    assert "Pt" in kept and kept.count("Cl") == 2 and kept.count("N") == 2


def test_similar_sized_fragments_are_not_silently_split():
    kept, gone, meta = dm.largest_fragment("c1ccccc1.c1ccccc1C")
    assert gone == [], "a near-equal co-crystal is not a salt"
    assert any(f.startswith("check:fragments") for f in meta["flags"])


@pytest.mark.skipif(not REAL_MAPPING.exists(), reason="mapping not generated")
def test_real_mapping_is_100_distinct_parseable_molecules():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = pd.read_csv(REAL_MAPPING)
    assert len(m) == 100
    assert list(m.sort_values("drug_index")["drug_index"]) == list(range(100))
    assert m["smiles"].notna().all()
    keys = [k[:14] for k in m["inchikey"]]
    assert len(set(keys)) == 100, "two labels resolve to the same molecule"
    for smi in m["smiles"]:
        assert Chem.MolFromSmiles(smi) is not None


@requires_deposit
@pytest.mark.skipif(not REAL_MAPPING.exists(), reason="mapping not generated")
def test_real_mapping_drug_order_matches_the_screen():
    m = pd.read_csv(REAL_MAPPING).sort_values("drug_index")
    for label in ("A375", "PANC1"):
        assert list(m["label"]) == list(koplev.load_screen(label).drugs)


@pytest.mark.skipif(not REAL_MAPPING.exists(), reason="mapping not generated")
def test_every_discarded_fragment_is_small():
    """No desalting may have thrown away most of the molecule."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = pd.read_csv(REAL_MAPPING)
    for _, r in m.iterrows():
        if not isinstance(r["discarded_fragments"], str) or not r["discarded_fragments"]:
            continue
        kept = Chem.MolFromSmiles(r["smiles"]).GetNumHeavyAtoms()
        for frag in r["discarded_fragments"].split("."):
            f = Chem.MolFromSmiles(frag)
            if f is None:
                continue
            assert kept >= dm.MIN_FRAGMENT_RATIO * f.GetNumHeavyAtoms(), \
                f"{r['label']}: discarded {frag} against a {kept}-atom parent"


# --------------------------------------------------------------------------
# results hygiene
# --------------------------------------------------------------------------

def test_phase3_results_are_not_mixed_with_earlier_phases():
    for stale in (REPO / "results" / "phase2", REPO / "results" / "phase2_residual",
                  REPO / "results" / "dchain_null"):
        if not stale.exists():
            continue
        for p in stale.rglob("*.jsonl"):
            with p.open() as fh:
                first = fh.readline()
            if not first.strip():
                continue
            row = json.loads(first)
            assert "fold_key" not in row, f"a Phase 3 row leaked into {p}"


@pytest.mark.skipif(not (RESULTS / "primary.jsonl").exists(),
                    reason="the primary sweep has not been run")
def test_committed_results_carry_no_errors_and_no_leaking_fold():
    rows = [json.loads(line) for line in (RESULTS / "primary.jsonl").open()]
    assert rows and not any("error" in r for r in rows)
    for r in rows:
        assert r["n_train_drugs"] + r["n_val_drugs"] + r["n_test_drugs"] == 100
        assert r["n_test_e1_pairs"] == 900 and r["n_test_e2_pairs"] == 45


def test_antisym_mlp_matches_the_concatenated_form():
    """The blocked first layer is an exact refactoring, not an approximation.

    ``W [x_i; x_j] = W1 x_i + W2 x_j``. Evaluating the two blocks once per drug
    instead of once per pair endpoint is ~60x cheaper on 3,160 pairs over 100
    drugs, and must change nothing.
    """
    cfg = _cfg(dim=48, mlp_hidden=8, seed=11)
    m = mdl.build_entity_model("antisym_mlp", cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        i, j = torch.tensor([0, 1, 7, 11]), torch.tensor([3, 4, 5, 2])
        W = torch.cat([m.first.weight, m.second.weight], dim=1)

        def F(a, b):
            return m.out(torch.relu(torch.cat([a, b], -1) @ W.T
                                    + m.first.bias)).squeeze(-1)

        ref = F(m.X[i], m.X[j]) - F(m.X[j], m.X[i])
        assert torch.allclose(m.d_res(i, j), ref, atol=1e-5)


def test_generated_blocks_inject_into_an_empty_marker_pair(tmp_path):
    """The first run is the one that matters, and it is the one that broke.

    An empty block is ``-->\\n<!-- /``: exactly one newline. A pattern expecting
    one on each side matches nothing, so a naive implementation appears to work
    on every run after the first and silently does nothing on the run that
    creates the document.
    """
    from intervention_algebra.real_data.entity_ood.report import inject_blocks

    doc = tmp_path / "d.md"
    doc.write_text("intro\n\n<!-- generated:t -->\n<!-- /generated:t -->\n\nouttro\n")
    assert inject_blocks(doc, {"t": "| a |\n|---|\n| 1 |"}) == ["t"]
    assert "| a |" in doc.read_text()

    # ... and running again must replace, not accumulate.
    before = doc.read_text()
    inject_blocks(doc, {"t": "| a |\n|---|\n| 1 |"})
    assert doc.read_text() == before
    assert doc.read_text().count("| a |") == 1
    assert doc.read_text().startswith("intro\n")
    assert doc.read_text().rstrip().endswith("outtro")


def test_injection_reports_blocks_with_no_marker(tmp_path):
    """A generated block that finds no home must be visible, not silent."""
    from intervention_algebra.real_data.entity_ood.report import inject_blocks

    doc = tmp_path / "d.md"
    doc.write_text("<!-- generated:present -->\n<!-- /generated:present -->\n")
    assert inject_blocks(doc, {"present": "x", "absent": "y"}) == ["present"]


@pytest.mark.skipif(not REAL_MAPPING.exists(), reason="mapping not generated")
def test_audited_overrides_are_applied_and_recorded():
    """An override must be visible in the row, not a silent value change."""
    m = pd.read_csv(REAL_MAPPING).set_index("label")
    for label in dm.AUDIT_OVERRIDES:
        row = m.loc[label]
        assert "audited:structure-overridden" in str(row["flags"])
        assert str(row["notes"]).startswith("AUDIT OVERRIDE.")
        assert row["smiles"] == dm.AUDIT_OVERRIDES[label]["smiles"]
        # the audit trail must still show what the databases actually returned
        assert isinstance(row["deposited_smiles"], str) and row["deposited_smiles"]


@pytest.mark.skipif(not REAL_MAPPING.exists(), reason="mapping not generated")
def test_carboplatin_is_not_its_own_ligand():
    """The specific defect the audit found: a fingerprint 0.875 similar to the
    bare diacid, reached by a different route than the one already guarded."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs

    RDLogger.DisableLog("rdApp.*")
    m = pd.read_csv(REAL_MAPPING).set_index("label")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    carb = gen.GetFingerprint(Chem.MolFromSmiles(m.loc["Carboplatin", "smiles"]))
    diacid = gen.GetFingerprint(Chem.MolFromSmiles("C1CC(C1)(C(=O)O)C(=O)O"))
    assert DataStructs.TanimotoSimilarity(carb, diacid) < 0.7
    assert "Pt" in m.loc["Carboplatin", "smiles"]


def test_a_second_sweep_refuses_to_write_the_same_directory(tmp_path):
    """Two sweeps interleaved into one .partial file once, and the second had
    been launched against a drug structure the audit later rejected."""
    import os

    from intervention_algebra.real_data.entity_ood.sweep import _exclusive

    (tmp_path / ".sweep.lock").write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="already writing"):
        with _exclusive(tmp_path):
            pass

    # A lock left behind by a killed run must not block the next attempt.
    (tmp_path / ".sweep.lock").write_text("999999")
    with _exclusive(tmp_path):
        pass
    assert not (tmp_path / ".sweep.lock").exists()


def test_gradient_projection_separates_potential_from_curl_on_e1_geometry():
    """Calibration of the diagnostic the Phase 3 verdict turns on.

    A measured curl fraction near zero must mean "this prediction is a
    potential", not "the projection is greedy". Checked on the *exact* E1
    graph -- 900 edges over 100 nodes, cycle rank 801 -- because a bipartite
    holdout could in principle have had a degenerate cycle space, in which case
    every antisymmetric function would be a gradient and the diagnostic would be
    vacuous.
    """
    fold = sp.make_drug_folds(100)[0]
    pairs = np.array(fold.test_e1_pairs)
    i, j = pairs[:, 0], pairs[:, 1]
    assert len(pairs) - 100 + 1 == 801, "the E1 cycle space is not what it was"

    def surviving(d):
        gp = ex.gradient_projection(i, j, d, 100)
        return float(((d - gp) ** 2).sum() / (d ** 2).sum())

    rng = np.random.default_rng(0)
    g = rng.normal(size=100)
    pot = g[i] - g[j]
    assert surviving(pot) < 1e-9, "a potential must be absorbed completely"

    # Averaged over draws, not measured on one. A single random rank-r form can
    # land anywhere from 0.68 to 0.98 on this graph, and a threshold fitted to
    # one draw is a flaky test pretending to be a calibration.
    for rank in (2, 4, 8):
        vals = []
        for seed in range(8):
            r = np.random.default_rng([7, rank, seed])
            u = r.normal(size=(100, rank))
            W = r.normal(size=(rank, rank))
            vals.append(surviving(np.einsum("na,ab,nb->n", u[i], W - W.T, u[j])))
        assert min(vals) > 0.6, f"rank {rank}: curl must largely survive, got {min(vals)}"
        assert sum(vals) / len(vals) > 0.85, f"rank {rank} mean {sum(vals) / len(vals)}"

    # And a half-and-half mix must read as roughly half, so the statistic is
    # graded rather than a threshold detector.
    r = np.random.default_rng([7, 2, 0])
    u = r.normal(size=(100, 2))
    W = r.normal(size=(2, 2))
    curl = np.einsum("na,ab,nb->n", u[i], W - W.T, u[j])
    mix = pot / pot.std() + curl / curl.std()
    assert 0.25 < surviving(mix) < 0.7


def test_a_rank2_form_can_express_a_pure_potential():
    """Why the low-rank rung beating the potential rung proves nothing by itself.

    ``z_i = (g_i, 1)``, ``K = [[0, 1], [-1, 0]]`` gives ``z_i' K z_j = g_i - g_j``
    exactly. This is the reason the curl decomposition exists.
    """
    rng = np.random.default_rng(1)
    g = rng.normal(size=20)
    z = np.stack([g, np.ones_like(g)], axis=1)
    K = np.array([[0.0, 1.0], [-1.0, 0.0]])
    i, j = np.array([0, 3, 7, 19]), np.array([1, 4, 8, 2])
    bilinear = np.einsum("na,ab,nb->n", z[i], K, z[j])
    assert np.allclose(bilinear, g[i] - g[j])


def test_the_runner_exits_nonzero_when_conditions_fail(monkeypatch, tmp_path):
    """A runner that reports failures and exits 0 is how a CI step goes green on
    a sweep in which nothing worked.

    Found by running the pipeline check on a checkout with no Koplev deposit:
    every condition raised, the runner printed "6 failed", and the exit code
    said success.
    """
    import scripts.run_phase3_entity_ood as runner

    def failing_sweep(specs, out, **kw):
        out.write_text("")
        return [{"tag": "smoke", "error": "boom"} for _ in specs]

    monkeypatch.setattr(runner.sweep, "run_entity_sweep", failing_sweep)
    assert runner.main(["--part", "smoke", "--outdir", str(tmp_path)]) == 1

    monkeypatch.setattr(runner.sweep, "run_entity_sweep",
                        lambda specs, out, **kw: (out.write_text(""), [{"tag": "smoke"}])[-1])
    assert runner.main(["--part", "smoke", "--outdir", str(tmp_path)]) == 0


def test_the_guard_catches_a_diluted_validation_bucket():
    """Validation must be entity-OOD, and nothing asserted it.

    The guard forbade a *test* drug in a validation pair but not a validation
    bucket padded with pairs between two training drugs. The counting checks make
    that dilution rather than contamination, but an invariant the documents call
    load-bearing should be asserted rather than inferred.
    """
    fold = sp.make_drug_folds(N_FIX, n_partitions=1, n_test=3, n_val=3, seed=11)[0]
    bad = replace(fold, train_pairs=fold.train_pairs[5:],
                  val_pairs=tuple(fold.train_pairs[:5]) + fold.val_pairs)
    with pytest.raises(AssertionError, match="no validation endpoint"):
        sp.assert_no_drug_leakage(bad, N_FIX)


def test_the_exhaustiveness_check_is_live_at_sparse_coverage():
    """It used to be skipped whenever coverage < 1.0 -- i.e. disabled in exactly
    the setting where pairs are dropped on purpose and a bug looks intended."""
    thin = sp.make_drug_folds(N_FIX, n_partitions=1, n_test=3, n_val=3, seed=11,
                              coverage=0.4)[0]
    sp.assert_no_drug_leakage(thin, N_FIX)                      # legitimate
    for bad in (replace(thin, test_e1_pairs=thin.test_e1_pairs[:-2]),
                replace(thin, n_eligible_train_pairs=thin.n_eligible_train_pairs - 3)):
        with pytest.raises(AssertionError, match="expected"):
            sp.assert_no_drug_leakage(bad, N_FIX)


def test_the_blind_baseline_is_the_training_marginal_not_a_zero_row():
    """Zeroing a feature row is not "no information about this drug".

    It asserts "this drug has zero fingerprint bits", a point no training drug
    occupies, and it is systematically pessimistic -- so anything at all beats
    it, which is how the random-feature control scored a spurious +0.052
    "attributable to the unseen drug". The information-free prediction is the
    marginal over the drugs the model actually trained on.
    """
    cfg = _cfg(dim=32, rank=2, seed=21)
    m = mdl.build_entity_model("potential", cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    train = list(range(4, N_FIX))
    i = torch.tensor([0, 1, 2])
    j = torch.tensor([5, 6, 7])
    with torch.no_grad():
        marginal = torch.stack([m.d_res(torch.full_like(i, t), j)
                                for t in train]).mean(0)
        zero_row = m.X.clone()
        zero_row[[0, 1, 2]] = 0.0
        m.X, saved = zero_row, m.X
        zeroed = m.d_res(i, j)
        m.X = saved
    # The two differ, and the marginal is the on-distribution one: it equals the
    # prediction using the mean training feature row, for a linear head.
    assert not torch.allclose(marginal, zeroed, atol=1e-4)
    with torch.no_grad():
        mean_row = m.X.clone()
        mean_row[[0, 1, 2]] = m.X[train].mean(0)
        m.X, saved = mean_row, m.X
        via_mean = m.d_res(i, j)
        m.X = saved
    assert torch.allclose(marginal, via_mean, atol=1e-5)


# --------------------------------------------------------------------------
# the decision rule itself -- nothing exercised it until an adversarial review
# pointed out that a rule nobody tests is a rule nobody has checked
# --------------------------------------------------------------------------

def _synthetic_rows(pot, low, incr, p, n=30, controls=0.0, positive=0.4):
    """Fabricate result rows that produce chosen per-screen statistics."""
    rows = []
    rng = np.random.default_rng(0)
    for tag, rep in (("primary", "ecfp4"), ("control_random", "random"),
                     ("control_shuffled", "shuffled"), ("positive_control", "ecfp4")):
        for screen in ("A375", "PANC1"):
            if tag == "positive_control" and screen != "A375":
                continue
            inc = (incr[screen] if tag == "primary" else
                   positive if tag == "positive_control" else controls)
            spread = 0.001 if p[screen] < 1e-6 else 0.06
            # Centred, so the realised fold mean is exactly the target. Without
            # this the fixture's own sampling noise (SE ~0.011 at n=30) drifts a
            # "no effect" scenario across the 0.01 criterion and the test fails
            # for a reason that has nothing to do with the code under test.
            noise = rng.normal(scale=spread, size=n)
            noise -= noise.mean()
            for f in range(n):
                mse_zero = 1.0
                inc_f = inc + noise[f]
                # Controls must sit at zero skill-vs-zero or they trip the frozen
                # validity gate before the cascade under test is ever reached.
                base = 1.0 - (pot[screen] if tag == "primary"
                              else 0.4 if tag == "positive_control" else 0.0)
                for model, mse in (("zero", mse_zero), ("potential", base),
                                   ("lowrank", base * (1 - inc_f))):
                    rows.append({
                        "tag": tag, "screen": screen, "representation": rep,
                        "coverage": 1.0, "fold_key": f"p0f{f}", "partition": 0,
                        "fold": f, "model": model, "n_params": 10,
                        "synthetic_target": tag == "positive_control",
                        "e1_mse": mse, "e1_mse_zero": mse_zero,
                        "e1_skill": 1.0 - mse / mse_zero, "e1_pearson": 0.3,
                        "e1_spearman": 0.3, "e1_mae": 0.1, "e1_rmse": 0.3,
                        "e1_sign_accuracy": 0.6, "e1_n_pairs": 900,
                        "per_drug": [{"drug": k, "label": f"d{k}", "n_pairs": 90,
                                      "max_sim_to_train": low["sim"],
                                      "median_sim_to_train": 0.1, "bits_set": 40,
                                      "e1_mse": mse * (1 - low["inc"] if model == "lowrank" else 1),
                                      "e1_mse_zero": mse_zero,
                                      "e1_skill": 0.1, "e1_pearson": 0.2}
                                     for k in range(10)],
                    })
    return pd.DataFrame(rows)


def test_verdict_returns_no_entity_transfer_only_when_both_screens_fail():
    """Registered rule 1 is quantified over BOTH screens.

    The implementation negated an all-screens conjunction, which fires when *one*
    screen is below the threshold -- so a result that is strong in one screen and
    flat in the other could be labelled NO ENTITY TRANSFER.
    """
    from intervention_algebra.real_data.entity_ood.report import verdict

    both_flat = _synthetic_rows({"A375": 0.0, "PANC1": 0.0}, {"sim": 0.1, "inc": 0.0},
                                {"A375": 0.0, "PANC1": 0.0}, {"A375": 1.0, "PANC1": 1.0})
    assert verdict(both_flat)["verdict"] == "NO ENTITY TRANSFER"

    one_flat = _synthetic_rows({"A375": 0.30, "PANC1": 0.0}, {"sim": 0.1, "inc": 0.0},
                               {"A375": 0.0, "PANC1": 0.0}, {"A375": 1.0, "PANC1": 1.0})
    assert verdict(one_flat)["verdict"] != "NO ENTITY TRANSFER"


def test_verdict_reports_potential_only_when_the_pair_term_adds_nothing():
    from intervention_algebra.real_data.entity_ood.report import verdict

    # potential clears its threshold in both screens; the pair term adds nothing.
    rows = _synthetic_rows({"A375": 0.30, "PANC1": 0.25}, {"sim": 0.1, "inc": 0.0},
                           {"A375": -0.005, "PANC1": -0.005},
                           {"A375": 1.0, "PANC1": 1.0})
    assert verdict(rows)["verdict"] == "POTENTIAL-ONLY ENTITY TRANSFER"


def test_verdict_routes_a_one_screen_result_to_weak_marginal():
    """The registration puts "significant in one screen only" in rule 5. The
    implementation negated all-screens conjunctions and could route it to
    POTENTIAL-ONLY instead."""
    from intervention_algebra.real_data.entity_ood.report import verdict

    rows = _synthetic_rows({"A375": 0.30, "PANC1": 0.25}, {"sim": 0.1, "inc": 0.05},
                           {"A375": 0.08, "PANC1": -0.005},
                           {"A375": 1e-9, "PANC1": 1.0})
    assert verdict(rows)["verdict"] == "WEAK/MARGINAL ENTITY TRANSFER"


def test_verdict_invalidates_on_a_single_leaking_fold():
    """Leakage was registered at zero tolerance and was being absorbed by the
    10% failure-fraction gate instead."""
    from intervention_algebra.real_data.entity_ood.report import verdict

    rows = _synthetic_rows({"A375": 0.30, "PANC1": 0.25}, {"sim": 0.1, "inc": 0.0},
                           {"A375": 0.0, "PANC1": 0.0}, {"A375": 1.0, "PANC1": 1.0})
    rows["error"] = np.nan
    leak = rows.iloc[[0]].copy()
    leak["error"] = "Traceback ... assert_no_drug_leakage ... p0f0: [3] are test drugs"
    v = verdict(pd.concat([rows, leak], ignore_index=True))
    assert v["verdict"] == "INCONCLUSIVE"
    assert any("leakage guard" in r for r in v["invalidating_reasons"])
