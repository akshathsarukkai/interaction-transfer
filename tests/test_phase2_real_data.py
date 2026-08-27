"""Phase 2 invariants: the ways the real-data pipeline could be quietly wrong.

Scoped deliberately. These cover the load-bearing risks -- pair canonicalisation,
split leakage, the directional target's sign, the architectural constraints, and
determinism -- and not every function in the package. Everything here runs on a
12-drug fixture in ``tests/fixtures/koplev_tiny`` and needs no download, so CI
stays fast and never depends on an external host.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.evaluate import directional_frame
from intervention_algebra.real_data.models import (FAMILIES, RealModelConfig,
                                                   build_model, pair_param_count)
from intervention_algebra.real_data.splits import (assert_no_pair_leakage,
                                                   canonical_pair, degrees,
                                                   make_coverage_splits)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "koplev_tiny"
#: The fixture has 66 unordered pairs, so the grid is scaled to it; the
#: experiment's own grid is in ``real_data.sweep.COVERAGES``.
COVERAGES = (0.30, 0.50, 0.70)


@pytest.fixture(scope="module")
def screen():
    return koplev.load_screen("A375", FIXTURE)


@pytest.fixture(scope="module")
def splits(screen):
    return make_coverage_splits(screen.frame, screen.n_drugs, COVERAGES,
                                split_seed=0, min_train_degree=1,
                                min_eligible_test_pairs=5)


# --------------------------------------------------------------- ingestion
def test_self_pairs_are_removed_and_counted(screen):
    assert screen.n_raw_rows == 144
    assert screen.n_self_rows == 12
    assert len(screen.frame) == 132
    assert not (screen.frame["i"] == screen.frame["j"]).any()


def test_pair_ids_are_unordered_and_canonical(screen):
    for (a, b), i, j in zip(screen.frame["pair"], screen.frame["i"], screen.frame["j"]):
        assert a < b
        assert {a, b} == {i, j}
        assert canonical_pair(i, j) == (a, b) == canonical_pair(j, i)


def test_canonical_pair_rejects_self_pairs():
    with pytest.raises(ValueError):
        canonical_pair(3, 3)


def test_both_directions_present_for_every_pair(screen):
    ordered = set(zip(screen.frame["i"], screen.frame["j"]))
    for a, b in set(screen.frame["pair"]):
        assert (a, b) in ordered and (b, a) in ordered


def test_drug_name_normalisation_does_not_merge_distinct_drugs(screen):
    # The fixture carries the deposit's whitespace oddities on purpose.
    assert "Drugene; Ara-X" in screen.drugs
    assert "drugene 07" in screen.drugs
    assert len(set(screen.drugs)) == len(screen.drugs) == 12


def test_ingestion_is_deterministic(screen):
    again = koplev.load_screen("A375", FIXTURE)
    assert again.drugs == screen.drugs
    pd.testing.assert_frame_equal(again.frame, screen.frame)


def test_duplicate_ordered_rows_are_rejected(tmp_path):
    """One value per ordered combination is an assumption, so it is checked.

    The deposit has no replicate rows; if a future release adds them, silently
    keeping the first would bias the fit rather than fail. This is the guard.
    """
    for name in ("Data Table 1.csv", "Data Table 2.csv", "Data Table 4.csv"):
        (tmp_path / name).write_text((FIXTURE / name).read_text())
    p = tmp_path / "Data Table 1.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[2]]) + "\n")
    with pytest.raises(RuntimeError, match="duplicated ordered pairs"):
        koplev.load_screen("A375", tmp_path)


def test_audit_reports_a_complete_pair_matrix(screen):
    a = koplev.audit_screen(screen)
    assert a["n_unordered_pairs"] == 66
    assert a["n_ordered_pairs"] == 132
    assert a["n_pairs_with_both_directions"] == 66
    assert a["pair_matrix_completeness"] == 1.0
    assert np.isclose(a["var_symmetric_share"] + a["var_antisymmetric_share"], 1.0,
                      atol=1e-9)


def test_verify_raw_detects_the_duplicated_table(tmp_path):
    """The finding that decides which deposit tables are usable, as a test."""
    for name in ("Data Table 1.csv", "Data Table 2.csv", "Data Table 4.csv"):
        (tmp_path / name).write_text((FIXTURE / name).read_text())
    for name, src in (("Data Table 3.csv", "Data Table 1.csv"),
                      ("Data Table 5.csv", "Data Table 4.csv")):
        (tmp_path / name).write_text((FIXTURE / src).read_text())
    t3 = pd.read_csv(tmp_path / "Data Table 1.csv")
    pd.DataFrame({
        "pretreatment_treatment": t3["first_compound"] + "_" + t3["second_compound"],
        "AsPC1": t3["synergy_measure"],
    }).to_csv(tmp_path / "Data Table 3.csv", index=False)
    report = koplev.verify_raw(tmp_path)
    assert report["table4_vs_table2"]["identical"] is True
    assert report["table1_vs_table2"]["identical"] is False


# ------------------------------------------------------------------ splits
def test_no_pair_leakage_at_every_coverage(screen, splits):
    for split in splits.values():
        assert_no_pair_leakage(split, screen.frame)


def test_both_directions_of_a_pair_move_together(screen, splits):
    for split in splits.values():
        for which in ("train", "val", "test"):
            rows = split.rows(screen.frame, which)
            counts = Counter(rows["pair"])
            # Every pair present in a part is present with BOTH of its rows.
            assert set(counts.values()) == {2}, (which, counts)


def test_train_val_test_partition_is_disjoint_and_complete(screen, splits):
    all_pairs = set(screen.frame["pair"])
    for split in splits.values():
        tr, va, te = (set(split.train_pairs), set(split.val_pairs),
                      set(split.test_pairs))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te <= all_pairs


def test_test_pool_is_identical_across_coverages(splits):
    pools = {tuple(s.test_pairs) for s in splits.values()}
    assert len(pools) == 1, "the coverage curve must be scored on one fixed pool"


def test_training_sets_are_nested_across_coverages(splits):
    ordered = sorted(splits)
    for lo, hi in zip(ordered, ordered[1:]):
        a = set(splits[lo].train_pairs) | set(splits[lo].val_pairs)
        b = set(splits[hi].train_pairs) | set(splits[hi].val_pairs)
        assert a <= b, f"coverage {lo} pool is not a subset of {hi}"


def test_every_test_endpoint_is_trained_on_elsewhere(screen, splits):
    for split in splits.values():
        rows = split.rows(screen.frame, "train")
        trained = set(rows["i"]) | set(rows["j"])
        assert {d for p in split.test_pairs for d in p} <= trained


def test_leakage_assertion_actually_fires(screen, splits):
    """A leakage check that cannot fail is not a leakage check."""
    split = splits[0.50]
    poisoned = replace(split, train_pairs=split.train_pairs + (split.test_pairs[0],))
    with pytest.raises(AssertionError, match="in both train and test"):
        assert_no_pair_leakage(poisoned, screen.frame)


def test_eligibility_ignores_outcomes(screen):
    """Test-pair selection must not be able to prefer easy pairs."""
    shuffled = screen.frame.copy()
    rng = np.random.default_rng(0)
    shuffled["y"] = rng.permutation(shuffled["y"].to_numpy())
    kw = dict(min_train_degree=1, min_eligible_test_pairs=5)
    a = make_coverage_splits(screen.frame, screen.n_drugs, COVERAGES, 0, **kw)
    b = make_coverage_splits(shuffled, screen.n_drugs, COVERAGES, 0, **kw)
    for c in COVERAGES:
        assert a[c].test_pairs == b[c].test_pairs
        assert a[c].train_pairs == b[c].train_pairs


def test_degrees_counts_each_endpoint_once():
    assert list(degrees(3, [(0, 1), (0, 2)])) == [2, 1, 1]


# ---------------------------------------------------------- the target D
def test_directional_target_flips_sign_under_reversal(screen, splits):
    split = splits[0.50]
    rows = split.rows(screen.frame, "test")
    pred = rows["y"].to_numpy()                    # perfect predictor
    d = directional_frame(rows, pred)
    swapped = rows.rename(columns={"i": "j", "j": "i"})
    d_rev = directional_frame(swapped, pred)
    key = {(a, b): v for a, b, v in zip(d["i"], d["j"], d["D_true"])}
    rev = {(a, b): v for a, b, v in zip(d_rev["i"], d_rev["j"], d_rev["D_true"])}
    assert key.keys() == rev.keys()
    for k in key:
        assert np.isclose(key[k], -rev[k])
    # D_true and D_pred must be built the SAME way round. They are two adjacent
    # lines in evaluate.py and nothing above catches an inconsistent flip --
    # verified by mutation: reversing D_true alone leaves the whole suite green.
    # An inconsistent flip turns ordering accuracy a into 1-a and flips
    # D_pearson for both order-expressing families, i.e. it would report the
    # Phase 2 conclusion exactly reversed, with no other symptom.
    assert np.allclose(d["D_pred"], d["D_true"]), (
        "pred is the true value here, so the two differences must agree; "
        "they are constructed independently and can silently disagree in sign")


def test_directional_join_is_one_to_one(screen, splits):
    rows = splits[0.50].rows(screen.frame, "test")
    d = directional_frame(rows, rows["y"].to_numpy())
    assert len(d) == len(rows) // 2
    assert (d["i"] < d["j"]).all()


def test_directional_join_refuses_a_half_pair(screen, splits):
    rows = splits[0.50].rows(screen.frame, "test")
    with pytest.raises(AssertionError, match="one-to-one"):
        directional_frame(rows.iloc[1:], rows["y"].to_numpy()[1:])


# ------------------------------------------------------------------ models
@pytest.fixture(scope="module")
def cfg():
    return RealModelConfig(n_drugs=12, emb_dim=6, pair_hidden=16, seed=0)


def _ij():
    i = torch.tensor([0, 1, 2, 3, 4])
    j = torch.tensor([1, 2, 3, 4, 0])
    return i, j


def test_structured_M_is_exactly_symmetric(cfg):
    m = build_model("structured", cfg)
    i, j = _ij()
    with torch.no_grad():
        assert torch.allclose(m.M(i, j), m.M(j, i), atol=0)


def test_structured_A_is_exactly_antisymmetric(cfg):
    m = build_model("structured", cfg)
    i, j = _ij()
    with torch.no_grad():
        assert torch.allclose(m.A(i, j), -m.A(j, i), atol=0)


def test_structured_invariants_survive_training(cfg):
    """Tripwire, not evidence.

    ``M = (f + f∘swap)/2`` and ``A = (f - f∘swap)/2`` are symmetric and
    antisymmetric for *any* ``f`` -- that is the identity decomposition, and it
    is exactly how ``implied_A`` is defined for every family. These assertions
    therefore say nothing about the structured family in particular; they catch
    a future edit that breaks the symmetrisation. The line that carries content
    is the last one: that the two halves reconstruct the prediction.
    """
    m = build_model("structured", cfg)
    for p in m.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    i, j = _ij()
    with torch.no_grad():
        assert torch.allclose(m.M(i, j), m.M(j, i), atol=0)
        assert torch.allclose(m.A(i, j), -m.A(j, i), atol=0)
        assert torch.allclose(m(i, j), m.M(i, j) + m.A(i, j), atol=1e-6)


def test_order_insensitive_is_order_blind(cfg):
    m = build_model("order_insensitive", cfg)
    for p in m.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    i, j = _ij()
    with torch.no_grad():
        assert torch.allclose(m(i, j), m(j, i), atol=0)
        assert torch.all(m.implied_A(i, j) == 0)


def test_unrestricted_is_not_order_blind(cfg):
    """A baseline that cannot express order would make the comparison vacuous."""
    m = build_model("unrestricted", cfg)
    for p in m.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    i, j = _ij()
    with torch.no_grad():
        assert not torch.allclose(m(i, j), m(j, i))


def test_pair_capacity_is_matched_at_the_production_config():
    """The tolerance that matters is the one at the width the experiment runs."""
    prod = RealModelConfig(n_drugs=100, emb_dim=16, pair_hidden=48)
    ref = build_model("structured", prod).n_pair_params()
    for fam in ("order_insensitive", "unrestricted"):
        got = build_model(fam, prod).n_pair_params()
        assert abs(got - ref) / ref < 0.01, (fam, got, ref)
        # The baselines must never be the *smaller* of the two by a margin that
        # could be mistaken for the effect.
        assert got > 0.99 * ref, (fam, got, ref)


def test_pair_capacity_is_matched_at_fixture_scale(cfg):
    """At small widths the integer search cannot land as close; bound it anyway."""
    ref = build_model("structured", cfg).n_pair_params()
    for fam in ("order_insensitive", "unrestricted"):
        got = build_model(fam, cfg).n_pair_params()
        assert abs(got - ref) / ref < 0.02, (fam, got, ref)


def test_every_family_shares_the_first_order_term(cfg):
    counts = {f: build_model(f, cfg).n_first_order_params() for f in FAMILIES}
    assert len(set(counts.values())) == 1, counts
    assert counts["structured"] == 2 * cfg.n_drugs + 1


def test_pair_heads_start_at_zero(cfg):
    """Every family begins at the two-way additive fit, not at O(1) noise."""
    i, j = _ij()
    for fam in ("order_insensitive", "unrestricted", "structured"):
        with torch.no_grad():
            assert torch.all(build_model(fam, cfg).pair_term(i, j) == 0), fam


def test_additive_has_no_pair_term(cfg):
    m = build_model("additive", cfg)
    i, j = _ij()
    with torch.no_grad():
        assert torch.all(m.pair_term(i, j) == 0)
    assert m.n_pair_params() == 0
    assert pair_param_count("additive", cfg, 99) == 0


def test_build_model_rejects_unknown_family(cfg):
    with pytest.raises(ValueError, match="unknown family"):
        build_model("algebra", cfg)


def test_model_init_is_deterministic_in_the_seed(cfg):
    a = build_model("structured", cfg)
    b = build_model("structured", cfg)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    c = build_model("structured", replace(cfg, seed=1))
    assert not all(torch.equal(pa, pc)
                   for pa, pc in zip(a.parameters(), c.parameters()))


def test_connectivity_floor_is_delivered_not_just_declared(screen):
    """``min_train_degree`` must hold against the TRAINING pairs, not the pool.

    Eligibility was once computed over the sparsest coverage's whole pool, with
    its validation pairs still in it. At the sparsest coverage validation takes
    15% of the pool, so an endpoint with three pool pairs could reach training
    with one -- a stated floor of 3 delivering 1, exactly where the data is
    thinnest and the claim most fragile.
    """
    floor = 2
    splits = make_coverage_splits(screen.frame, screen.n_drugs, COVERAGES,
                                  split_seed=0, min_train_degree=floor,
                                  min_eligible_test_pairs=5)
    for cov, split in splits.items():
        deg = split.train_degree
        for a, b in split.test_pairs:
            assert deg[a] >= floor and deg[b] >= floor, (cov, a, b, deg[a], deg[b])


def test_val_pairs_never_enter_the_training_rows(screen, splits):
    """Early stopping must not be scored on data the model was fitted to."""
    for split in splits.values():
        train = split.rows(screen.frame, "train")
        val_pairs = set(split.val_pairs)
        assert not (set(train["pair"]) & val_pairs)


def test_structured_contains_unrestricted(cfg):
    """At equal head width the structured family CONTAINS the unrestricted one.

    The docs said the opposite for a while. Setting ``F_M = F_A = G`` makes the
    structured pair term ``0.5(G_ij+G_ji) + 0.5(G_ij-G_ji) == G_ij`` exactly, so
    "the baseline is a strict superset, therefore a win cannot be capacity" was
    not just unsupported but backwards. Pinned here so the claim cannot drift
    back into the write-up.
    """
    from intervention_algebra.real_data.models import Structured, Unrestricted
    u, st = Unrestricted(cfg), Structured(cfg)
    st.emb.load_state_dict(u.emb.state_dict())
    st.first_a.load_state_dict(u.first_a.state_dict())
    st.first_b.load_state_dict(u.first_b.state_dict())
    with torch.no_grad():
        st.bias.copy_(u.bias)
    st.F_M.load_state_dict(u.G.state_dict())
    st.F_A.load_state_dict(u.G.state_dict())
    i, j = _ij()
    with torch.no_grad():
        assert torch.allclose(st(i, j), u(i, j), atol=1e-6)


def test_collapse_detector_fires_on_a_dead_head(cfg):
    """A diagnostic that reports health for a dead head is worse than none.

    ``implied_A`` includes the first-order term ``((a_i-b_i)-(a_j-b_j))/2``,
    which every family carries; with ``F_A`` pinned to identically zero it still
    reads a perfectly healthy value. Collapse is therefore measured on the pair
    head alone.
    """
    m = build_model("structured", cfg)
    with torch.no_grad():
        m.first_a.weight.normal_(0.0, 0.5)
        m.first_b.weight.normal_(0.0, 0.5)
        for p in m.F_M.parameters():
            p.normal_(0.0, 0.3)
        for p in m.F_A.parameters():
            p.zero_()
    i, j = _ij()
    with torch.no_grad():
        assert m.pair_head_A(i, j).abs().max() == 0.0
        assert m.implied_A(i, j).abs().mean() > 0.05, (
            "the total antisymmetric readout should still look healthy -- that "
            "is precisely why it cannot be the detector")


def test_order_insensitive_head_is_symmetric_not_absent(cfg):
    """Its head is real and doing work; only its antisymmetric part is zero."""
    m = build_model("order_insensitive", cfg)
    with torch.no_grad():
        for p in m.F_M.parameters():
            p.normal_(0.0, 0.3)
    i, j = _ij()
    with torch.no_grad():
        assert m.pair_head_A(i, j).abs().max() == 0.0
        assert m.pair_term(i, j).abs().mean() > 0.0


def test_verify_raw_guard_actually_fires(tmp_path):
    """The Table-4 duplication guard must fail when the duplication stops holding.

    Verified by mutation that this was unpinned: deleting the guard in
    ``koplev.verify_raw`` left the entire suite green, while
    ``docs/phase2_dataset.md`` claimed a test pinned it. The recorded sha256
    digests protect against a revised *deposit*, but nothing re-hashes files
    already on disk, so a locally edited or half-written Table 4 would pass.
    """
    for name in ("Data Table 1.csv", "Data Table 2.csv", "Data Table 4.csv"):
        (tmp_path / name).write_text((FIXTURE / name).read_text())
    for name, src in (("Data Table 3.csv", "Data Table 1.csv"),
                      ("Data Table 5.csv", "Data Table 4.csv")):
        (tmp_path / name).write_text((FIXTURE / src).read_text())
    t3 = pd.read_csv(tmp_path / "Data Table 1.csv")
    pd.DataFrame({
        "pretreatment_treatment": t3["first_compound"] + "_" + t3["second_compound"],
        "AsPC1": t3["synergy_measure"],
    }).to_csv(tmp_path / "Data Table 3.csv", index=False)

    t4 = pd.read_csv(tmp_path / "Data Table 4.csv")
    t4.loc[0, "synergy_measure"] = float(t4.loc[0, "synergy_measure"]) + 1.0
    t4.to_csv(tmp_path / "Data Table 4.csv", index=False)
    with pytest.raises(RuntimeError, match="no longer byte-identical"):
        koplev.verify_raw(tmp_path)


def test_parity_grid_overrides_the_selected_learning_rate():
    """The parity block must actually vary the rate it claims to vary.

    ``apply_hparams`` writes the *selected* rate onto the config, and the parity
    grid then overrides it. If that override were dropped -- or applied before
    ``apply_hparams`` rather than after -- every arm would silently collapse onto
    the shipped setting, the block would report "re-tuning changes nothing", and
    the tuning artifact it exists to detect would be invisible.
    """
    from intervention_algebra.real_data.sweep import PARITY_LRS, parity_grid
    hp = {"structured": {"lr": 0.01, "weight_decay": 0.01, "emb_dim": 16},
          "unrestricted": {"lr": 0.03, "weight_decay": 0.01, "emb_dim": 16}}
    specs = parity_grid(cells=(("A375", 0.10),), split_seeds=(0,), hparams=hp)
    assert len(specs) == 2 * len(PARITY_LRS)
    for fam in ("structured", "unrestricted"):
        rates = sorted(c.train.lr for c in specs if c.family == fam)
        assert rates == sorted(PARITY_LRS), (fam, rates)
    # Non-rate hyperparameters must still come from the selection.
    assert {c.emb_dim for c in specs} == {16}
    assert {c.train.weight_decay for c in specs} == {0.01}


def test_main_grid_pool_coverages_do_not_move_the_evaluation_pool():
    """A subset of rungs must not silently redefine the split it is scored on.

    ``Phase2Config.coverages`` fixes the evaluation pool, so passing a shortened
    ``coverages`` moves the goalposts rather than running fewer conditions. The
    init-variance block runs two rungs and must stay comparable to the headline,
    which is what ``pool_coverages`` is for. This was a live bug: the committed
    init-variance rows were scored on 5,940 test rows against the headline's
    1,696-2,006, which understated initialisation noise by roughly 4x.
    """
    from intervention_algebra.real_data.sweep import COVERAGES, main_grid
    naive = main_grid(screens=("A375",), coverages=(0.10, 0.40),
                      split_seeds=(0,), families=("structured",), hparams={})
    pooled = main_grid(screens=("A375",), coverages=(0.10, 0.40),
                       pool_coverages=COVERAGES, split_seeds=(0,),
                       families=("structured",), hparams={})
    assert {c.coverages for c in naive} == {(0.10, 0.40)}
    assert {c.coverages for c in pooled} == {COVERAGES}
    assert len(naive) == len(pooled) == 2
