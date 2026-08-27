"""Tests for the report's cell-selection logic.

The report is the last place a mistake can hide, because by then the numbers
look authoritative. The specific failure these tests exist to prevent already
happened once: the headline was built from ``tag == "main"``, so the command
advertised as "this regenerates the reported results" silently regenerated the
five-seed pilot instead of the seventeen-seed replication the conclusions rest
on, and the two are indistinguishable by eye.

So the properties asserted here are the ones that make that class of error
impossible rather than merely unlikely: seeds pool iff the configuration is
identical, duplicates raise instead of halving a p-value, and a control
condition cannot enter the headline whatever it is tagged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from intervention_algebra import analysis, phase1, report

RESULTS = Path(__file__).resolve().parents[1] / "results" / "phase1.jsonl"


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    if not RESULTS.exists():
        pytest.skip("authoritative results file not present")
    return analysis.to_frame(analysis.load(RESULTS))


@pytest.fixture(scope="module")
def a_run() -> dict:
    """One real run dict, with its nested config, to mutate in key tests."""
    if not RESULTS.exists():
        pytest.skip("authoritative results file not present")
    with RESULTS.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("tag") == "main" and row.get("family") == "algebra":
                return row
    pytest.skip("no main/algebra run in results")


# ------------------------------------------------------------- condition keys


def test_condition_key_ignores_seed_and_tag(a_run):
    """Replicates of one experiment must land in one cell."""
    other = copy.deepcopy(a_run)
    other["seed"] = a_run["seed"] + 999
    for section in ("system", "split", "model", "train"):
        other["config"][section]["seed"] += 999
    other["tag"] = "some_replication_batch_added_later"
    other["config"]["tag"] = other["tag"]
    assert analysis.condition_key(other) == analysis.condition_key(a_run)


@pytest.mark.parametrize("section,field,value", [
    ("split", "pair_coverage", 0.7),      # the omission that pooled coverages
    ("train", "patience", 600),           # early-stopping vs fixed-length protocol
    ("train", "max_epochs", 2000),
    ("train", "lr", 0.01),
    ("system", "s_scale", 0.6),           # the weak-interaction control
    ("system", "sparsity_mode", "random"),
    ("system", "regime", "symmetric"),
    ("system", "simultaneity_defect", 0.8),
    ("system", "observation_map", "tanh"),
])
def test_condition_key_separates_different_experiments(a_run, section, field, value):
    other = copy.deepcopy(a_run)
    assert other["config"][section][field] != value, "test would be vacuous"
    other["config"][section][field] = value
    assert analysis.condition_key(other) != analysis.condition_key(a_run)


def test_condition_key_uses_the_width_actually_trained(a_run):
    """Capacity matching rewrites the configured width; the key must follow it."""
    other = copy.deepcopy(a_run)
    other["pair_hidden"] = a_run["pair_hidden"] * 2
    assert analysis.condition_key(other) != analysis.condition_key(a_run)

    # ...and must ignore the pre-resolution value the JSONL happens to store.
    same = copy.deepcopy(a_run)
    same["config"]["model"]["pair_hidden"] = 999
    assert analysis.condition_key(same) == analysis.condition_key(a_run)


def test_condition_key_ignores_fields_resolved_from_the_system(a_run):
    """`n_interventions`/`out_dim` are stored unresolved and mean nothing here."""
    same = copy.deepcopy(a_run)
    same["config"]["model"]["n_interventions"] = 12345
    same["config"]["model"]["out_dim"] = 7
    assert analysis.condition_key(same) == analysis.condition_key(a_run)


def test_spec_key_matches_the_runs_that_spec_produced(a_run):
    """The bridge the whole report stands on: declaration == observation.

    `spec_condition_key` resolves a phase1 spec the way the worker would; if it
    ever drifts from what the worker actually ran, the report would select
    nothing and every cell would silently go empty.
    """
    spec = phase1.headline_spec(a_run["family"], a_run["pair_coverage"])
    assert analysis.spec_condition_key(spec) == analysis.condition_key(a_run)


# ----------------------------------------------------------------- selection


def test_powered_cells_pool_replications_and_are_balanced(real_df):
    powered = report.powered_frame(real_df)
    assert not powered.empty

    counts = powered.groupby(["pair_coverage", "family"])["seed"].nunique()
    for cov, g in counts.groupby(level=0):
        assert g.nunique() == 1, (
            f"ragged n at coverage {cov}: {g.to_dict()} -- a paired comparison "
            f"across families with different seed sets is not paired")
        if cov <= 0.05:
            continue   # deliberately unpowered: below the identifiability floor
        assert g.iloc[0] > len(phase1.REPORT_SEEDS), (
            f"coverage {cov} has only {g.iloc[0]} seeds; the report has fallen "
            f"back to the five-seed pilot")

    # and it genuinely pooled across batches rather than reading one tag
    tags = set()
    for t in powered["contributing_tags"]:
        tags.update(t.split(","))
    assert len(tags) > 1


def test_powered_cells_are_seed_complete_and_unique(real_df):
    powered = report.powered_frame(real_df)
    dup = powered.duplicated(subset=["pair_coverage", "family", "seed"])
    assert not dup.any()


def test_powered_cells_exclude_every_control_condition(real_df):
    """No control may enter the headline, whatever it happens to be tagged."""
    powered = report.powered_frame(real_df)
    assert set(powered["regime"]) == {"both"}
    assert set(powered["sparsity_mode"]) == {"latent"}
    assert set(powered["observation_map"]) == {"identity"}
    assert set(powered["patience"]) == {phase1.TRAIN_BASE["patience"]}
    assert set(powered["max_epochs"]) == {phase1.TRAIN_BASE["max_epochs"]}
    if "simultaneity_defect" in powered.columns:
        assert set(powered["simultaneity_defect"].dropna()) <= {0.0}
    # the ceiling model is a reference, not a competitor
    assert "wellspecified" not in set(powered["family"])
    # exactly one pair width per family: mixing capacity arms would confound it
    assert (powered.groupby("family")["pair_hidden"].nunique() == 1).all()


def test_duplicate_replicates_raise_rather_than_halving_p_values(real_df):
    """A directory glob produces this shape; it must not be survivable."""
    doubled = pd.concat([real_df, real_df], ignore_index=True)
    with pytest.raises(analysis.DuplicateRunsError):
        report.powered_frame(doubled)


def test_superseded_protocol_cannot_enter_a_powered_cell(real_df):
    """A run identical except for early stopping is a different experiment."""
    superseded = Path(__file__).resolve().parents[1] / \
        "results" / "SUPERSEDED_main_confounded.jsonl"
    if not superseded.exists():
        pytest.skip("superseded evidence file not present")
    mixed = analysis.to_frame(analysis.load(RESULTS) + analysis.load(superseded))
    powered = report.powered_frame(mixed)
    assert set(powered["patience"]) == {phase1.TRAIN_BASE["patience"]}


# ---------------------------------------------------------------- statistics


def test_powered_skill_is_defined_for_every_seed(real_df):
    """The tag-keyed merge NaN'd 12 of 17 seeds; the coverage-keyed one must not."""
    powered = report.add_powered_skill(report.powered_frame(real_df))
    per_cell = (powered.groupby(["pair_coverage", "family"])
                ["skill_vs_additive_family"].apply(lambda s: s.notna().sum()))
    expected = powered.groupby(["pair_coverage", "family"])["seed"].nunique()
    assert (per_cell == expected).all(), \
        f"skill is NaN for some seeds:\n{pd.concat([per_cell, expected], axis=1)}"
    # the additive family is its own reference, so its skill is identically zero
    add = powered.loc[powered["family"] == "additive", "skill_vs_additive_family"]
    assert (add.abs() < 1e-12).all()


def test_seeds_favouring_column_follows_the_metric_direction():
    """Higher-is-better metrics must not report the lower-is-better count."""
    rows = []
    for seed in range(5):
        rows.append(dict(family="algebra", seed=seed, pair_coverage=0.1,
                         test_S_pearson=0.6, tag="t"))
        rows.append(dict(family="unconstrained", seed=seed, pair_coverage=0.1,
                         test_S_pearson=0.4, tag="t"))
    df = pd.DataFrame(rows)

    def favouring(table: str) -> str:
        header, _, body = table.partition("\n| ---")
        cols = [c.strip() for c in header.strip().strip("|").split("|")]
        cells = [c.strip() for c in
                 body.strip().splitlines()[-1].strip().strip("|").split("|")]
        return dict(zip(cols, cells))["seeds_favouring_algebra"]

    # algebra scores 0.6 against 0.4 on all five seeds: it is better on all five
    # under a higher-is-better metric and on none under a lower-is-better one.
    hi = report.paired_table(df, "algebra", "unconstrained",
                             metric="test_S_pearson", lower_is_better=False)
    lo = report.paired_table(df, "algebra", "unconstrained",
                             metric="test_S_pearson", lower_is_better=True)
    assert float(favouring(hi)) == 5
    assert float(favouring(lo)) == 0


def test_build_report_runs_and_names_its_denominators(real_df):
    body = report.build_report(real_df)
    assert "Which runs are in each headline cell" in body
    assert "contributing tags" in body
    # every powered cell must print its own n
    assert "(n=" in body
    # the report must never claim to be the five-seed pilot
    powered = report.powered_frame(real_df)
    n = int(powered.groupby(["pair_coverage", "family"])["seed"].nunique().max())
    assert f"(n={n})" in body


def test_no_headline_cell_mixes_denominators(real_df):
    """Within a cell, a metric is either defined for every seed or for none.

    Nan-safe means are the right behaviour -- `test_A_pearson` is genuinely
    undefined where a readout is constant -- but they let one row average 17
    seeds in one column and 3 in the next, understating one metric and
    overstating another with nothing visible in the table (review finding R2).
    The report prints every cell's `n`, and this asserts the stronger property
    that the denominators do not diverge in the first place.

    The `additive` family is exempt: it has no interaction readout at all, so
    its S/A columns are all-NaN by construction and render as "—".
    """
    powered = report.powered_frame(real_df)
    metrics = ["test_mse", "test_S_pearson", "test_A_pearson",
               "final_test_mse"]
    offenders = []
    for (cov, fam), g in powered.groupby(["pair_coverage", "family"]):
        if fam == "additive":
            continue
        for m in metrics:
            if m not in g.columns:
                continue
            n = int(g[m].notna().sum())
            if n not in (0, len(g)):
                offenders.append((cov, fam, m, n, len(g)))
    assert not offenders, f"partial denominators: {offenders}"


# ------------------------------------------------------------------ injection


def test_inject_does_not_eat_prose_after_the_generated_block(tmp_path):
    """The regression that cost several hundred lines of hand-written analysis.

    `inject` originally replaced everything from `<!-- RESULTS -->` to the next
    `##` heading. On this README the whole of section 8's argument lived there,
    so the command documented as "regenerates the reported tables" deleted the
    analysis and left tables in its place. It had simply never been run against a
    README that had prose in that span.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_make_report",
        Path(__file__).resolve().parents[1] / "scripts" / "make_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Title\n\n## 8. Results\n\n<!-- RESULTS -->\n\nOLD TABLE\n"
        "<!-- END RESULTS -->\n\n### Analysis\n\nirreplaceable prose\n\n"
        "## 9. Next\n\nmore\n")
    mod.inject(doc, "NEW TABLE")
    out = doc.read_text()

    assert "NEW TABLE" in out and "OLD TABLE" not in out
    assert "irreplaceable prose" in out, "inject destroyed prose after the block"
    assert "### Analysis" in out
    assert "## 9. Next" in out


def test_inject_refuses_to_guess_where_the_block_ends(tmp_path):
    """Without an end marker it must fail, not fall back to eating the section."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_make_report2",
        Path(__file__).resolve().parents[1] / "scripts" / "make_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    doc = tmp_path / "doc.md"
    doc.write_text("<!-- RESULTS -->\n\nold\n\n### Analysis\n\nprose\n")
    with pytest.raises(SystemExit):
        mod.inject(doc, "new")
    assert "prose" in doc.read_text()
