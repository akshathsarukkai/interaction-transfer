"""The single authoritative path from ``results/phase1.jsonl`` to the README.

``scripts/make_report.py`` is a thin CLI over this module; the logic lives here
so that it can be imported and tested (``tests/test_report.py``).

Why this module exists
----------------------
The first version of the report built every headline table from ``tag == "main"``.
That was wrong in a way that is easy to miss and expensive to believe: ``main``
holds five seeds, while every conclusion Phase 1 actually reaches rests on
seventeen. The replication seeds live under other tags (``rep020``,
``rep10_matched1x``, ``power10_algebra``, ...), so the command advertised as
"this regenerates the reported numbers" regenerated a strictly weaker set of
numbers that happened to look like the real ones.

The fix is not a longer list of tags. A tag records which *batch* a run was
launched in; it is provenance, not experimental design, and a report keyed on
tags goes stale every time a replication is added. Instead every cell is named
by its **condition** -- the hash of the fully resolved run configuration with the
seed removed (:func:`analysis.condition_key`). Two runs pool iff they differ only
in their seed. That makes the four hazards structural rather than
matters-of-care:

* *double counting* -- a duplicated ``(condition, seed)`` raises
  :class:`analysis.DuplicateRunsError` rather than halving a p-value silently;
* *incompatible conditions* -- ``max_epochs``/``patience`` are inside the hash,
  so an early-stopping run can never pool with a fixed-length one;
* *missing ``pair_coverage``* -- the split config is inside the hash;
* *control tags leaking into the headline* -- a different ``s_scale``,
  ``sparsity_mode``, ``simultaneity_defect`` or pair width is a different hash.

The conditions themselves are declared in :mod:`intervention_algebra.phase1`
(:func:`phase1.headline_conditions`, :func:`phase1.matched2x_conditions`), i.e.
by the same module that defines the experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis, phase1

__all__ = [
    "powered_frame",
    "add_powered_skill",
    "cell_table",
    "paired_table",
    "provenance_table",
    "build_report",
]

#: Metrics the powered headline reports, with the direction that counts as better.
HEADLINE_METRICS = [
    ("test_mse", True, "Held-out-pair prediction error"),
    ("skill_vs_additive_family", False, "Skill against the trained additive null"),
    ("test_S_pearson", False, "Recovery of the symmetric interaction S"),
    ("test_A_pearson", False, "Recovery of the antisymmetric interaction A"),
]


class ReportIntegrityError(RuntimeError):
    """Raised when the selected rows cannot support the table being built."""


#: Declared headline cells that genuinely have no runs, each with a reason.
#: Everything else missing is treated as a defect, not as an empty table cell.
KNOWN_ABSENT: dict[tuple[str, float], str] = {}


# ------------------------------------------------------------------ selection


def powered_frame(df: pd.DataFrame, conditions: dict | None = None) -> pd.DataFrame:
    """Select exactly the runs belonging to ``conditions``.

    ``conditions`` maps ``(family, coverage) -> spec`` (see
    :func:`phase1.headline_conditions`). Returns the concatenated rows with a
    ``contributing_tags`` column recording, per cell, which sweep batches the
    replicates came from -- so the provenance stays visible even though it plays
    no part in the selection.

    Raises :class:`ReportIntegrityError` if a selected row's family does not
    match the cell it was selected for (which would mean the hash collided), or
    if a ``(coverage, family, seed)`` appears twice.
    """
    if conditions is None:
        conditions = phase1.headline_conditions()
    if df.empty:
        return df

    frames = []
    for (family, coverage), spec in conditions.items():
        key = analysis.spec_condition_key(spec)
        sub = analysis.select_conditions(df, key)   # raises on duplicate replicates
        if sub.empty:
            # A declared cell with no rows is either an unrun experiment or --
            # more dangerously -- a config field added since the rows were
            # written, which changes the hash and makes a *present* cell
            # disappear. Silently skipping renders it as an em-dash in a table
            # whose neighbours all say n=17, which reads as "measured, and
            # nothing there".
            if (family, coverage) in KNOWN_ABSENT:
                continue
            raise ReportIntegrityError(
                f"no runs for the declared cell (family={family!r}, "
                f"coverage={coverage}); condition {key}. Either the experiment "
                f"was never run -- add it to report.KNOWN_ABSENT with a reason "
                f"-- or a config field changed and the stored rows no longer "
                f"hash to their own condition.")
        bad = set(sub["family"]) - {family}
        if bad:
            raise ReportIntegrityError(
                f"condition {key} was declared for family {family!r} but "
                f"matched rows with family {sorted(bad)!r}")
        sub = sub.copy()
        sub["pair_coverage"] = coverage
        sub["contributing_tags"] = ",".join(sorted(set(sub["tag"].dropna())))
        frames.append(sub)

    if not frames:
        return df.iloc[0:0]
    out = pd.concat(frames, ignore_index=True)

    dup = out.duplicated(subset=["pair_coverage", "family", "seed"], keep=False)
    if dup.any():
        raise ReportIntegrityError(
            "the same (coverage, family, seed) was selected more than once:\n"
            + str(out.loc[dup, ["pair_coverage", "family", "seed", "tag"]]))
    return out


def add_powered_skill(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute ``skill_vs_additive_family`` pairing on coverage and seed.

    :func:`analysis.add_skill_vs_additive_family` matches each run to its
    additive reference on ``CONDITION_COLS``, which include ``tag``. Inside a
    single sweep that is right. Across a *powered* cell it is not: at coverage
    0.10 the algebra replicates carry ``tag="power10_algebra"`` while the
    additive replicates carry ``tag="rep10_matched1x"``, so the merge finds no
    reference and returns NaN for twelve of the seventeen seeds -- a table that
    reports n=5 while every neighbouring column reports n=17.

    Within a powered frame the coverage already identifies the condition (the
    frame contains nothing else), so pairing on ``(pair_coverage, seed)`` is both
    correct and tag-free.
    """
    if df.empty or "test_mse" not in df.columns:
        return df
    df = df.copy()
    ref = (df[df["family"] == "additive"]
           .groupby(["pair_coverage", "seed"], dropna=False)["test_mse"]
           .mean().rename("_additive_mse"))
    df = df.merge(ref, how="left",
                  left_on=["pair_coverage", "seed"], right_index=True)
    df["skill_vs_additive_family"] = 1.0 - df["test_mse"] / df["_additive_mse"]
    return df.drop(columns=["_additive_mse"])


# --------------------------------------------------------------------- tables


def cell_table(df: pd.DataFrame, metric: str,
               families=("additive", "unconstrained", "shared_pair", "algebra"),
               ) -> str:
    """``mean ± sd (n)`` per coverage per family. ``n`` is always shown.

    The earlier table printed ``n`` only when it fell below the cell's seed
    count, which is fine when every column has the same denominator and
    dangerous when the whole point of the table is that the denominator grew
    from 5 to 17.
    """
    if df.empty or metric not in df.columns:
        return "_(no rows)_"
    fams = [f for f in families if f in set(df["family"])]
    rows = []
    for cov, g in df.groupby("pair_coverage", sort=True):
        rec = {"pair_coverage": f"{cov:.2f}"}
        for f in fams:
            vals = g.loc[g["family"] == f, metric].astype(float).dropna()
            if len(vals) == 0:
                rec[f] = "—"
            elif len(vals) == 1:
                rec[f] = f"{vals.mean():.3f} (n=1)"
            else:
                rec[f] = (f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f} "
                          f"(n={len(vals)})")
        rows.append(rec)
    return analysis.markdown_table(pd.DataFrame(rows))


def paired_table(df: pd.DataFrame, family_a: str, family_b: str,
                 metric: str = "test_mse", lower_is_better: bool = True) -> str:
    """Paired-by-seed ``family_a − family_b`` per coverage, with n and p.

    Reports the paired t p-value **and** the Wilcoxon signed-rank p. At n = 17
    with one heavy-tailed seed the two can disagree by an order of magnitude --
    at coverage 0.40 the t test gives 0.12 and the signed-rank test 0.0046 on
    the identical, untouched sample, because a single outlier inflates the
    variance the t statistic divides by. Reporting only the t there would have
    understated a real effect; reporting only whichever is smaller would be
    worse. Both are shown at every cell so the choice is never made per cell.
    """
    cmp = analysis.paired_comparison(df, family_a, family_b, metric=metric,
                                     group_cols=("pair_coverage",))
    if cmp.empty:
        return "_(not computable)_"
    keep = cmp[["pair_coverage", "mean_diff", "std_diff", "n_seeds",
                "n_a_lt_b", "p_value"]].copy()
    keep["p_wilcoxon"] = _wilcoxon_by_group(df, family_a, family_b, metric,
                                            cmp["pair_coverage"].tolist())
    keep["seeds_favouring_" + family_a] = (keep["n_a_lt_b"] if lower_is_better
                                           else keep["n_seeds"] - keep["n_a_lt_b"])
    keep = keep.drop(columns=["n_a_lt_b"])
    keep["ci95_lo"], keep["ci95_hi"] = _ci95(cmp)
    return analysis.markdown_table(keep)


def _wilcoxon_by_group(df, family_a, family_b, metric, coverages):
    """Wilcoxon signed-rank p per coverage, on the same paired differences."""
    from scipy import stats

    out = []
    for cov in coverages:
        g = df[df["pair_coverage"] == cov]
        piv = g.pivot_table(index="seed", columns="family", values=metric,
                            aggfunc="mean")
        if family_a not in piv.columns or family_b not in piv.columns:
            out.append(np.nan)
            continue
        d = (piv[family_a] - piv[family_b]).dropna()
        if len(d) < 6 or float(np.abs(d).sum()) == 0.0:
            out.append(np.nan)     # too few pairs for the rank test to mean much
            continue
        try:
            out.append(float(stats.wilcoxon(d).pvalue))
        except Exception:
            out.append(np.nan)
    return out


def _ci95(cmp: pd.DataFrame):
    """95 % t interval on the paired mean difference (n is 5-17, not large)."""
    from scipy import stats

    lo, hi = [], []
    for _, r in cmp.iterrows():
        n, m, s = r["n_seeds"], r["mean_diff"], r["std_diff"]
        if not n or n < 2 or not np.isfinite(s):
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        half = float(stats.t.ppf(0.975, n - 1)) * s / np.sqrt(n)
        lo.append(m - half)
        hi.append(m + half)
    return lo, hi


def provenance_table(df: pd.DataFrame) -> str:
    """Which sweep batches supplied each powered cell, and how many seeds."""
    if df.empty:
        return "_(no rows)_"
    rows = []
    for (cov, fam), g in df.groupby(["pair_coverage", "family"], sort=True):
        rows.append({
            "pair_coverage": f"{cov:.2f}",
            "family": fam,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": _compact_seeds(sorted(int(s) for s in g["seed"].unique())),
            "contributing tags": g["contributing_tags"].iloc[0],
            "condition": g["condition"].iloc[0],
        })
    return analysis.markdown_table(pd.DataFrame(rows))


def _compact_seeds(seeds: list[int]) -> str:
    if not seeds:
        return "—"
    out, start, prev = [], seeds[0], seeds[0]
    for s in seeds[1:] + [None]:
        if s is not None and s == prev + 1:
            prev = s
            continue
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
        if s is not None:
            start = prev = s
    return ",".join(out)


# --------------------------------------------------------------- report body


def _section(title: str, body: str) -> str:
    return f"### {title}\n\n{body}\n"


def _families_present(df: pd.DataFrame) -> list[str]:
    order = ["additive", "unconstrained", "shared_pair", "algebra", "wellspecified"]
    have = set(df["family"])
    return [f for f in order if f in have]


def _param_table(df: pd.DataFrame) -> str:
    cols = [c for c in ("n_params", "n_pair_params", "pair_hidden") if c in df.columns]
    if not cols:
        return "_(parameter counts not recorded)_"
    g = df.groupby("family")[cols].agg(["min", "max"])
    rows = []
    for fam in _families_present(df):
        if fam not in g.index:
            continue
        rec = {"family": fam}
        for c in cols:
            lo, hi = g.loc[fam, (c, "min")], g.loc[fam, (c, "max")]
            rec[c] = f"{int(lo)}" if lo == hi else f"{int(lo)}–{int(hi)}"
        rows.append(rec)
    return analysis.markdown_table(pd.DataFrame(rows))


#: Tags that are the headline experiment or a replication of it, not a control.
#: Listing them under "controls" was not merely untidy: it re-presented the
#: coverage-0.40 headline column as though it were a control condition, and
#: showed `power10_algebra_wide` -- the 4x-capacity arm this repository
#: explicitly calls the *wrong* comparison -- unlabelled next to real ones.
_NOT_CONTROLS = ("main", "ceiling", "ceiling_fixedlen", "cov040", "rep020",
                 "rep10_matched1x", "power10_algebra", "power10_algebra_m2x",
                 "power10_algebra_wide", "power10_double")


def poolable_tag_groups(df: pd.DataFrame) -> list[list[str]]:
    """Group sweep tags that are replications of the *same* conditions.

    Several controls were piloted at three seeds and later powered under a
    different tag (`misspec_0.3` then `rep_misspec_0.3`; `regime_symmetric` then
    `rep_regime_sym`). Rendering those as separate rows shows n=3 next to prose
    that correctly cites the pooled n=13, which is the denominator problem this
    report exists to remove.

    Two tags pool iff they share at least one family and, for **every** family
    they share, their condition hashes are equal -- i.e. the only difference is
    which batch the seeds were launched in. That rule keeps
    `control_double_capacity` (unconstrained@120) apart from
    `control_unmatched_capacity` (unconstrained@48), which differ in a way that
    matters, without anyone maintaining a list.
    """
    per_tag: dict[str, dict[str, str]] = {}
    for tag, g in df.groupby("tag"):
        per_tag[str(tag)] = {str(f): str(sub["condition"].iloc[0])
                             for f, sub in g.groupby("family")}

    groups: list[list[str]] = []
    for tag in sorted(per_tag):
        for grp in groups:
            head = per_tag[grp[0]]
            shared = set(head) & set(per_tag[tag])
            if shared and all(head[f] == per_tag[tag][f] for f in shared):
                grp.append(tag)
                break
        else:
            groups.append([tag])
    return groups


def _control_table(df: pd.DataFrame, metric: str = "test_mse") -> str:
    """One row per control *condition* -- pooled across replication batches."""
    if "tag" not in df.columns or "condition" not in df.columns:
        return "_(no tags)_"
    fams = _families_present(df)
    rows = []
    for grp in poolable_tag_groups(df):
        if any(t in _NOT_CONTROLS for t in grp):
            continue
        sub_all = df[df["tag"].isin(grp)]
        label = " + ".join(sorted(grp))
        for cov, g in sub_all.groupby("pair_coverage", sort=True):
            rec = {"condition": label, "coverage": f"{cov:.2f}"}
            for f in fams:
                v = g.loc[g["family"] == f, metric].astype(float).dropna()
                rec[f] = f"{v.mean():.3f} (n={len(v)})" if len(v) else "—"
            rows.append(rec)
    return analysis.markdown_table(pd.DataFrame(rows)) if rows else "_(no controls)_"


def _invariant_table(df: pd.DataFrame) -> str:
    cols = [c for c in ("sym_residual", "antisym_residual",
                        "pair_net_order_asymmetry") if c in df.columns]
    if not cols:
        return "_(no invariant diagnostics recorded)_"
    g = df.groupby("family")[cols].max().reset_index()
    g = g[g["family"].isin(_families_present(df))]
    return analysis.markdown_table(g, floatfmt="{:.2e}")


def build_report(df: pd.DataFrame) -> str:
    """The full generated body, powered cells first."""
    powered = add_powered_skill(powered_frame(df))
    m2x = add_powered_skill(powered_frame(df, phase1.matched2x_conditions()))

    parts: list[str] = []
    n_ok = len(df)
    parts.append(
        f"All numbers below are regenerated from `results/phase1.jsonl` "
        f"({n_ok} successful runs) by `scripts/make_report.py`, which selects "
        f"cells by **experimental condition** (a hash of the fully resolved run "
        f"config with the seed removed), never by sweep tag. Every cell shows "
        f"its own `n`.\n")

    parts.append(_section(
        "Which runs are in each headline cell",
        "Selection is by condition; the tags are shown only so the provenance "
        "stays legible. A cell pools two batches iff every configured value "
        "except the seed is identical.\n\n"
        + provenance_table(powered)))

    for metric, lower_better, title in HEADLINE_METRICS:
        body = cell_table(powered, metric)
        direction = "Lower is better." if lower_better else "Higher is better."
        body = f"{direction}\n\n{body}"
        body += ("\n\nPaired `algebra − unconstrained` (matched seeds):\n\n"
                 + paired_table(powered, "algebra", "unconstrained",
                                metric=metric, lower_is_better=lower_better))
        if metric == "test_mse":
            body += ("\n\nPaired `algebra − additive` (matched seeds) — the "
                     "no-interaction null:\n\n"
                     + paired_table(powered, "algebra", "additive",
                                    metric=metric, lower_is_better=lower_better))
        parts.append(_section(title + " (headline cells)", body))

    parts.append(_section(
        "Capacity control: every family at ~2× its headline pair parameters",
        "`algebra`@78 (23 096 pair params) against `unconstrained`@120 "
        "(23 288). If the advantage were capacity it would not survive here.\n\n"
        + cell_table(m2x, "test_mse", families=("unconstrained", "algebra"))
        + "\n\nPaired `algebra − unconstrained`:\n\n"
        + paired_table(m2x, "algebra", "unconstrained", metric="test_mse")
        + "\n\nS recovery:\n\n"
        + paired_table(m2x, "algebra", "unconstrained",
                       metric="test_S_pearson", lower_is_better=False)
        + "\n\nA recovery:\n\n"
        + paired_table(m2x, "algebra", "unconstrained",
                       metric="test_A_pearson", lower_is_better=False)))

    parts.append(_section(
        "Checkpoint-selection control (final epoch vs best validation)",
        "`final_test_mse` scores the same runs at their last epoch instead of "
        "their best-validation checkpoint. If the headline only exists at one "
        "of the two readouts it is a selection artifact.\n\n"
        + cell_table(powered, "final_test_mse")
        + "\n\nPaired `algebra − unconstrained` at the final epoch:\n\n"
        + paired_table(powered, "algebra", "unconstrained",
                       metric="final_test_mse")))

    parts.append(_section(
        "Controls (mean held-out MSE over seeds)",
        _control_table(df)))

    parts.append(_section(
        "Parameter counts (powered headline cells)",
        "Capacity matching is on; `control_unmatched_capacity` and the 2× "
        "control above remove it in the two opposite directions.\n\n"
        + _param_table(powered)))

    parts.append(_section(
        "Architectural invariants (worst case over all runs)",
        "`sym_residual` = ‖S(i,j) − S(j,i)‖, `antisym_residual` = ‖A(i,j) + "
        "A(j,i)‖. These are tripwires: they must be ~0 by construction, and a "
        "nonzero value means the implementation stopped matching the "
        "formalism.\n\n"
        + _invariant_table(df)))

    return "\n".join(parts)
