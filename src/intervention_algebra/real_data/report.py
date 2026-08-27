"""Aggregation, paired tests and figures for the Phase 2 real-data experiment.

Aggregation rule, stated once and applied everywhere: a *run* is one
(screen, coverage, family, split_seed, init_seed). Init seeds are averaged
first, so the unit of the paired test is the **split seed** -- which pairs
happened to be observed. That is the source of variation the claim is about, and
treating two init seeds as two independent observations would double the
apparent n for free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FAMILY_ORDER = ("additive", "order_insensitive", "unrestricted", "structured")

#: Metrics where a SMALLER value is better. Everything else is larger-is-better.
#: This exists because the seed-win count beside every headline delta was
#: computed as ``(delta < 0).sum()`` for all metrics -- correct for MSE, exactly
#: inverted for Pearson r and ordering accuracy. On the real data that printed
#: "-0.011 (7/8)" for the primary directional metric on a cell where the
#: structured model won on ONE of eight seeds: the reader is told the opposite
#: of the finding, beside a delta that is itself correct. Phase 1 had this bug,
#: fixed it, and pinned it; Phase 2 reintroduced it.
LOWER_IS_BETTER = frozenset({
    "test_mse", "test_mae", "test_rmse", "test_D_mae", "test_D_rmse",
    "val_loss", "train_loss", "const_baseline_mse", "const_baseline_mae",
})
#: Metrics with no "better" direction -- a ratio whose ideal is 1.0, not an
#: extremum. No win count is emitted for these.
NO_DIRECTION = frozenset({"test_pred_A_rms_ratio", "test_head_A_over_sym"})
NICE = {"additive": "additive null", "order_insensitive": "order-insensitive",
        "unrestricted": "unrestricted", "structured": "structured"}


def load_runs(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = [r for r in rows if "error" in r]
    if bad:
        raise RuntimeError(f"{len(bad)} runs failed; refusing to report over a "
                           f"grid with holes. First:\n{bad[0]['error'][-800:]}")
    df = pd.DataFrame(rows)
    df["nice"] = df["family"].map(NICE)
    # Surfaced as a column because the tuning-parity analysis selects on it.
    df["lr"] = [r.get("config", {}).get("train", {}).get("lr") for r in rows]
    assert_balanced_init_seeds(df)
    for tag in df["tag"].unique():
        assert_grid_complete(df, tag)
    return df


def assert_grid_complete(df: pd.DataFrame, tag: str = "main") -> None:
    """Every (screen, coverage, family) cell must carry the same split seeds.

    ``load_runs`` rejects rows that recorded an error, but a sweep that was
    killed, or one being read while it is still running, leaves no error rows at
    all -- it leaves *holes*. ``main_table`` then means over whatever exists
    while ``paired_delta`` intersects, so a single printed row can mix an 8-seed
    mean with a 2-seed one and nothing says so. Worse, the pool completes
    families in cost order, so a mid-sweep read is biased toward whichever
    family trains fastest -- the additive null first, the structured model last.
    This was live: PANC1 coverage 0.20 once stood at additive 8 / order-
    insensitive 5 / unrestricted 2 / structured 0, and the report would have
    printed it as a finished row.
    """
    sub = df[df["tag"] == tag]
    if sub.empty:
        return
    cells = sub.groupby(["screen", "coverage", "family"])["split_seed"].apply(
        lambda x: frozenset(x))
    expected = max(cells, key=len)
    holes = {k: sorted(expected - v) for k, v in cells.items() if v != expected}
    # Expected cells are the (screen, coverage) combinations this tag actually
    # scheduled, crossed with its families -- not screens x coverages x families.
    # The headline grid fills the full cross-product either way, but a block that
    # deliberately runs a subset of cells (the tuning-parity grid runs four)
    # would otherwise be reported as full of holes it never had.
    missing_cells = []
    for screen, cov in sub[["screen", "coverage"]].drop_duplicates().itertuples(index=False):
        for fam in sub["family"].unique():
            if (screen, cov, fam) not in cells.index:
                missing_cells.append((screen, cov, fam))
    if holes or missing_cells:
        raise RuntimeError(
            f"tag={tag!r} grid is incomplete -- refusing to report over holes.\n"
            f"  cells missing split seeds: { {str(k): v for k, v in list(holes.items())[:6]} }\n"
            f"  cells absent entirely: {missing_cells[:6]}\n"
            f"If the sweep is still running, wait for it; if it died, rerun it.")


def assert_balanced_init_seeds(df: pd.DataFrame) -> None:
    """Every split seed in a cell must carry the same number of init seeds.

    Averaging init seeds before the paired test is only sound if the average is
    over the same number of runs everywhere. An unbalanced cell still aggregates
    silently, still reports ``n = 8`` split seeds, and quietly weights the
    headline mean toward whichever seeds happened to be run more often. This
    exact bug was introduced once, by an init-variance block reusing the main
    grid and inheriting its tag.
    """
    keys = ["tag", "screen", "coverage", "family"]
    counts = df.groupby(keys + ["split_seed"]).size().rename("n").reset_index()
    bad = (counts.groupby(keys)["n"].nunique() > 1)
    if bad.any():
        offenders = bad[bad].index.tolist()
        raise RuntimeError(
            "unbalanced init seeds across split seeds in "
            f"{offenders[:5]} -- aggregating these would weight the headline "
            "mean. Check that every grid emits under its own tag.")


def by_split_seed(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Collapse init seeds, leaving one value per (screen, coverage, family, split)."""
    keys = ["tag", "screen", "coverage", "family", "split_seed"]
    return (df.groupby(keys, as_index=False)[metric]
              .mean().rename(columns={metric: "value"}))


def paired_delta(df: pd.DataFrame, metric: str, a: str, b: str,
                 tag: str = "main") -> pd.DataFrame:
    """``a - b`` per (screen, coverage), paired on split seed.

    Paired because the two families see the *same* pairs on a given split seed;
    an unpaired test would throw away the design and inflate the variance with
    between-split differences that cancel exactly.
    """
    s = by_split_seed(df[df["tag"] == tag], metric)
    out = []
    for (screen, cov), g in s.groupby(["screen", "coverage"]):
        va = g[g["family"] == a].set_index("split_seed")["value"]
        vb = g[g["family"] == b].set_index("split_seed")["value"]
        common = sorted(set(va.index) & set(vb.index))
        if len(common) < 3:
            continue
        d = va.loc[common].to_numpy() - vb.loc[common].to_numpy()
        t = stats.ttest_rel(va.loc[common], vb.loc[common])
        w = (stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan"))
        se = d.std(ddof=1) / np.sqrt(len(d))
        crit = stats.t.ppf(0.975, len(d) - 1)
        row = {
            "screen": screen, "coverage": cov, "n_split_seeds": len(d),
            "metric": metric, "family_a": a, "family_b": b,
            "mean_a": float(va.loc[common].mean()),
            "mean_b": float(vb.loc[common].mean()),
            "delta": float(d.mean()),
            "ci_lo": float(d.mean() - crit * se), "ci_hi": float(d.mean() + crit * se),
            "p_ttest": float(t.pvalue), "p_wilcoxon": float(w),
        }
        if metric in NO_DIRECTION:
            row["n_seeds_favouring_a"] = None
            row["direction"] = "none"
        else:
            lower = metric in LOWER_IS_BETTER
            row["n_seeds_favouring_a"] = int((d < 0).sum() if lower else (d > 0).sum())
            row["direction"] = "lower_is_better" if lower else "higher_is_better"
        out.append(row)
    return pd.DataFrame(out)


def main_table(df: pd.DataFrame, metric: str, tag: str = "main") -> pd.DataFrame:
    s = by_split_seed(df[df["tag"] == tag], metric)
    piv = (s.groupby(["screen", "coverage", "family"])["value"]
             .agg(["mean", "std", "count"]).reset_index())
    return piv.sort_values(["screen", "coverage", "family"])


def collapse_report(df: pd.DataFrame) -> pd.DataFrame:
    """How often the antisymmetric pair head fell into its degenerate basin.

    Counted at two levels: whole runs whose head emits an antisymmetric RMS
    under 5% of the head's own symmetric RMS, and individual restarts whose
    training-set head RMS is at machine zero. Reported for every family, not
    only the structured one -- a diagnostic that only ever runs on the model
    under suspicion cannot tell you whether the number is unusual -- with
    ``zero_A_is_expected`` marking the families whose head is symmetric by
    construction and therefore pinned at zero.

    Measured on the **head**, not on the model's total antisymmetric output: the
    latter includes the first-order term every family carries, and reports a
    healthy 0.42 for a pair head pinned to identically zero.
    """
    g = df.groupby(["tag", "screen", "coverage", "family"], as_index=False)
    out = g.agg(
        n_runs=("collapsed", "size"),
        n_collapsed_runs=("collapsed", "sum"),
        n_collapsed_restarts=("n_restarts_collapsed", "sum"),
        mean_head_A_over_sym=("test_head_A_over_sym", "mean"),
        min_head_A_over_sym=("test_head_A_over_sym", "min"),
        mean_first_order_A_rms=("test_first_order_A_rms_std_units", "mean"),
        worst_restart_ratio=("restart_train_loss_ratio", "max"),
    )
    # Two families emit A == 0 by construction and must not be read as having
    # failed: `order_insensitive` is symmetrised end to end, and `additive` has
    # no pair term (its nonzero A comes from the first-order term alone). The
    # column says which rows the collapse count is even a question about.
    out["zero_A_is_expected"] = ~out["family"].isin(["unrestricted", "structured"])
    return out


def figures(df: pd.DataFrame, outdir: Path, screen: str = "A375") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    main = df[(df["tag"] == "main") & (df["screen"] == screen)]
    paths = []
    colors = {"additive": "#888888", "order_insensitive": "#4C72B0",
              "unrestricted": "#DD8452", "structured": "#55A868"}
    styles = {"additive": "--"}

    # 1. held-out error vs coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam in FAMILY_ORDER:
        s = by_split_seed(main[main["family"] == fam], "test_mse")
        g = s.groupby("coverage")["value"].agg(["mean", "std", "count"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"] / np.sqrt(g["count"]),
                    marker="o", capsize=3, label=NICE[fam], color=colors[fam])
    ax.set_xscale("log")
    ax.set_xticks(sorted(main["coverage"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("pair coverage (fraction of unordered pairs seen in training)")
    ax.set_ylabel("held-out ordered MSE")
    ax.set_title(f"{screen}: held-out prediction error vs pair coverage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = outdir / "fig1_heldout_mse_vs_coverage.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 2. structured - unrestricted, the Phase 1 shape question
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, label, better) in zip(axes, [
            ("test_mse", "held-out ordered MSE", "lower is better"),
            ("test_D_pearson", "directional Pearson r", "higher is better")]):
        for sc in sorted(df["screen"].unique()):
            d = paired_delta(df[df["screen"] == sc], metric, "structured", "unrestricted")
            if d.empty:
                continue
            ax.errorbar(d["coverage"], d["delta"],
                        yerr=[d["delta"] - d["ci_lo"], d["ci_hi"] - d["delta"]],
                        marker="o", capsize=3, label=sc)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(sorted(main["coverage"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("pair coverage")
        ax.set_ylabel(f"structured − unrestricted\n{label}")
        ax.set_title(f"{label}\n({better})", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = outdir / "fig2_structured_minus_unrestricted.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 3. directional effect, measured vs predicted
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    # The additive null belongs on this figure, not only in the tables. At the
    # sparsest coverage the two pair models sit exactly on top of it, and a
    # reader seeing 0.57 directional r without the null reads it as pair-model
    # skill when it is entirely the per-drug tendency.
    for fam in ("additive", "unrestricted", "structured"):
        s = by_split_seed(main[main["family"] == fam], "test_D_pearson")
        g = s.groupby("coverage")["value"].agg(["mean", "std", "count"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"] / np.sqrt(g["count"]),
                    marker="o", capsize=3, label=NICE[fam], color=colors[fam],
                    ls=styles.get(fam, "-"))
    ctl = df[(df["tag"] == "direction_shuffle") & (df["screen"] == screen)]
    if not ctl.empty:
        s = by_split_seed(ctl[ctl["family"] == "structured"], "test_D_pearson")
        g = s.groupby("coverage")["value"].agg(["mean", "std", "count"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"] / np.sqrt(g["count"]),
                    marker="s", ls="--", capsize=3, color="#C44E52",
                    label="structured, direction shuffled")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sorted(main["coverage"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("pair coverage")
    ax.set_ylabel(r"Pearson $r$, predicted vs measured $D_{ij}$")
    ax.set_title(f"{screen}: directional-effect prediction on held-out pairs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = outdir / "fig3_directional_effect.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def distribution_figure(screens: dict, outdir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, screen in screens.items():
        M = screen.matrix("y")
        n = screen.n_drugs
        iu = np.triu(np.ones((n, n), dtype=bool), 1)
        D = (M - M.T)[iu]
        D = D[np.isfinite(D)]
        ax.hist(D, bins=80, histtype="step", density=True, label=f"{label} (n={len(D)})")
    ax.set_xlabel(r"measured schedule effect $D_{ij} = y_{i\to j} - y_{j\to i}$")
    ax.set_ylabel("density")
    ax.set_title("Distribution of measured schedule effects")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = outdir / "fig4_schedule_effect_distribution.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def parity_delta(parity: pd.DataFrame, main: pd.DataFrame, metric: str,
                 a: str = "structured", b: str = "unrestricted") -> pd.DataFrame:
    """``a - b`` with each family at its own validation-selected learning rate.

    The shipped grid reuses one hyperparameter choice, made on A375 at coverage
    0.10, at every coverage and on both screens. Where that choice does not
    transfer, the family contrast is partly a comparison of learning rates. This
    recomputes it with the tuning artifact removed.

    Selection is per (screen, coverage, family, split seed) on **validation
    loss**, pooling the parity runs with the shipped ``main`` run for that cell
    so the selected rate is drawn from the full grid including the rate actually
    shipped. Test metrics are never consulted -- picking the rate that minimises
    the reported metric would replace a tuning artifact with a worse one.
    """
    cols = ["screen", "coverage", "family", "split_seed", "val_loss", metric, "lr"]
    pool = pd.concat([parity[cols], main[main["tag"] == "main"][cols]],
                     ignore_index=True).dropna(subset=["val_loss", metric])
    pool = pool[pool["family"].isin([a, b])]
    # Argmin on validation loss within each cell; ties broken by lower lr for
    # determinism (they do not occur in practice, but a silent tie-break by row
    # order would make the output depend on file concatenation order).
    pool = pool.sort_values(["val_loss", "lr"])
    best = pool.groupby(["screen", "coverage", "family", "split_seed"],
                        as_index=False).first()

    out = []
    for (screen, cov), g in best.groupby(["screen", "coverage"]):
        va = g[g["family"] == a].set_index("split_seed")
        vb = g[g["family"] == b].set_index("split_seed")
        common = sorted(set(va.index) & set(vb.index))
        if len(common) < 3:
            continue
        xa = va.loc[common, metric].to_numpy()
        xb = vb.loc[common, metric].to_numpy()
        d = xa - xb
        t = stats.ttest_rel(xa, xb)
        se = d.std(ddof=1) / np.sqrt(len(d))
        crit = stats.t.ppf(0.975, len(d) - 1)
        lower = metric in LOWER_IS_BETTER
        out.append({
            "screen": screen, "coverage": cov, "metric": metric,
            "n_split_seeds": len(d),
            "lrs_a": sorted(va.loc[common, "lr"].unique().tolist()),
            "lrs_b": sorted(vb.loc[common, "lr"].unique().tolist()),
            "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
            "delta": float(d.mean()),
            "ci_lo": float(d.mean() - crit * se),
            "ci_hi": float(d.mean() + crit * se),
            "p_ttest": float(t.pvalue),
            "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else float("nan"),
            "n_seeds_favouring_a": int((d < 0).sum() if lower else (d > 0).sum()),
        })
    return pd.DataFrame(out)
