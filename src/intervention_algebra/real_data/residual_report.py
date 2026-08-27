"""Aggregation, statistics and figures for the residual-directionality result.

Two rules are enforced here rather than trusted to the reader.

**The split seed is the unit of replication.** Pairs within a run are not
independent (they share a fitted additive baseline, a fitted model and a
training set), and the two orientations of a pair are one observation, not two.
Every test in this module first collapses to one number per split seed via
:func:`by_split_seed` and only then runs a paired test over the 8 seeds. A
p-value computed over 848 held-out pairs would be roughly 10x too small and
would turn every table into a wall of significance.

**Residual rows and Phase 2 rows are different experiments.** They share a
split, a screen and a coverage grid, and nothing else: different target
(``D_res`` versus ``y``), different null (zero versus the additive family) and
different metric names. :func:`load_residual_runs` refuses a file carrying Phase
2 columns, so the two can never be concatenated by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .residual_models import FLEXIBLE_RUNGS, LADDER_ORDER

NICE = {
    "zero": "zero (null)",
    "potential": "potential c_i−c_j",
    "lowrank": "low-rank antisym.",
    "mlp": "antisym. MLP",
    "mlp_ordered": "ordered-residual MLP",
}

#: Columns that only a Phase 2 run row can have. Their presence means the file
#: is the wrong experiment.
_PHASE2_ONLY = ("family", "test_D_pearson", "test_head_A_over_sym")


def load_residual_runs(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    err = [r for r in rows if "error" in r]
    if err:
        raise RuntimeError(
            f"{len(err)} failed runs in {path}; first traceback:\n"
            f"{err[0]['error'][-1200:]}")
    df = pd.DataFrame(rows)
    clash = [c for c in _PHASE2_ONLY if c in df.columns]
    if clash:
        raise ValueError(
            f"{path} carries Phase 2 columns {clash}. That file measures a "
            f"different target ('can a family predict y') against a different "
            f"null. Pooling the two would average a raw-target metric with a "
            f"residual-target metric of the same name.")
    if "rung" not in df.columns:
        raise ValueError(f"{path} has no 'rung' column; not a residual run file")
    df["rung"] = pd.Categorical(df["rung"], categories=list(LADDER_ORDER),
                                ordered=True)
    return df


def assert_residual_grid_complete(df: pd.DataFrame, tag: str = "main") -> None:
    """Every (screen, coverage, rung) cell must have every split seed.

    A silently short cell would be averaged over fewer seeds than its neighbours
    and would still print an ``n`` -- the Phase 2 audit found exactly this shape
    of bug twice, so it is checked rather than eyeballed.
    """
    sub = df[df["tag"] == tag]
    if sub.empty:
        raise AssertionError(f"no rows tagged {tag!r}")
    seeds = sorted(sub["split_seed"].unique())
    holes = []
    for key, g in sub.groupby(["screen", "coverage", "rung"], observed=True):
        got = sorted(g["split_seed"].unique())
        if got != seeds or len(g) != len(seeds):
            holes.append((key, len(g), got))
    if holes:
        raise AssertionError(
            f"{len(holes)} incomplete cells in tag {tag!r}, e.g. {holes[:3]} "
            f"(expected {len(seeds)} split seeds {seeds})")


def by_split_seed(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One value per (screen, coverage, rung, split seed).

    Averages over anything else that varies -- today only ``init_seed``, which
    is fixed at 0 in the shipped grids. Kept general so that adding an extra
    initialisation later cannot silently double a cell's weight, which is the
    mistake the Phase 2 init-variance block made.
    """
    keys = ["screen", "coverage", "rung", "split_seed"]
    out = (df.groupby(keys, observed=True)[metric].mean().reset_index()
             .rename(columns={metric: "value"}))
    return out


def _ci(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    h = float(stats.t.ppf(0.975, len(x) - 1) * se)
    return (float(x.mean() - h), float(x.mean() + h))


def skill_summary(df: pd.DataFrame, metric: str = "cal_skill",
                  tag: str = "main") -> pd.DataFrame:
    """Per cell: mean skill over split seeds, with both tests reported.

    Both the t-test and the Wilcoxon signed-rank test are always reported, never
    whichever one is nicer. With n=8 seeds the Wilcoxon's smallest attainable
    two-sided p is 0.0078, so a cell can be unanimous and still not reach 0.001
    -- that is a property of the test at this n and not a weak result.
    """
    s = by_split_seed(df[df["tag"] == tag], metric)
    out = []
    for (screen, cov, rung), g in s.groupby(["screen", "coverage", "rung"],
                                            observed=True):
        raw = g["value"].to_numpy()
        # Correlation metrics are NaN for a constant predictor -- the zero rung
        # emits one by construction. Dropping the NaNs silently would leave a
        # cell reporting an n it does not have, so ``n_split_seeds`` counts the
        # values that survived and ``n_missing`` says how many did not.
        x = raw[~np.isnan(raw)]
        lo, hi = _ci(x) if len(x) else (float("nan"), float("nan"))
        row = {"screen": screen, "coverage": cov, "rung": rung, "metric": metric,
               "n_split_seeds": len(x), "n_missing": int(len(raw) - len(x)),
               "mean": float(x.mean()) if len(x) else float("nan"),
               "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
               "ci_lo": lo, "ci_hi": hi,
               "n_seeds_positive": int((x > 0).sum())}
        # The zero rung's skill is identically 0 by construction, so a
        # one-sample test against 0 is degenerate rather than significant.
        if rung == "zero" or len(x) < 2 or np.allclose(x, 0.0):
            row["p_ttest"] = float("nan")
            row["p_wilcoxon"] = float("nan")
        else:
            row["p_ttest"] = float(stats.ttest_1samp(x, 0.0).pvalue)
            try:
                row["p_wilcoxon"] = float(
                    stats.wilcoxon(x, alternative="two-sided").pvalue)
            except (ValueError, ZeroDivisionError):     # pragma: no cover
                row["p_wilcoxon"] = float("nan")
        row["flexible_diagnostic"] = rung in FLEXIBLE_RUNGS
        out.append(row)
    return pd.DataFrame(out).sort_values(["screen", "coverage", "rung"])


def paired_rung_delta(df: pd.DataFrame, metric: str, a: str, b: str,
                      tag: str = "main") -> pd.DataFrame:
    """Paired contrast between two rungs, split seed by split seed."""
    s = by_split_seed(df[df["tag"] == tag], metric)
    wide = s.pivot_table(index=["screen", "coverage", "split_seed"],
                         columns="rung", values="value", observed=True)
    out = []
    for (screen, cov), g in wide.groupby(level=[0, 1]):
        if a not in g or b not in g:
            continue
        d = (g[a] - g[b]).dropna().to_numpy()
        if len(d) < 2:
            continue
        lo, hi = _ci(d)
        out.append({"screen": screen, "coverage": cov, "metric": metric,
                    "rung_a": a, "rung_b": b, "n_split_seeds": len(d),
                    "mean_a": float(g[a].mean()), "mean_b": float(g[b].mean()),
                    "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                    "p_ttest": float(stats.ttest_1samp(d, 0.0).pvalue),
                    "n_seeds_favouring_a": int((d > 0).sum())})
    return pd.DataFrame(out)


DECOMP_COLS = ("dec_D_mean_square", "dec_D_res_mean_square",
               "dec_frac_D_removed_by_potential", "dec_D_res_std", "dec_D_std",
               "dec_D_res_mean_abs", "dec_D_pearson_add", "dec_additive_test_r2_y",
               "dec_additive_lambda", "dec_D_res_abs_q50", "dec_D_abs_q50",
               "dec_y_var", "dec_D_var", "dec_D_res_var", "dec_D_res_mean")


def decomposition_table(df: pd.DataFrame, tag: str = "main") -> pd.DataFrame:
    """The additive decomposition, averaged over split seeds.

    Read off the ``zero`` rung only. The decomposition is a property of the
    additive fit and the split, not of the rung, so it is identical across rungs
    within a split seed -- averaging over all five would report the same numbers
    with a fake 5x sample size. ``test_decomposition_is_rung_invariant`` checks
    the identity holds so this filter is safe rather than convenient.
    """
    sub = df[(df["tag"] == tag) & (df["rung"] == "zero")]
    if sub.empty:
        raise AssertionError("no zero-rung rows; the decomposition is read off "
                             "them so that it is counted once per split seed")
    cols = [c for c in DECOMP_COLS if c in sub.columns]
    g = sub.groupby(["screen", "coverage"], observed=True)
    out = g[cols].mean().reset_index()
    sd = g[cols].std(ddof=1).reset_index()
    for c in cols:
        out[c + "_sd"] = sd[c].to_numpy()
    out["n_split_seeds"] = g["split_seed"].nunique().reset_index(drop=True).to_numpy()
    return out


# ------------------------------------------------------------------ figures
def figures(df: pd.DataFrame, outdir: Path, power: pd.DataFrame | None = None,
            honest: pd.DataFrame | None = None,
            hodge: dict | None = None,
            primary_coverage: float = 0.70) -> list[Path]:
    """The four diagnostic figures. Nothing decorative.

    ``primary_coverage`` is where figure 3 is drawn, and it is an argument with
    a pre-registered default rather than a choice made after looking: it is the
    densest rung, the one with the most training pairs and therefore the most
    favourable case for finding residual structure. Picking the coverage that
    happened to look best would be choosing a result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _coverage_axis(ax, covs):
        """Log x with only the five real rungs labelled.

        Matplotlib's log locator adds its own minor ticks (5x10^-2, 3x10^-1 ...)
        which overprint the coverage labels and made the first version of these
        figures unreadable at 0.05.
        """
        ax.set_xscale("log")
        ax.set_xticks(sorted(covs))
        ax.set_xticklabels([f"{c:g}" for c in sorted(covs)])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("pair coverage")

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    main = df[df["tag"] == "main"]

    # --- Figure 1: raw vs residual directional scale, by screen
    dec = decomposition_table(main)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, screen in zip(axes, sorted(dec["screen"].unique())):
        d = dec[dec["screen"] == screen].sort_values("coverage")
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["dec_D_mean_square"], 0.4, label="raw  mean $D^2$",
               color="0.35")
        ax.bar(x + 0.2, d["dec_D_res_mean_square"], 0.4,
               label="residual  mean $D_{res}^2$", color="tab:red")
        if hodge and screen in hodge:
            # The exact in-sample cyclic mean square: what a *perfect* per-drug
            # potential leaves behind. The held-out residual approaches it from
            # above as coverage grows, which is the check that the train-only
            # residualisation is converging on the right quantity rather than on
            # whatever the ridge happened to shrink.
            ax.axhline(hodge[screen]["curl_mean_square"], color="tab:blue",
                       ls="--", lw=1.2,
                       label="exact cyclic floor (Hodge, in-sample)")
        for k, (_, r) in enumerate(d.iterrows()):
            ax.text(k, r["dec_D_mean_square"] * 1.02,
                    f"−{100 * r['dec_frac_D_removed_by_potential']:.0f}%",
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:g}" for c in d["coverage"]])
        ax.set_xlabel("pair coverage")
        ax.set_title(screen)
    axes[0].set_ylabel("held-out directional mean square")
    axes[0].set_ylim(0, float(dec["dec_D_mean_square"].max()) * 1.22)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Fig 1  What the per-drug potential $g_i-g_j$ removes "
                 "(held-out pairs, train-only fit)", fontsize=10)
    fig.tight_layout()
    p = outdir / "fig1_residual_directional_variance.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Figure 2: skill vs coverage, one line per rung
    s = skill_summary(main, "cal_skill")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, screen in zip(axes, sorted(s["screen"].unique())):
        d = s[s["screen"] == screen]
        for rung in LADDER_ORDER:
            r = d[d["rung"] == rung].sort_values("coverage")
            if r.empty:
                continue
            ax.errorbar(r["coverage"], r["mean"],
                        yerr=[r["mean"] - r["ci_lo"], r["ci_hi"] - r["mean"]],
                        marker="o", ms=4, capsize=3, label=NICE[rung],
                        ls="--" if rung in FLEXIBLE_RUNGS else "-")
        if honest is not None and not honest.empty:
            h = skill_summary(honest, "cal_skill", tag="honest_alpha")
            h = h[(h["screen"] == screen) & (h["rung"] == "lowrank")].sort_values("coverage")
            if not h.empty:
                ax.plot(h["coverage"], h["mean"], marker="D", ms=4, lw=1.2,
                        color="k", ls=":",
                        label="low-rank, shrinkage on held-out validation")
        ax.axhline(0.0, color="k", lw=1)
        _coverage_axis(ax, d["coverage"].unique())
        ax.set_title(screen)
    axes[0].set_ylabel("residual skill  $1-\\mathrm{MSE}/\\mathrm{MSE}_0$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 2  Can anything predict $D_{res}$ for an unseen pair? "
                 "(0 = tie with the zero predictor; dashed = flexible diagnostic)",
                 fontsize=10)
    fig.tight_layout()
    p = outdir / "fig2_residual_skill_vs_coverage.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Figure 3: predicted vs measured D_res at the pre-registered coverage
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, rung in zip(axes, ("lowrank", "mlp")):
        sub = main[(main["screen"] == "A375") & (main["coverage"] == primary_coverage)
                   & (main["rung"] == rung)]
        if sub.empty:
            continue
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.8)
        # Per-seed summary rather than a scatter of pairs: individual pairs are
        # not independent replicates and a 6,784-point cloud would suggest they
        # are. Each marker is one split seed.
        ax.scatter(sub["heldout_pearson"], sub["cal_skill"], s=28, color="tab:red")
        ax.set_xlabel("held-out Pearson r($D_{res}$, $\\hat{D}_{res}$)")
        ax.set_ylabel("calibrated residual skill")
        ax.set_title(f"{NICE[rung]}  (A375, coverage {primary_coverage:g}, "
                     f"one point per split seed)", fontsize=9)
    fig.suptitle("Fig 3  Correlation and skill are not the same claim",
                 fontsize=10)
    fig.tight_layout()
    p = outdir / "fig3_predicted_vs_measured.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Figure 4: raw-direction vs residual-direction predictability
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for screen, marker in (("A375", "o"), ("PANC1", "s")):
        rawr = by_split_seed(main[(main["screen"] == screen)
                                  & (main["rung"] == "lowrank")], "raw_pearson")
        resr = by_split_seed(main[(main["screen"] == screen)
                                  & (main["rung"] == "lowrank")], "cal_pearson")
        m1 = rawr.groupby("coverage", observed=True)["value"].mean()
        m2 = resr.groupby("coverage", observed=True)["value"].mean()
        ax.plot(m1.index, m1.to_numpy(), marker=marker, color="0.35",
                label=f"{screen}: raw $D$ (potential + model)")
        ax.plot(m2.index, m2.to_numpy(), marker=marker, color="tab:red",
                label=f"{screen}: residual $D_{{res}}$")
    ax.axhline(0, color="k", lw=1)
    _coverage_axis(ax, main["coverage"].unique())
    ax.set_ylabel("held-out Pearson r")
    ax.legend(fontsize=8)
    ax.set_title("Fig 4  How much of the predictable direction is the per-drug\n"
                 "potential rather than the pair (low-rank rung)", fontsize=10)
    fig.tight_layout()
    p = outdir / "fig4_raw_vs_residual_correlation.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Figure 5 (only when the positive control was run): detection power
    if power is not None and not power.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for tag, colour in zip(sorted(power["tag"].unique()),
                               ("tab:blue", "tab:green", "tab:purple")):
            for rung, ls in (("lowrank", "-"), ("mlp", "--")):
                d = by_split_seed(power[(power["tag"] == tag)
                                        & (power["rung"] == rung)], "cal_skill")
                if d.empty:
                    continue
                m = d.groupby("coverage", observed=True)["value"].mean()
                ax.plot(m.index, m.to_numpy(), ls, marker="o", ms=4,
                        color=colour, label=f"{tag} · {NICE[rung]}")
        ax.axhline(0, color="k", lw=1)
        _coverage_axis(ax, power["coverage"].unique())
        ax.set_ylabel("recovered skill on injected signal")
        ax.legend(fontsize=7)
        ax.set_title("Fig 5  Positive control: skill recovered when a known\n"
                     "antisymmetric signal IS present (A375)", fontsize=10)
        fig.tight_layout()
        p = outdir / "fig5_power_positive_control.png"
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


# --------------------------------------------------- generated doc tables
#: Markdown blocks emitted by ``scripts/report_phase2_residual.py`` into
#: ``summary/doc_tables.md`` and required to appear **verbatim** in
#: ``docs/phase2_residual_directionality.md``.
#:
#: This exists because of what an audit of the finished writeup found: 31 of its
#: findings were transcription errors and scope overstatements in tables that
#: had been hand-copied from a terminal. Four p-values in the table carrying the
#: pre-registered primary contrast did not reproduce, two of them matching no
#: run in the repository; a Pearson column labelled as coming from the corrected
#: estimator came from the uncorrected one. None of it changed the decision, and
#: all of it was avoidable: a number a human retypes is a number nobody checks.
#: So the load-bearing tables are generated here and pinned by
#: ``test_document_tables_are_generated_not_transcribed``.
DOC_TABLE_MARKER = "<!-- generated: {name} -->"


def _md(rows: list[list[str]], header: list[str], align: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def primary_contrast_table(df: pd.DataFrame, tag: str = "main",
                           metric: str = "cal_skill",
                           a: str = "lowrank", b: str = "potential") -> str:
    """The pre-registered contrast, formatted exactly as the writeup prints it."""
    s = by_split_seed(df[df["tag"] == tag], metric)
    wide = s.pivot_table(index=["screen", "coverage", "split_seed"],
                         columns="rung", values="value", observed=True)
    rows = []
    for (screen, cov), g in wide.groupby(level=[0, 1]):
        d = (g[a] - g[b]).dropna().to_numpy()
        lo, hi = _ci(d)
        pw = float(stats.wilcoxon(d).pvalue)
        rows.append([screen, f"{cov:g}", f"{d.mean():+.3f}",
                     f"[{lo:+.3f},{hi:+.3f}]",
                     f"{stats.ttest_1samp(d, 0.0).pvalue:.1e}",
                     f"{pw:.4f}", f"{int((d > 0).sum())}/{len(d)}"])
    return _md(rows,
               ["screen", "coverage", "Δ", "95% CI", "p (t)", "p (Wilcoxon)",
                "seeds favouring low-rank"],
               ["---", "---:", "---:", "---", "---:", "---:", "---:"])


def rung_metric_table(df: pd.DataFrame, rung: str, tag: str = "main",
                      metrics=("cal_skill", "cal_pearson", "cal_sign_accuracy")
                      ) -> str:
    """One row per (screen, coverage) for a single rung, several metrics."""
    parts = {mname: skill_summary(df, mname, tag=tag) for mname in metrics}
    base = parts[metrics[0]]
    base = base[base["rung"] == rung]
    rows = []
    for _, r in base.sort_values(["screen", "coverage"]).iterrows():
        cells = [r["screen"], f"{r['coverage']:g}",
                 f"{r['mean']:+.3f}", f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]",
                 f"{r['n_seeds_positive']}/{r['n_split_seeds']}"]
        for mname in metrics[1:]:
            q = parts[mname]
            q = q[(q["rung"] == rung) & (q["screen"] == r["screen"])
                  & (q["coverage"] == r["coverage"])]
            cells.append(f"{q['mean'].iloc[0]:.3f}" if len(q) else "—")
        rows.append(cells)
    return _md(rows,
               ["screen", "coverage", "skill", "95% CI", "seeds > 0",
                "Pearson r", "sign acc."],
               ["---", "---:", "---:", "---", "---:", "---:", "---:"])


def doc_tables(main: pd.DataFrame, honest: pd.DataFrame | None,
               power: pd.DataFrame | None) -> dict[str, str]:
    out = {"primary_contrast_as_run": primary_contrast_table(main)}
    if honest is not None and not honest.empty:
        out["primary_contrast_honest_alpha"] = primary_contrast_table(
            honest, tag="honest_alpha")
        out["lowrank_honest_alpha"] = rung_metric_table(
            honest, "lowrank", tag="honest_alpha")
    if power is not None and not power.empty:
        parts = []
        for tag in sorted(power["tag"].unique()):
            s = skill_summary(power, "cal_skill", tag=tag)
            s = s.assign(block=tag)
            parts.append(s)
        allp = pd.concat(parts, ignore_index=True)
        rows = []
        for _, r in allp.sort_values(["block", "screen", "rung", "coverage"]).iterrows():
            rows.append([r["block"], r["screen"], str(r["rung"]), f"{r['coverage']:g}",
                         f"{r['mean']:+.3f}",
                         f"{r['n_seeds_positive']}/{r['n_split_seeds']}"])
        out["power"] = _md(rows,
                           ["block", "screen", "rung", "coverage",
                            "recovered skill", "seeds > 0"],
                           ["---", "---", "---", "---:", "---:", "---:"])
    return out
