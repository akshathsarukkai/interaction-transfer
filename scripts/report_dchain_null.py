"""Tables, figures and the preregistered verdict for the d-chain null.

    python scripts/report_dchain_null.py

Reads ``results/dchain_null/metrics.jsonl`` and the committed Phase 2R
artifacts, and writes ``results/dchain_null/summary/``:

    comparison.csv        null distribution beside both real screens
    mechanism.csv         the zero-free-parameter test of the predicted artifact
    per_screen.csv        one row per simulated screen, block and coverage
    structure.csv         matrix-structure metrics, no model involved
    controls.csv          oracle / unshared / noise-sweep blocks
    verdict.json          the decision rule, executed, with its inputs
    doc_tables.md         the markdown tables that must appear verbatim in
                          docs/dchain_null_falsification.md
    fig1_curl_fraction.png
    fig2_skill_at_070.png
    fig3_coverage_curves.png
    fig4_curl_spectrum.png
    fig5_artifact_matrix.png
    fig6_mechanism.png    the zero-free-parameter test of the predicted artifact

The verdict is computed, not typed. Every real value is read from
``results/phase2_residual/summary/`` on each run, never transcribed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from intervention_algebra.real_data.dchain_null import report as R  # noqa: E402

OUT = R.OUT_DIR / "summary"
REAL_COLOURS = {"A375": "#c0392b", "PANC1": "#2471a3"}


def _fmt(x, nd=3, signed=False):
    """Format a number for a markdown table.

    ``signed`` only where a sign carries meaning. Skill is signed -- "did the
    rung beat predicting nothing" is the whole question -- while a fraction of
    energy, a correlation floor or a spread is not, and writing "+0.970" for a
    cyclic fraction reads as a change rather than a level.
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def table_comparison(cmp_df: pd.DataFrame, kind: str, title: str) -> str:
    sel = cmp_df[cmp_df["kind"] == kind]
    tag = "comparison_decision" if kind == "decision" else "comparison_descriptive"
    head = ("| metric | null median | null 95% interval | null max | real A375 | "
            "real PANC1 | real percentile under null | p (one-sided) |")
    rule = "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
    if kind != "decision":
        head = ("| metric | null median | null 95% interval | real A375 | "
                "real PANC1 |")
        rule = "| --- | ---: | --- | ---: | ---: |"
    lines = [f"<!-- generated: {tag} -->", f"**{title}**", "", head, rule]
    for _, r in sel.iterrows():
        sgn = "skill" in r["metric"]
        if kind == "decision":
            pct = ("—" if not np.isfinite(r["pct_A375"])
                   else f"{r['pct_A375']:.0f}% / {r['pct_PANC1']:.0f}%")
            pv = ("—" if not np.isfinite(r["p_A375"])
                  else f"{r['p_A375']:.3f} / {r['p_PANC1']:.3f}")
            lines.append(
                f"| {r['metric']} | {_fmt(r['null_median'], signed=sgn)} | "
                f"[{_fmt(r['null_q025'], signed=sgn)}, {_fmt(r['null_q975'], signed=sgn)}] | "
                f"{_fmt(r['null_max'], signed=sgn)} | {_fmt(r['real_A375'], signed=sgn)} | "
                f"{_fmt(r['real_PANC1'], signed=sgn)} | {pct} | {pv} |")
        else:
            lines.append(
                f"| {r['metric']} | {_fmt(r['null_median'], signed=sgn)} | "
                f"[{_fmt(r['null_q025'], signed=sgn)}, {_fmt(r['null_q975'], signed=sgn)}] | "
                f"{_fmt(r['real_A375'], signed=sgn)} | {_fmt(r['real_PANC1'], signed=sgn)} |")
    return "\n".join(lines)


def table_mechanism(mech: pd.DataFrame) -> str:
    lines = ["<!-- generated: mechanism -->",
             "| block | n | offset error ε (RMS) | gain m̃ mean | m̃ sd | "
             "template R² | subspace overlap | split-half r(D) | selector on |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in mech.iterrows():
        if not np.isfinite(r.get("offset_error_rms", np.nan)):
            continue
        lines.append(
            f"| `{r['block']}` | {int(r['n'])} | {r['offset_error_rms']:.4f} | "
            f"{r['second_position_gain_mean']:.3f} | "
            f"{r['second_position_gain_sd']:.3f} | "
            f"{_fmt(r['template_r2'])} | {_fmt(r['template_subspace_overlap'])} | "
            f"{_fmt(r['split_half_pearson_D'])} | "
            f"{_fmt(r['selector_on_fraction'])} |")
    return "\n".join(lines)


def table_controls(st: pd.DataFrame, ps: pd.DataFrame, primary: str) -> str:
    """Every block except the experiment itself. Controls A, C and D."""
    st = st[st["tag"] != primary]
    lines = ["<!-- generated: controls -->",
             "| block | n | true pair effect | est. synergy RMS | curl fraction | "
             "top-2 curl energy | rank-2 skill @ 0.70 |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for tag, g in st.groupby("tag", observed=True):
        sel = ps[(ps["tag"] == tag) & (ps["block"] == "rank2")
                 & (np.isclose(ps["coverage"], 0.70))]
        truth = ("zero" if bool(g["true_pair_interaction_is_zero"].all())
                 else f"independent, RMS {g['true_synergy_rms'].median():.4f}")
        lines.append(
            f"| `{tag}` | {len(g)} | {truth} | "
            f"{g['est_synergy_rms'].median():.4f} | "
            f"{_fmt(g['curl_fraction'].median())} | "
            f"{_fmt(g['top2'].median())} | "
            f"{_fmt(sel['cal_skill'].median(), signed=True) if len(sel) else '—'} |")
    return "\n".join(lines)


def fig_null_vs_real(values, real, title, xlabel, path, extra=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(values):
        ax.hist(values, bins=max(6, min(20, len(values) // 2)),
                color="#7f8c8d", alpha=0.75, edgecolor="white",
                label=f"null ensemble (n={len(values)})")
        ax.axvline(float(np.median(values)), color="#2c3e50", ls="--", lw=1.4,
                   label=f"null median {np.median(values):+.3f}")
    for screen, v in real.items():
        if v is not None and np.isfinite(v):
            ax.axvline(v, color=REAL_COLOURS[screen], lw=2.4,
                       label=f"real {screen} {v:+.3f}")
    if extra:
        for label, v, c in extra:
            ax.axvline(v, color=c, lw=1.2, ls=":", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("simulated screens")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_coverage(ps: pd.DataFrame, real: dict, path: Path, tag="primary"):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sel = ps[(ps["tag"] == tag) & (ps["block"] == "rank2")]
    if len(sel):
        g = sel.groupby("coverage", observed=True)["cal_skill"]
        cov = np.array(sorted(sel["coverage"].unique()))
        med = g.median().reindex(cov).to_numpy()
        lo = g.quantile(0.025).reindex(cov).to_numpy()
        hi = g.quantile(0.975).reindex(cov).to_numpy()
        ax.fill_between(cov, lo, hi, color="#95a5a6", alpha=0.35,
                        label="null 95% interval")
        ax.plot(cov, med, "o-", color="#2c3e50", label="null median")
    for screen, r in real.items():
        c = sorted(r["rank2_skill"])
        ax.plot(c, [r["rank2_skill"][k] for k in c], "s-",
                color=REAL_COLOURS[screen], label=f"real {screen}")
    ax.axhline(0.0, color="black", lw=0.9, zorder=1)
    ax.set_xscale("log")
    # Label the coverages that were actually run. Matplotlib's default log
    # ticks render these as "6 x 10^-2", which is unreadable for five values a
    # reader needs to match against the Phase 2R tables.
    ticks = sorted({float(c) for c in sel["coverage"].unique()} if len(sel)
                   else set(real["A375"]["rank2_skill"]))
    ax.set_xticks(ticks, minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([f"{c:g}" for c in ticks])
    ax.set_xlabel("pair coverage (fraction of the 4,950 unordered pairs trained on)")
    ax.set_ylabel("held-out residual skill, rank-2 rung")
    ax.set_title("Fixed rank-2 detector: real screens vs the zero-interaction null",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_spectrum(st: pd.DataFrame, real: dict, path: Path, tag="primary"):
    ks = [1, 2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sel = st[st["tag"] == tag]
    if len(sel):
        med = [sel[f"top{k}"].median() for k in ks]
        lo = [sel[f"top{k}"].quantile(0.025) for k in ks]
        hi = [sel[f"top{k}"].quantile(0.975) for k in ks]
        ax.fill_between(ks, lo, hi, color="#95a5a6", alpha=0.35,
                        label="null 95% interval")
        ax.plot(ks, med, "o-", color="#2c3e50", label="null median")
    for screen, r in real.items():
        ax.plot(ks, [r["top_k_energy"][str(k)] for k in ks], "s-",
                color=REAL_COLOURS[screen], label=f"real {screen}")
    ax.axhline(R.DECISION["noise_floor_top2"], color="#8e44ad", ls=":", lw=1.2,
               label=f"i.i.d. noise, top 2 ({R.DECISION['noise_floor_top2']:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (leading singular directions of the cyclic part)")
    ax.set_ylabel("fraction of cyclic energy in the top k")
    ax.set_title("How concentrated the cyclic component is", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_artifact(st: pd.DataFrame, path: Path):
    """Control B: what the estimator added, by block."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    tags, vals = [], []
    for tag, g in st.groupby("tag", observed=True):
        v = g["artifact_rms"].dropna().to_numpy()
        if len(v):
            tags.append(tag)
            vals.append(v)
    if vals:
        ax.boxplot(vals, tick_labels=tags, showfliers=False)
        for k, v in enumerate(vals, 1):
            ax.plot(np.full(len(v), k) + np.linspace(-.08, .08, len(v)), v,
                    ".", color="#2980b9", ms=4, alpha=0.8)
    ax.set_ylabel("RMS of (estimate − truth), synergy units")
    ax.set_title("Control B: the size of what the estimator added, by block",
                 fontsize=10)
    ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def table_verdict(v: dict) -> str:
    """The decision rule's own output, as a table. Generated, never typed.

    Phase 2R's audit found four hand-copied p-values in its write-up. The
    verdict is the single most consequential number this experiment produces, so
    it is emitted here and pinned into the document by
    ``test_the_document_states_the_verdict_the_rule_computed``.
    """
    lines = ["<!-- generated: verdict -->",
             f"**Verdict: {v['verdict']}**", "",
             f"Computed by `report.verdict()` from {v['n_runs']} usable runs of "
             f"the {v['tag']} block "
             f"({v.get('n_failed', 0)} failed, {v.get('n_incomplete', 0)} "
             f"incomplete and excluded under the preregistered rule).", ""]
    if v.get("criterion_D_reasons"):
        lines += ["Criterion D triggers that fired:", ""]
        lines += [f"* {r}" for r in v["criterion_D_reasons"]] + [""]
    crit = v.get("criteria")
    if crit:
        lines += ["| clause | value |", "| --- | :--: |"]
        label = {"artifact_skill": "null median skill ≥ half the weaker real screen, both coverages",
                 "artifact_spectral": "null median rank-2 share of D² ≥ half the real mean",
                 "real_inside_null_95": "a real value lies inside the null 95% interval at coverage 0.70",
                 "partial": "null median clearly positive at coverage 0.70",
                 "little": "null below the positive threshold, real above the null maximum, spectral below",
                 "null_skill_crushed": "null median ≤ 0 at both coverages and real above the null maximum"}
        for k, val in crit.items():
            lines.append(f"| {label.get(k, k)} | {'**yes**' if val else 'no'} |")
        lines.append("")
    sk = v.get("skill", {})
    if sk:
        lines += ["| coverage | null median | null max | null 95% | real A375 | real PANC1 | "
                  "artifact threshold | p (A375 / PANC1) |",
                  "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
        for key in sorted(sk):
            c = sk[key]
            lines.append(
                f"| {c['coverage']:.2f} | {_fmt(c['null_median'], signed=True)} | "
                f"{_fmt(c['null_max'], signed=True)} | "
                f"[{_fmt(c['null_q025'], signed=True)}, {_fmt(c['null_q975'], signed=True)}] | "
                f"{_fmt(c['real_A375'], signed=True)} | {_fmt(c['real_PANC1'], signed=True)} | "
                f"{_fmt(c['artifact_threshold'], signed=True)} | "
                f"{_fmt(c['p_A375'])} / {_fmt(c['p_PANC1'])} |")
        lines.append("")
    for name, block in (("rank-2 share of D²", v.get("rank2_share_of_D")),
                        ("rank-2 cyclic energy, absolute", v.get("rank2_energy_absolute"))):
        if not block:
            continue
        lines.append(f"*{name}:* null median "
                     f"{_fmt(block.get('null_median'), 5)}, "
                     f"real {_fmt(block.get('real_A375', block.get('real_mean')), 5)} (A375) / "
                     f"{_fmt(block.get('real_PANC1', block.get('real_mean')), 5)} (PANC1).")
    for k, lbl in (("selector_on_fraction_median", "combination selector on-fraction"),
                   ("split_half_pearson_D_median", "split-half r(D)"),
                   ("posterior_noise_fraction_median", "posterior noise fraction of D"),
                   ("oracle_max_skill", "Control A: maximum oracle rank-2 skill")):
        if k in v:
            lines.append(f"*{lbl}:* {_fmt(v[k], 4)}")
    return "\n".join(lines)


def fig_mechanism(st: pd.DataFrame, path: Path, tag="primary"):
    """The zero-free-parameter test: is what is there the predicted artifact?

    Left: the size of the shared first-position offset error, which is the
    factor that carries the artifact's magnitude. Right: how much of the
    estimated cyclic component the fully-specified template ``eps ^ mtilde``
    explains, against the subspace overlap. Both are reported and neither is in
    the decision rule -- the verdict is about size, this is about shape.
    """
    sel = st[(st["tag"] == tag) & st["offset_error_rms"].notna()]
    if sel.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    axes[0].hist(sel["offset_error_rms"], bins=max(5, len(sel) // 2),
                 color="#7f8c8d", edgecolor="white")
    axes[0].set_xlabel("RMS of ε = û − u  (log viability units)")
    axes[0].set_ylabel("simulated screens")
    axes[0].set_title("How badly the shared first-position" + "\n"
                      + "offset is estimated", fontsize=9)
    axes[1].scatter(sel["template_r2"], sel["template_subspace_overlap"],
                    s=26, color="#2980b9", alpha=0.85)
    axes[1].axvline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("R² of the estimated curl on curl(ε ∧ m̃)")
    axes[1].set_ylabel("subspace overlap")
    axes[1].set_title("Does the cyclic component have" + "\n"
                      + "the predicted shape?", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()

    rows = R.load_rows()
    real = R.real_reference()
    OUT.mkdir(parents=True, exist_ok=True)

    ps = R.per_screen(rows)
    st = R.structure_frame(rows)
    ps.to_csv(OUT / "per_screen.csv", index=False)
    st.to_csv(OUT / "structure.csv", index=False)
    st[st["tag"] != args.tag].to_csv(OUT / "controls.csv", index=False)

    have_primary = (st["tag"] == args.tag).any()
    if have_primary:
        cmp_df = R.comparison_table(rows, real, tag=args.tag)
        cmp_df.to_csv(OUT / "comparison.csv", index=False)
        v = R.verdict(rows, real, tag=args.tag)
        (OUT / "verdict.json").write_text(json.dumps(v, indent=2, default=float) + "\n")
        print(f"VERDICT: {v['verdict']}")
    else:
        cmp_df = pd.DataFrame()
        v = {"verdict": "not yet run", "tag": args.tag}
        # Delete rather than leave. A stale comparison.csv or verdict.json from
        # an earlier run -- or, as happened here, from a dry run on fabricated
        # rows -- is a results file that no longer corresponds to any metrics
        # file, and nothing downstream can tell.
        for stale in ("comparison.csv", "verdict.json"):
            (OUT / stale).unlink(missing_ok=True)
        print(f"no rows tagged {args.tag!r} yet; controls only, and any stale "
              f"comparison/verdict removed")

    mech = R.mechanism_table(rows, tag=args.tag)
    mech.to_csv(OUT / "mechanism.csv", index=False)
    blocks = [table_controls(st, ps, args.tag), table_mechanism(mech)]
    if have_primary:
        blocks.insert(0, table_verdict(v))
    if have_primary:
        blocks.insert(0, table_comparison(
            cmp_df, "descriptive",
            "Descriptive — how the two worlds compare. No p-values: these are "
            "not discriminators, and the pre-registration says why."))
        blocks.insert(0, table_comparison(
            cmp_df, "decision",
            "The decision statistics. The null unit is one simulated screen; "
            "the minimum reportable one-sided p is 1/(n+1)."))
    (OUT / "doc_tables.md").write_text("\n\n".join(blocks) + "\n")

    if have_primary:
        prim = st[st["tag"] == args.tag]
        fig_null_vs_real(
            prim["curl_fraction"], {k: real[k]["curl_fraction"] for k in real},
            "Figure 1 — cyclic fraction. Note the i.i.d.-noise floor: at n=100 "
            "any unstructured\nmatrix sits near 0.98, so this statistic does not "
            "discriminate. Reported, not decisive.",
            "fraction of directional energy that no per-drug potential can express",
            OUT / "fig1_curl_fraction.png",
            extra=[("i.i.d. noise at n=100 (0.980)", 0.980, "#8e44ad")])
        sel = ps[(ps["tag"] == args.tag) & (ps["block"] == "rank2")
                 & (np.isclose(ps["coverage"], 0.70))]
        fig_null_vs_real(
            sel["cal_skill"],
            {k: real[k]["rank2_skill"].get(0.70) for k in real},
            "Figure 2 — held-out residual skill of the fixed rank-2 detector at "
            "coverage 0.70.\nThe primary comparison: unstructured noise cannot "
            "produce positive skill.",
            "held-out residual skill (1 − MSE/MSE_zero)",
            OUT / "fig2_skill_at_070.png",
            extra=[("zero", 0.0, "black")])
        fig_coverage(ps, real, OUT / "fig3_coverage_curves.png", tag=args.tag)
        fig_spectrum(st, real, OUT / "fig4_curl_spectrum.png", tag=args.tag)
    fig_artifact(st, OUT / "fig5_artifact_matrix.png")
    if fig_mechanism(st, OUT / "fig6_mechanism.png", tag=args.tag):
        print("wrote fig6_mechanism.png")

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
