"""Tables and figures for the residual-directionality diagnostic.

    python scripts/report_phase2_residual.py

Reads ``results/phase2_residual/*.jsonl`` and writes machine-readable summaries
and figures into ``results/phase2_residual/summary/``. Prints the decomposition,
the skill ladder, the pre-registered primary contrast and the two controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.residual import hodge_decomposition
from intervention_algebra.real_data.residual_models import LADDER_ORDER
from intervention_algebra.real_data.residual_report import (
    DOC_TABLE_MARKER, NICE, assert_residual_grid_complete, by_split_seed,
    decomposition_table, doc_tables, figures, load_residual_runs,
    paired_rung_delta, skill_summary)

RESULTS = Path("results/phase2_residual")

#: The comparison the decision rests on, fixed in docs/PREREGISTRATIONS.md before
#: the grid was run. ``lowrank`` contains ``c_i - c_j`` exactly, so beating
#: ``zero`` is not evidence of pair-specific structure; beating ``potential`` is.
PRIMARY = {"metric": "cal_skill", "rung": "lowrank",
           "contrast": ("lowrank", "potential")}


def _fmt(df: pd.DataFrame, cols: dict[str, str], floatfmt: str = "{:.4f}") -> str:
    head = "| " + " | ".join(cols.values()) + " |"
    rule = "| " + " | ".join("---:" if k != list(cols)[0] else "---"
                             for k in cols) + " |"
    lines = [head, rule]
    for _, r in df.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--raw", type=Path, default=koplev.DEFAULT_RAW_DIR)
    args = ap.parse_args()
    out = args.results / "summary"
    out.mkdir(parents=True, exist_ok=True)

    main_df = load_residual_runs(args.results / "runs.jsonl")
    assert_residual_grid_complete(main_df, "main")
    optional = {}
    for name, fn in (("controls", "controls.jsonl"),
                     ("power", "power.jsonl"),
                     ("sensitivity", "sensitivity.jsonl"),
                     ("honest_alpha", "honest_alpha.jsonl"),
                     ("rank2", "rank2.jsonl"),
                     ("titration", "ridge_titration.jsonl")):
        p = args.results / fn
        if p.exists():
            optional[name] = load_residual_runs(p)

    # ---------------------------------------- model-free, no split involved
    hodge = {}
    for screen in sorted(main_df["screen"].unique()):
        try:
            hodge[screen] = hodge_decomposition(
                koplev.load_screen(screen, args.raw).__dict__["frame"],
                int(main_df[main_df["screen"] == screen]["n_drugs"].iloc[0]))
        except FileNotFoundError:
            pass
    if hodge:
        (out / "hodge_decomposition.json").write_text(
            json.dumps(hodge, indent=2) + "\n")
        print("\n## Gradient / curl split of the measured directional effect "
              "(exact, in-sample, no model)\n")
        print("| screen | mean D² | potential part | cyclic part | sd(cyclic) | "
              "curl energy in top 4 / 16 singular dirs |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for sc, h in hodge.items():
            e = h["curl_rank_energy"]
            print(f"| {sc} | {h['D_mean_square']:.4f} | "
                  f"{100 * h['grad_fraction']:.1f}% | "
                  f"{100 * h['curl_fraction']:.1f}% | "
                  f"{h['curl_std_offdiag']:.4f} | "
                  f"{100 * e['4']:.0f}% / {100 * e['16']:.0f}% |")

    # ------------------------------------------------------------- tables
    dec = decomposition_table(main_df)
    dec.to_csv(out / "decomposition.csv", index=False)
    print("\n## Additive decomposition on held-out pairs (train-only fit)\n")
    print(_fmt(dec.assign(cov=dec["coverage"]), {
        "screen": "screen", "cov": "coverage",
        "dec_D_mean_square": "mean D²", "dec_D_res_mean_square": "mean D_res²",
        "dec_frac_D_removed_by_potential": "removed by g_i−g_j",
        "dec_D_res_std": "sd(D_res)", "dec_D_std": "sd(D)",
        "dec_additive_test_r2_y": "additive R²(y)"}))

    frames = {}
    for metric in ("cal_skill", "heldout_skill", "cal_pearson", "cal_spearman",
                   "cal_sign_accuracy", "cal_rmse", "cal_mae"):
        s = skill_summary(main_df, metric)
        s.to_csv(out / f"summary_{metric}.csv", index=False)
        frames[metric] = s

    print("\n## Residual skill vs the zero predictor (mean over 8 split seeds)\n")
    for screen in sorted(main_df["screen"].unique()):
        print(f"\n### {screen}\n")
        s = frames["cal_skill"]
        s = s[s["screen"] == screen]
        rows = []
        for cov in sorted(s["coverage"].unique()):
            cell = {"coverage": f"{cov:g}"}
            for rung in LADDER_ORDER:
                r = s[(s["coverage"] == cov) & (s["rung"] == rung)]
                if r.empty:
                    cell[rung] = "—"
                    continue
                r = r.iloc[0]
                cell[rung] = (f"{r['mean']:+.3f} [{r['ci_lo']:+.3f},"
                              f"{r['ci_hi']:+.3f}] {r['n_seeds_positive']}/8")
            rows.append(cell)
        t = pd.DataFrame(rows)
        print("| " + " | ".join(["coverage"] + [NICE[r] for r in LADDER_ORDER]) + " |")
        print("| " + " | ".join(["---"] * (1 + len(LADDER_ORDER))) + " |")
        for _, r in t.iterrows():
            print("| " + " | ".join(str(r[c]) for c in
                                    ["coverage"] + list(LADDER_ORDER)) + " |")

    # ------------------------------------------- the pre-registered contrast
    a, b = PRIMARY["contrast"]
    prim = paired_rung_delta(main_df, PRIMARY["metric"], a, b)
    prim.to_csv(out / "primary_contrast.csv", index=False)
    print(f"\n## PRIMARY (pre-registered): {a} − {b}, metric {PRIMARY['metric']}\n")
    print(_fmt(prim, {"screen": "screen", "coverage": "coverage",
                      "mean_a": f"{a}", "mean_b": f"{b}", "delta": "Δ",
                      "ci_lo": "CI lo", "ci_hi": "CI hi", "p_ttest": "p (t)",
                      "n_seeds_favouring_a": "seeds"}))
    # Wilcoxon on the same paired differences, always reported alongside.
    wil = []
    from scipy import stats as _st
    s = by_split_seed(main_df[main_df["tag"] == "main"], PRIMARY["metric"])
    wide = s.pivot_table(index=["screen", "coverage", "split_seed"],
                         columns="rung", values="value", observed=True)
    for (screen, cov), g in wide.groupby(level=[0, 1]):
        d = (g[a] - g[b]).dropna().to_numpy()
        wil.append({"screen": screen, "coverage": cov,
                    "p_wilcoxon": float(_st.wilcoxon(d).pvalue)})
    pd.DataFrame(wil).to_csv(out / "primary_contrast_wilcoxon.csv", index=False)
    print("\n" + _fmt(pd.DataFrame(wil), {"screen": "screen",
                                          "coverage": "coverage",
                                          "p_wilcoxon": "p (Wilcoxon)"}))

    # ------------------------------------------------------------ controls
    for name in ("controls", "power", "sensitivity", "honest_alpha", "rank2",
                 "titration"):
        if name not in optional:
            continue
        df = optional[name]
        tags = sorted(df["tag"].unique())
        parts = []
        for tag in tags:
            s = skill_summary(df, "cal_skill", tag=tag)
            s.insert(0, "block", tag)
            parts.append(s)
        allb = pd.concat(parts, ignore_index=True)
        allb.to_csv(out / f"{name}.csv", index=False)
        print(f"\n## {name}\n")
        print(_fmt(allb, {"block": "block", "screen": "screen",
                          "coverage": "coverage", "rung": "rung",
                          "mean": "mean skill", "ci_lo": "CI lo",
                          "ci_hi": "CI hi", "n_seeds_positive": "seeds>0"}))

    contaminated = args.results / "contaminated_diagnostic.jsonl"
    if contaminated.exists():
        c = load_residual_runs(contaminated)
        clean = main_df[(main_df["coverage"] == 0.10)
                        & (main_df["screen"] == "A375")
                        & (main_df["split_seed"].isin(c["split_seed"].unique()))]
        print("\n## Control C — additive baseline deliberately fitted on the "
              "held-out pairs (NOT a result)\n")
        print(f"  fraction of held-out directional signal removed: "
              f"clean {clean['dec_frac_D_removed_by_potential'].mean():.3f} -> "
              f"contaminated {c['dec_frac_D_removed_by_potential'].mean():.3f}")
        print(f"  skill of the lowrank rung on the contaminated residual: "
              f"{c[c['rung'] == 'lowrank']['cal_skill'].mean():+.3f}")

    tables = doc_tables(main_df, optional.get("honest_alpha"),
                        optional.get("power"))
    (out / "doc_tables.md").write_text("\n\n".join(
        DOC_TABLE_MARKER.format(name=k) + "\n" + v
        for k, v in tables.items()) + "\n")
    print(f"\nwrote {len(tables)} generated doc tables to {out / 'doc_tables.md'}")

    paths = figures(main_df, out, power=optional.get("power"),
                    honest=optional.get("honest_alpha"), hodge=hodge)
    payload = {
        "primary": PRIMARY,
        "n_runs": {k: int(len(v)) for k, v in
                   {"main": main_df, **optional}.items()},
        "decomposition": json.loads(dec.to_json(orient="records")),
        "skill": json.loads(frames["cal_skill"].to_json(orient="records")),
        "primary_contrast": json.loads(prim.to_json(orient="records")),
        "primary_contrast_wilcoxon": wil,
        "hodge": hodge,
    }
    (out / "residual_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    main_df.drop(columns=["config", "grid_val_losses"], errors="ignore") \
           .to_csv(out / "runs.csv", index=False)
    print(f"\nwrote {len(paths)} figures and the summary tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
