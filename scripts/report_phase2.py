"""Tables and figures for the Phase 2 real-data experiment.

    python scripts/report_phase2.py

Reads ``results/phase2/runs.jsonl`` and writes machine-readable results plus the
four figures into ``results/phase2/summary/``. Prints the headline table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.report import (FAMILY_ORDER, NICE,
                                                   by_split_seed, collapse_report,
                                                   distribution_figure, figures,
                                                   load_runs, main_table,
                                                   paired_delta)

METRICS = ("test_mse", "test_mae", "test_r2", "test_D_pearson", "test_D_spearman",
           "test_D_rmse", "test_ordering_accuracy", "test_ordering_balanced_accuracy",
           "test_pred_A_rms_ratio", "test_head_A_over_sym",
           "test_first_order_A_rms_std_units")

#: The one comparison the conclusion is allowed to rest on, fixed before the
#: grid was run (docs/phase2_koplev.md section 1). Everything else in these
#: tables is exploratory: 2 screens x 5 coverages x ~11 metrics x 4 contrasts is
#: several hundred paired tests, and an uncorrected p < 0.05 among them is
#: expected by chance rather than informative.
PRIMARY = {"screen": "A375", "metric": "test_D_pearson",
           "contrast": ("structured", "unrestricted"),
           "coverages": (0.05, 0.10)}


def markdown_headline(df: pd.DataFrame, metric: str, screen: str) -> str:
    sub = df[(df["tag"] == "main") & (df["screen"] == screen)]
    if sub.empty:
        return f"_(no `main` runs for {screen})_"
    s = by_split_seed(sub, metric)
    g = s.groupby(["coverage", "family"])["value"].mean().unstack()
    d = paired_delta(df[df["screen"] == screen], metric, "structured", "unrestricted")
    delta = d.set_index("coverage") if not d.empty else d
    lines = [f"| coverage | " + " | ".join(NICE[f] for f in FAMILY_ORDER)
             + " | structured − unrestricted |",
             "| --- | " + " | ".join("---:" for _ in FAMILY_ORDER) + " | ---: |"]
    for cov in sorted(g.index):
        cells = []
        for f in FAMILY_ORDER:
            if f not in g.columns or pd.isna(g.loc[cov, f]):
                # The order-insensitive family predicts D == 0 identically, so a
                # correlation with the measured D is undefined rather than zero.
                # Printing "nan" invites a reader to treat it as a low score.
                cells.append("n/a" if f == "order_insensitive"
                             and metric.startswith("test_D_") else "—")
            else:
                cells.append(f"{g.loc[cov, f]:.4f}")
        if not delta.empty and cov in delta.index:
            r = delta.loc[cov]
            if r["n_seeds_favouring_a"] is None:
                won = ""
            else:
                won = (f", {int(r['n_seeds_favouring_a'])}/"
                       f"{int(r['n_split_seeds'])} seeds favour structured")
            d = f"**{r['delta']:+.4f}** (p={r['p_ttest']:.3f}{won})"
        else:
            d = "—"
        lines.append(f"| {cov:.2f} | " + " | ".join(cells) + f" | {d} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("results/phase2/runs.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("results/phase2/summary"))
    ap.add_argument("--raw", type=Path, default=koplev.DEFAULT_RAW_DIR)
    args = ap.parse_args()

    df = load_runs(args.runs)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- machine-readable ----
    tables = {}
    for m in METRICS:
        tables[m] = main_table(df, m).to_dict(orient="records")
    deltas = {}
    for m in METRICS:
        for a, b in (("structured", "unrestricted"),
                     ("structured", "order_insensitive"),
                     ("unrestricted", "order_insensitive"),
                     ("unrestricted", "additive")):
            d = paired_delta(df, m, a, b)
            if not d.empty:
                deltas[f"{m}::{a}-{b}"] = d.to_dict(orient="records")
    (args.outdir / "main_table.json").write_text(json.dumps(tables, indent=2) + "\n")
    (args.outdir / "paired_deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")
    collapse = collapse_report(df)
    collapse.to_csv(args.outdir / "collapse_diagnostics.csv", index=False)
    by_split_seed(df, "test_mse").to_csv(args.outdir / "runs_by_split_seed.csv",
                                         index=False)
    df.drop(columns=["restarts", "config"], errors="ignore").to_csv(
        args.outdir / "runs.csv", index=False)

    # control
    ctl = df[df["tag"] == "direction_shuffle"]
    if not ctl.empty:
        c = (by_split_seed(ctl, "test_D_pearson")
             .groupby(["screen", "coverage", "family"])["value"]
             .agg(["mean", "std", "count"]).reset_index())
        c.to_csv(args.outdir / "direction_shuffle_control.csv", index=False)

    # ---- figures ----
    paths = figures(df, args.outdir)
    try:
        screens = {lab: koplev.load_screen(lab, args.raw) for lab in ("A375", "PANC1")}
        paths.append(distribution_figure(screens, args.outdir))
    except FileNotFoundError:
        print("raw data absent; skipping the schedule-effect distribution figure")

    # ---- printed summary ----
    print("\n" + "=" * 78)
    print("PRIMARY (pre-registered): " + json.dumps(PRIMARY))
    print("Every other cell below is EXPLORATORY. Do not read an uncorrected")
    print("p < 0.05 from a non-primary cell as evidence either way.")
    print("=" * 78)
    prim = paired_delta(df[df["screen"] == PRIMARY["screen"]],
                        PRIMARY["metric"], *PRIMARY["contrast"])
    if not prim.empty:
        print(prim[prim["coverage"].isin(PRIMARY["coverages"])].to_string(index=False))

    print("\n## Held-out ordered MSE (A375, lower is better)\n")
    print(markdown_headline(df, "test_mse", "A375"))
    print("\n## Directional-effect Pearson r (A375, higher is better)\n")
    print(markdown_headline(df, "test_D_pearson", "A375"))
    print("\n## Ordering accuracy (A375, higher is better)\n")
    print(markdown_headline(df, "test_ordering_accuracy", "A375"))
    print("\n## Held-out ordered MSE (PANC1)\n")
    print(markdown_headline(df, "test_mse", "PANC1"))
    print("\n## Directional-effect Pearson r (PANC1)\n")
    print(markdown_headline(df, "test_D_pearson", "PANC1"))
    print("\n## Antisymmetric-collapse diagnostics\n")
    print(collapse[collapse["tag"] == "main"].to_string(index=False))
    print(f"\nwrote {len(paths)} figures and 6 result files to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
