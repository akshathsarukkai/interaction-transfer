#!/usr/bin/env python
"""Generate every Phase 3 table, figure and summary from the result files.

Nothing in ``docs/phase3_entity_ood.md`` is typed by hand: this script rewrites
the marked blocks, and CI fails if the committed document disagrees with the
committed metrics. Phase 2R's audit found four hand-copied p-values that matched
no run in the repository, which is what this arrangement exists to prevent.

    python scripts/report_phase3_entity_ood.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from intervention_algebra.real_data.entity_ood import report as R

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase3_entity_ood"
SUMMARY = RESULTS / "summary"
FIGURES = ROOT / "figures"
DOC = ROOT / "docs" / "phase3_entity_ood.md"

BLOCKS = ("primary", "contrast", "blind_contrast", "blind", "curl", "similarity", "representation",
          "e1_e2", "verdict", "per_drug_extremes", "analogue", "cluster", "composition", "e2",
          "coverage", "counts")


def result_files() -> list[Path]:
    return sorted(p for p in RESULTS.glob("*.jsonl") if p.stem != "smoke")


def per_drug_extremes(per_drug: pd.DataFrame, n: int = 8) -> str:
    inc = R.per_drug_incremental(per_drug)
    if inc.empty:
        return "_no per-drug rows_"
    pooled = inc.groupby(["drug", "label"], as_index=False).agg(
        incremental_skill=("incremental_skill", "mean"),
        skill_base=("skill_base", "mean"), skill_better=("skill_better", "mean"),
        max_sim_to_train=("max_sim_to_train", "mean"))
    pooled["stratum"] = R.stratum(pooled["max_sim_to_train"].to_numpy())
    top = pooled.nlargest(n, "incremental_skill")
    bot = pooled.nsmallest(n, "incremental_skill")
    frame = pd.concat([top, bot])
    return R._md(frame, {
        "label": "held-out drug", "max_sim_to_train": "mean max Tanimoto to train",
        "stratum": "stratum", "skill_base": "potential skill",
        "skill_better": "low-rank skill", "incremental_skill": "incremental",
    }, fmt={"max_sim_to_train": ".3f", "incremental_skill": "+.4f"})


def coverage_table(frame: pd.DataFrame) -> str:
    ok = frame[frame["error"].isna()] if "error" in frame.columns else frame
    tab = R.fold_table(ok)
    rows = []
    for coverage, g in tab.groupby("coverage"):
        tagged = ok[np.isclose(ok["coverage"], coverage)]
        for screen, gs in g.groupby("screen"):
            inc = R.incremental(tagged[tagged["screen"] == screen])
            pot = gs[gs["model"] == "potential"]["skill"]
            low = gs[gs["model"] == "lowrank"]["skill"]
            rows.append({
                "coverage": coverage, "screen": screen,
                "n_train_pairs": int(round(coverage * 3160)),
                "potential": float(pot.mean()) if len(pot) else float("nan"),
                "lowrank": float(low.mean()) if len(low) else float("nan"),
                "incremental": (float(inc["incremental_skill"].mean())
                                if len(inc) else float("nan")),
            })
    return R._md(pd.DataFrame(rows).sort_values(["screen", "coverage"]), {
        "screen": "screen", "coverage": "coverage",
        "n_train_pairs": "train pairs", "potential": "potential skill",
        "lowrank": "low-rank skill", "incremental": "incremental",
    }, fmt={"n_train_pairs": "d", "coverage": ".2f", "potential": "+.4f",
            "lowrank": "+.4f", "incremental": "+.4f"})


def counts_table(frame: pd.DataFrame) -> str:
    rows = []
    for (tag, rep), g in frame.groupby(["tag", "representation"], sort=False):
        nerr = int(g["error"].notna().sum()) if "error" in g.columns else 0
        rows.append({"tag": tag, "representation": rep, "n": len(g),
                     "n_failed": nerr, "models": ", ".join(sorted(set(g["model"])))})
    return R._md(pd.DataFrame(rows), {
        "tag": "block", "representation": "representation", "models": "rungs",
        "n": "conditions", "n_failed": "failed",
    }, fmt={"n": "d", "n_failed": "d"})


def figures(frame: pd.DataFrame, per_drug: pd.DataFrame, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    ok = frame[frame["error"].isna()] if "error" in frame.columns else frame
    tab = R.fold_table(ok)
    prim = tab[(tab["tag"] == "primary") & (tab["representation"] == "ecfp4")]

    # Figure 1 -- entity-OOD skill by model, E1
    if len(prim):
        order = ["zero", "potential", "lowrank", "antisym_mlp", "pair_only"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, screen in zip(axes, R.SCREENS):
            g = prim[prim["screen"] == screen]
            data = [g[g["model"] == m]["skill"].dropna().to_numpy() for m in order]
            keep = [(m, d) for m, d in zip(order, data) if len(d)]
            ax.axhline(0, color="0.6", lw=1)
            ax.boxplot([d for _, d in keep], tick_labels=[m for m, _ in keep],
                       showmeans=True)
            ax.set_title(f"{screen} — E1, one unseen drug")
            ax.tick_params(axis="x", rotation=20)
        axes[0].set_ylabel("held-out skill vs zero (per fold)")
        fig.suptitle("Figure 1 — entity-OOD skill by model, 30 drug-level folds")
        fig.tight_layout()
        p = outdir / "phase3_fig1_skill_by_model.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # Figure 2 -- incremental pair skill per fold
    inc = R.incremental(ok[ok["tag"] == "primary"])
    if len(inc):
        inc = inc[inc["representation"] == "ecfp4"]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for k, screen in enumerate(R.SCREENS):
            g = inc[inc["screen"] == screen].sort_values("fold_key")
            ax.scatter(np.arange(len(g)) + 0.18 * (k - 0.5), g["incremental_skill"],
                       s=26, label=screen, alpha=0.85)
        ax.axhline(0, color="0.4", lw=1)
        ax.axhline(R.DECISION["min_incremental_skill"], color="C3", ls="--", lw=1,
                   label=f"registered floor {R.DECISION['min_incremental_skill']}")
        ax.set_xlabel("entity fold"); ax.set_ylabel("1 − MSE(low-rank)/MSE(potential)")
        ax.set_title("Figure 2 — incremental pair skill over the feature potential, per fold")
        ax.legend()
        fig.tight_layout()
        p = outdir / "phase3_fig2_incremental_by_fold.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # Figure 3 -- performance vs chemical distance from the training set
    pdi = R.per_drug_incremental(per_drug)
    if len(pdi):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].axhline(0, color="0.6", lw=1)
        for screen in R.SCREENS:
            g = pdi[pdi["screen"] == screen]
            axes[0].scatter(g["max_sim_to_train"], g["skill_base"], s=20, alpha=0.7,
                            label=f"{screen} potential")
        for x in (R.DECISION["sim_q33"], R.DECISION["sim_q66"]):
            axes[0].axvline(x, color="0.7", ls=":", lw=1)
        axes[0].set_ylabel("held-out skill vs zero"); axes[0].legend(fontsize=8)
        axes[0].set_title("potential model")
        axes[1].axhline(0, color="0.6", lw=1)
        for screen in R.SCREENS:
            g = pdi[pdi["screen"] == screen]
            axes[1].scatter(g["max_sim_to_train"], g["incremental_skill"], s=20,
                            alpha=0.7, label=screen)
        for x in (R.DECISION["sim_q33"], R.DECISION["sim_q66"]):
            axes[1].axvline(x, color="0.7", ls=":", lw=1)
        axes[1].set_ylabel("incremental pair skill"); axes[1].legend(fontsize=8)
        axes[1].set_title("low-rank over potential")
        for ax in axes:
            ax.set_xlabel("max ECFP4 Tanimoto to any training drug")
        fig.suptitle("Figure 3 — entity-OOD performance vs chemical distance "
                     "(dotted: pre-registered strata)")
        fig.tight_layout()
        p = outdir / "phase3_fig3_similarity.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # Figure 5 -- what the E1 skill is actually made of. The other four figures
    # show skill; this one shows that skill against zero is largely the seen
    # partner, and that what the pair term adds lives in the curl.
    if len(prim) and "e1_blind_skill" in ok.columns:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        models = ["potential", "lowrank", "antisym_mlp"]
        width = 0.35
        for ax, screen in zip(axes, R.SCREENS):
            g = ok[(ok["tag"] == "primary") & (ok["screen"] == screen)]
            xs = np.arange(len(models))
            blind = [g[g["model"] == m]["e1_blind_skill"].astype(float).mean() for m in models]
            attrib = [g[g["model"] == m]["e1_skill"].astype(float).mean() - b
                      for m, b in zip(models, blind)]
            grad = [g[g["model"] == m]["e1_grad_skill"].astype(float).mean() for m in models]
            curl = [g[g["model"] == m]["e1_skill"].astype(float).mean() - gr
                    for m, gr in zip(models, grad)]
            ax.bar(xs - width / 2, blind, width, label="from the seen partner alone",
                   color="0.75")
            ax.bar(xs - width / 2, attrib, width, bottom=blind,
                   label="needs the unseen drug", color="C0")
            ax.bar(xs + width / 2, grad, width, label="expressible as a potential",
                   color="0.55")
            ax.bar(xs + width / 2, curl, width, bottom=grad,
                   label="pair-specific (curl)", color="C3")
            ax.set_xticks(xs)
            ax.set_xticklabels(models, rotation=12)
            ax.axhline(0, color="0.3", lw=1)
            ax.set_title(screen)
        axes[0].set_ylabel("E1 skill vs zero")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Figure 5 — what the E1 skill is made of: left bar splits by "
                     "whether the unseen drug is needed,\nright bar by whether a "
                     "per-drug potential can express it")
        fig.tight_layout()
        p5 = outdir / "phase3_fig5_decomposition.png"
        fig.savefig(p5, dpi=150); plt.close(fig); made.append(p5)

    # Figure 4 -- A375 vs PANC1, paired by fold
    if len(prim):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, model in zip(axes, ("potential", "lowrank")):
            a = prim[(prim["screen"] == "A375") & (prim["model"] == model)]
            b = prim[(prim["screen"] == "PANC1") & (prim["model"] == model)]
            j = a.set_index("fold_key")["skill"].to_frame("A375").join(
                b.set_index("fold_key")["skill"].to_frame("PANC1"), how="inner")
            if not len(j):
                continue
            ax.scatter(j["A375"], j["PANC1"], s=28, alpha=0.85)
            lim = [float(min(j.min())) - 0.02, float(max(j.max())) + 0.02]
            ax.plot(lim, lim, color="0.6", lw=1)
            ax.axhline(0, color="0.85", lw=1); ax.axvline(0, color="0.85", lw=1)
            ax.set_xlabel("A375 skill"); ax.set_ylabel("PANC1 skill")
            r = float(np.corrcoef(j["A375"], j["PANC1"])[0, 1]) if len(j) > 2 else float("nan")
            ax.set_title(f"{model} (fold-paired r = {r:+.2f})")
        fig.suptitle("Figure 4 — does entity-OOD transfer replicate across screens?")
        fig.tight_layout()
        p = outdir / "phase3_fig4_screens.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=SUMMARY)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    files = result_files()
    if not files:
        print("no Phase 3 result files; nothing to report")
        return 0
    frame = R.load_runs(files)
    if "error" not in frame.columns:
        frame["error"] = np.nan
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_drug = R.per_drug_table(frame)
    v = R.verdict(frame)
    vp = R.verdict_posthoc(frame)

    blocks = {
        "primary": R.primary_table(frame),
        "contrast": R.contrast_table(frame),
        "blind_contrast": R.blind_contrast_table(frame),
        "blind": R.blind_table(frame),
        "curl": R.curl_table(frame),
        "similarity": R.similarity_table(frame),
        "analogue": R.analogue_table(frame),
        "cluster": R.cluster_robust_table(frame),
        "composition": R.e1_composition_table(frame),
        "e2": R.e2_table(frame),
        "representation": R.representation_table(frame),
        "e1_e2": R.e1_vs_e2_table(frame),
        "verdict": R.verdict_block(v) + "\n\n### The same rule with one statistic corrected (post-hoc)\n\n"
                   + R.verdict_block({**vp, "verdict": vp["verdict_posthoc"]}),
        "per_drug_extremes": per_drug_extremes(per_drug),
        "coverage": coverage_table(frame[frame["tag"].str.startswith("coverage")]
                                   if len(frame) else frame),
        "counts": counts_table(frame),
    }
    for name, body in blocks.items():
        (args.outdir / f"{name}.md").write_text(body + "\n")
    (args.outdir / "verdict.json").write_text(json.dumps(v, indent=2, default=float) + "\n")
    (args.outdir / "verdict_posthoc.json").write_text(
        json.dumps(vp, indent=2, default=float) + "\n")
    R.fold_table(frame).to_csv(args.outdir / "folds.csv", index=False)
    if not per_drug.empty:
        per_drug.to_csv(args.outdir / "per_drug.csv", index=False)
        R.per_drug_incremental(per_drug).to_csv(
            args.outdir / "per_drug_incremental.csv", index=False)
    inc = R.incremental(frame[frame["tag"] == "primary"])
    if len(inc):
        inc.to_csv(args.outdir / "incremental.csv", index=False)

    replaced = R.inject_blocks(DOC, blocks)
    missing = sorted(set(blocks) - set(replaced))
    if DOC.exists() and missing:
        raise SystemExit(f"{DOC.name} has no marker for generated blocks: {missing}")
    if not args.no_figures:
        for p in figures(frame, per_drug, FIGURES):
            print(f"  figure {p.relative_to(ROOT)}")

    print(f"{len(frame)} rows from {len(files)} files")
    print(f"  frozen rule:   {v['verdict']}")
    print(f"  post-hoc rule: {vp['verdict_posthoc']}")
    if v["invalidating_reasons"]:
        for r in v["invalidating_reasons"]:
            print(f"  INVALID: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
