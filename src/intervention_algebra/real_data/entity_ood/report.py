"""Aggregation, the pre-registered decision rule, and the generated tables.

Two commitments this module exists to keep.

**The decision rule is executed, not narrated.** :func:`verdict` implements the
categories from the pre-registration in the order they were registered, reads its
thresholds from :data:`DECISION`, and returns which criteria fired. Phase 2R's
audit found four hand-copied p-values matching no run in the repository; every
number in the Phase 3 documents is generated from the result files, and CI fails
if a committed document has drifted from its committed metrics.

**The unit of replication is the fold, and it is enforced here.** Nine hundred E1
rows in one fold share ten held-out drugs and are nowhere near independent.
Every inferential statistic is computed over folds, and the per-drug summary --
which is the more interpretable of the two -- averages each drug's three folds
before entering a drug-level statistic. Nothing in this module ever treats a pair
row as a replicate.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

#: Every threshold the decision rule reads, exactly as pre-registered. Kept in
#: one dict so the document can print it and a reader can check that what ran is
#: what was registered.
DECISION = {
    "potential_beats_zero": 0.02,
    "min_incremental_skill": 0.01,
    "alpha": 0.05,
    "min_folds_favouring": 20,
    "n_folds": 30,
    "control_ceiling": 0.02,
    "control_invalidates_above": 0.05,
    "positive_control_floor": 0.05,
    "max_failure_fraction": 0.10,
    "sim_q33": 0.2340,
    "sim_q66": 0.5161,
}

SCREENS = ("A375", "PANC1")
#: The primary endpoint. E2 and the metal-excluded re-scoring use the same code
#: with a different prefix, and are never substituted for this one.
PRIMARY_PREFIX = "e1"


def load_runs(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "fold_key" not in frame.columns:
        raise ValueError(f"{paths}: not Phase 3 rows -- no fold_key column")
    return frame


def errors(frame: pd.DataFrame) -> pd.DataFrame:
    if "error" not in frame.columns:
        return frame.iloc[:0]
    return frame[frame["error"].notna()]


def _ok(frame: pd.DataFrame) -> pd.DataFrame:
    if "error" in frame.columns:
        frame = frame[frame["error"].isna()]
    return frame


def fold_table(frame: pd.DataFrame, prefix: str = PRIMARY_PREFIX) -> pd.DataFrame:
    """One row per (tag, screen, representation, coverage, fold, model).

    The join key for every paired comparison. ``mse`` and ``mse_zero`` are kept
    rather than only ``skill``, because the incremental contrast is a ratio of
    MSEs within a fold and reconstructing it from two skills would go through
    ``mse_zero`` twice.
    """
    frame = _ok(frame)
    cols = {
        "tag": "tag", "screen": "screen", "representation": "representation",
        "coverage": "coverage", "fold_key": "fold_key", "partition": "partition",
        "fold": "fold", "model": "model", "n_params": "n_params",
        "synthetic_target": "synthetic_target",
    }
    out = frame[[c for c in cols if c in frame.columns]].copy()
    for metric in ("mse", "mse_zero", "skill", "pearson", "spearman", "mae",
                   "rmse", "sign_accuracy", "n_pairs"):
        key = f"{prefix}_{metric}"
        if key in frame.columns:
            out[metric] = frame[key]
    return out.reset_index(drop=True)


def incremental(frame: pd.DataFrame, better: str = "lowrank",
                base: str = "potential", prefix: str = PRIMARY_PREFIX) -> pd.DataFrame:
    """Per-fold ``1 - MSE_better / MSE_base``, the primary quantity.

    Computed **within** a fold and only then averaged. A ratio of pooled means
    would be dominated by whichever folds happen to have the largest ``D``
    variance, which is a property of which ten drugs were held out rather than of
    either model.
    """
    tab = fold_table(frame, prefix)
    keys = ["tag", "screen", "representation", "coverage", "fold_key"]
    a = tab[tab["model"] == better].set_index(keys)
    b = tab[tab["model"] == base].set_index(keys)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(index=common).reset_index()
    out["mse_better"] = a.loc[common, "mse"].to_numpy()
    out["mse_base"] = b.loc[common, "mse"].to_numpy()
    out["skill_better"] = a.loc[common, "skill"].to_numpy()
    out["skill_base"] = b.loc[common, "skill"].to_numpy()
    out["incremental_skill"] = 1.0 - out["mse_better"] / out["mse_base"]
    out["better"] = better
    out["base"] = base
    return out


def paired_summary(values: np.ndarray) -> dict:
    """Mean, SD, 95% t CI, paired t and Wilcoxon, and the sign count.

    Both tests are reported because they answer slightly different questions and
    the pre-registration requires both: at n = 30 the t-test is sensitive to a
    few large folds and Wilcoxon is not, so requiring both is a cheap guard
    against a result carried by outliers.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    out = {"n": n, "mean": float(v.mean()) if n else float("nan"),
           "sd": float(v.std(ddof=1)) if n > 1 else float("nan"),
           "n_positive": int((v > 0).sum())}
    if n > 1:
        se = out["sd"] / math.sqrt(n)
        crit = stats.t.ppf(0.975, n - 1)
        out["ci_lo"] = out["mean"] - crit * se
        out["ci_hi"] = out["mean"] + crit * se
        out["p_ttest"] = float(stats.ttest_1samp(v, 0.0).pvalue)
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(v).pvalue)
        except ValueError:                       # all differences exactly zero
            out["p_wilcoxon"] = 1.0
    else:
        out.update({"ci_lo": float("nan"), "ci_hi": float("nan"),
                    "p_ttest": float("nan"), "p_wilcoxon": float("nan")})
    return out


def model_summary(frame: pd.DataFrame, tag: str = "primary",
                  representation: str = "ecfp4", coverage: float = 1.0,
                  prefix: str = PRIMARY_PREFIX) -> pd.DataFrame:
    """Fold-mean metrics per (screen, model). Rows are folds, never pairs."""
    tab = fold_table(frame, prefix)
    sel = tab[(tab["tag"] == tag) & (tab["representation"] == representation)
              & (np.isclose(tab["coverage"], coverage))]
    rows = []
    for (screen, model), g in sel.groupby(["screen", "model"], sort=False):
        s = paired_summary(g["skill"].to_numpy())
        rows.append({
            "screen": screen, "model": model, "n_folds": s["n"],
            "mse": float(g["mse"].mean()), "rmse": float(g["rmse"].mean()),
            "mae": float(g["mae"].mean()), "mse_zero": float(g["mse_zero"].mean()),
            "skill": s["mean"], "skill_sd": s["sd"],
            "skill_ci_lo": s["ci_lo"], "skill_ci_hi": s["ci_hi"],
            "n_folds_positive": s["n_positive"], "p_ttest": s["p_ttest"],
            "pearson": float(np.nanmean(g["pearson"])) if "pearson" in g else float("nan"),
            "spearman": float(np.nanmean(g["spearman"])) if "spearman" in g else float("nan"),
            "sign_accuracy": (float(np.nanmean(g["sign_accuracy"]))
                              if "sign_accuracy" in g else float("nan")),
            "n_params_median": int(np.median(g["n_params"])),
            "n_params_max": int(g["n_params"].max()),
            "n_pairs": int(g["n_pairs"].mean()),
        })
    order = {m: k for k, m in enumerate(
        ("zero", "potential", "lowrank", "antisym_mlp", "pair_only"))}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["screen", "model"],
                           key=lambda c: c.map(order) if c.name == "model" else c
                           ).reset_index(drop=True)


def per_drug_table(frame: pd.DataFrame, tag: str = "primary",
                   representation: str = "ecfp4") -> pd.DataFrame:
    """One row per (held-out drug, model), averaging that drug's folds.

    The interpretable unit: a drug is an entity, and "did this drug's behaviour
    transfer?" is a question about the drug rather than about the ten-drug group
    it happened to be binned with. Also the guard against a fold mean carried by
    one or two easy analogues -- with 100 drugs the distribution is visible.
    """
    frame = _ok(frame)
    sel = frame[(frame["tag"] == tag) & (frame["representation"] == representation)]
    rows = []
    for _, r in sel.iterrows():
        for d in r.get("per_drug") or []:
            rows.append({
                "screen": r["screen"], "model": r["model"], "fold_key": r["fold_key"],
                "drug": d["drug"], "label": d["label"], "n_pairs": d["n_pairs"],
                "max_sim_to_train": d["max_sim_to_train"],
                "median_sim_to_train": d["median_sim_to_train"],
                "bits_set": d["bits_set"], "mse": d["e1_mse"],
                "mse_zero": d["e1_mse_zero"], "skill": d["e1_skill"],
                "pearson": d.get("e1_pearson", float("nan")),
            })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    agg = long.groupby(["screen", "model", "drug", "label"], as_index=False).agg(
        n_folds=("fold_key", "nunique"), n_pairs=("n_pairs", "sum"),
        mse=("mse", "mean"), mse_zero=("mse_zero", "mean"), skill=("skill", "mean"),
        pearson=("pearson", "mean"), max_sim_to_train=("max_sim_to_train", "mean"),
        median_sim_to_train=("median_sim_to_train", "mean"),
        bits_set=("bits_set", "max"))
    agg["stratum"] = stratum(agg["max_sim_to_train"].to_numpy())
    return agg


def stratum(max_sim: np.ndarray) -> np.ndarray:
    """Similarity strata at the pre-registered, outcome-free cut points."""
    lo, hi = DECISION["sim_q33"], DECISION["sim_q66"]
    return np.where(max_sim < lo, "low", np.where(max_sim < hi, "medium", "high"))


def per_drug_incremental(per_drug: pd.DataFrame, better: str = "lowrank",
                         base: str = "potential") -> pd.DataFrame:
    """Per-drug ``1 - MSE_better / MSE_base``, carrying the similarity stratum."""
    if per_drug.empty:
        return per_drug
    keys = ["screen", "drug", "label"]
    a = per_drug[per_drug["model"] == better].set_index(keys)
    b = per_drug[per_drug["model"] == base].set_index(keys)
    common = a.index.intersection(b.index)
    out = pd.DataFrame(index=common).reset_index()
    out["incremental_skill"] = (1.0 - a.loc[common, "mse"].to_numpy()
                               / b.loc[common, "mse"].to_numpy())
    out["skill_better"] = a.loc[common, "skill"].to_numpy()
    out["skill_base"] = b.loc[common, "skill"].to_numpy()
    out["max_sim_to_train"] = a.loc[common, "max_sim_to_train"].to_numpy()
    out["n_pairs"] = a.loc[common, "n_pairs"].to_numpy()
    out["stratum"] = stratum(out["max_sim_to_train"].to_numpy())
    return out


def stratified_incremental(per_drug: pd.DataFrame) -> pd.DataFrame:
    inc = per_drug_incremental(per_drug)
    if inc.empty:
        return inc
    rows = []
    for (screen, strat), g in inc.groupby(["screen", "stratum"]):
        s = paired_summary(g["incremental_skill"].to_numpy())
        rows.append({"screen": screen, "stratum": strat, "n_drugs": s["n"],
                     "incremental_skill": s["mean"], "sd": s["sd"],
                     "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "n_positive": s["n_positive"],
                     "skill_base": float(g["skill_base"].mean()),
                     "skill_better": float(g["skill_better"].mean()),
                     "max_sim_lo": float(g["max_sim_to_train"].min()),
                     "max_sim_hi": float(g["max_sim_to_train"].max())})
    order = {"low": 0, "medium": 1, "high": 2}
    return pd.DataFrame(rows).sort_values(
        ["screen", "stratum"], key=lambda c: c.map(order) if c.name == "stratum" else c
    ).reset_index(drop=True)


def control_skill(frame: pd.DataFrame, representation: str,
                  model: str = "lowrank", prefix: str = PRIMARY_PREFIX) -> dict:
    """Mean E1 skill of one control representation, per screen and pooled."""
    tab = fold_table(frame, prefix)
    sel = tab[(tab["representation"] == representation) & (tab["model"] == model)]
    if sel.empty:
        return {"present": False}
    out = {"present": True, "n_folds": int(len(sel)),
           "mean": float(sel["skill"].mean()), "max": float(sel["skill"].max())}
    for screen, g in sel.groupby("screen"):
        out[screen] = float(g["skill"].mean())
    return out


def _classify(invalid, have_both, criteria, pair_specific,
              per_screen_pot_low, per_screen_incr_fails) -> str:
    """The registered cascade, in the registered order, quantified as written."""
    if invalid:
        return "INCONCLUSIVE"
    if not have_both:
        return "INCONCLUSIVE"
    if all(per_screen_pot_low):                       # rule 1, both screens
        return "NO ENTITY TRANSFER"
    if pair_specific:                                 # rule 3 / rule 4
        return ("PAIR-SPECIFIC ENTITY TRANSFER" if criteria["g_not_analogue_confined"]
                else "ANALOGUE-ONLY TRANSFER")
    if not any(per_screen_pot_low) and all(per_screen_incr_fails):   # rule 2
        return "POTENTIAL-ONLY ENTITY TRANSFER"
    return "WEAK/MARGINAL ENTITY TRANSFER"            # rule 5


def verdict(frame: pd.DataFrame) -> dict:
    """The pre-registered rule, executed in the order it was registered.

    Returns the category, every criterion's value, and -- when a validity gate
    fires -- the reason, so the document can print why the experiment is
    uninterpretable rather than reporting a category anyway.
    """
    d = DECISION
    err = errors(frame)
    ok = _ok(frame)
    n_attempted = len(frame)
    fail_fraction = len(err) / n_attempted if n_attempted else 1.0

    # ---- validity gates -------------------------------------------------
    invalid: list[str] = []
    # The registered gate list has four entries and only three were implemented.
    # A fold that fails assert_no_drug_leakage raises inside run_entity_condition,
    # and the sweep's blanket `except Exception` turns it into an ordinary error
    # row -- which the 10% failure-fraction gate would tolerate. Leakage was
    # registered at zero tolerance and is now checked at zero tolerance.
    leaks = [r for _, r in err.iterrows()
             if "assert_no_drug_leakage" in str(r.get("error", ""))
             or "leakage" in str(r.get("error", "")).lower()]
    if leaks:
        invalid.append(f"{len(leaks)} condition(s) failed the entity-leakage guard; "
                       f"the registered tolerance is zero")
    if fail_fraction > d["max_failure_fraction"]:
        invalid.append(f"{len(err)}/{n_attempted} conditions failed "
                       f"({fail_fraction:.1%} > {d['max_failure_fraction']:.0%})")

    pos = incremental(ok[ok["tag"] == "positive_control"]) if "tag" in ok else pd.DataFrame()
    pos_mean = float(pos["incremental_skill"].mean()) if len(pos) else float("nan")
    if not len(pos):
        invalid.append("the synthetic positive control was not run")
    elif pos_mean <= d["positive_control_floor"]:
        invalid.append(f"the positive control recovers only {pos_mean:+.3f} incremental "
                       f"skill from a planted rank-2 signal "
                       f"(<= {d['positive_control_floor']}), so the machinery cannot "
                       f"detect what it is looking for")

    controls = {rep: control_skill(ok, rep) for rep in ("random", "shuffled")}
    for rep, c in controls.items():
        if c.get("present") and c["mean"] > d["control_invalidates_above"]:
            invalid.append(f"the {rep} representation posts mean E1 skill "
                           f"{c['mean']:+.3f} > {d['control_invalidates_above']}, "
                           f"so something is leaking")

    # ---- the primary statistics ----------------------------------------
    inc = incremental(ok[(ok["tag"] == "primary")]) if "tag" in ok else pd.DataFrame()
    summary = model_summary(ok)
    per_screen: dict[str, dict] = {}
    for screen in SCREENS:
        g = inc[inc["screen"] == screen]
        s = paired_summary(g["incremental_skill"].to_numpy()) if len(g) else {}
        row = {"incremental": s}
        for model in ("potential", "lowrank"):
            m = summary[(summary["screen"] == screen) & (summary["model"] == model)]
            row[model] = m.iloc[0].to_dict() if len(m) else {}
        per_screen[screen] = row

    def crit(name) -> bool:
        return all(name(per_screen[s]) for s in SCREENS if per_screen.get(s))

    have_both = all(per_screen.get(s, {}).get("incremental", {}).get("n", 0) > 1
                    for s in SCREENS)
    criteria = {
        "potential_beats_zero": crit(
            lambda r: r["potential"].get("skill", float("-inf")) > d["potential_beats_zero"]),
        "a_both_positive": crit(
            lambda r: r["potential"].get("skill", float("-inf")) > 0
            and r["lowrank"].get("skill", float("-inf")) > 0),
        "b_incremental_above_floor": crit(
            lambda r: r["incremental"].get("mean", float("-inf")) > d["min_incremental_skill"]),
        "c_both_tests_significant": crit(
            lambda r: (r["incremental"].get("p_ttest", 1.0) < d["alpha"]
                       and r["incremental"].get("p_wilcoxon", 1.0) < d["alpha"])),
        "d_folds_favouring": crit(
            lambda r: r["incremental"].get("n_positive", 0) >= d["min_folds_favouring"]),
        "e_pearson_positive": crit(
            lambda r: r["lowrank"].get("pearson", float("-inf")) > 0),
        "f_controls_collapse": all(
            (not c.get("present")) or c["mean"] <= d["control_ceiling"]
            for c in controls.values()),
    }

    strat = stratified_incremental(per_drug_table(ok))
    low = strat[strat["stratum"] == "low"] if len(strat) else strat
    criteria["g_not_analogue_confined"] = bool(
        len(low) and (low["incremental_skill"] > 0).any())

    pair_terms = [k for k in ("a_both_positive", "b_incremental_above_floor",
                              "c_both_tests_significant", "d_folds_favouring",
                              "e_pearson_positive", "f_controls_collapse")]
    pair_specific = all(criteria[k] for k in pair_terms)

    # Rules 1 and 2 are quantified over screens *separately*, as registered.
    # Negating an all-screens conjunction is not the same statement: it fires
    # when at least one screen qualifies, and it routed one-screen-only outcomes
    # away from WEAK/MARGINAL, where the registration puts them.
    per_screen_pot_low = [per_screen[s]["potential"].get("skill", float("-inf"))
                          <= d["potential_beats_zero"] for s in SCREENS if s in per_screen]
    per_screen_incr_fails = [
        not (per_screen[s]["incremental"].get("mean", float("-inf"))
             > d["min_incremental_skill"]
             or (per_screen[s]["incremental"].get("p_ttest", 1.0) < d["alpha"]
                 and per_screen[s]["incremental"].get("p_wilcoxon", 1.0) < d["alpha"]))
        for s in SCREENS if s in per_screen]
    label = _classify(invalid, have_both, criteria, pair_specific,
                      per_screen_pot_low, per_screen_incr_fails)
    if label == "INCONCLUSIVE" and not invalid and not have_both:
        invalid = ["one or both screens produced no paired folds"]

    return {
        "verdict": label,
        "criteria": criteria,
        "_per_screen_pot_low": per_screen_pot_low,
        "_per_screen_incr_fails": per_screen_incr_fails,
        "invalidating_reasons": invalid,
        "n_attempted": n_attempted, "n_failed": len(err),
        "positive_control_incremental": pos_mean,
        "controls": controls,
        "per_screen": {s: {
            "potential_skill": per_screen[s]["potential"].get("skill"),
            "lowrank_skill": per_screen[s]["lowrank"].get("skill"),
            "lowrank_pearson": per_screen[s]["lowrank"].get("pearson"),
            "incremental": per_screen[s]["incremental"],
        } for s in SCREENS if s in per_screen},
        "stratified_low_similarity": (
            low.to_dict("records") if len(low) else []),
        "thresholds": dict(d),
    }


# --------------------------------------------------------------------------
# generated tables
# --------------------------------------------------------------------------

def _md(frame: pd.DataFrame, cols: dict[str, str], fmt: dict[str, str] | None = None) -> str:
    """A markdown table. Every number in the Phase 3 documents comes through here."""
    fmt = fmt or {}
    head = "| " + " | ".join(cols.values()) + " |"
    rule = "|" + "|".join("---:" if k not in ("screen", "model", "stratum", "label",
                                              "representation", "regime", "criterion")
                          else "---" for k in cols) + "|"
    lines = [head, rule]
    for _, r in frame.iterrows():
        cells = []
        for k in cols:
            v = r.get(k)
            if isinstance(v, float) and not np.isfinite(v):
                cells.append("—")
            elif k in fmt:
                cells.append(format(v, fmt[k]))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def primary_table(frame: pd.DataFrame, prefix: str = PRIMARY_PREFIX,
                  tag: str = "primary", representation: str = "ecfp4") -> str:
    """The headline: every rung against zero, plus its incremental skill."""
    summ = model_summary(_ok(frame), tag=tag, representation=representation, prefix=prefix)
    if summ.empty:
        return "_no runs_"
    inc = incremental(_ok(frame)[_ok(frame)["tag"] == tag], prefix=prefix)
    inc = inc[inc["representation"] == representation] if len(inc) else inc
    imap = {}
    for model in ("lowrank", "antisym_mlp", "pair_only"):
        g = incremental(_ok(frame)[_ok(frame)["tag"] == tag], better=model, prefix=prefix)
        if len(g):
            g = g[g["representation"] == representation]
            for screen, gg in g.groupby("screen"):
                imap[(screen, model)] = float(gg["incremental_skill"].mean())
    summ["incremental_skill"] = [imap.get((r["screen"], r["model"]), float("nan"))
                                 for _, r in summ.iterrows()]
    return _md(summ, {
        "screen": "screen", "model": "model",
        "n_params_median": "params (median fold)", "n_params_max": "max",
        "mse": "MSE", "skill": "zero skill",
        "incremental_skill": "incremental pair skill",
        "pearson": "Pearson", "spearman": "Spearman",
        "sign_accuracy": "sign acc",
    }, fmt={"n_params_median": "d", "n_params_max": "d", "mse": ".5f"})


def contrast_table(frame: pd.DataFrame, prefix: str = PRIMARY_PREFIX,
                   tag: str = "primary", representation: str = "ecfp4") -> str:
    """The primary contrast with its fold-level inference."""
    inc = incremental(_ok(frame)[_ok(frame)["tag"] == tag], prefix=prefix)
    if inc.empty:
        return "_no paired folds_"
    inc = inc[inc["representation"] == representation]
    rows = []
    for screen, g in inc.groupby("screen"):
        s = paired_summary(g["incremental_skill"].to_numpy())
        rows.append({"screen": screen, "n_folds": s["n"], "mean": s["mean"],
                     "sd": s["sd"], "ci": f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]",
                     "n_positive": f"{s['n_positive']}/{s['n']}",
                     "p_ttest": s["p_ttest"], "p_wilcoxon": s["p_wilcoxon"]})
    return _md(pd.DataFrame(rows), {
        "screen": "screen", "n_folds": "folds", "mean": "incremental skill",
        "sd": "SD", "ci": "95% CI", "n_positive": "folds favouring low-rank",
        "p_ttest": "paired t", "p_wilcoxon": "Wilcoxon",
    }, fmt={"n_folds": "d", "mean": "+.4f", "sd": ".4f",
            "p_ttest": ".3g", "p_wilcoxon": ".3g"})


def similarity_table(frame: pd.DataFrame) -> str:
    strat = stratified_incremental(per_drug_table(_ok(frame)))
    if strat.empty:
        return "_no per-drug rows_"
    strat["range"] = [f"{r.max_sim_lo:.2f}–{r.max_sim_hi:.2f}" for r in strat.itertuples()]
    return _md(strat, {
        "screen": "screen", "stratum": "stratum", "range": "mean max Tanimoto to train",
        "n_drugs": "drugs", "skill_base": "potential skill",
        "skill_better": "low-rank skill", "incremental_skill": "incremental",
        "ci_lo": "CI lo", "ci_hi": "CI hi",
    }, fmt={"n_drugs": "d", "incremental_skill": "+.4f",
            "ci_lo": "+.4f", "ci_hi": "+.4f"})


def representation_table(frame: pd.DataFrame) -> str:
    """Every representation and control side by side, on its own folds."""
    ok = _ok(frame)
    tab = fold_table(ok)
    rows = []
    for (tag, rep), g in tab.groupby(["tag", "representation"], sort=False):
        for screen, gs in g.groupby("screen"):
            pot = gs[gs["model"] == "potential"]["skill"]
            low = gs[gs["model"] == "lowrank"]["skill"]
            sub = ok[(ok["tag"] == tag) & (ok["representation"] == rep)]
            inc = incremental(sub)
            inc = inc[inc["screen"] == screen] if len(inc) else inc
            rows.append({
                "tag": tag, "representation": rep, "screen": screen,
                "n_folds": int(len(low)),
                "potential": float(pot.mean()) if len(pot) else float("nan"),
                "lowrank": float(low.mean()) if len(low) else float("nan"),
                "incremental": (float(inc["incremental_skill"].mean())
                                if len(inc) else float("nan")),
            })
    return _md(pd.DataFrame(rows), {
        "tag": "block", "representation": "representation", "screen": "screen",
        "n_folds": "folds", "potential": "potential skill",
        "lowrank": "low-rank skill", "incremental": "incremental",
    }, fmt={"n_folds": "d", "potential": "+.4f", "lowrank": "+.4f",
            "incremental": "+.4f"})


def e1_vs_e2_table(frame: pd.DataFrame) -> str:
    rows = []
    for prefix, regime in (("e1", "E1: one unseen"), ("e2", "E2: both unseen"),
                           ("e1x", "E1, metals excluded")):
        summ = model_summary(_ok(frame), prefix=prefix)
        if summ.empty:
            continue
        inc = incremental(_ok(frame)[_ok(frame)["tag"] == "primary"], prefix=prefix)
        for screen in SCREENS:
            s = summ[summ["screen"] == screen]
            g = inc[inc["screen"] == screen] if len(inc) else inc
            def sk(model):
                r = s[s["model"] == model]
                return float(r.iloc[0]["skill"]) if len(r) else float("nan")
            rows.append({
                "regime": regime, "screen": screen,
                "n_pairs": int(s["n_pairs"].max()) if len(s) else 0,
                "potential": sk("potential"), "lowrank": sk("lowrank"),
                "incremental": (float(g["incremental_skill"].mean())
                                if len(g) else float("nan")),
            })
    return _md(pd.DataFrame(rows), {
        "regime": "regime", "screen": "screen", "n_pairs": "pairs/fold",
        "potential": "potential skill", "lowrank": "low-rank skill",
        "incremental": "incremental",
    }, fmt={"n_pairs": "d", "potential": "+.4f", "lowrank": "+.4f",
            "incremental": "+.4f"})


def verdict_block(v: dict) -> str:
    lines = [f"**{v['verdict']}**", ""]
    if v["invalidating_reasons"]:
        lines.append("Validity gates that fired:")
        lines += [f"* {r}" for r in v["invalidating_reasons"]]
        lines.append("")
    rows = pd.DataFrame([{"criterion": k, "met": "yes" if x else "no"}
                         for k, x in v["criteria"].items()])
    lines.append(_md(rows, {"criterion": "criterion", "met": "met"}))
    return "\n".join(lines)


def inject_blocks(doc: Path, blocks: dict[str, str]) -> list[str]:
    """Replace ``<!-- generated:NAME -->`` … ``<!-- /generated:NAME -->`` regions.

    Returns the names actually replaced, so a caller can fail loudly when a block
    it generated found no home. Silence here is how a document drifts.

    This function has a history and the shape of it is deliberate.

    * It slices **between** the markers with a regex rather than rebuilding a
      delimiter and calling ``str.replace``. A previous version rebuilt the
      marker without its space, so every block after the first was compared
      against a string that could not occur, and nothing was updated.
    * A sibling bug in the same family used ``str.index`` to find a boundary, hit
      an earlier occurrence, produced a reversed slice, and called
      ``text.replace("", new)`` -- which inserts between every character. It
      turned a README into 551,827 inserted lines.
    * The newline handling is why this is written out rather than inlined. An
      *empty* block is ``-->\\n<!-- /``: there is exactly one newline, and a
      pattern that expects one on each side matches nothing. First runs silently
      write empty blocks, so a naive pattern works on every subsequent run and
      fails only on the first -- which is the run where the document is created.
    """
    if not doc.exists():
        return []
    text = doc.read_text()
    replaced = []
    for name, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- generated:{re.escape(name)} -->\n)"
            rf"(?:.*?\n)??"
            rf"(<!-- /generated:{re.escape(name)} -->)",
            re.DOTALL)
        if not pattern.search(text):
            continue
        text = pattern.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2),
                           text, count=1)
        replaced.append(name)
    doc.write_text(text)
    return replaced


# --------------------------------------------------------------------------
# The two diagnostics that decide what "entity transfer" means here
# --------------------------------------------------------------------------

def blind_table(frame: pd.DataFrame) -> str:
    """How much of E1 survives when the model is blind to the unseen drug.

    "Blind" is the **marginal over the training drugs**: the unseen endpoint is
    replaced by each of the 80 drugs the model trained on and the prediction is
    averaged. An earlier version zeroed the drug's feature row instead, which is
    not "no information" but the assertion "this drug has zero fingerprint bits"
    -- a systematically pessimistic point that anything at all beats. Against
    that baseline the random-feature control scored a spurious +0.052
    "attributable to the unseen drug" at p = 0.049.

    Which is why **every representation is shown here, not just ECFP4.** Showing
    only the real features would repeat exactly the mistake that discredited
    skill-against-zero: a diagnostic is only evidence if a representation that
    cannot contain the answer fails it.
    """
    ok = _ok(frame)
    if "e1_blind_skill" not in ok.columns:
        return "_no blind diagnostics_"
    rows = []
    for (tag, rep, screen, model), g in ok.groupby(
            ["tag", "representation", "screen", "model"], sort=False):
        if model == "zero" or g["e1_blind_skill"].isna().all():
            continue
        full = g["e1_skill"].astype(float)
        blind = g["e1_blind_skill"].astype(float)
        s = paired_summary((full - blind).to_numpy())
        rows.append({"tag": tag, "representation": rep, "screen": screen,
                     "model": model, "n": s["n"],
                     "full": float(full.mean()), "blind": float(blind.mean()),
                     "attributable": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "p_ttest": s["p_ttest"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return "_no blind diagnostics_"
    order = {m: k for k, m in enumerate(
        ("potential", "lowrank", "antisym_mlp", "pair_only"))}
    out = out.sort_values(["tag", "screen", "model"],
                          key=lambda c: c.map(order) if c.name == "model" else c)
    return _md(out, {
        "tag": "block", "representation": "representation", "screen": "screen",
        "model": "model", "n": "folds", "full": "E1 skill",
        "blind": "blind to unseen drug", "attributable": "attributable",
        "ci_lo": "CI lo", "ci_hi": "CI hi", "p_ttest": "paired t",
    }, fmt={"n": "d", "full": "+.4f", "blind": "+.4f", "attributable": "+.4f",
            "ci_lo": "+.4f", "ci_hi": "+.4f", "p_ttest": ".3g"})


def curl_table(frame: pd.DataFrame) -> str:
    """Is the low-rank model's prediction a potential, or is it pair-specific?

    A rank-2 antisymmetric form contains the potential exactly (``z_i = (g_i,
    1)``, ``K = [[0,1],[-1,0]]``), so beating the potential rung does not by
    itself demonstrate pair structure. ``curl fraction`` is the share of the
    model's own predicted energy that no per-drug potential can express;
    ``curl gain`` is the skill the full prediction has over its own gradient
    projection. The positive-control row is the calibration: there the target
    really does contain a planted rank-2 term.
    """
    ok = _ok(frame)
    if "e1_pred_curl_fraction" not in ok.columns:
        return "_no decomposition_"
    sel = ok[ok["model"].isin(("lowrank", "antisym_mlp", "pair_only", "potential"))]
    rows = []
    for (tag, rep, screen, model), g in sel.groupby(
            ["tag", "representation", "screen", "model"], sort=False):
        cf = g["e1_pred_curl_fraction"].astype(float)
        if cf.isna().all():
            continue
        rows.append({
            "tag": tag, "representation": rep, "screen": screen, "model": model,
            "n_folds": len(g), "curl_fraction": float(cf.mean()),
            "curl_fraction_max": float(cf.max()),
            "grad_skill": float(g["e1_grad_skill"].astype(float).mean()),
            "full_skill": float(g["e1_skill"].astype(float).mean()),
            "curl_gain": float(g["e1_curl_gain"].astype(float).mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return "_no decomposition_"
    return _md(out, {
        "tag": "block", "representation": "representation", "screen": "screen",
        "model": "model", "n_folds": "folds", "curl_fraction": "curl share of Dhat",
        "curl_fraction_max": "max", "grad_skill": "skill of its gradient part",
        "full_skill": "full skill", "curl_gain": "gain from the curl part",
    }, fmt={"n_folds": "d", "curl_fraction": ".2e", "curl_fraction_max": ".2e",
            "grad_skill": "+.4f", "full_skill": "+.4f", "curl_gain": "+.5f"})


def analogue_table(frame: pd.DataFrame) -> str:
    """Is the pair-specific gain confined to drugs with a close training analogue?

    The pre-registered criterion (g) asked only that the low-similarity stratum's
    mean incremental skill be **> 0**, with no significance requirement. That was
    too weak a bar, written before it was clear how much of E1 the seen partner
    supplies, and it is reported here as written alongside the test it should
    have carried.
    """
    inc = per_drug_incremental(per_drug_table(_ok(frame)))
    if inc.empty:
        return "_no per-drug rows_"
    rows = []
    for screen, g in inc.groupby("screen"):
        for strat in ("low", "medium", "high"):
            v = g[g["stratum"] == strat]["incremental_skill"].to_numpy()
            if not len(v):
                continue
            s = paired_summary(v)
            rows.append({"screen": screen, "stratum": strat, "n_drugs": s["n"],
                         "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                         "n_positive": f"{s['n_positive']}/{s['n']}",
                         "p": s["p_ttest"]})
        lo = g[g["stratum"] == "low"]["incremental_skill"].to_numpy()
        hi = g[g["stratum"] == "high"]["incremental_skill"].to_numpy()
        if len(lo) > 1 and len(hi) > 1:
            rows.append({
                "screen": screen, "stratum": "high vs low", "n_drugs": len(hi) + len(lo),
                "mean": float(hi.mean() - lo.mean()), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_positive": "—",
                "p": float(stats.mannwhitneyu(hi, lo).pvalue)})
        rho = stats.spearmanr(g["max_sim_to_train"], g["incremental_skill"])
        rows.append({"screen": screen, "stratum": "Spearman rho vs similarity",
                     "n_drugs": len(g), "mean": float(rho.statistic),
                     "ci_lo": float("nan"), "ci_hi": float("nan"),
                     "n_positive": "—", "p": float(rho.pvalue)})
    return _md(pd.DataFrame(rows), {
        "screen": "screen", "stratum": "stratum / test", "n_drugs": "drugs",
        "mean": "incremental skill", "ci_lo": "CI lo", "ci_hi": "CI hi",
        "n_positive": "drugs positive", "p": "p",
    }, fmt={"n_drugs": "d", "mean": "+.4f", "ci_lo": "+.4f", "ci_hi": "+.4f",
            "p": ".3g"})


def verdict_posthoc(frame: pd.DataFrame) -> dict:
    """The frozen rule with **one** change: the control criterion's statistic.

    Not a re-run of the analysis with thresholds moved until something passes.
    Exactly one substitution, forced by a fact the pre-registration did not
    anticipate and the results made unmissable: in E1 one endpoint is a training
    drug, so skill-against-zero measures the seen partner and not entity
    transfer. The random-feature control -- containing no chemistry at all --
    posts *higher* E1 skill than real fingerprints, which is a demonstration
    that the registered statistic cannot distinguish the two things it was
    written to distinguish.

    So criteria (f) and the matching validity gate are evaluated on the
    **incremental pair skill**, which is the experiment's actual primary
    endpoint and on which the controls do collapse. Everything else -- every
    threshold, every other criterion, the ordering -- is untouched, and
    :func:`verdict` still reports the rule exactly as registered.
    """
    ok = _ok(frame)
    out = verdict(frame)
    controls = {}
    for rep, tag in (("random", "control_random"), ("shuffled", "control_shuffled")):
        inc = incremental(ok[ok["tag"] == tag])
        controls[rep] = ({"present": False} if inc.empty else
                         {"present": True, "n_folds": int(len(inc)),
                          "mean": float(inc["incremental_skill"].mean()),
                          "max": float(inc["incremental_skill"].max())})
    d = DECISION
    invalid = [r for r in out["invalidating_reasons"] if "representation posts" not in r]
    for rep, c in controls.items():
        if c.get("present") and c["mean"] > d["control_invalidates_above"]:
            invalid.append(f"the {rep} representation posts mean incremental pair skill "
                           f"{c['mean']:+.4f} > {d['control_invalidates_above']}")
    criteria = dict(out["criteria"])
    criteria["f_controls_collapse"] = all(
        (not c.get("present")) or c["mean"] <= d["control_ceiling"]
        for c in controls.values())

    pair_terms = ("a_both_positive", "b_incremental_above_floor",
                  "c_both_tests_significant", "d_folds_favouring",
                  "e_pearson_positive", "f_controls_collapse")
    pair_specific = all(criteria[k] for k in pair_terms)
    label = _classify(invalid, True, criteria, pair_specific,
                      out["_per_screen_pot_low"], out["_per_screen_incr_fails"])

    # What criterion (g) would have said with the significance requirement it
    # should have carried. Reported, never substituted -- the rule is the rule.
    inc = per_drug_incremental(per_drug_table(ok))
    low_significant = {}
    if not inc.empty:
        for screen, g in inc.groupby("screen"):
            v = g[g["stratum"] == "low"]["incremental_skill"].to_numpy()
            s = paired_summary(v)
            low_significant[screen] = {
                "n_drugs": s["n"], "mean": s["mean"], "p": s["p_ttest"],
                "significant_at_0.05": bool(s["p_ttest"] < d["alpha"] and s["mean"] > 0)}
    return {
        "verdict_posthoc": label,
        "single_change": "criterion (f) and its validity gate read incremental pair "
                         "skill instead of skill-against-zero",
        "criteria": criteria,
        "invalidating_reasons": invalid,
        "controls_on_incremental": controls,
        "low_similarity_stratum": low_significant,
        "g_as_written_passes": criteria["g_not_analogue_confined"],
    }


def e1_composition_table(frame: pd.DataFrame, tag: str = "primary",
                         representation: str = "ecfp4") -> str:
    """E1 split by whether the *partner* was trained on.

    E1 is not homogeneous and the documents said it was. A pair with exactly one
    test endpoint lands in E1 whether its partner is a training drug or a
    **validation** drug — and validation drugs appear in no training pair at all.
    So 800 of the 900 rows per fold are test-x-trained and 100 are
    test-x-untrained: a second both-unseen regime sitting inside the primary one.
    Since the whole reinterpretation of this phase rests on what the partner
    supplies, the split is reported rather than left implicit.
    """
    ok = _ok(frame)
    if "e1tr_skill" not in ok.columns:
        return "_no composition split_"
    sel = ok[(ok["tag"] == tag) & (ok["representation"] == representation)]
    rows = []
    for (screen, model), g in sel.groupby(["screen", "model"], sort=False):
        if model == "zero":
            continue
        rows.append({
            "screen": screen, "model": model,
            "n_tr": int(g["e1tr_n_pairs"].mean()), "tr": float(g["e1tr_skill"].mean()),
            "n_va": int(g["e1va_n_pairs"].mean()), "va": float(g["e1va_skill"].mean()),
            "all": float(g["e1_skill"].mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return "_no composition split_"
    order = {m: k for k, m in enumerate(
        ("potential", "lowrank", "antisym_mlp", "pair_only"))}
    out = out.sort_values(["screen", "model"],
                          key=lambda c: c.map(order) if c.name == "model" else c)
    return _md(out, {
        "screen": "screen", "model": "model",
        "n_tr": "test x trained pairs", "tr": "skill there",
        "n_va": "test x untrained pairs", "va": "skill there", "all": "E1 overall",
    }, fmt={"n_tr": "d", "n_va": "d", "tr": "+.4f", "va": "+.4f", "all": "+.4f"})


def e2_table(frame: pd.DataFrame, tag: str = "primary",
             representation: str = "ecfp4") -> str:
    """E2 with intervals and a power figure, not a bare point estimate.

    E2 has 45 pairs per fold against E1's 900. Reporting its negative fold-mean
    as "nothing transfers" states an established null from an arm that could only
    have detected transfer of nearly E1 magnitude, so the interval and the
    minimum detectable effect are printed alongside.
    """
    ok = _ok(frame)
    if "e2_skill" not in ok.columns:
        return "_no E2 rows_"
    sel = ok[(ok["tag"] == tag) & (ok["representation"] == representation)]
    rows = []
    for (screen, model), g in sel.groupby(["screen", "model"], sort=False):
        if model == "zero":
            continue
        v = g["e2_skill"].astype(float).to_numpy()
        s = paired_summary(v)
        # Pair-pooled skill: a mean of per-fold ratios over 45 pairs is a biased
        # summary, so the pooled MSE ratio is given beside it.
        pooled = 1.0 - float(g["e2_mse"].sum()) / float(g["e2_mse_zero"].sum())
        mde = ((stats.norm.ppf(0.975) + stats.norm.ppf(0.80))
               * s["sd"] / math.sqrt(s["n"])) if s["n"] > 1 else float("nan")
        rows.append({"screen": screen, "model": model, "n": s["n"],
                     "fold_mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "pooled": pooled,
                     "n_positive": f"{s['n_positive']}/{s['n']}",
                     "p": s["p_ttest"], "mde": mde})
    out = pd.DataFrame(rows)
    if out.empty:
        return "_no E2 rows_"
    return _md(out, {
        "screen": "screen", "model": "model", "n": "folds",
        "fold_mean": "fold-mean skill", "ci_lo": "CI lo", "ci_hi": "CI hi",
        "pooled": "pair-pooled skill", "n_positive": "folds positive",
        "p": "p vs zero", "mde": "detectable at 80%",
    }, fmt={"n": "d", "fold_mean": "+.4f", "ci_lo": "+.4f", "ci_hi": "+.4f",
            "pooled": "+.4f", "p": ".3g", "mde": "+.4f"})


def chemical_clusters(features, threshold: float = 0.5) -> np.ndarray:
    """Single-linkage clusters of drugs at a Tanimoto threshold.

    The per-drug analysis treats 100 drugs as 100 independent observations, and
    they are not: the high-similarity stratum is built out of congener families —
    nucleoside analogues, vincas, taxanes, anthracyclines — which are similar to
    *each other*, so a family contributes one piece of evidence, not seven. This
    is the grouping the cluster bootstrap resamples.
    """
    from .features import tanimoto_matrix

    sim = np.nan_to_num(tanimoto_matrix(features), nan=0.0)
    n = len(sim)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if sim[a, b] >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(a) for a in range(n)])


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat,
                      n_boot: int = 2000, seed: int = 20260826) -> dict:
    """Resample whole clusters, not drugs, and report the interval that gives.

    A naive per-drug interval assumes the 100 drugs are independent. Resampling
    congener families instead is the honest version of the same statistic, and it
    is what decides whether an analogue-gradient claim survives.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        if len(idx) < 3:
            continue
        try:
            boots.append(stat(values[idx], idx))
        except Exception:                                # noqa: BLE001
            continue
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    point = stat(values, np.arange(len(values)))
    if not len(boots):
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_two_sided": float("nan"), "n_clusters": len(uniq)}
    return {
        "point": float(point),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        # Fraction of resamples on the far side of zero, doubled: the bootstrap
        # analogue of a two-sided p, and deliberately not a t-test on clusters,
        # of which there are too few for one.
        "p_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
        "n_clusters": int(len(uniq)),
    }


def cluster_robust_table(frame: pd.DataFrame, mapping=None,
                         threshold: float = 0.5) -> str:
    """The analogue-gradient statistics, resampled by congener family."""
    from .features import fingerprint_matrix

    if mapping is None:
        import pandas as _pd

        from .experiment import DEFAULT_MAPPING

        if not DEFAULT_MAPPING.exists():
            return "_mapping not available_"
        mapping = _pd.read_csv(DEFAULT_MAPPING)
    feats = fingerprint_matrix(mapping)
    clusters = chemical_clusters(feats, threshold)

    inc = per_drug_incremental(per_drug_table(_ok(frame)))
    if inc.empty:
        return "_no per-drug rows_"
    rows = []
    for screen, g in inc.groupby("screen"):
        g = g.sort_values("drug")
        drugs = g["drug"].to_numpy()
        cl = clusters[drugs]
        vals = g["incremental_skill"].to_numpy()
        sims = g["max_sim_to_train"].to_numpy()
        strat = g["stratum"].to_numpy()

        def diff(v, idx):
            hi, lo = v[strat[idx] == "high"], v[strat[idx] == "low"]
            return hi.mean() - lo.mean() if len(hi) and len(lo) else float("nan")

        def rho(v, idx):
            return stats.spearmanr(sims[idx], v).statistic

        for name, fn in (("high minus low", diff), ("Spearman rho vs similarity", rho)):
            b = cluster_bootstrap(vals, cl, fn)
            rows.append({"screen": screen, "statistic": name,
                         "n_drugs": len(vals), "n_clusters": b["n_clusters"],
                         "point": b["point"], "ci_lo": b["ci_lo"],
                         "ci_hi": b["ci_hi"], "p": b["p_two_sided"]})
        # And the same rho restricted to drugs below the high cut, which tests
        # whether the relation is a gradient or a step at the top.
        below = strat != "high"
        if below.sum() > 5:
            r = stats.spearmanr(sims[below], vals[below])
            rows.append({"screen": screen, "statistic": "Spearman rho, below the high cut",
                         "n_drugs": int(below.sum()), "n_clusters": len(np.unique(cl[below])),
                         "point": float(r.statistic), "ci_lo": float("nan"),
                         "ci_hi": float("nan"), "p": float(r.pvalue)})
    return _md(pd.DataFrame(rows), {
        "screen": "screen", "statistic": "statistic", "n_drugs": "drugs",
        "n_clusters": "congener families", "point": "value",
        "ci_lo": "CI lo", "ci_hi": "CI hi", "p": "p",
    }, fmt={"n_drugs": "d", "n_clusters": "d", "point": "+.4f",
            "ci_lo": "+.4f", "ci_hi": "+.4f", "p": ".3g"})


def blind_contrast_table(frame: pd.DataFrame, better: str = "lowrank",
                         base: str = "potential") -> str:
    """Does the pair term's advantage require the unseen drug's own structure?

    The sharpest form of the blind diagnostic, and the one the conclusion rests
    on. Rather than asking how much skill a single model loses when blinded --
    a quantity that depends on where the baseline sits -- it recomputes the
    **primary contrast itself** under blinding:

        full     = 1 - MSE(low-rank) / MSE(potential)
        blinded  = the same ratio when the unseen endpoint is replaced by the
                   marginal over the 80 training drugs, for both models

    Both models are blinded identically, so the comparison is immune to the
    baseline's absolute position; what is left is exactly "does knowing which
    molecule this is buy the pair term anything". The positive-control row
    calibrates it: there the planted rank-2 signal is genuinely a function of
    the drugs' features, and blinding removes essentially all of it.
    """
    ok = _ok(frame)
    if "e1_blind_mse" not in ok.columns:
        return "_no blind diagnostics_"
    rows = []
    for (tag, rep), sub in ok.groupby(["tag", "representation"], sort=False):
        for screen, g in sub.groupby("screen"):
            a = g[g["model"] == better].set_index("fold_key")
            b = g[g["model"] == base].set_index("fold_key")
            k = a.index.intersection(b.index)
            if not len(k):
                continue
            full = 1.0 - a.loc[k, "e1_mse"].to_numpy() / b.loc[k, "e1_mse"].to_numpy()
            blind = (1.0 - a.loc[k, "e1_blind_mse"].to_numpy()
                     / b.loc[k, "e1_blind_mse"].to_numpy())
            s = paired_summary(full - blind)
            rows.append({"tag": tag, "representation": rep, "screen": screen,
                         "n": s["n"], "full": float(full.mean()),
                         "blinded": float(blind.mean()), "difference": s["mean"],
                         "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                         "p": s["p_ttest"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return "_no blind diagnostics_"
    return _md(out, {
        "tag": "block", "representation": "representation", "screen": "screen",
        "n": "folds", "full": "incremental, full", "blinded": "incremental, blinded",
        "difference": "difference", "ci_lo": "CI lo", "ci_hi": "CI hi", "p": "paired t",
    }, fmt={"n": "d", "full": "+.4f", "blinded": "+.4f", "difference": "+.4f",
            "ci_lo": "+.4f", "ci_hi": "+.4f", "p": ".3g"})
