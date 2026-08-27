"""Ensemble analysis, the real-vs-null comparison, and the preregistered verdict.

The decision rule is **executed**, not applied by hand: :func:`verdict` reads the
thresholds from :data:`DECISION` -- which encodes the rule committed in
``docs/PREREGISTRATIONS.md`` before the ensemble ran -- and returns the classification
together with every quantity that produced it. A verdict a person types is a
verdict nobody can check.

Real reference values are read from the generated Phase 2R artifacts on every
call, never transcribed, so this module cannot drift away from the numbers the
comparison is against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2R = Path("results/phase2_residual")
OUT_DIR = Path("results/dchain_null")
METRICS = OUT_DIR / "metrics.jsonl"

#: The preregistered thresholds. Changing any value here after the ensemble has
#: run is changing the decision rule after seeing the result; the numbers are
#: quoted in ``docs/PREREGISTRATIONS.md`` under the pre-registration and
#: ``test_decision_thresholds_match_the_preregistration`` pins them to it.
DECISION = {
    "coverages": (0.40, 0.70),
    #: Criterion A: the null median reaches this fraction of the real value.
    "artifact_fraction_of_real": 0.5,
    #: Criterion B / C: "clearly positive" skill. Ten times the largest
    #: |cal_skill| the real `potential` rung reaches at coverage >= 0.20.
    "clearly_positive_skill": 0.02,
    #: Criterion D: fraction of ensemble runs allowed to fail or be incomplete.
    "max_failure_fraction": 0.20,
    #: Criterion D: fraction of the *planned* ensemble that must be present. A
    #: partial file must not be able to produce a confident verdict.
    "min_fraction_of_planned": 0.80,
    #: Criterion D: the combination selector must not be shut. The measure
    #: carries lambda_AB as a multiplicative factor, so a null in which the
    #: posterior gate closes is a world where the artifact CANNOT express, and a
    #: negative result there says nothing about the estimator. The real deposit
    #: runs at mean |lambda| = 0.492 (A375) / 0.464 (PANC1).
    "min_selector_on_fraction": 0.10,
    #: Criterion D: reproducibility of the analysed matrix. Below this, the
    #: posterior-mean directional matrix is more chain noise than structure and
    #: "no skill" would be uninformative.
    #:
    #: This replaced a first version that used the *posterior* noise fraction
    #: 2*mean(sd^2)/mean(D^2) against the real screens' 0.205 / 0.192. A pipeline
    #: check on 60 drugs showed why that was broken: under a correct null the
    #: true D is zero, so the posterior means shrink toward zero while each keeps
    #: its own posterior uncertainty, and the ratio comes out above 1 **by
    #: construction**. A criterion that fires on every correctly-built null is
    #: not a criterion. The quantity that actually bears on whether there is
    #: something to find is how reproducible the analysed *means* are, which is
    #: the split-half agreement -- and it is a quantity the real deposit cannot
    #: report at all, since the published fit was one unseeded chain.
    #: The posterior noise fraction is still recorded, and reported beside the
    #: real values with that caveat.
    "min_split_half_pearson_D": 0.50,
    #: The i.i.d.-noise floor for top-2 curl energy at n=100. Registered as
    #: 0.076 from 5 draws; remeasured on 20 draws as 0.0747 +/- 0.0038, which is
    #: the value used. Reported, never decisive.
    "noise_floor_top2": 0.0747,
    #: Descriptive only. The first draft made this a criterion-D trigger, which
    #: was an error: cal_skill is *exactly* invariant to the size of D (verified
    #: over a 250x range), so the band policed a quantity the primary metric
    #: provably cannot see, and it would have fired INCONCLUSIVE precisely in the
    #: "real but small artifact" case the reconstruction says is likely.
    "scale_factor_tolerance": 5.0,
}


def load_rows(path: Path = METRICS) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/run_dchain_null.py")
    return [json.loads(line) for line in open(path) if line.strip()]


def real_reference() -> dict:
    """The committed Phase 2R values this null is compared against.

    Read from ``hodge_decomposition.json`` and ``rank2.csv`` rather than typed,
    because a transcribed reference is a reference nobody rechecks -- the exact
    failure mode Phase 2R's own audit found in its section 8.3.
    """
    hodge = json.loads((PHASE2R / "summary" / "hodge_decomposition.json").read_text())
    rank2 = pd.read_csv(PHASE2R / "summary" / "rank2.csv")
    out: dict[str, dict] = {}
    for screen, h in hodge.items():
        r = rank2[(rank2["screen"] == screen) & (rank2["metric"] == "cal_skill")]
        skill = {float(c): float(v) for c, v in zip(r["coverage"], r["mean"])}
        out[screen] = {
            "curl_fraction": float(h["curl_fraction"]),
            "grad_fraction": float(h["grad_fraction"]),
            "top_k_energy": {k: float(v) for k, v in h["curl_rank_energy"].items()},
            "rank2_skill": skill,
            "D_mean_square": float(h["D_mean_square"]),
            "D_std_offdiag": float(h["D_std_offdiag"]),
        }
    # The searched low-rank rung under the shrinkage the decision quotes, so the
    # secondary block has a matched reference too.
    ha = pd.read_csv(PHASE2R / "summary" / "honest_alpha.csv")
    ha = ha[(ha["rung"] == "lowrank") & (ha["metric"] == "cal_skill")]
    for screen, g in ha.groupby("screen", observed=True):
        out[screen]["honest_lowrank_skill"] = {
            float(c): float(v) for c, v in zip(g["coverage"], g["mean"])}
    return out


def is_usable(row: dict) -> bool:
    """The preregistered exclusion rule, and the only one.

    "A run is excluded only if the sampler exits nonzero or if
    ``n_samples != n_samples_expected``." No run is excluded on the basis of the
    structure it produces.

    This was written into the pre-registration and **not implemented**: the
    verdict counted incomplete runs and then used them anyway. An independent
    reviewer showed the cost -- two truncated runs out of twenty, at 10% and so
    inside the 20% failure allowance, drag the null's 97.5th percentile up far
    enough that a real value lands inside the null interval and the verdict flips
    from "little evidence for estimator artifact" to "estimator artifact
    reproduces result". One outlier is nearly enough on its own.
    """
    if "error" in row:
        return False
    d = row.get("diagnostics") or {}
    exp = d.get("n_samples_expected")
    if exp is not None and d.get("n_samples") != exp:
        return False
    return True


def _phase2r_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per (simulated screen, block, coverage, rung), averaged over splits."""
    recs = []
    for r in rows:
        if not is_usable(r) or not r.get("phase2r"):
            continue
        for p in r["phase2r"]:
            recs.append({
                "tag": r["tag"], "estimator": r["estimator"],
                "variant": r["variant"], "sigma_obs": r["sigma_obs"],
                "sim_seed": r["sim_seed"], "est_seed": r["est_seed"],
                "block": p["block"], "coverage": p["coverage"],
                "rung": p["rung"], "split_seed": p["split_seed"],
                "cal_skill": p["cal_skill"], "cal_pearson": p.get("cal_pearson"),
                "cal_spearman": p.get("cal_spearman"),
                "cal_sign_accuracy": p.get("cal_sign_accuracy"),
                "heldout_skill": p["heldout_skill"],
                "n_params": p["n_params"],
            })
    return pd.DataFrame(recs)


def per_screen(rows: list[dict]) -> pd.DataFrame:
    """The ensemble unit: one row per simulated screen, per block and coverage.

    The unit of the null distribution is the **simulated screen**, not the split
    seed: split seeds within a screen share the same estimated matrix and are not
    independent draws from the null. Averaging them first and treating the screen
    as the unit is the only way the empirical percentile means what it says.
    """
    df = _phase2r_frame(rows)
    if df.empty:
        return df
    g = df.groupby(["tag", "estimator", "variant", "sigma_obs", "sim_seed",
                    "est_seed", "block", "coverage", "rung"],
                   observed=True, dropna=False)
    out = g.agg(cal_skill=("cal_skill", "mean"),
                cal_skill_sd=("cal_skill", "std"),
                cal_pearson=("cal_pearson", "mean"),
                cal_spearman=("cal_spearman", "mean"),
                cal_sign_accuracy=("cal_sign_accuracy", "mean"),
                n_split_seeds=("split_seed", "nunique"),
                n_params=("n_params", "max")).reset_index()
    return out


def structure_frame(rows: list[dict]) -> pd.DataFrame:
    """Matrix-structure metrics per simulated screen, with no model involved."""
    recs = []
    for r in rows:
        if not is_usable(r):
            continue
        e = r["estimated_decomposition"]
        t = r["true_decomposition"]
        a = r.get("artifact_decomposition", {})
        d = r.get("diagnostics", {})
        rec = {
            "tag": r["tag"], "estimator": r["estimator"], "variant": r["variant"],
            "sigma_obs": r["sigma_obs"], "sim_seed": r["sim_seed"],
            "est_seed": r["est_seed"], "seconds": r.get("seconds"),
            "true_synergy_rms": r["true_synergy_rms"],
            "true_pair_interaction_is_zero": r["true_pair_interaction_is_zero"],
            "true_curl_fraction": t.get("curl_fraction"),
            "true_D_is_zero": t.get("D_is_identically_zero"),
            "est_synergy_rms": e.get("synergy_rms"),
            "est_D_std": e.get("D_std_offdiag"),
            "est_D_mean_square": e.get("D_mean_square"),
            "curl_fraction": e.get("curl_fraction"),
            "grad_fraction": e.get("grad_fraction"),
            "artifact_rms": r.get("artifact_rms"),
            "artifact_curl_fraction": a.get("curl_fraction"),
            "estimate_truth_pearson": r.get("estimate_truth_pearson"),
            "converged": d.get("converged"),
            "n_samples": d.get("n_samples"),
            "n_samples_expected": d.get("n_samples_expected"),
            "selector_on_fraction": d.get("selector_on_fraction"),
            "split_half_pearson_D": d.get("split_half_pearson_D"),
            "posterior_noise_fraction_of_D": d.get("posterior_noise_fraction_of_D"),
            "offset_error_rms": d.get("offset_error_rms"),
            "second_position_gain_mean": d.get("second_position_gain_mean"),
            "second_position_gain_sd": d.get("second_position_gain_sd"),
            "template_r2": (r.get("artifact_template") or {}).get("r2"),
            "template_subspace_overlap":
                (r.get("artifact_template") or {}).get("subspace_overlap"),
        }
        # The rank-2 cyclic energy in the screen's OWN units, not as a
        # fraction. Both the skill criterion and the rank-2 *share* are ratios,
        # so neither answers "how much of the real screen's directional energy
        # could this artifact account for". This does. Reported, never a
        # criterion -- it was added while the ensemble ran and before any
        # primary condition had completed, and adding a criterion at that point
        # would not have been legitimate.
        cf, t2 = e.get("curl_fraction"), e.get("top_k_energy", {}).get("2")
        dms = e.get("D_mean_square")
        rec["rank2_energy_absolute"] = (
            float(cf * t2 * dms) if None not in (cf, t2, dms) else float("nan"))
        for k in ("1", "2", "4", "8", "16", "32", "64"):
            rec[f"top{k}"] = e.get("top_k_energy", {}).get(k)
            rec[f"artifact_top{k}"] = a.get("top_k_energy", {}).get(k)
        recs.append(rec)
    return pd.DataFrame(recs)


def percentile_of(value: float, null: np.ndarray) -> dict:
    """Where a real value falls in the null ensemble, and the one-sided tail.

    The tail probability is the standard ``(1 + #{null >= value}) / (n + 1)``
    permutation-style estimate, which is bounded below by ``1/(n+1)`` and never
    reports zero -- 20 simulations cannot support a p smaller than 1/21.
    """
    null = np.asarray([v for v in null if np.isfinite(v)], dtype=float)
    n = len(null)
    if not np.isfinite(value):
        # A comparison that was never made must not report a percentile of 0 and
        # the most extreme p the design can produce, which is what the first
        # draft did for the rows whose real value is not defined.
        return {"n_null": n, "percentile": float("nan"),
                "p_one_sided": float("nan"),
                **({} if n == 0 else {
                    "null_median": float(np.median(null)),
                    "null_mean": float(null.mean()),
                    "null_min": float(null.min()),
                    "null_max": float(null.max()),
                    "null_q025": float(np.quantile(null, 0.025)),
                    "null_q975": float(np.quantile(null, 0.975))})}
    if n == 0:
        return {"n_null": 0, "percentile": float("nan"), "p_one_sided": float("nan")}
    ge = int((null >= value).sum())
    return {
        "n_null": n,
        "null_median": float(np.median(null)),
        "null_mean": float(null.mean()),
        "null_min": float(null.min()),
        "null_max": float(null.max()),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "percentile": float(100.0 * (null < value).mean()),
        "p_one_sided": float((1 + ge) / (n + 1)),
        "real_minus_null_median": float(value - np.median(null)),
        # Standardised, not a ratio. A ratio is undefined at a null median of
        # zero -- which is the MODAL case here, because the shrinkage selects
        # alpha = 0 whenever the rung finds nothing and cal_skill is then
        # exactly 0 -- and it silently flips sign on a negative median.
        "real_minus_null_median_over_sd": (
            float((value - np.median(null)) / null.std())
            if n > 1 and null.std() > 0 else float("nan")),
    }


def _null_skill(ps: pd.DataFrame, tag: str, block: str, rung: str,
                coverage: float) -> np.ndarray:
    sel = ps[(ps["tag"] == tag) & (ps["block"] == block)
             & (ps["rung"] == rung) & (np.isclose(ps["coverage"], coverage))]
    return sel["cal_skill"].to_numpy()


def comparison_table(rows: list[dict], real: dict | None = None,
                     tag: str = "primary") -> pd.DataFrame:
    """The headline table: null distribution beside both real screens.

    ``kind`` separates the rows a verdict may be read from (``decision``) from
    the ones that only describe the two worlds (``descriptive``). A p-value is
    emitted **only** for decision rows: the first draft attached one to the
    cyclic fraction -- which the pre-registration explicitly rejects as a
    discriminator -- and to a comparison whose real value is undefined, where it
    reported the most extreme value the design can produce for a comparison
    never made.
    """
    real = real or real_reference()
    ps, st = per_screen(rows), structure_frame(rows)
    if st.empty:
        return pd.DataFrame()
    s = st[st["tag"] == tag]
    out = []

    def add(metric, null_vals, a_val, p_val, kind="decision"):
        sa, sb = percentile_of(a_val, null_vals), percentile_of(p_val, null_vals)
        out.append({
            "metric": metric, "kind": kind, "n_null": sa["n_null"],
            "null_median": sa.get("null_median"),
            "null_q025": sa.get("null_q025"), "null_q975": sa.get("null_q975"),
            "null_min": sa.get("null_min"), "null_max": sa.get("null_max"),
            "real_A375": a_val, "real_PANC1": p_val,
            "pct_A375": sa["percentile"] if kind == "decision" else float("nan"),
            "pct_PANC1": sb["percentile"] if kind == "decision" else float("nan"),
            "p_A375": sa["p_one_sided"] if kind == "decision" else float("nan"),
            "p_PANC1": sb["p_one_sided"] if kind == "decision" else float("nan"),
            "gap_sd_A375": sa.get("real_minus_null_median_over_sd", float("nan")),
        })

    # --- the two decision statistics ----------------------------------------
    for cov in DECISION["coverages"]:
        add(f"rank-2 held-out skill @ coverage {cov:.2f}",
            _null_skill(ps, tag, "rank2", "lowrank", cov),
            real["A375"]["rank2_skill"].get(cov, float("nan")),
            real["PANC1"]["rank2_skill"].get(cov, float("nan")))
    add("rank-2 cyclic share of D² (curl fraction × top-2)",
        (s["curl_fraction"] * s["top2"]).to_numpy(),
        real["A375"]["curl_fraction"] * real["A375"]["top_k_energy"]["2"],
        real["PANC1"]["curl_fraction"] * real["PANC1"]["top_k_energy"]["2"])

    # --- the secondary block, matched to the honest-alpha real values -------
    for cov in DECISION["coverages"]:
        add(f"searched low-rank skill @ coverage {cov:.2f}",
            _null_skill(ps, tag, "honest_alpha", "lowrank", cov),
            real["A375"].get("honest_lowrank_skill", {}).get(cov, float("nan")),
            real["PANC1"].get("honest_lowrank_skill", {}).get(cov, float("nan")))

    # --- the coverage transition H_artifact-null predicts --------------------
    for cov in sorted({float(c) for c in ps["coverage"].unique()}):
        if cov in DECISION["coverages"]:
            continue
        add(f"rank-2 held-out skill @ coverage {cov:.2f}",
            _null_skill(ps, tag, "rank2", "lowrank", cov),
            real["A375"]["rank2_skill"].get(cov, float("nan")),
            real["PANC1"]["rank2_skill"].get(cov, float("nan")))

    # --- description of the two worlds. No p-values. -------------------------
    add("cyclic fraction of D  (i.i.d. noise at n=100: 0.980)",
        s["curl_fraction"].to_numpy(), real["A375"]["curl_fraction"],
        real["PANC1"]["curl_fraction"], kind="descriptive")
    for k in ("2", "4", "16"):
        add(f"curl energy in top {k}  (noise floor {DECISION['noise_floor_top2']:.3f} at k=2)",
            s[f"top{k}"].to_numpy(), real["A375"]["top_k_energy"][k],
            real["PANC1"]["top_k_energy"][k], kind="descriptive")
    add("spread of D (sd, off-diagonal)", s["est_D_std"].to_numpy(),
        real["A375"]["D_std_offdiag"], real["PANC1"]["D_std_offdiag"],
        kind="descriptive")
    add("combination selector on-fraction", s["selector_on_fraction"].to_numpy(),
        0.4916, 0.4635, kind="descriptive")
    add("posterior noise fraction of D  (see DECISION)",
        s["posterior_noise_fraction_of_D"].to_numpy(), 0.205, 0.192,
        kind="descriptive")
    # In the screens' own units: mean(D²) × curl fraction × top-2. The only
    # quantity here that is not a ratio, and the one that answers "how much of
    # the real directional energy could this artifact account for".
    add("rank-2 cyclic energy, absolute (mean D² × curl frac × top-2)",
        s["rank2_energy_absolute"].to_numpy(),
        real["A375"]["D_mean_square"] * real["A375"]["curl_fraction"]
        * real["A375"]["top_k_energy"]["2"],
        real["PANC1"]["D_mean_square"] * real["PANC1"]["curl_fraction"]
        * real["PANC1"]["top_k_energy"]["2"], kind="descriptive")
    return pd.DataFrame(out)


def mechanism_table(rows: list[dict], tag: str = "primary") -> pd.DataFrame:
    """The zero-free-parameter test of the predicted artifact, and its inputs."""
    st = structure_frame(rows)
    if st.empty:
        return pd.DataFrame()
    cols = ["offset_error_rms", "second_position_gain_mean",
            "second_position_gain_sd", "template_r2",
            "template_subspace_overlap", "split_half_pearson_D",
            "selector_on_fraction"]
    out = []
    for t_, g in st.groupby("tag", observed=True):
        rec = {"block": t_, "n": len(g)}
        for c in cols:
            v = g[c].dropna() if c in g.columns else pd.Series(dtype=float)
            rec[c] = float(v.median()) if len(v) else float("nan")
            rec[f"{c}_max"] = float(v.max()) if len(v) else float("nan")
        out.append(rec)
    return pd.DataFrame(out)


def _planned(tag: str | None = None) -> int:
    """How many runs the preregistered ensemble specifies, for a tag or in total.

    Read from the grids, so an interrupted run is detected as an incomplete
    ensemble rather than silently treated as a complete one.
    """
    from . import grids
    try:
        jobs = grids.part_jobs("all")
    except Exception:                                    # pragma: no cover
        return 0
    return len(jobs if tag is None else [c for c in jobs if c.null.tag == tag])


def ensemble_completeness(rows: list[dict]) -> dict:
    """Which preregistered blocks are present, and which are missing entirely.

    The first amendment said "a partial ensemble must not produce a confident
    verdict" and the check was written **per tag** -- so an ensemble missing the
    whole realism arm, the whole noise sweep, the whole convergence block and
    half of Control A still returned no criterion-D reason, because the primary
    tag was complete. A final adversarial reviewer caught it after the primary
    verdict had been committed.

    That omission was not neutral. The project's own second amendment measured
    the realism arm as the one where the artifact is **larger** -- selector gate
    0.33 against strict's 0.21, artifact top-2 curl energy 0.294 against the real
    screens' 0.340/0.321 -- and raised it from 10 seeds to 20 for that reason.
    Declaring on the strict arm alone is declaring on the weaker one.
    """
    from . import grids
    planned: dict[str, int] = {}
    for c in grids.part_jobs("all"):
        planned[c.null.tag] = planned.get(c.null.tag, 0) + 1
    present: dict[str, int] = {}
    for r in rows:
        if is_usable(r):
            present[r.get("tag")] = present.get(r.get("tag"), 0) + 1
    return {
        "planned": planned, "present": {k: present.get(k, 0) for k in planned},
        "n_planned": sum(planned.values()),
        "n_present": sum(present.get(k, 0) for k in planned),
        "missing_blocks": sorted(k for k in planned if present.get(k, 0) == 0),
        "incomplete_blocks": sorted(k for k in planned
                                    if 0 < present.get(k, 0) < planned[k]),
    }


def verdict(rows: list[dict], real: dict | None = None,
            tag: str = "primary") -> dict:
    """Execute the preregistered decision rule. Returns the class and its inputs."""
    real = real or real_reference()
    ps, st = per_screen(rows), structure_frame(rows)
    if st.empty or "tag" not in st.columns:
        # No runs at all. There is no comparison to make, and the one thing this
        # function must never do is turn missing data into a finding.
        return {"tag": tag, "n_runs": 0, "verdict": "INCONCLUSIVE RECONSTRUCTION",
                "criterion_D_reasons": ["no runs present for this tag"]}
    prim = st[st["tag"] == tag]                       # usable runs only
    tagged = [r for r in rows if r.get("tag") == tag]
    n_failed = len([r for r in tagged if "error" in r])
    n_unconverged = len([r for r in tagged
                         if "error" not in r and not is_usable(r)])

    ev: dict = {"tag": tag, "n_runs": int(len(prim)),
                "n_failed": n_failed, "n_incomplete": n_unconverged,
                "n_unconverged": n_unconverged,
                "n_attempted": len(tagged)}

    # --- criterion D triggers, checked first -------------------------------
    d_reasons = []
    # The whole preregistered ensemble, not just this tag. See
    # ensemble_completeness for why the per-tag version was not enough.
    comp = ensemble_completeness(rows)
    ev["ensemble"] = comp
    if comp["n_planned"] and comp["n_present"] < (
            DECISION["min_fraction_of_planned"] * comp["n_planned"]):
        d_reasons.append(
            f"only {comp['n_present']} of {comp['n_planned']} preregistered "
            f"conditions have run ({comp['n_present']/comp['n_planned']:.0%}, "
            f"below the {DECISION['min_fraction_of_planned']:.0%} floor)"
            + (f"; blocks with no runs at all: "
               f"{', '.join(comp['missing_blocks'])}" if comp["missing_blocks"]
               else ""))
    elif comp["missing_blocks"]:
        d_reasons.append(
            f"these preregistered blocks have no runs at all: "
            f"{', '.join(comp['missing_blocks'])}")
    n_planned = _planned(tag)
    if n_planned and len(prim) < DECISION["min_fraction_of_planned"] * n_planned:
        d_reasons.append(
            f"only {len(prim)} of {n_planned} planned runs are present, below "
            f"the {DECISION['min_fraction_of_planned']:.0%} floor. A partial "
            f"ensemble must not produce a confident verdict")
    if n_planned and (n_failed + n_unconverged) / max(n_planned, 1) > \
            DECISION["max_failure_fraction"]:
        d_reasons.append(
            f"{n_failed} failed and {n_unconverged} incomplete of {n_planned}, "
            f"above the {DECISION['max_failure_fraction']:.0%} limit")

    # The oracle control. Under the STRICT null the true directional matrix is
    # exactly zero, so held-out skill on it is 0/0 and comes back NaN -- which
    # means the skill form of this control CANNOT fail there and is not the
    # check to rely on. The check that works under STRICT is that the true
    # matrix is identically zero; the skill check is the one that works under
    # NUISANCE, where the true matrix is nonzero but unpredictable.
    oracle = ps[ps["estimator"] == "oracle"]
    oracle_rank2 = oracle[(oracle["block"] == "rank2")
                          & (oracle["rung"] == "lowrank")]
    ev["oracle_max_skill"] = (float(oracle_rank2["cal_skill"].max())
                              if len(oracle_rank2) else float("nan"))
    if np.isfinite(ev["oracle_max_skill"]) and \
            ev["oracle_max_skill"] >= DECISION["clearly_positive_skill"]:
        d_reasons.append(
            f"the oracle control reaches rank-2 skill {ev['oracle_max_skill']:.3f} "
            f"on the TRUE matrix; the generative null is malformed")
    strict_true_zero = st[(st["estimator"] == "oracle")
                          & (st["variant"] == "strict")]["true_D_is_zero"]
    ev["oracle_strict_true_D_is_zero"] = (bool(strict_true_zero.all())
                                          if len(strict_true_zero) else None)
    if len(strict_true_zero) and not bool(strict_true_zero.all()):
        d_reasons.append("the strict null's true directional matrix is not "
                         "identically zero; the generative null is malformed")

    # The gate. The measure is lambda_AB * (...), so a null in which the
    # posterior selector closes is a world where the artifact cannot express,
    # and a negative result there is about the world, not the estimator.
    gate = prim["selector_on_fraction"].dropna()
    ev["selector_on_fraction_median"] = float(gate.median()) if len(gate) else float("nan")
    ev["selector_on_fraction_real"] = {"A375": 0.4916, "PANC1": 0.4635}
    if len(gate) and ev["selector_on_fraction_median"] < DECISION["min_selector_on_fraction"]:
        d_reasons.append(
            f"the combination selector is shut in the null (median "
            f"{ev['selector_on_fraction_median']:.3f} against 0.46-0.49 in the "
            f"deposit); the artifact has no channel to express through and a "
            f"negative result would be a property of the simulated world")

    # Is the analysed matrix reproducible? Split-half agreement of the
    # posterior-mean directional matrix between the two halves of the chain.
    sh = prim["split_half_pearson_D"].dropna()
    ev["split_half_pearson_D_median"] = float(sh.median()) if len(sh) else float("nan")
    if len(sh) and ev["split_half_pearson_D_median"] < DECISION["min_split_half_pearson_D"]:
        d_reasons.append(
            f"the null's directional matrix agrees between chain halves at only "
            f"r = {ev['split_half_pearson_D_median']:.2f}; it is more chain noise "
            f"than structure and 'no skill' would be uninformative")
    # Recorded and reported, not a trigger -- see DECISION.
    nf = prim["posterior_noise_fraction_of_D"].dropna()
    ev["posterior_noise_fraction_median"] = float(nf.median()) if len(nf) else float("nan")
    ev["posterior_noise_fraction_real"] = {"A375": 0.205, "PANC1": 0.192}

    # Descriptive only -- see DECISION["scale_factor_tolerance"].
    real_spread = np.mean([real[k]["D_std_offdiag"] for k in real])
    null_spread = float(prim["est_D_std"].median()) if len(prim) else float("nan")
    ev["null_D_std_median"] = null_spread
    ev["real_D_std_mean"] = float(real_spread)
    ev["scale_ratio"] = float(null_spread / real_spread) if real_spread else float("nan")

    ev["criterion_D_reasons"] = d_reasons

    # --- the skill comparison ------------------------------------------------
    # One cell per decision coverage. Each cell carries the null ensemble
    # summary, both real values, and the two derived quantities the rule reads:
    # the artifact threshold (half the weaker real screen) and whether either
    # real value falls inside the null's central 95%.
    ev["skill"] = {}
    for cov in DECISION["coverages"]:
        null = _null_skill(ps, tag, "rank2", "lowrank", cov)
        r_a = real["A375"]["rank2_skill"].get(cov, float("nan"))
        r_p = real["PANC1"]["rank2_skill"].get(cov, float("nan"))
        r_min = float(np.nanmin([r_a, r_p]))
        st_a, st_p = percentile_of(r_a, null), percentile_of(r_p, null)
        ev["skill"][f"{cov:.2f}"] = {
            "coverage": cov,
            "n_null": st_a["n_null"],
            "null_median": st_a.get("null_median", float("nan")),
            "null_min": st_a.get("null_min", float("nan")),
            "null_max": st_a.get("null_max", float("nan")),
            "null_q025": st_a.get("null_q025", float("nan")),
            "null_q975": st_a.get("null_q975", float("nan")),
            "real_A375": r_a, "real_PANC1": r_p, "real_min": r_min,
            "artifact_threshold": DECISION["artifact_fraction_of_real"] * r_min,
            "pct_A375": st_a["percentile"], "pct_PANC1": st_p["percentile"],
            "p_A375": st_a["p_one_sided"], "p_PANC1": st_p["p_one_sided"],
            # "the real result lies comfortably inside the simulated null" --
            # sufficient on its own for criterion A, for either screen. The
            # pre-registration states this at coverage 0.70; evaluating it at
            # every coverage would double the chances for a clause that is
            # already the most outlier-sensitive part of the rule, since a 95%
            # interval from 20 draws is essentially min-to-max.
            "real_inside_null_95": bool(
                cov == DECISION["coverages"][-1]
                and (st_a.get("null_q025", np.inf) <= r_a <= st_a.get("null_q975", -np.inf)
                     or st_p.get("null_q025", np.inf) <= r_p <= st_p.get("null_q975", -np.inf))),
        }

    # --- the spectral comparison --------------------------------------------
    # In ABSOLUTE units: curl_fraction x top2 is the share of the *directional*
    # energy that is rank-2 cyclic. The first draft used top2 alone, which is a
    # fraction of the curl and therefore scale-free -- and under the strict null
    # 100% of the curl is estimator artifact, so "is at least 13% of the
    # estimator's cyclic error rank-2?" is answered yes at ANY magnitude, down
    # to magnitudes a thousandfold below the real screen. That made criterion
    # A's spectral clause near-automatic and criterion C's near-impossible.
    t2_real = float(np.mean([real[k]["curl_fraction"] * real[k]["top_k_energy"]["2"]
                             for k in real]))
    abs_null = (prim["curl_fraction"] * prim["top2"]).dropna().to_numpy() \
        if len(prim) else np.array([])
    t2_null = float(np.median(abs_null)) if len(abs_null) else float("nan")
    ev["rank2_share_of_D"] = {
        "null_median": t2_null, "real_mean": t2_real,
        "real_A375": real["A375"]["curl_fraction"] * real["A375"]["top_k_energy"]["2"],
        "real_PANC1": real["PANC1"]["curl_fraction"] * real["PANC1"]["top_k_energy"]["2"],
        "artifact_threshold": DECISION["artifact_fraction_of_real"] * t2_real,
        **percentile_of(t2_real, abs_null)}
    # Reported beside it: the *ceiling* the real data puts on the artifact, from
    # the real spectrum alone. A pure rank-2 artifact has top-2 = 1 and pure
    # noise has 0.0747, so the artifact share of the real cyclic energy is at
    # most (top2 - floor) / (1 - floor).
    floor = DECISION["noise_floor_top2"]
    ev["artifact_ceiling_from_real_data"] = {
        k: {"max_share_of_curl": float((real[k]["top_k_energy"]["2"] - floor)
                                       / (1 - floor)),
            "max_share_of_D": float((real[k]["top_k_energy"]["2"] - floor)
                                    / (1 - floor) * real[k]["curl_fraction"])}
        for k in real}
    abs_e = prim["rank2_energy_absolute"].dropna() if len(prim) else pd.Series(dtype=float)
    real_abs = {k: real[k]["D_mean_square"] * real[k]["curl_fraction"]
                * real[k]["top_k_energy"]["2"] for k in real}
    ev["rank2_energy_absolute"] = {
        "null_median": float(abs_e.median()) if len(abs_e) else float("nan"),
        "null_max": float(abs_e.max()) if len(abs_e) else float("nan"),
        **{f"real_{k}": v for k, v in real_abs.items()},
        "null_median_over_real_A375": (float(abs_e.median() / real_abs["A375"])
                                       if len(abs_e) else float("nan")),
        "note": ("reported, not a criterion: added while the ensemble ran and "
                 "before any primary condition completed"),
    }
    ev["top2_fraction_of_curl"] = {
        "null_median": float(prim["top2"].median()) if len(prim) else float("nan"),
        "real_A375": real["A375"]["top_k_energy"]["2"],
        "real_PANC1": real["PANC1"]["top_k_energy"]["2"],
        "noise_floor": floor}

    if d_reasons:
        ev["verdict"] = "INCONCLUSIVE RECONSTRUCTION"
        return ev

    covs = [f"{c:.2f}" for c in DECISION["coverages"]]
    cells = [ev["skill"][c] for c in covs]
    # A verdict must never be produced from an empty comparison. If any decision
    # coverage has no null draws at all, that is a reconstruction failure, not a
    # finding of "no artifact".
    if any(c["n_null"] == 0 for c in cells):
        ev["verdict"] = "INCONCLUSIVE RECONSTRUCTION"
        ev["criterion_D_reasons"] = d_reasons + [
            "at least one decision coverage has no null draws"]
        return ev

    thr = DECISION["clearly_positive_skill"]
    artifact_skill = all(np.isfinite(c["null_median"])
                         and c["null_median"] >= c["artifact_threshold"]
                         for c in cells)
    artifact_spec = (np.isfinite(t2_null)
                     and t2_null >= ev["rank2_share_of_D"]["artifact_threshold"])
    inside = any(c["real_inside_null_95"] for c in cells)

    dense = cells[-1]
    partial = (np.isfinite(dense["null_median"]) and dense["null_median"] >= thr
               and dense["null_q025"] > 0.0)
    little = (all(np.isfinite(c["null_median"]) and c["null_median"] < thr
                  and c["real_A375"] > c["null_max"]
                  and c["real_PANC1"] > c["null_max"] for c in cells)
              and np.isfinite(t2_null)
              and t2_null < ev["rank2_share_of_D"]["artifact_threshold"])

    ev["criteria"] = {"artifact_skill": bool(artifact_skill),
                      "artifact_spectral": bool(artifact_spec),
                      "real_inside_null_95": bool(inside),
                      "partial": bool(partial), "little": bool(little)}

    if (artifact_skill and artifact_spec) or inside:
        ev["verdict"] = "ESTIMATOR ARTIFACT REPRODUCES RESULT"
    elif little:
        ev["verdict"] = "LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT"
    else:
        # PARTIAL is not a catch-all. The first draft made it the `else`, which
        # meant a null whose skill was clearly NEGATIVE at both coverages -- the
        # artifact hypothesis crushed -- was reported as "partial estimator
        # contribution". A null that finds strictly less than nothing is
        # evidence against an artifact, not for a small one.
        crushed = all(np.isfinite(c["null_median"]) and c["null_median"] <= 0.0
                      and c["real_A375"] > c["null_max"]
                      and c["real_PANC1"] > c["null_max"] for c in cells)
        ev["criteria"]["null_skill_crushed"] = bool(crushed)
        ev["verdict"] = ("LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT" if crushed
                         else "PARTIAL ESTIMATOR CONTRIBUTION")
    return ev
