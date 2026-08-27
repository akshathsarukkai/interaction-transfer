"""Aggregation and plotting for intervention-algebra sweep results.

Input is the JSONL file written by :func:`intervention_algebra.sweep.run_sweep`
-- one JSON object per run.  Rows that carry an ``error`` key are failed runs and
are dropped (and counted) at load time.

Typical use::

    python -m intervention_algebra.analysis --results results.jsonl --outdir figs

which prints markdown summary tables (family x pair_coverage, one block per
regime) to stdout and writes the four standard figures into ``--outdir``.

Everything here is nan-safe: metrics are undefined in some regimes (e.g.
``test_A_pearson`` in the symmetric-only regime), so all-NaN groups must
aggregate to NaN rather than raise.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "FAMILY_COLORS",
    "FAMILY_ORDER",
    "DEFAULT_METRICS",
    "load",
    "to_frame",
    "aggregate",
    "summary_table",
    "markdown_table",
    "paired_comparison",
    "condition_fields",
    "condition_key",
    "spec_condition_key",
    "select_conditions",
    "DuplicateRunsError",
    "make_figures",
    "main",
]

# ---------------------------------------------------------------- constants --

FAMILY_ORDER = ["additive", "unconstrained", "shared_pair", "algebra"]

#: Stable colour per model family so every figure reads the same way.
FAMILY_COLORS = {
    "additive": "#7f7f7f",
    "unconstrained": "#1f77b4",
    "shared_pair": "#ff7f0e",
    "algebra": "#2ca02c",
}
_FALLBACK_COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

#: Metrics shown in the default summary table, in display order.
DEFAULT_METRICS = [
    "test_mse",
    "test_rmse",
    "test_mse_sim",
    "test_mse_ord",
    "skill_vs_additive_family",
    "test_S_pearson",
    "test_A_pearson",
    "test_S_nrmse",
    "test_A_nrmse",
    "test_topo_auprc_S",
    "test_topo_auprc_A",
    "sym_residual",
    "antisym_residual",
    "pair_net_order_asymmetry",
]

#: Identifier columns -- never aggregated as metrics.
ID_COLS = [
    "family",
    "condition",
    "regime",
    "pair_coverage",
    "seed",
    "sparsity_mode",
    "observation_map",
    "tag",
]


def family_color(family: str) -> str:
    if family in FAMILY_COLORS:
        return FAMILY_COLORS[family]
    idx = abs(hash(family)) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), "")
    except ValueError:
        return (len(FAMILY_ORDER), str(family))


def _sorted_families(values: Iterable[str]) -> list[str]:
    return sorted({str(v) for v in values}, key=_family_sort_key)


# -------------------------------------------------------------------- load --


def load(path, return_errors: bool = False):
    """Read a sweep JSONL file, dropping (and counting) failed runs.

    Returns the list of successful run dicts.  With ``return_errors=True``
    returns ``(rows, error_rows)`` instead.
    """
    path = Path(path)
    rows: list[dict] = []
    errors: list[dict] = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[analysis] skipping unparseable line {lineno} of {path}: {exc}",
                      file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue
            if "error" in obj:
                errors.append(obj)
            else:
                rows.append(obj)
    if errors:
        print(f"[analysis] dropped {len(errors)} failed run(s) from {path}; "
              f"first error: {errors[0].get('error')!r}", file=sys.stderr)
    if return_errors:
        return rows, errors
    return rows


def to_frame(rows: Sequence[dict]) -> pd.DataFrame:
    """Build a tidy DataFrame from run dicts.

    Nested payloads (``config`` and friends) are dropped; everything that looks
    numeric is coerced to float so NaNs survive as NaNs rather than strings.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(list(rows))

    # Lift the few nested config fields the analysis needs before the nested
    # blob is dropped. Done here rather than recorded per-run so that it also
    # works on JSONL written before these columns existed.
    if "config" in df.columns:
        def _dig(cfg, *path):
            for key in path:
                if not isinstance(cfg, dict):
                    return None
                cfg = cfg.get(key)
            return cfg
        for name, path in (("max_epochs", ("train", "max_epochs")),
                           ("patience", ("train", "patience")),
                           ("emb_dim", ("model", "emb_dim")),
                           ("pair_mode", ("model", "pair_mode")),
                           ("n_interventions", ("system", "n_interventions")),
                           ("simultaneity_defect", ("system", "simultaneity_defect"))):
            if name not in df.columns:
                df[name] = df["config"].map(lambda c, p=path: _dig(c, *p))

    # The experimental-condition hash must be computed here, from the raw run
    # dicts, because the `config` blob it reads is dropped two lines below.
    if "config" in df.columns:
        df["condition"] = [condition_key(r) for r in rows]

    # Drop non-scalar columns (e.g. the full nested `config` blob).
    drop = [c for c in df.columns
            if df[c].map(lambda v: isinstance(v, (dict, list, tuple))).any()]
    if drop:
        df = df.drop(columns=drop)

    for col in df.columns:
        if col in ("family", "regime", "sparsity_mode", "observation_map", "tag"):
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # only accept the conversion if it did not destroy information
            if converted.notna().sum() >= df[col].notna().sum():
                df[col] = converted
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return add_skill_vs_additive_family(df)


#: Columns that identify one experimental *condition*; runs sharing these values
#: differ only in the model family, which is what makes them comparable.
CONDITION_COLS = ["tag", "regime", "sparsity_mode", "observation_map",
                  "pair_coverage", "seed"]


# --------------------------------------------------- experimental conditions --

#: Model-config fields that ``experiment._resolve_model_config`` overwrites at
#: run time from the *system* config. They are stored in the JSONL at their
#: pre-resolution default (e.g. ``n_interventions=48`` on a 64-intervention
#: system), so hashing them would compare values no run ever trained with. The
#: system fields they are derived from are in the key already.
_RESOLVED_FROM_SYSTEM = ("n_interventions", "out_dim", "observation_head")

#: Fields that identify a *replicate* rather than a condition.
_REPLICATE_FIELDS = ("seed",)


def condition_fields(row: dict) -> dict:
    """The experimental condition a run belongs to, as a flat sorted dict.

    Two runs share a condition **iff they differ only in their random seed** --
    that is, iff they are replicates of the same experiment and may be pooled
    into one cell. Everything that could make two runs incomparable is in here:
    every system parameter (regime, scales, sparsity mode, observation map,
    simultaneity defect), every split parameter (pair coverage, eval-pool
    fractions), every model parameter, and every training parameter
    (``max_epochs``/``patience`` -- so a run from the early-stopping protocol can
    never be pooled with one from the fixed-length protocol).

    Two deliberate substitutions:

    * ``model.pair_hidden`` is replaced by the run's top-level ``pair_hidden``,
      which is the width actually trained. Capacity matching rewrites the
      configured width, so the stored config says 48 for a network that ran at
      76.
    * ``tag`` is *excluded*. A tag is a provenance label chosen by whoever
      launched the sweep; it records which batch a run came from, not what
      experiment it was. Keying on the tag is what forces the headline to be
      either five seeds from ``main`` or a hand-maintained allow-list of
      replication tags -- the exact failure this function exists to remove.
    """
    cfg = row.get("config") or {}
    out: dict = {"family": row.get("family", cfg.get("family"))}
    for section in ("system", "split", "model", "train"):
        sub = cfg.get(section) or {}
        for k, v in sub.items():
            if k in _REPLICATE_FIELDS:
                continue
            if section == "model" and k in _RESOLVED_FROM_SYSTEM:
                continue
            if section == "model" and k == "pair_hidden":
                continue
            out[f"{section}.{k}"] = v
    ph = row.get("pair_hidden", (cfg.get("model") or {}).get("pair_hidden"))
    out["pair_hidden"] = int(ph) if ph is not None else None
    return dict(sorted(out.items()))


def condition_key(row: dict) -> str:
    """A short stable hash of :func:`condition_fields`."""
    import hashlib

    blob = json.dumps(condition_fields(row), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def spec_condition_key(spec: dict) -> str:
    """The condition key a :mod:`intervention_algebra.phase1` spec *would* produce.

    Builds the same :class:`RunConfig` that :func:`sweep._worker` builds and
    applies the same capacity resolution, without training anything. This is
    what lets the report define its cells from the experiment definition rather
    than from labels attached to past batches: "the headline cell at coverage c
    for family f is every run whose condition equals the condition of
    ``phase1``'s headline spec", which cannot drift as new tags are added.
    """
    from .experiment import RunConfig, _resolve_model_config
    from .generator import SystemConfig
    from .models import ModelConfig
    from .splits import SplitConfig
    from .train import TrainConfig

    cfg = RunConfig(
        family=spec["family"],
        system=SystemConfig(**spec.get("system", {})),
        split=SplitConfig(**spec.get("split", {})),
        model=ModelConfig(**spec.get("model", {})),
        train=TrainConfig(**spec.get("train", {})),
        capacity_match=spec.get("capacity_match", True),
        tag=spec.get("tag", ""),
    )
    resolved = _resolve_model_config(cfg)
    return condition_key({"family": cfg.family, "config": cfg.to_dict(),
                          "pair_hidden": resolved.pair_hidden})


class DuplicateRunsError(RuntimeError):
    """Raised when one (condition, seed) is present more than once."""


def select_conditions(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Rows whose condition is in ``keys``, with duplicate replicates refused.

    ``keys`` may be a single key, an iterable of keys, or a mapping whose values
    are keys. Raises :class:`DuplicateRunsError` if any ``(condition, seed)``
    appears twice: that is the shape a directory glob produces, and it halves
    every p-value silently.
    """
    if df.empty:
        return df
    if isinstance(keys, dict):
        keys = list(keys.values())
    elif isinstance(keys, str):
        keys = [keys]
    keys = set(keys)
    if "condition" not in df.columns:
        raise KeyError("frame has no `condition` column; build it with to_frame()")
    sub = df[df["condition"].isin(keys)].copy()
    if not sub.empty and "seed" in sub.columns:
        dup = sub.duplicated(subset=["condition", "seed"], keep=False)
        if dup.any():
            offenders = (sub.loc[dup, ["condition", "family", "seed", "tag"]]
                         .sort_values(["condition", "seed"]))
            raise DuplicateRunsError(
                f"{int(dup.sum())} duplicated (condition, seed) rows -- the "
                f"results file contains the same experiment twice:\n{offenders}")
    return sub


def flag_capped_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Mark runs that ended at their epoch cap, and the cells containing them.

    A run that ends at its cap was still improving when the budget expired, so
    it is not "that family's best" and comparing families across such runs
    partly measures the budget.

    ``capped_cell`` marks the whole ``(tag, regime, coverage, seed)`` cell --
    *every* family in it -- not just the offending run. Dropping only the capped
    runs would condition membership on the outcome, and convergence speed
    correlates with the final metric, so that exclusion biases the comparison in
    a direction that cannot be signed in advance. Excluding whole cells applies
    the criterion symmetrically across families and preserves the paired-by-seed
    structure.
    """
    if df.empty or "epochs_run" not in df.columns:
        return df
    if "max_epochs" in df.columns:
        cap = df["max_epochs"]
    else:
        return df
    df = df.copy()
    df["capped"] = df["epochs_run"] >= cap
    cell = [c for c in ("tag", "regime", "pair_coverage", "seed") if c in df.columns]
    if cell:
        df["capped_cell"] = df.groupby(cell, dropna=False)["capped"].transform("any")
    else:
        df["capped_cell"] = df["capped"]
    return df


def add_skill_vs_additive_family(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``skill_vs_additive_family`` = 1 - test_mse / test_mse(additive).

    The reference is the *trained* additive family from the identical condition
    (same regime, coverage and seed), which is the honest "no interactions"
    null: it is fitted on the same training rows and scored on the same test
    pairs.  This replaces the earlier skill score against an additive predictor
    built from the true ``v``, which was not a lower bound -- the trained
    additive model beat it routinely, so positive skill against it did not imply
    that any interaction structure had been learned (see
    ``metrics.true_v_additive_mse``).

    Positive values mean "better than modelling no interactions at all".
    """
    if df.empty or "family" not in df.columns or "test_mse" not in df.columns:
        return df
    cols = [c for c in CONDITION_COLS if c in df.columns]
    if not cols:
        return df
    add = (df[df["family"] == "additive"]
           .groupby(cols, dropna=False)["test_mse"].mean()
           .rename("_additive_mse"))
    df = df.merge(add, how="left", left_on=cols, right_index=True)
    df["skill_vs_additive_family"] = 1.0 - df["test_mse"] / df["_additive_mse"]
    return df.drop(columns=["_additive_mse"])


# --------------------------------------------------------------- aggregate --


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are plausibly metrics (excludes identifiers)."""
    out = []
    for col in df.columns:
        if col in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def warn_if_regime_pools_tags(df: pd.DataFrame) -> None:
    """Warn when ``regime`` is used as a selector across several sweep tags.

    ``regime`` is NOT a safe selector on a combined results file. The controls,
    the misspecification levels, the extra power seeds and the ceiling model all
    carry ``regime="both"``, so grouping or filtering by regime alone silently
    pools five different system configurations -- a seed can then enter the same
    average six times under different conditions. Filter on ``tag`` first.
    """
    if df.empty or "regime" not in df.columns or "tag" not in df.columns:
        return
    for regime, g in df.groupby("regime"):
        tags = sorted(set(g["tag"].dropna()))
        if len(tags) > 1:
            print(f"[analysis] WARNING: regime={regime!r} spans {len(tags)} tags "
                  f"({', '.join(map(str, tags))}). `regime` is not a safe "
                  f"selector here -- filter on `tag` first.", file=sys.stderr)


def aggregate(df: pd.DataFrame,
              group_cols: Sequence[str],
              metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """Nan-safe mean/std over seeds.

    Returns one row per group with ``<metric>_mean``/``<metric>_std`` columns
    plus ``n_seeds``.  All-NaN groups yield NaN (no exception, no warning).
    """
    group_cols = [c for c in group_cols if c in df.columns]
    if df.empty or not group_cols:
        return pd.DataFrame()
    if metrics is None:
        metrics = numeric_metric_columns(df)
    metrics = [m for m in metrics if m in df.columns]

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(group_cols, keys))
            if "seed" in sub.columns:
                rec["n_seeds"] = int(sub["seed"].nunique(dropna=True)) or len(sub)
            else:
                rec["n_seeds"] = len(sub)
            for m in metrics:
                vals = pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                rec[f"{m}_mean"] = float(np.mean(vals)) if vals.size else np.nan
                rec[f"{m}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
                rec[f"{m}_n"] = int(vals.size)
            records.append(rec)

    out = pd.DataFrame.from_records(records)
    ordered = list(group_cols) + ["n_seeds"]
    for m in metrics:
        ordered += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    out = out[[c for c in ordered if c in out.columns]]
    if "family" in out.columns:
        out = out.sort_values(
            by=[c for c in group_cols if c != "family"] + ["family"],
            key=lambda s: s.map(_family_sort_key) if s.name == "family" else s,
        ).reset_index(drop=True)
    return out


def summary_table(df: pd.DataFrame,
                  regime: str | None = None,
                  metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """family x pair_coverage summary for a single regime.

    ``regime=None`` picks the only regime present (or the first, alphabetically,
    when several are).  Returns mean columns with std columns alongside.
    """
    if df.empty:
        return pd.DataFrame()
    sub = df
    if "regime" in df.columns:
        regimes = sorted(df["regime"].dropna().unique().tolist())
        if regime is None:
            if not regimes:
                return pd.DataFrame()
            regime = regimes[0]
        sub = df[df["regime"] == regime]
    if sub.empty:
        return pd.DataFrame()

    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in sub.columns]
    metrics = [m for m in metrics if m in sub.columns]

    agg = aggregate(sub, ["family", "pair_coverage"], metrics)
    if agg.empty:
        return agg
    # `{m}_n` is kept, not dropped. These are nan-safe means, and a metric can
    # be NaN for exactly the runs that failed -- `test_A_pearson` is undefined
    # wherever the antisymmetric readout is constant -- so two columns of the
    # same row can silently carry different denominators. `n_seeds` is the cell
    # size; `{m}_n` is what that metric was actually averaged over, and the gap
    # between them is the thing a reader needs to see (review finding R2).
    keep = ["family", "pair_coverage", "n_seeds"]
    for m in metrics:
        keep += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    agg = agg[[c for c in keep if c in agg.columns]]
    agg.attrs["regime"] = regime
    return agg


# ---------------------------------------------------------------- markdown --


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if df is None or len(df) == 0:
        return "_(no rows)_"

    frame = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()

    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return floatfmt.format(v)
        if isinstance(v, (int, np.integer, bool, np.bool_)):
            return str(v)
        return str(v)

    cols = [str(c) for c in frame.columns]
    body = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in body)) if body else len(cols[i])
              for i in range(len(cols))]

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    out = [line(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out += [line(r) for r in body]
    return "\n".join(out)


def headline_table(df: pd.DataFrame, metric: str = "test_mse",
                   tag: str = "main",
                   families: Sequence[str] = ("additive", "unconstrained",
                                              "shared_pair", "algebra"),
                   lower_is_better: bool = True) -> str:
    """One compact `mean ± std over seeds` table: metric by coverage by family.

    The auto-generated summary table carries every metric and is unreadable as a
    result artifact.  This is the version that goes in the README, plus the
    paired algebra-vs-unconstrained difference and its p-value, which is the
    actual headline comparison.
    """
    if df.empty or metric not in df.columns:
        return "_(no rows)_"
    sub = df[df["tag"] == tag] if "tag" in df.columns else df
    if sub.empty:
        return f"_(no rows with tag={tag!r})_"

    fams = [f for f in families if f in set(sub["family"])]
    n_seeds_expected = int(sub["seed"].nunique()) if "seed" in sub.columns else 0
    rows = []
    for cov, g in sub.groupby("pair_coverage", sort=True):
        rec = {"pair_coverage": f"{cov:.2f}"}
        for f in fams:
            vals = g.loc[g["family"] == f, metric].astype(float).dropna()
            n = len(vals)
            if n == 0:
                rec[f] = "—"
                continue
            cell = f"{vals.mean():.3f}" + (f" ± {vals.std(ddof=1):.3f}" if n > 1 else "")
            # Show the per-metric n whenever it is below the number of seeds in
            # the cell. A metric can be NaN for exactly the runs that failed
            # (e.g. A_pearson is undefined when the antisymmetric head collapses
            # to a constant), so a nan-safe mean silently changes denominator
            # between columns of the same row -- understating one metric and
            # overstating another. Where n differs, the reader must see it.
            if n_seeds_expected and n < n_seeds_expected:
                cell += f" (n={n})"
            rec[f] = cell
        rows.append(rec)
    table = markdown_table(pd.DataFrame(rows))

    if {"algebra", "unconstrained"} <= set(fams):
        cmp = paired_comparison(sub, "algebra", "unconstrained", metric=metric,
                                group_cols=("pair_coverage",))
        if not cmp.empty:
            keep = cmp[["pair_coverage", "mean_diff", "std_diff", "n_seeds",
                        "n_a_lt_b", "p_value"]].copy()
            direction = "algebra better" if lower_is_better else "unconstrained better"
            keep = keep.rename(columns={"n_a_lt_b": f"seeds {direction}"})
            table += ("\n\nPaired difference `algebra − unconstrained` "
                      f"({metric}, matched seeds):\n\n"
                      + markdown_table(keep))
    return table


# -------------------------------------------------------- paired statistics --


def paired_comparison(df: pd.DataFrame,
                      family_a: str,
                      family_b: str,
                      metric: str = "test_mse",
                      group_cols: Sequence[str] = ("tag", "regime", "pair_coverage")) -> pd.DataFrame:
    """Per-group paired difference ``family_a - family_b`` across shared seeds.

    Seed-to-seed variance is large relative to the effect we care about, so the
    comparison is paired: only seeds present (and non-NaN) for *both* families
    contribute.  Reports mean/std of the difference, ``n_seeds``, the
    ``scipy.stats.ttest_rel`` p-value, and how often ``a < b``.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in df.columns]
    sub = df[df["family"].isin([family_a, family_b])]
    if sub.empty or "seed" not in sub.columns:
        return pd.DataFrame()

    try:
        from scipy import stats as _stats
    except Exception:  # pragma: no cover - scipy is a hard dep in practice
        _stats = None

    records = []
    groups = sub.groupby(group_cols, dropna=False, sort=True) if group_cols \
        else [((), sub)]
    for keys, g in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        pivot = (g.pivot_table(index="seed", columns="family", values=metric,
                               aggfunc="mean", dropna=False))
        if family_a not in pivot.columns or family_b not in pivot.columns:
            continue
        pair = pivot[[family_a, family_b]].dropna()
        a = pair[family_a].to_numpy(dtype=float)
        b = pair[family_b].to_numpy(dtype=float)
        diff = a - b
        rec.update({
            "metric": metric,
            "family_a": family_a,
            "family_b": family_b,
            "n_seeds": int(diff.size),
            f"{family_a}_mean": float(np.mean(a)) if a.size else np.nan,
            f"{family_b}_mean": float(np.mean(b)) if b.size else np.nan,
            "mean_diff": float(np.mean(diff)) if diff.size else np.nan,
            "std_diff": float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan,
            "n_a_lt_b": int(np.sum(diff < 0)),
        })
        pval = np.nan
        if _stats is not None and diff.size > 1 and float(np.std(diff)) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    pval = float(_stats.ttest_rel(a, b).pvalue)
                except Exception:
                    pval = np.nan
        rec["p_value"] = pval
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    if not out.empty and group_cols:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out


# ----------------------------------------------------------------- figures --


def _coverage_axis(ax, coverages: Sequence[float]) -> None:
    """Log x-axis labelled with the coverages actually run.

    Matplotlib's default log ticks render these as ``3x10^-1 4x10^-1``, which
    overlap into an unreadable string at the crowded end and never show the
    values the experiment was actually run at. The measured coverages are the
    only meaningful ticks here, so they are the ticks.
    """
    vals = sorted({float(c) for c in coverages
                   if c is not None and not pd.isna(c)})
    if vals and min(vals) > 0:
        ax.set_xscale("log")
        ax.set_xticks(vals)
        ax.set_xticklabels([f"{v:g}" for v in vals])
        ax.minorticks_off()
    ax.set_xlabel("pair coverage (fraction of pairs seen in training)")


def _nanmean(series) -> float:
    vals = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _plot_metric_by_family(ax, sub: pd.DataFrame, metric: str) -> bool:
    """Mean +/- std of ``metric`` vs coverage, one line per family."""
    agg = aggregate(sub, ["family", "pair_coverage"], [metric])
    if agg.empty:
        return False
    drew = False
    for fam in _sorted_families(agg["family"]):
        g = agg[agg["family"] == fam].sort_values("pair_coverage")
        y = g[f"{metric}_mean"].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue
        yerr = g[f"{metric}_std"].to_numpy(dtype=float)
        yerr = np.where(np.isnan(yerr), 0.0, yerr)
        ax.errorbar(g["pair_coverage"].to_numpy(dtype=float), y, yerr=yerr,
                    marker="o", capsize=3, lw=1.6, ms=5,
                    color=family_color(fam), label=fam)
        drew = True
    if drew:
        _coverage_axis(ax, agg["pair_coverage"].tolist())
    return drew


def _legend(ax) -> None:
    """Legend only when something labelled was actually drawn."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    # bbox_inches="tight" keeps the suptitle (drawn above the axes) in frame.
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _regimes_present(df: pd.DataFrame, wanted: Sequence[str] | None = None) -> list[str]:
    if "regime" not in df.columns:
        return []
    present = [r for r in df["regime"].dropna().unique().tolist()]
    order = ["independent", "symmetric", "antisymmetric", "both"]
    present = sorted(present, key=lambda r: (order.index(r) if r in order else len(order), str(r)))
    if wanted is not None:
        present = [r for r in present if r in wanted]
    return present


def _fig_mse_vs_coverage(df: pd.DataFrame, outdir: Path) -> Path | None:
    regimes = _regimes_present(df)
    if not regimes or "test_mse" not in df.columns:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(1 + 4.6 * len(regimes), 4.2),
                             squeeze=False)
    any_drawn = False
    for ax, regime in zip(axes[0], regimes):
        sub = df[df["regime"] == regime]
        drew = _plot_metric_by_family(ax, sub, "test_mse")
        any_drawn = any_drawn or drew
        nf = _nanmean(sub["noise_floor_mse"]) if "noise_floor_mse" in sub.columns else np.nan
        if not np.isnan(nf):
            ax.axhline(nf, ls=":", color="k", lw=1.2, label="noise floor")
        ref = (_nanmean(sub["test_true_v_additive_mse"])
               if "test_true_v_additive_mse" in sub.columns else np.nan)
        if not np.isnan(ref):
            ax.axhline(ref, ls="--", color="#8c564b", lw=1.2,
                       label="true-v additive reference")
        ax.set_yscale("log")
        ax.set_ylabel("held-out-pair test MSE")
        ax.set_title(f"regime = {regime}")
        ax.grid(alpha=0.25, which="both")
        _legend(ax)
    if not any_drawn:
        plt.close(fig)
        return None
    fig.suptitle("Held-out-pair prediction error vs pair coverage", y=1.02)
    return _finish(fig, outdir / "fig1_heldout_mse_vs_coverage.png")


def _fig_component_recovery(df: pd.DataFrame, outdir: Path, *, metric: str,
                            regimes_wanted: Sequence[str], filename: str,
                            ylabel: str, title: str) -> Path | None:
    if metric not in df.columns:
        return None
    regimes = _regimes_present(df, regimes_wanted)
    if not regimes:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(1 + 4.6 * len(regimes), 4.2),
                             squeeze=False, sharey=True)
    any_drawn = False
    for ax, regime in zip(axes[0], regimes):
        sub = df[df["regime"] == regime]
        drew = _plot_metric_by_family(ax, sub, metric)
        any_drawn = any_drawn or drew
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_ylim(-0.1, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(f"regime = {regime}")
        ax.grid(alpha=0.25, which="both")
        _legend(ax)
    if not any_drawn:
        plt.close(fig)
        return None
    fig.suptitle(title, y=1.02)
    return _finish(fig, outdir / filename)


def _fig_topology(df: pd.DataFrame, outdir: Path) -> Path | None:
    metric = "test_topo_auprc_S"
    if metric not in df.columns:
        return None
    regimes = _regimes_present(df, ["symmetric", "both"])
    if not regimes:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(1 + 4.6 * len(regimes), 4.2),
                             squeeze=False, sharey=True)
    any_drawn = False
    for ax, regime in zip(axes[0], regimes):
        sub = df[df["regime"] == regime]
        drew = _plot_metric_by_family(ax, sub, metric)
        any_drawn = any_drawn or drew
        base = (_nanmean(sub["test_topo_base_rate"])
                if "test_topo_base_rate" in sub.columns else np.nan)
        if not np.isnan(base):
            ax.axhline(base, ls="--", color="k", lw=1.2,
                       label=f"chance AUPRC ({base:.2f})")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("AUPRC, symmetric interaction support")
        ax.set_title(f"regime = {regime}")
        ax.grid(alpha=0.25, which="both")
        _legend(ax)
    if not any_drawn:
        plt.close(fig)
        return None
    fig.suptitle("Recovery of symmetric interaction topology", y=1.02)
    return _finish(fig, outdir / "fig4_topology_auprc.png")


def make_figures(df: pd.DataFrame, outdir, tag: str | None = "main") -> list[Path]:
    """Write the four standard figures; returns the paths actually written.

    ``tag`` defaults to the headline sweep. This is not cosmetic: *every* sweep
    tag carries ``regime="both"`` -- the controls, the misspecification levels,
    the extra power seeds and the ceiling model included -- so filtering on
    regime alone silently averages the headline curve together with the
    weak-interaction control, the tanh control and the ceiling family. Pass
    ``tag=None`` to plot everything deliberately.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return []
    if tag is not None and "tag" in df.columns:
        df = df[df["tag"] == tag]
        if df.empty:
            return []

    paths = [
        _fig_mse_vs_coverage(df, outdir),
        _fig_component_recovery(
            df, outdir, metric="test_S_pearson",
            regimes_wanted=["symmetric", "both"], filename="fig2_S_recovery.png",
            ylabel="Pearson r, true vs predicted S",
            title="Recovery of the symmetric component S"),
        _fig_component_recovery(
            df, outdir, metric="test_A_pearson",
            regimes_wanted=["antisymmetric", "both"], filename="fig3_A_recovery.png",
            ylabel="Pearson r, true vs predicted A",
            title="Recovery of the antisymmetric component A"),
        _fig_topology(df, outdir),
    ]
    return [p for p in paths if p is not None]


# --------------------------------------------------------------------- CLI --


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m intervention_algebra.analysis",
        description="Aggregate sweep results into summary tables and figures.")
    p.add_argument("--results", required=True, help="sweep results .jsonl")
    p.add_argument("--outdir", required=True, help="directory for the PNG figures")
    p.add_argument("--tag", default="main",
                   help="restrict to one sweep tag (default: main). Every tag "
                        "shares regime='both' -- controls, misspecification "
                        "levels, extra power seeds and the ceiling model "
                        "included -- so without this the headline sweep is "
                        "silently averaged together with all of them. Pass "
                        "--tag '' to disable.")
    p.add_argument("--regime", action="append", default=None,
                   help="restrict to this regime (repeatable; default: all present)")
    p.add_argument("--metric", action="append", default=None,
                   help="metric column for the summary tables (repeatable)")
    p.add_argument("--compare", nargs=2, metavar=("FAMILY_A", "FAMILY_B"),
                   default=None, help="also print a paired family comparison")
    p.add_argument("--compare-metric", default="test_mse",
                   help="metric for --compare (default: test_mse)")
    p.add_argument("--no-figures", action="store_true",
                   help="print tables only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"error: results file not found: {results_path}", file=sys.stderr)
        return 2

    rows, errors = load(results_path, return_errors=True)
    if not rows:
        print(f"error: no successful runs in {results_path} "
              f"({len(errors)} failed rows)", file=sys.stderr)
        return 1

    df = to_frame(rows)
    if getattr(args, "tag", None) and "tag" in df.columns:
        before = len(df)
        df = df[df["tag"] == args.tag]
        print(f"- tag filter: `{args.tag}` ({len(df)} of {before} runs)")
        if df.empty:
            print(f"error: no rows with tag={args.tag!r}", file=sys.stderr)
            return 1
    if args.regime:
        wanted = set(args.regime)
        missing = wanted - set(df.get("regime", pd.Series(dtype=object)).unique())
        for r in sorted(missing):
            print(f"warning: regime {r!r} not present in results", file=sys.stderr)
        df = df[df["regime"].isin(wanted)]
        if df.empty:
            print("error: no rows left after --regime filtering", file=sys.stderr)
            return 1

    metrics = args.metric or None

    print(f"# Intervention-algebra sweep summary\n")
    print(f"- results: `{results_path}`")
    print(f"- runs: {len(rows)} successful, {len(errors)} failed")
    if "family" in df.columns:
        print(f"- families: {', '.join(_sorted_families(df['family']))}")
    if "seed" in df.columns:
        print(f"- seeds: {int(df['seed'].nunique())}")
    print()

    warn_if_regime_pools_tags(df)
    for regime in _regimes_present(df):
        table = summary_table(df, regime=regime, metrics=metrics)
        if table.empty:
            continue
        print(f"## regime = {regime}\n")
        print(markdown_table(table))
        print()

    if args.compare:
        fam_a, fam_b = args.compare
        cmp_df = paired_comparison(df, fam_a, fam_b, metric=args.compare_metric)
        print(f"## paired comparison: {fam_a} - {fam_b} ({args.compare_metric})\n")
        print(markdown_table(cmp_df))
        print()

    if not args.no_figures:
        paths = make_figures(df, args.outdir)
        print("## figures\n")
        if paths:
            for p in paths:
                print(f"- `{p}`")
        else:
            print("_(no figures written -- required columns/regimes absent)_")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
