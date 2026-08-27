#!/usr/bin/env python
"""Regenerate every Phase 4 table, document and figure from the result files.

    python scripts/report_phase4_chemlex.py
    python scripts/report_phase4_chemlex.py --no-figures      # what CI runs

Writes `results/phase4_chemlex/summary/`, `results/phase4_chemlex/README_PHASE4.md`,
`docs/phase4_chemlex_dataset.md`, `docs/phase4_chemlex_mapping.md` and
`docs/phase4_chemlex_interactions.md`.

Nothing here is typed by hand. Phase 2R's audit found four hand-copied p-values
that matched no run in the repository, and the fix that stuck was to generate
the documents and have CI diff them against a regeneration on a clean checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intervention_algebra.real_data.chemlex import dataset as ds
from intervention_algebra.real_data.chemlex import report as rp
from intervention_algebra.real_data.chemlex.features import fingerprints
from intervention_algebra.real_data.chemlex.splits import (group_report,
                                                           split_groups)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=rp.RESULTS)
    ap.add_argument("--raw-dir", type=Path, default=ds.DEFAULT_RAW_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    rp.RESULTS = args.results
    summary = args.results / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    primary = rp.load_rows("primary")
    controls = rp.load_rows("controls")
    positive = rp.load_rows("positive")
    trans = rp.load_rows("transductive")
    sens = rp.load_rows("sensitivity")
    entity_frames = [f for f in (rp.load_per_entity("primary"),
                                 rp.load_per_entity("controls"),
                                 rp.load_per_entity("sensitivity"))
                     if not f.empty]
    per_entity = (pd.concat(entity_frames, ignore_index=True) if entity_frames
                  else pd.DataFrame(columns=[
                      "contrast", "endpoint", "block", "usable", "screen",
                      "bucket", "role", "entity", "incremental",
                      "incremental_blind", "max_similarity_to_train",
                      "n_rows"]))

    if primary.empty:
        raise SystemExit(f"no primary results under {args.results}")
    if per_entity.empty:
        print("WARNING: no per-entity records; the inferential unit of this "
              "phase is the held-out entity, so the tables that read it will be "
              "empty and the verdict cannot be evaluated", flush=True)

    raw = ds.load_raw(args.raw_dir)
    screens = {name: ds.load_screen(name, raw=raw) for name in ds.SCREENS}
    a = ds.audit(args.raw_dir, raw=raw)
    roles = {name: ds.role_check(s) for name, s in screens.items()}

    groups = {}
    ref = screens["all"]
    for role, smis in (("acid", ref.acids), ("amine", ref.amines)):
        fp = fingerprints(smis, role)
        groups[role] = group_report(smis, split_groups(smis, fp.x), role)

    tables = {
        "counts": rp.counts_table(primary),
        "models_yield": rp.model_table(primary, "yield"),
        "models_feasible": rp.model_table(primary, "feasible"),
        "folds": rp.fold_level_table(primary, "yield"),
        "folds_feasible": rp.fold_level_table(primary, "feasible"),
        "condition_geometry": rp.condition_table(screens),
        "condition_stratified": rp.condition_stratified_table(primary),
        "adaptive_condition": rp.adaptive_condition_table(screens),
    }
    per_entity = rp.attach_common_denominator(per_entity, primary)
    tables["incremental"], _ = rp.incremental_table(per_entity)
    tables["incremental_corrected"], _ = rp.incremental_table(
        per_entity, statistic="common")
    tables["statistic_comparison"] = rp.statistic_comparison(per_entity)
    tables["multiplicity"] = rp.multiplicity_table(primary, per_entity)
    tables["role_relevant_controls"] = rp.role_relevant_control_table(
        primary, controls)
    tables["controls"], control_summary = rp.control_table(primary, controls)
    tables["positive"], positive_summary = rp.positive_table(positive)
    tables["blind"], _ = rp.blind_table(
        {"primary": primary, "control": controls, "positive": positive})
    tables["projection"], _ = rp.projection_table(primary)
    tables["similarity"], _ = rp.similarity_table(per_entity)
    tables["similarity_fixed"], _ = rp.similarity_table(per_entity, fixed=True)
    tables["similarity_corrected"], _ = rp.similarity_table(
        per_entity, statistic="common")
    tables["congener"], _ = rp.congener_table(per_entity, screens)
    tables["congener_corrected"], _ = rp.congener_table(
        per_entity, screens, statistic="common")
    tables["transductive"], _ = rp.transductive_table(trans)
    tables["sensitivity"] = rp.sensitivity_table(sens, primary)

    v = rp.verdict(primary, controls, positive, trans, per_entity, screens)
    vp = rp.verdict_posthoc(primary, controls, positive, trans, per_entity,
                            screens)

    for name, text in tables.items():
        (summary / f"{name}.md").write_text(text + "\n")
    (summary / "verdict.json").write_text(json.dumps(v, indent=2, default=str) + "\n")
    (summary / "verdict_posthoc.json").write_text(
        json.dumps(vp, indent=2, default=str) + "\n")
    (summary / "verdict.md").write_text(
        rp.verdict_markdown(v)
        + "\n\n### The same rule with one statistic corrected (post-hoc)\n\n"
        + f"Single change: {vp['single_change']}.\n\n"
        + rp.verdict_markdown(dict(vp, verdict=vp["verdict_posthoc"])) + "\n")
    # `smiles` is dropped for the same reason it is dropped in
    # chemlex/sweep.py: it is the CC BY-NC deposit's substrate inventory,
    # not ours to redistribute. `role` + `entity` is the join key.
    per_entity.drop(columns=["smiles"], errors="ignore").to_csv(
        summary / "per_entity.csv", index=False)

    # An explicit allow-list, not a glob. The glob this replaces picked up
    # every .jsonl in the directory, so adding a diagnostic beside the results
    # -- axes.jsonl and pair_terms.jsonl are both regenerable diagnostics that
    # feed no decision -- silently rewrote the index's headline from 173
    # conditions to 285. A results index that counts its own diagnostics as
    # results is worse than no index.
    counts = {}
    for name in ("primary", "controls", "positive", "transductive",
                 "sensitivity"):
        f = args.results / f"{name}.jsonl"
        if f.exists():
            counts[f.name] = sum(1 for l in f.read_text().splitlines()
                                 if l.strip())
    (args.results / "README_PHASE4.md").write_text(
        rp.readme(v, counts, vp=vp, screens=screens))

    Path("docs/phase4_chemlex_dataset.md").write_text(
        rp.dataset_document(a, screens, roles))
    Path("docs/phase4_chemlex_mapping.md").write_text(
        rp.mapping_document(screens, groups))
    Path("docs/phase4_chemlex_interactions.md").write_text(
        rp.interactions_document(v, tables, screens, vp=vp,
                                 defect=rp.denominator_defect(per_entity)))

    if not args.no_figures:
        made = rp.figures(primary, controls, positive, per_entity)
        for p in made:
            print(f"  figure {p}")

    print(f"verdict (frozen):   {v['verdict']}")
    print(f"verdict (post-hoc): {vp['verdict_posthoc']}")
    print(f"  {v['n_attempted']} conditions, {v['n_failed']} failed")
    for r in v["invalidating_reasons"]:
        print(f"  GATE: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
