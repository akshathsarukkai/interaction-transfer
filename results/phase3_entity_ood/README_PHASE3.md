# Phase 3 — entity-level out-of-distribution results

The unit held out is a **drug**, not a pair. For a test drug no Koplev
measurement enters training, validation, feature preprocessing, target scaling,
hyperparameter selection or shrinkage calibration; the only thing the model knows
about it is what the molecule is.

These rows must never be pooled with Phase 2 / Phase 2R / d-chain-null rows.
They answer a different question against a different target — raw `D`, not the
additively-residualised `D_res` — and a Phase 3 row is identifiable by its
`fold_key` column. `test_phase3_results_are_not_mixed_with_earlier_phases`
asserts no Phase 3 row has leaked into an earlier phase's files.

## Files

| file | what it holds |
|---|---|
| `primary.jsonl` | every rung, both screens, all 30 entity folds, ECFP4 |
| `controls.jsonl` | Control A (random entity features) and Control B (shuffled fingerprints) |
| `positive.jsonl` | the synthetic positive control: a target that *is* potential + a known rank-2 form |
| `targets.jsonl` | the secondary ChEMBL-mechanism representation, and Control D (its shuffle) |
| `coverage.jsonl` | sparser pair coverage among training entities only |
| `summary/` | generated tables, `verdict.json`, `folds.csv`, `per_drug.csv` |
| `smoke.jsonl` | a pipeline check, gitignored — not a result |

## Reading a row

`e1_*` is the primary regime, one unseen endpoint. `e2_*` is both unseen and is
never pooled with it. `e1x_*` / `e2x_*` are the same fits re-scored with the four
metal compounds excluded — a re-scoring, not a refit, so the arms are exactly
paired. `va_*` is validation, reported for diagnosis only. `per_drug` carries one
entry per held-out drug with its E1 metrics and its chemical distance from that
fold's training set.

`skill` is against the zero predictor. It is **not** the headline. Most of `D`'s
energy is per-drug potential, so a model can post a healthy skill against zero
while knowing nothing about interaction. The headline is
`1 - MSE(lowrank)/MSE(potential)`, computed within a fold, in
`summary/incremental.csv`.

## Reproduction

```
python scripts/prepare_phase3_drugs.py          # PubChem + ChEMBL -> mapping
python scripts/prepare_phase3_targets.py        # ChEMBL mechanisms -> targets
python scripts/audit_phase3_drugs.py            # mapping statistics
python scripts/run_phase3_entity_ood.py --counts
python scripts/run_phase3_entity_ood.py --part all
python scripts/report_phase3_entity_ood.py
```

Condition counts are derived from `entity_ood.sweep.PART_GRIDS` and printed by
`--counts`. They are deliberately not written here: that is exactly the number
that goes stale, and Phase 2R had to come back and close that hole once already.
