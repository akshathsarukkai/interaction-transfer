# Phase 4 results — ChemLex acid–amine entity-OOD

**Frozen verdict: INCONCLUSIVE**

**The same rule with one statistic corrected: ANALOGUE-ONLY CHEMICAL TRANSFER** — and that is the reported reading. Single change: the per-entity statistic's denominator is the fold's baseline MSE rather than the entity's own, which is not bounded away from zero. See `docs/phase4_chemlex_interactions.md` for why the frozen verdict is reported and not believed.

173 conditions across every block, 0 failed
(150 of them read by the decision rule; the sensitivity block is
reported but does not enter it). Authoritative fold seed 20260904,
k = 5, 3 partitions.

| file | rows |
|---:|---:|
| `results/phase4_chemlex/controls.jsonl` | 40 |
| `results/phase4_chemlex/positive.jsonl` | 30 |
| `results/phase4_chemlex/primary.jsonl` | 60 |
| `results/phase4_chemlex/sensitivity.jsonl` | 23 |
| `results/phase4_chemlex/transductive.jsonl` | 20 |

`summary/` holds the generated tables. `docs/phase4_chemlex_interactions.md` is
the document they compose; `docs/phase4_chemlex_dataset.md` and
`docs/phase4_chemlex_mapping.md` cover the deposit and molecular identity.

Regenerate everything with

    python scripts/report_phase4_chemlex.py

Reproduce the results themselves with

    python scripts/download_chemlex.py
    python scripts/run_phase4_chemlex.py --part all --workers 6

`smoke.jsonl` is a pipeline check and is gitignored: a pipeline check is not a
result.

## How to read the numbers

The headline is **incremental pair skill**, `1 − MSE(pair) / MSE(additive)`, from
paired predictions on identical rows. It is not a skill against zero — Phase 3's
registered gate read that statistic and fired on a control containing no
chemistry — and it is not a difference of two separately reported R²s.

Read every R² against the replicate ceiling. Two measurements of the same
nominal reaction in this deposit correlate at Pearson all 0.59, hatu 0.61, giving an R²
ceiling of all 0.49, hatu 0.54 — so roughly half the variance of the endpoint is
unavailable to any deterministic predictor of (acid, amine, condition).
