# What is in results/

## Read this first

**`phase1.jsonl` is the single authoritative file.** It holds all Phase 1 runs
and contains **no duplicate cells** — one row per experimental *condition* and
seed, where the condition is a hash of the fully resolved run configuration
(`analysis.condition_key`). That is a stricter guarantee than the old
`(tag, family, seed, coverage, pair_hidden)` key, because it also separates runs
that differ only in a training-protocol field such as `patience`. Every number in the README
is computed from it. `scripts/make_report.py` and
the analysis CLI both default to it.

**Phase 2 results are not indexed here.** They live in `results/phase2/` and are
catalogued by [`phase2/README_PHASE2.md`](phase2/README_PHASE2.md). Phase 2 uses
different data, different models and a different target; nothing in this file
should ever be aggregated with anything in that one.

> **Do not glob `results/*.jsonl`.** All 17 `phase1_*.jsonl` files below are
> exact **subsets** of `phase1.jsonl`, kept as the per-sweep record of how the
> results were produced — and so, since its coverage-0.05 rows were promoted, is
> `SUPERSEDED_main_intermediate.jsonl` (55/55 rows present). Only
> `SUPERSEDED_main_confounded.jsonl` and `SUPERSEDED_featuremap_testmetric.jsonl`
> are genuinely outside `phase1.jsonl`.
> Globbing the directory double-counts every run: seeds appear twice, paired
> tests silently double their `n`, and p-values shrink accordingly. Read
> `phase1.jsonl` alone, or read the individual sweep files, never both.

Two further selector hazards, both of which produced wrong numbers during
review before they were caught:

* **`regime` is not a selector.** `regime="both"` spans 15 tags — the main
  sweep plus the random-topology, tanh, weak-interaction, double-capacity and
  both misspecification controls, which are different systems. Filter on `tag`.
* **`tag` alone is not a selector either.** The `main` tag spans coverages
  0.05, 0.10 and 0.20; a seed-keyed lookup that omits `pair_coverage` silently
  keeps whichever coverage came last. (An earlier version of this warning said
  the tag spans "0.05–0.70", which was never true of this file.)
* **And `tag` is the wrong axis regardless.** A tag records which *batch* a run
  was launched in. The powered cells pool several batches — coverage 0.10's
  `algebra` arm is `main` + `power10_algebra`, its baselines are `main` +
  `rep10_matched1x` — so anything keyed on tags reports five seeds where the
  result has seventeen. `scripts/make_report.py` selects by condition and prints
  the contributing tags as provenance only; see `src/intervention_algebra/report.py`.

## Authoritative results

All of these ran under the **confound-free protocol**: `max_epochs=5000`,
`patience=5000` (early stopping disabled, so every run trains the full budget),
`n_restarts=2`. All are subsets of `phase1.jsonl`.

| file | rows | what it is |
|---|---|---|
| `phase1_ceiling_fixedlen.jsonl` | 25 | the identifiability ceiling re-measured under the **fixed-length** protocol, so the "% of achievable" ratios no longer span two protocols. Tag `ceiling_fixedlen`. These are the ceiling numbers the docs quote. |
| `phase1_cov040.jsonl` | 68 | **coverage 0.40, seeds 0–16, all four families** — the Phase 1 closure cell, run against a prediction registered in the private research notebook this repository was cut from before it started. `phase1.closure_sweep`, tag `cov040`. |
| `phase1_decisive.jsonl` | 40 | main sweep, coverages 0.10/0.20, seeds 0–4. Carries both `test_*` (best-validation checkpoint) and `final_test_*` (final epoch) as the checkpoint-selection control. |
| `phase1_replication_cov010.jsonl` | 36 | out-of-sample replication of the headline at coverage 0.10, seeds 5–16. Supplies the matched baseline arms that made the headline computable outside the original five seeds. |
| `phase1_replication_cov020.jsonl` | 48 | coverage 0.20 powered to seeds 5–16. Turned the apparent collapse into a significant reversal. |
| `phase1_power10.jsonl` | 36 | `algebra`@48, `unconstrained`@120 and `algebra`@120 at seeds 5–16. |
| `phase1_matched2x.jsonl`, `phase1_matched2x_s04.jsonl` | 12 + 5 | `algebra`@78 (23 096 pair params) — the capacity-**matched** 2× arm against `unconstrained`@120 (23 288). Seeds 5–16 and 0–4. |
| `phase1_bigcap.jsonl` | 10 | `unconstrained`/`shared_pair` at `pair_hidden=120`, seeds 0–4. |
| `phase1_regime.jsonl` | 36 | single-component and no-interaction regimes, seeds 0–2. **Same protocol as everything else** (`patience=5000`) — poolable with the replications below, and pooled in the reported n=15. |
| `phase1_rep_regime_antisym.jsonl` | 48 | `regime_antisymmetric` powered to seeds 5–16. |
| `phase1_rep_regime_sym.jsonl`, `phase1_rep_regime_sym_fill.jsonl` | 32 + 16 | `regime_symmetric` powered to seeds 5–16. |
| `phase1_misspec.jsonl` | 24 | simultaneity-defect 0.3 / 0.8, seeds 0–2. |
| `phase1_rep_misspec.jsonl` | 40 | the same, powered to seeds 5–16. |
| `phase1_control.jsonl` | 42 | i.i.d.-topology, tanh-observation, weak-interaction and unmatched-capacity controls. |

**One protocol exception, now superseded rather than merely disclosed:**
`phase1_ceiling.jsonl` (25 rows, the `wellspecified` reference model) ran with
`patience=600`, i.e. early stopping **on**. It is retained as the historical
artifact and is reproducible via `ceiling_sweep(early_stopping=True)`;
`phase1_ceiling_fixedlen.jsonl` is what the docs now quote. The two agree to
three decimals above coverage 0.10. Originally this said only: That is defensible for a ceiling — stopping on validation cannot flatter
it — but it means the fraction-of-ceiling denominators mix protocols, and the
README says so where it uses them.

## Selection history — inputs to the protocol, not results

* `featuremap_selection.jsonl` (47) — feature-map selection on **validation**
  MSE, budget matched across arms. This is the one that counts.
* `hparam_search.jsonl` (6), `hparam_search_v2.jsonl` (76),
  `hparam_search_stage2.jsonl` (48), `hparams.json`, `hparams_stage1.json` —
  hyperparameter selection history. **Read with this caveat in hand**, which an
  internal review raised and which is the reason these files are listed as
  selection inputs rather than as results: every row in the weight-decay stage
  ran `pair_mode="concat_outer"`, which the feature-map re-selection later
  replaced with `"outer"`. No artifact here
  varies weight decay under the shipped architecture.
  Note `configs/hparams.json` is what `phase1.py` actually loads;
  `results/hparams.json` is a superseded artifact read by nothing.
* `hparam_search_stage1.log` — the only surviving evidence for stage-1
  selection, after its raw JSONL was overwritten by an accidental re-launch.
  Tracked deliberately despite `results/*.log` being ignored.

## Not results — retained as evidence of retracted work

**These three are named `SUPERSEDED_*` rather than `phase1_*` on purpose.** The
confounded main sweep shares 40 identical `(tag, family, seed, coverage)` keys
with the clean `phase1_decisive.jsonl` — same identifiers, different protocol
(`patience=600` vs `5000`). Under the old names, a `glob("results/phase1_*")`
would silently mix confounded and clean runs under indistinguishable keys. With
the rename, `phase1_*.jsonl` matches only clean data: verified 0 keys with
conflicting protocols across all 15 matching files.

* `SUPERSEDED_main_confounded.jsonl` (80) — **the file exhibiting the run-length confound**,
  where held-out MSE correlated with run length at r = −0.52/−0.55. Ran with
  `patience=600`. Retained because it is the evidence *for* that confound. Must
  not be aggregated with the authoritative results. On the `(tag, family, seed, coverage)` key this
  document uses elsewhere, **60** of its 80 cells now appear in `phase1.jsonl`
  (it was 40 before the coverage-0.05 promotion); on the *condition* key this
  document declares canonical at the top, **none** of them do, since `patience`
  differs. Either way the file must not be aggregated with the authoritative
  results.
* `SUPERSEDED_main_intermediate.jsonl` (55) — an **interrupted** main sweep, not
  a superseded protocol. All 55 rows ran the confound-free protocol
  (`patience=5000`, `n_restarts=2`), and the 35 rows it shares with
  `phase1_decisive.jsonl` are identical on every key they share (the decisive
  rows additionally carry the 25 `final_test_*` checkpoint-control keys, which
  did not exist when the intermediate sweep ran — which is also why the
  checkpoint-selection control shows no rows at coverage 0.05). The name is kept because renaming
  files has broken references here before, but it is misleading and this note is
  the correction. Its 20 coverage-0.05 rows were **unique** — the README quoted
  them while no authoritative file contained them — and have since been promoted
  into `phase1.jsonl`.
* `SUPERSEDED_featuremap_testmetric.jsonl` (8) — the single-seed check that selected a feature
  map on a *test* metric. Superseded by `featuremap_selection.jsonl`; kept
  because the retraction is part of the record.


## Phase 3 — entity-level out-of-distribution (`results/phase3_entity_ood/`)

The unit held out is a **drug**. These rows answer a different question against a
different target from every file above — raw `D`, not the additively
residualised `D_res` — and must never be pooled with Phase 2, Phase 2R or the
d-chain null. A Phase 3 row is identifiable by its `fold_key` column.
`results/phase3_entity_ood/README_PHASE3.md` is the detailed index.

* `phase3_entity_ood/primary.jsonl` (300) — every rung, both screens, all 30
  entity folds, ECFP4 fingerprints. **The authoritative Phase 3 result.**
* `phase3_entity_ood/controls.jsonl` (120) — Control A (random entity features)
  and Control B (fingerprints shuffled among drugs), partition 0.
* `phase3_entity_ood/positive.jsonl` (30) — the synthetic positive control: a
  target that *is* a feature potential plus a known rank-2 antisymmetric form.
  Distinguishes "no transferable pair signal" from "the machinery cannot detect
  one".
* `phase3_entity_ood/targets.jsonl` (120) — the secondary ChEMBL-mechanism
  representation and Control D, its shuffle. Secondary on annotation grounds: 16
  of 100 drugs have a target vector orthogonal to every other drug.
* `phase3_entity_ood/coverage.jsonl` (120) — sparser pair coverage among
  *training* entities only; test entities excluded at every coverage.
* `phase3_entity_ood/summary/` — generated tables, `verdict.json` (the frozen
  pre-registered rule), `verdict_posthoc.json` (the same rule with one statistic
  corrected), `folds.csv`, `per_drug.csv`, `mapping_stats.json`.
* `phase3_entity_ood/smoke.jsonl` — a pipeline check, gitignored. Not a result.


## Phase 4 — external chemical validation (`results/phase4_chemlex/`)

A completely different experimental system: the ChemLex acid–amine coupling
high-throughput screen (Zenodo doi:10.5281/zenodo.17596563), 11,669 measured
wet-lab reactions over 272 acids and 230 amines. The unit held out is a
**reactant**, the interaction is **bipartite** — an acid and an amine are
different entity types, so there is no antisymmetry here and there should not be
— and the endpoint is an uncalibrated LC-MS UV area ratio rather than a fitted
posterior. Nothing in this directory may be pooled with any earlier phase: a
Phase 4 row is identifiable by carrying both a `block` and a `screen` column.
`results/phase4_chemlex/README_PHASE4.md` is the detailed index.

Paths below are written **relative to the repository root**, deliberately. Phase
3's results went unindexed for a while because the index keyed on basenames and
`primary.jsonl` collided with an earlier phase's; a bare filename is not a
reference.

* `results/phase4_chemlex/primary.jsonl` (60) — the whole model ladder, both
  screens, both endpoints, all 15 authoritative entity folds, ECFP4.
  **The authoritative Phase 4 result.**
* `results/phase4_chemlex/primary_per_entity.csv` — one row per held-out
  reactant per fold: incremental pair skill, its blinded counterpart, and the
  entity's maximum Tanimoto to a training reactant of the same role. **This is
  the inferential unit**; the JSONL's fold-level numbers are the summary, not
  the evidence.
* `results/phase4_chemlex/controls.jsonl` (40) — shuffled acid features,
  shuffled amine features, both shuffled, and random features. Partition 0.
  Measured as an *increment over the additive baseline*, never as a skill
  against zero.
* `results/phase4_chemlex/positive.jsonl` (30) — a planted rank-3 acid–amine
  interaction on the exact ChemLex entity graph, at three planted sizes and
  against the both-shuffled control. Establishes the smallest interaction the
  pipeline can resolve, which is the number that bounds a negative result.
* `results/phase4_chemlex/transductive.jsonl` (20) — the ceiling: **pairs** held
  out, entities not. Never an entity-generalisation result; the report refuses
  to place it in an entity-OOD table.
* `results/phase4_chemlex/sensitivity.jsonl` (23) — **four of the five**
  registered sensitivities: the 8-level condition encoding, replicate-cell
  aggregation, the classical-amine subset, and k = 8 folds. The fifth —
  incremental pair skill on rows with `Conversion > 0` only — was registered and
  never implemented. It is named here because a registered analysis that
  silently does not appear is indistinguishable from one that was run and
  disliked. It was registered as outcome-conditioned and so could never have
  been a headline, which is why its absence changes no verdict.
* `results/phase4_chemlex/summary/` — every generated table, `verdict.json` (the
  registered rule), `verdict.md`, `per_entity.csv`.
* `results/phase4_chemlex/smoke.jsonl` — a pipeline check, gitignored. Not a
  result.

Raw data is **not** in this repository. The deposit is CC BY-NC 4.0 and is
fetched, digest-verified, by `python scripts/download_chemlex.py` into the
gitignored `data/raw/chemlex2025/`.

Two further Phase 4 artifacts are diagnostics that need a model refit, so they
have their own files rather than columns in `primary.jsonl` — which keeps the
authoritative results exactly what `run()` produced while leaving both
regenerable from committed code:

* `results/phase4_chemlex/pair_terms.jsonl` — the standard deviation of each
  rung's **fitted** interaction term on the rows it is scored on. A term that
  never leaves its zero initialisation reports exactly 0.0 incremental skill,
  which in a results table is indistinguishable from a genuine finding of no
  benefit; the flexible comparator does exactly that here. Regenerate with
  `python scripts/measure_pair_terms.py`.
* `results/phase4_chemlex/axes.jsonl` — Spearman correlations of the learned
  interaction axes, rotated into the singular basis of `W`, against a descriptor
  list fixed in advance, with a Benjamini-Hochberg q over the whole table.
  Exploratory and correlational; nothing here feeds the decision rule.
  Regenerate with `python scripts/interpret_phase4.py`.
