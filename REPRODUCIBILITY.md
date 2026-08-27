# Reproducing this work

Everything in `results/` was produced by the code in this repository. This
document says exactly which command produces which artifact, which artifacts are
authoritative, and — where it matters — which are expensive.

## Install

```
git clone https://github.com/akshathsarukkai/interaction-transfer
cd interaction-transfer
python -m pip install -e ".[dev]"
pytest
```

Python 3.11 or 3.12. The 3.11 floor is real: `sweep.run_sweep` passes
`max_tasks_per_child` to `ProcessPoolExecutor`, which 3.10 does not accept, and
on 3.10 the resulting `TypeError` is swallowed by the pool's serial fallback — so
the sweep would run correctly and about eight times slower with no error, which
is a worse failure than not installing.

The suite is ~600 tests and takes about two minutes. Tests that need a
third-party deposit or the fetched C++ sampler skip themselves, so a clean clone
with no network passes.

## What you can reproduce without any download

| Command | What it does | Cost |
|---|---|---|
| `pytest` | invariants, leakage guards, split integrity, document/result drift | ~2 min |
| `python scripts/run_phase1.py --smoke --workers 2` | Phase 1 pipeline end to end on a tiny grid | ~1 min |
| `python scripts/make_report.py --results results/phase1.jsonl --outdir results/summary --inject docs/PHASE1_SYNTHETIC.md` | regenerate every Phase 1 number from the committed results | seconds |
| `python scripts/run_dchain_null.py --part smoke --workers 2 --raw-dir tests/fixtures/koplev_tiny` | Phase 2N pipeline, no MCMC and no network | ~1 min |
| `python scripts/run_phase3_entity_ood.py --part smoke --workers 2 --raw-dir tests/fixtures/koplev_tiny --mapping tests/fixtures/koplev_tiny/drug_mapping.csv --fold-geometry 1 3 3` | Phase 3 pipeline on the generated fixture | ~1 min |
| `python scripts/run_phase4_chemlex.py --part smoke --workers 2 --raw-dir tests/fixtures/chemlex_tiny` | Phase 4 pipeline on the generated fixture | ~1 min |
| `python scripts/report_phase3_entity_ood.py --no-figures` | regenerate the Phase 3 documents from committed results | seconds |
| `python scripts/report_dchain_null.py` | regenerate the falsification document from committed metrics | seconds |
| `python scripts/report_phase2_residual.py` | regenerate the Phase 2R tables and figures | seconds |

These are exactly the paths CI runs. A `--smoke` run writes to its own
gitignored `smoke.jsonl`: a pipeline check is not a result, and it is kept out of
the results tree so it can never be aggregated with one.

## Acquiring the external data

No third-party dataset is committed here. See
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for licences.

```
python scripts/download_koplev.py       # Koplev deposit, CC BY 4.0, ~1.7 MB
python scripts/download_chemlex.py      # ChemLex deposit, CC BY-NC 4.0, ~550 kB
python scripts/prepare_dchain_null.py   # d-chain source, GPL-3.0: fetch, patch, build, verify
```

Each verifies a recorded SHA-256 and fails loudly if the upstream bytes have
changed. `prepare_dchain_null.py` additionally **refuses the build** unless the
patched sampler reproduces the unpatched one byte for byte on the authors' own
example data at the published seed. It needs a C++17 compiler and takes about ten
seconds.

## Phase by phase

### Phase 1 — the synthetic benchmark

```
python scripts/run_phase1.py --smoke --workers 2       # pipeline check, ~1 min
python scripts/select_hparams.py                       # selection, dev seeds only
python scripts/run_phase1.py --workers 8               # the sweep: hours of CPU
python scripts/make_report.py --results results/phase1.jsonl \
    --outdir results/summary --inject docs/PHASE1_SYNTHETIC.md
```

**Authoritative:** `results/phase1.jsonl` — 563 runs, one row per experimental
*condition* and seed, no duplicate cells. Every Phase 1 number is computed from
it.

**Do not glob `results/*.jsonl`.** Seventeen `phase1_*.jsonl` files are exact
subsets of `phase1.jsonl`, kept as the per-sweep record of how the results were
produced. Globbing double-counts every run: seeds appear twice, paired tests
silently double their `n`, and p-values shrink accordingly. Read
`phase1.jsonl` alone, or read the individual sweep files, never both.

**Not fully regenerable.** The Phase 1 sweeps were launched in batches over
time, and the batch structure is a historical record rather than a script:
`results/README_RESULTS.md` documents which file came from which sweep. One
artifact is explicitly irrecoverable — `results/hparam_search_stage1.log` is the
only surviving evidence for stage-1 hyperparameter selection, because its raw
JSONL was overwritten by an accidental re-launch. It is tracked deliberately even
though `results/*.log` is otherwise ignored. Three `SUPERSEDED_*.jsonl` files are
retained as evidence of retracted work and must never be aggregated with the
authoritative results.

### Phase 2 — the Koplev screen, original architecture

```
python scripts/download_koplev.py
python scripts/prepare_koplev.py                  # ingestion + dataset audit
python scripts/select_phase2_hparams.py
python scripts/run_phase2.py --part all           # 448 runs, ~83 min on 8 cores
python scripts/run_phase2.py --part parity        # 192 runs, the tuning-parity re-run
python scripts/report_phase2.py
```

**Authoritative:** `results/phase2/runs.jsonl` (448 rows). `tuning_parity.jsonl`
(192) carries the re-tuned contrast, and it is the one to quote: the original
comparison gave the two model families unequal tuning, and the numbers reported
in [`docs/phase2_koplev.md`](docs/phase2_koplev.md) §7.8 are the corrected ones.
`hparams.jsonl` is a selection input and must never be aggregated with
`runs.jsonl`.

### Phase 2R — residual directionality

```
python scripts/run_phase2_residual.py --counts     # what each part costs
python scripts/run_phase2_residual.py --part all   # 1,560 runs, ~2.5 h on 7 workers
python scripts/report_phase2_residual.py
```

`--part all` runs every block: main 400, sensitivity 160, controls 104, power 480,
robustness 416. Those totals are computed from the grid functions themselves by
`residual_sweep.part_counts()` and pinned to this document by
`test_documented_run_counts_match_the_grids`, so a grid change that makes this
paragraph stale fails the suite rather than going unnoticed.

**Authoritative:** `results/phase2_residual/honest_alpha.jsonl` (240) — the three
rungs the primary contrast depends on, rerun with the shrinkage coefficient
fitted on validation pairs withheld from model selection. **These are the numbers
the decision quotes.** `runs.jsonl` (400) is the main grid;
`contaminated_diagnostic.jsonl` is a deliberate control and is **not a result** —
every row carries `contaminated: true`.

There is deliberately no separate tuning stage: hyperparameters are selected
inside every run against that run's own validation pairs, because a shared tuning
stage is what produced the Phase 2 fairness failure.

### Phase 2N — the d-chain estimator null

```
python scripts/download_koplev.py                 # once, for the real reference values
python scripts/prepare_dchain_null.py             # once, ~10 s: fetch, patch, build, verify
python scripts/run_dchain_null.py --counts
python scripts/run_dchain_null.py --part oracle     # Control A, seconds
python scripts/run_dchain_null.py --part unshared   # Control C, ~5 min
python scripts/run_dchain_null.py --part primary    # the experiment, hours
python scripts/validate_dchain_null.py              # fidelity evidence vs the deposit
python scripts/report_dchain_null.py
```

**This is the expensive one.** The `joint` parts run `dchain.cpp` at its own
compiled-in defaults — 500,000 iterations, 1,999 retained samples, the settings
the deposit is provably at — which is **roughly 100 minutes per simulated screen
on one core**, and the full ensemble is on the order of ten hours. CI never runs
the MCMC; it runs the null's *logic*.

**Authoritative:** `results/dchain_null/metrics.jsonl` plus `summary/` and
`configs/`. The raw simulation inputs and MCMC output are tens of megabytes per
seed and are regenerated on demand rather than committed.

### Phase 3 — entity-level OOD on Koplev

```
python scripts/prepare_phase3_drugs.py          # PubChem + ChEMBL -> mapping
python scripts/prepare_phase3_targets.py        # ChEMBL mechanisms -> targets
python scripts/audit_phase3_drugs.py            # mapping statistics
python scripts/run_phase3_entity_ood.py --counts
python scripts/run_phase3_entity_ood.py --part all
python scripts/report_phase3_entity_ood.py
```

The two `prepare_` scripts hit public web services. Set
`INTERACTION_TRANSFER_CONTACT` to your own email first — PubChem and ChEMBL both
ask for a contact in the User-Agent so they can reach whoever is loading their
service. The mapping CSVs are committed with SHA-256 provenance over their cached
responses, so you do not need to re-run them to reproduce the results; re-run
them to re-derive the mapping from scratch, and expect the digest to change if
the databases have moved.

`--part all` is roughly an hour on six cores. Condition counts come from
`entity_ood.sweep.PART_GRIDS` and are printed by `--counts` rather than written
down, deliberately: a hard-coded total in prose is a number that goes stale
silently, and Phase 2R had to come back and close that hole once already.

**Authoritative:** `results/phase3_entity_ood/primary.jsonl` (300).
`summary/verdict.json` is the frozen pre-registered rule and
`summary/verdict_posthoc.json` is the same rule with one statistic corrected;
both are reported. `targets.jsonl` is the **secondary** ChEMBL-mechanism
representation — secondary on annotation grounds, since 16 of 100 drugs have a
target vector orthogonal to every other drug.

### Phase 4 — ChemLex external validation

```
python scripts/download_chemlex.py
python scripts/run_phase4_chemlex.py --counts
python scripts/run_phase4_chemlex.py --part all     # 173 conditions, ~2 h on 7 cores
python scripts/report_phase4_chemlex.py
python scripts/measure_pair_terms.py                # is each rung's pair term alive?
python scripts/interpret_phase4.py                  # exploratory latent-axis correlations
```

**Authoritative:** `results/phase4_chemlex/primary.jsonl` (60) and
`primary_per_entity.csv`. **The per-entity file is the inferential unit** — the
JSONL's fold-level numbers are the summary, not the evidence, because reaction
rows sharing one held-out acid are not independent observations.

The per-entity CSVs carry `role` and `entity` but **not** reactant structures:
the deposit is CC BY-NC 4.0 and its substrate inventory is not ours to
redistribute. Fetch the deposit and the report will re-attach structures locally.

`transductive.jsonl` holds out **pairs**, not entities. It is the ceiling, never
an entity-generalisation result, and the report refuses to place it in an
entity-OOD table.

`measure_pair_terms.py` is not optional if you are reading the model ladder: it
reports the fitted standard deviation of each rung's interaction term, and it is
how the `flexible` rung was found to have never left its initialisation. A term
that never trains reports exactly 0.000 incremental skill, which in a results
table is indistinguishable from a genuine finding of no benefit.

## Regenerating the documents

Every table in the phase documents is generated; none is typed. CI fails if a
committed document has drifted from the committed results.

```
python scripts/make_report.py --results results/phase1.jsonl \
    --outdir results/summary --inject docs/PHASE1_SYNTHETIC.md
python scripts/report_phase2.py
python scripts/report_phase2_residual.py
python scripts/report_dchain_null.py
python scripts/report_phase3_entity_ood.py --no-figures
python scripts/report_phase4_chemlex.py --no-figures      # needs the ChemLex deposit
python scripts/audit_phase3_drugs.py
```

Pass `--no-figures` to skip plotting; figures are written to the gitignored
`figures/` directory and to each phase's `summary/`. The document checks are
scoped to the **documents**, which format every number to three or four decimals
and regenerate byte-identically anywhere. They are deliberately not scoped to
`summary/*.csv`, which carries full-precision floats that do not survive a change
of numpy/scipy/pandas version — that mistake was made once and had to be undone,
because it turns a drift detector into a library-version detector.

## Determinism

Every condition's seed is a `blake2b` digest of its resolved configuration key,
truncated to 31 bits. It was previously Python's `hash()`, which is salted per
process: three interpreters returned three different seeds for the same
specification, so the committed results could not be regenerated from the
committed code. A test now runs the seed function in a fresh subprocess three
times and requires agreement.
