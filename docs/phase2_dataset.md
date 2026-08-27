# Phase 2 dataset — the Koplev sequential anticancer screen

What this file is for: everything about the data that had to be established
*before* a target could be defined, including two things the deposit itself
gets wrong. Results live in [`phase2_koplev.md`](phase2_koplev.md).

## 1. Provenance

| | |
|---|---|
| Paper | Koplev S, Longden J, Ferkinghoff-Borg J, Blicher Bjerregård M, Cox TR, Erler JT, Pedersen JT, Voellmy F, Sommer MOA, Linding R. *Dynamic Rearrangement of Cell States Detected by Systematic Screening of Sequential Anticancer Treatments.* Cell Reports 20(12):2784–2791, 2017. |
| Paper DOI | [10.1016/j.celrep.2017.08.095](https://doi.org/10.1016/j.celrep.2017.08.095) (PMID 28930675) |
| Data | Mendeley Data, [10.17632/wgybvcvjwf.1](https://doi.org/10.17632/wgybvcvjwf.1), version 1 |
| Licence | CC BY 4.0 |
| Model source code | <https://github.com/skoplev/d-chain> (GPL-3.0) |
| Acquired | 2026-08-17 |
| Acquisition | `python scripts/download_koplev.py` — digest-verified, idempotent |

Only *modelled/aggregate* data is deposited. The raw nuclei counts are not
public, so nothing downstream of the authors' Bayesian fit can be recomputed
from source. This is the single most important limitation of Phase 2 and it is
carried into [`phase2_koplev.md`](phase2_koplev.md) rather than mentioned once
here.

The raw files are gitignored. They total 1.7 MB and the licence would permit
vendoring them, but a digest-verified fetch is a stronger record than a copy
that can drift silently from its source. `data/raw/koplev2017/PROVENANCE.json`
is written at download time with every URL, size and SHA-256.

## 2. Experimental design

Answering the questions that had to be answered before constructing a target.

**Interventions.** 100 compounds — the Approved Oncology Drug Set IV
(Developmental Therapeutics Program, NCI). Identity only; no doses vary across
rows and no molecular descriptors are used (see §7).

**Biological systems.** Two cell lines in the primary screen: **A375**
(malignant melanoma, ATCC CRL-1619) and **PANC1** (pancreatic adenocarcinoma,
ATCC CRL-1469). A separate 193-combination subset was run in four pancreatic
lines (AsPC-1, BXPC3, Capan1, DAN-G).

**What an ordered pair is.** Cells in 384-well plates were pretreated with drug
α at a **single 1 µM dose for 24 h**, then drug β was added in a **4-point dose
response** (10 µM / 1 µM / 100 nM / 10 nM; 5-point in the validation arm) and
incubated a **further 24 h**, after which nuclei were counted. So the time
separation between the two treatments is 24 h, and the readout is 24 h after the
second.

**There is no washout.** The second drug is added *on top* of the first
(35 µL + 5 µL + 10 µL), so the 1 µM pretreatment is diluted to 0.8 µM rather than
removed. The authors' model carries an explicit residual-effect term for this.
The only arm with drug removal is the α0 single-agent control. This matters for
interpretation: `i → j` is "j on top of 24 h of i", not "j after i has washed
out", and the asymmetry between `i → j` and `j → i` is therefore a mix of
scheduling and of which drug is still present.

**Both directions.** Measured, not imputed: *"for each of the 10,000 possible
drug combinations, nuclei counts were measured after single-drug exposure (α0,
αα, β0, and ββ) and sequential combination in both orders (αβ and βα)."* Each
primary table is a **complete 100 × 100 ordered matrix**, diagonal included.

**Self-combinations.** The 100 diagonal entries (α then α) are real measurements
and the paper interprets them ("self-synergy for cisplatin alone, λ = 0.735";
"gemcitabine was found to be highly self-antagonistic, λ = −1"). Phase 2
**excludes** them, for a reason unrelated to their quality: a self-pair has no
ordering, contributes `A(i,i) = 0` by definition, and would hand the structured
model a constraint that is true by construction rather than learned. The
exclusion is counted in the audit (100 rows per screen).

**Simultaneous treatment was never measured.** *"While simultaneous treatments
were not directly measured, and thus cannot be definitively excluded…"* This is
why the Phase 2 structured model does **not** reuse the Phase 1 decomposition —
see §6.

**Replicates.** All measurements were performed in triplicate (~250,000 data
points total), but replicates were reduced to sufficient statistics — *"the
experimental mean and variance of normalized cell counts across experimental
repeats"* — before the Bayesian model was fitted. **The deposit carries no
replicate rows**: exactly one modelled value per ordered combination. So
"replicates must not cross the split" is vacuous here, and the pipeline instead
asserts that duplicate ordered rows do not exist
(`test_duplicate_ordered_rows_are_rejected`).

**Missing values.** None. Both primary tables are complete; the audit reports
`missing_values_in_raw = 0` for each.

## 3. Fields

| column | meaning |
|---|---|
| `first_compound`, `second_compound` | the ordered schedule α → β |
| `synergy_measure` | the target. Posterior mean of the *area-based measure of synergy* |
| `lambda` | signed posterior probability of interaction, in [−1, 1] |
| `synergy_sd` | (tables 4/5 only) posterior SD of `synergy_measure` |

**`synergy_measure` is a derived, modelled quantity, not an observation.** From
the authors' own code (`post/interpretMCMC.R`), for each retained MCMC sample it
is the mean over 10 concentrations spanning [0.01, 10] µM of
(baseline viability − combination viability); the deposited value is the
posterior mean of that. Positive = synergy, negative = antagonism. The baseline
is built from the **second** drug's fitted single-agent curve under Bliss
independence, with the first drug's effect and its residual effect entering as
multiplicative factors that cancel in the difference.

Three consequences that constrain what Phase 2 can claim:

1. It is computed **per ordered combination** — the array is indexed
   `[first, second]` and filled over the full double loop with no
   symmetrisation. So `y(i→j)` and `y(j→i)` are separate quantities, which is
   what makes the directional target legitimate.
2. It is **already relative to single-agent response**, which is why the Phase 2
   models carry no separate single-intervention rows: there is nothing left for
   them to explain.
3. The values come from **one joint 45,000-parameter posterior**, not from
   10,000 independent fits. A held-out pair's value is therefore not
   statistically independent of the training pairs — they share the same fitted
   single-agent curves. Phase 2 predicts *the authors' modelled quantity*, not a
   fresh experiment, and §5 of [`phase2_koplev.md`](phase2_koplev.md) treats
   this as a first-class limitation rather than a footnote.

**`lambda`** is the average of a Boolean selector over MCMC samples — the
posterior probability that the combination needs its own dose-response
parameters at all — signed by the direction of the effect. It is an
*effect-existence* probability, not an effect size, and `p = 1 − |λ|`. Phase 2
uses it only to reproduce the paper's counts (§4); it is never used to filter
training data, because filtering on it would be filtering on the outcome.

**`synergy_sd` is posterior uncertainty, not replicate noise.** In the source
code it is `sd(synergy_index[, a, b])` — the SD across retained MCMC samples.
It is used in one place only: to set the magnitude threshold below which a
measured schedule effect is treated as too small to score for ordering accuracy.

**No column encodes both directions.** There is no schedule-difference or
sequence-effect score anywhere in the deposit, and the paper derives none — the
only place both orderings meet is a descriptive correlation. So training on
`y(i→j)` cannot leak `y(j→i)` through a derived field.

## 4. Which tables are usable — and why the deposit's descriptions are wrong

The deposit holds five CSVs. **Two of them are usable.** Establishing that
required comparing the files, not reading their descriptions.

| file | deposit says | actually | used |
|---|---|---|---|
| `Data Table 1.csv` | A375 combinatorial screen | A375, complete 100×100 ordered matrix | **yes** |
| `Data Table 2.csv` | PANC1 combinatorial screen | PANC1, complete 100×100 ordered matrix | **yes** |
| `Data Table 3.csv` | pancreatic panel, 4 cell lines | 193 ordered combinations × 4 lines — **zero reverse pairs** | no |
| `Data Table 4.csv` | **A375 validation screen** | **PANC1.** `synergy_measure` byte-identical to Table 2 on all 9,900 shared rows | `synergy_sd` only |
| `Data Table 5.csv` | PANC1 validation screen | a 190-row subset of Table 4 — zero reverse pairs | no |

**Table 4 is not A375 and not an independent screen.** Its `synergy_measure`
column matches Table 2 exactly (max absolute difference 0.0 over 9,900 rows);
it is Table 2 minus the 100 rows with Cytarabine as first compound, plus a
posterior-SD column and a sign-corrected `lambda` (860 rows differ from Table 2,
identical in magnitude and flipped in sign, and Table 4's sign agrees with
`sign(synergy_measure)` everywhere). Table 5 is an exact subset of Table 4.
**There is no A375 validation table in the deposit at all.**

Believing the descriptions would have produced a two-context experiment in which
one "cell line" is a copy of the other, at roughly 2× weight, under the wrong
label. `koplev.verify_raw()` re-derives this on every ingestion and raises if it
stops holding, and `test_verify_raw_detects_the_duplicated_table` pins it.

**Why the "validation" values are not independent estimates.** The validation
experiments were real new measurements, but *"these validation results were used
to further refine the model so that the updated synergy measures, presented
throughout this study, take into account both the primary screen and validation
data."* Tables 2, 4 and 5 are three views of one post-validation joint posterior.

**How Table 1 was pinned to A375 and Table 2 to PANC1**, given that one deposit
label is demonstrably wrong. Two independent fingerprints, both against the
paper:

* λ values quoted in the paper's Discussion. A375 bortezomib→vemurafenib
  λ = 0.926 and erlotinib→vemurafenib λ = 0.725 appear in Table 1
  (0.925962981, 0.724862431). PANC1 gemcitabine→erlotinib λ = 0.996 and
  cisplatin→gemcitabine λ = 0.805 appear in Tables 2/4/5.
* The paper's significance counts reproduce **exactly** — see §4 below.

## 5. Reproducing the publication's statistics

Run automatically by `scripts/download_koplev.py`, which exits non-zero if the
counts stop matching.

| statistic | paper | this pipeline |
|---|---|---|
| A375 synergistic, p < 0.05 | 707 | **707** ✓ |
| A375 antagonistic, p < 0.05 | 1,845 | **1,845** ✓ |
| PANC1 synergistic, p < 0.05 | 551 | **551** ✓ |
| PANC1 antagonistic, p < 0.05 | 1,464 | **1,464** ✓ |
| total high-confidence | 1,258 + 3,309 = 4,567 | **4,567** ✓ |
| fraction of all combinations | "approximately 23%" | **22.8%** ✓ |
| forward/reverse correlation, A375 | r = 0.25 | 0.237 |
| forward/reverse correlation, PANC1 | r = 0.23 | 0.218 |
| drugs / combinations per line | 100 / 10,000 | 100 / 10,000 ✓ |

The threshold is `|λ| > 0.95`, i.e. `p = 1 − |λ| < 0.05`, over the **full**
matrix including the 100 self-combinations — that is the paper's denominator.
The modelling pipeline excludes the diagonal; this check deliberately does not,
or it would not reproduce.

**Unresolved discrepancy, disclosed.** The paper reports a cross-cell-line
correlation of *"r = 0.19"* between PANC1 and A375 synergy measures. From the
deposited tables the same comparison gives **r = 0.388** (0.382 off-diagonal).
The reverse-direction correlations reproduce to within 0.013 and the
significance counts reproduce exactly, so the table identification is not in
doubt; the r = 0.19 figure appears to have been computed over some subset or
transform the paper does not specify. Nothing in Phase 2 depends on it — no
result is pooled across cell lines — but it is recorded rather than glossed.

## 6. Dataset audit

Emitted by `python scripts/prepare_koplev.py`; the committed copy is
`results/phase2/dataset_audit.json`.

| | A375 | PANC1 |
|---|---:|---:|
| rows in raw table | 10,000 | 10,000 |
| self-combination rows removed | 100 | 100 |
| rows with missing values removed | 0 | 0 |
| **rows used** | **9,900** | **9,900** |
| unique drugs | 100 | 100 |
| unordered pairs | 4,950 | 4,950 |
| ordered pairs | 9,900 | 9,900 |
| pairs with **both** directions | 4,950 (100%) | 4,950 (100%) |
| replicate rows per ordered pair | 1 | 1 |
| pair-graph degree, min / max | 99 / 99 | 99 / 99 |
| `synergy_measure` mean (sd) | −0.033 (0.180) | −0.024 (0.130) |
| variance that is **symmetric** | 61.8% | 60.9% |
| variance that is **antisymmetric** | **38.2%** | **39.1%** |
| corr(`y(i→j)`, `y(j→i)`) | 0.237 | 0.218 |
| mean \|D\| = \|y(i→j) − y(j→i)\| | 0.145 | 0.109 |
| high-confidence rows (\|λ\| > 0.95) | 25.4% | 20.1% |

The line that decides whether Phase 2 is worth running at all is the
antisymmetric variance share: **38–39% of the variance in this screen is
order-dependent**, and forward and reverse are only weakly correlated. There is
real directional signal to predict.

One structural fact that shapes the whole experiment: a plain two-way additive
fit `y(i→j) ≈ a_i + b_j` already explains **54%** of the variance in A375 and
**41%** in PANC1, and captures **54% / 40%** of the *antisymmetric* variance on
its own (an additive fit has antisymmetric part `g_i − g_j`). So most of the
directional signal in this screen is a per-drug "better first than second"
tendency, and only the remainder is genuine pair-specific ordering structure.
Every Phase 2 model therefore carries the same free additive term, and the
additive null is reported as a family so the size of that remainder is never
hidden inside a headline number.

## 7. What is deliberately not used

* **Molecular descriptors.** Drug identity embeddings only. The Phase 1 question
  being transferred is whether the *structural* inductive bias helps; adding
  chemistry would introduce a second source of improvement and confound it.
  Entity-level generalisation is a later phase.
* **`lambda` as a filter.** It is an outcome-derived confidence, so filtering
  training or test rows on it would be filtering on the outcome.
* **Tables 3 and 5.** Zero reverse pairs, so no directional target exists in them.
* **Table 4's `synergy_measure` and `lambda`.** Duplicates of Table 2.
* **The pancreatic four-line panel.** 193 one-directional combinations is not
  enough for a held-out-pair experiment even before the missing reverse
  direction rules it out.
