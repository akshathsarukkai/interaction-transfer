# Phase 2 — does the Phase 1 structural bias survive on real sequential interventions?

Phase 1 is frozen at tag `phase1-final`. Nothing here changes it, and the Phase 2
package imports nothing from the Phase 1 package.

Dataset semantics, provenance and the audit live in
[`phase2_dataset.md`](phase2_dataset.md). This file is the experiment.

## 1. The question

Phase 1's surviving claim, on synthetic data: *a structured cross-row
decomposition provides useful inductive bias when intervention-pair observations
are sparse, and the advantage disappears and then reverses as pair coverage
rises.*

The Phase 2 hypothesis is the narrowest real-data version of that:

> Under sparse observation of drug-pair schedules, an explicit
> symmetric/antisymmetric decomposition improves prediction of the **directional
> effect of completely held-out drug pairs**, relative to a capacity-matched
> unrestricted ordered-pair model.

Not claimed, and not testable here: any causal or biological mechanism, any
discovery about drug synergy, any clinical relevance. The experiment predicts a
modelled quantity from a 2017 screen and nothing more.

### What counts as the answer, fixed in advance

Written before the grid was run, because the design reports 2 screens x 5
coverages x several metrics and that is more than enough freedom to find a
favourable cell after the fact. Caveat a reader is entitled to: this document,
the code, the `PRIMARY` constant and the results all land in a single commit, so
the ordering is asserted here rather than demonstrated by the history. The
defence against multiplicity below leans on this, and cannot be independently
verified from the repository.

* **Primary comparison:** `structured` minus `unrestricted`, paired on split
  seed, on **A375** at the **sparse** coverages (0.05 and 0.10), on the two
  directional metrics -- Pearson *r* against measured `D_ij`, and ordering
  accuracy above the noise threshold. A375 is primary because it has the larger
  directional signal in absolute terms (mean |D| 0.145 against PANC1's 0.109);
  note the antisymmetric *variance share* is marginally higher on PANC1 (39.1%
  against 38.2%), so it is the scale that differs, not the fraction. PANC1 is a
  replication, not a second chance.
* **Secondary:** held-out ordered MSE; the same contrasts on PANC1; the *shape*
  of the structured-minus-unrestricted curve against coverage, which is the part
  that would resemble Phase 1.
* **Sanity conditions that must hold for any of it to mean anything:**
  `order_insensitive` must be clearly worse than `unrestricted` (the screen
  contains learnable order information at all), and the direction-shuffle
  control must remove the directional advantage.
* **A null result is a result.** If `structured` and `unrestricted` are
  effectively tied, that is reported as no transfer. The models are not tuned
  further in search of a win; the hyperparameter grid was fixed on validation
  before any test metric was computed and is not revisited.

## 2. Target

For an ordered schedule *i → j* the target is `synergy_measure`, the deposited
posterior-mean area-based synergy for that schedule. The derived quantity of
interest is the **directional effect**

    D_ij = y(i→j) − y(j→i)

computed from *predictions* on held-out pairs, never supplied as a training
label. Equivalently the model's antisymmetric component is `A_ij = D_ij / 2`.

Why this is legitimate on this outcome, checked before adopting it:

* `synergy_measure` is computed **per ordered combination**, not per unordered
  pair — the source code indexes it `[first, second]` and fills the full double
  loop with no symmetrisation.
* **No deposited column encodes both directions**, and the paper derives no
  schedule-difference score, so training on `y(i→j)` cannot leak `y(j→i)`.
* Both orderings are on the same scale (area-averaged viability difference
  against a Bliss baseline) and were measured under the same protocol, so the
  subtraction is the same comparison the paper itself makes when it correlates
  αβ against βα.

**The caveat that comes with it, and it is not small.** The baseline for
`y(i→j)` is built from drug *j*'s fitted single-agent curve, and the baseline for
`y(j→i)` from drug *i*'s. The protocol is also asymmetric by design: the first
drug is given at a single 1 µM dose, the second in a 4-point dose response. So a
nonzero `D_ij` can arise from *which drug occupies the dose-ranged position*
rather than from any biological schedule effect. This is not hypothetical — it is
measurable, and it accounts for the bulk of the antisymmetric signal:

> A two-way additive fit `y(i→j) ≈ a_i + b_j` captures **54% (A375) / 40%
> (PANC1)** of the antisymmetric variance on its own. An additive model's
> antisymmetric part is `g_i − g_j`, a per-drug "better first than second"
> tendency — exactly the shape the protocol asymmetry would produce.

Two consequences, both built into the design rather than noted and ignored:

1. **Every family gets the same free additive term** (`a_i` and `b_j`, two free
   scalars per drug). The families therefore compete only on the *pair-specific*
   remainder — the part that a per-drug tendency cannot explain, and the only
   part where "the algebra" could be doing work.
2. **The additive null is reported as a family**, not mentioned in prose.
   Without it, a directional correlation of 0.65 reads as a strong result when
   most of it is available to a 201-parameter model.

## 3. Split

The evaluation is **seen drugs, entirely unseen pair**. For a test pair `{i, j}`:

* both `i→j` and `j→i` are withheld — withholding one direction would hand the
  model the other and make `D_ij` half-observed;
* `i` and `j` each appear in other *training* pairs, so this is compositional
  generalisation and not cold-start prediction for an unseen drug;
* nothing derived from `{i, j}` appears in training. The screen contains no
  derived column, and `assert_no_pair_leakage` checks the ordered rows directly,
  on every run, at every coverage.

**Coverage nesting.** One permutation of the 4,950 unordered pairs is drawn per
split seed; training at coverage *c* is its first `round(c · 4950)` entries.
The training *pool* (train + validation) is therefore nested; the training set
itself is not, since validation takes a growing prefix of that pool, so a few
dozen pairs that train at coverage 0.05 become validation pairs at 0.10. What
the comparison needs is the weaker property, and it holds: the evaluation pool — pairs excluded at
the highest coverage, restricted to those whose endpoints clear a connectivity
floor at the *lowest* coverage — is **identical at every coverage**. A coverage
curve built this way varies only the amount of training data; it does not also
swap the test set underneath the comparison.

**Validation** pairs are carved out of the training pool at pair level, so early
stopping and restart selection never see a test pair.

**Connectivity.** Eligibility requires both endpoints to have training degree ≥ 3
in the sparsest training set. It depends only on the pair graph, never on `y` —
verified by a test that permutes the outcomes and asserts the split is unchanged,
so the filter cannot select easy pairs.

**Replicates** are not an issue here and the reason is documented rather than
assumed: the deposit carries exactly one modelled value per ordered combination
(the wet-lab triplicates were collapsed before the Bayesian fit), so there are no
replicate rows that could straddle a split. The pipeline asserts duplicates do
not exist rather than relying on that.

## 4. Models

Four families. Only the pair term differs; embeddings, the first-order term, the
pair feature map, depth, optimiser and training budget are identical.

    y(i→j)  =  a_i + b_j  +  ⟨pair term⟩

| family | pair term | role |
|---|---|---|
| `additive` | none | the null. How much of this screen needs no interaction model at all. |
| `order_insensitive` | `M(i,j)`, whole prediction symmetrised | how much order information the screen contains |
| `unrestricted` | `G(φ(e_i, e_j))`, unconstrained | **the primary baseline** |
| `structured` | `M(i,j) + A(i,j)`, `M(i,j)=M(j,i)` and `A(i,j)=−A(j,i)` by construction | the hypothesis |

**What the comparison is actually between — corrected.** An earlier draft of
this document said `unrestricted` is a strict *superset* of `structured` in
function class. **That is backwards**, and an independent audit caught it.
Setting `F_M = F_A = G` gives

    ½(G_ij + G_ji) + ½(G_ij − G_ji) = G_ij   identically,

so at equal head width `Structured(h)` contains `Unrestricted(h)` **exactly**
(verified to float32 roundoff — `test_structured_contains_unrestricted`). The
symmetric/antisymmetric split is the *identity* decomposition of an arbitrary
function, not a restriction of one.

At the widths actually run, capacity is matched by parameter count (two heads of
width 48 against one of width 86; 32,546 vs 32,423 parameters), so neither class
contains the other and both comfortably contain the truth. The hypothesis under
test is therefore **not** "does constraining the function class help" but:

> does splitting the pair term into a symmetric and an antisymmetric head, each
> with its own parameters, give a better optimisation and regularisation bias
> than one undivided head of the same total size?

A win for the structured family would be a win for *parameterisation*, not for
expressiveness — and equally, could not be dismissed as extra capacity. This
matters for what a result would mean: it is a weaker claim than the Phase 1
framing suggested, because Phase 2 has no simultaneous-treatment row and so
none of the Phase 1 cross-row constraint survives. Only weight sharing does.

**What is deliberately not carried over from Phase 1.** The Phase 1 model tied a
simultaneous-treatment row and two single-intervention rows to the same `S`. The
Koplev study never measured simultaneous treatment, and its single-agent
responses are already absorbed into the baseline `synergy_measure` is defined
against. Importing that structure would encode an assumption the data cannot
support. `M` here is only "the symmetric part of the ordered response", with no
claim that it equals a measurable co-treatment outcome.

**Parity.** Pair-parameter counts are matched by search, not by hope; the
baselines are widened to match the structured family's two heads, which can only
help them. The structured family evaluates its pair heads four times per row
against the unrestricted family's once, so it costs more *compute* at matched
*parameters* — reported, not hidden.

**Hyperparameters** are selected per family on validation loss at dev split seeds
100/101, disjoint from the evaluation seeds 0–7, **on A375 at coverage 0.10
only**, with one restart. The winning setting is then reused unchanged at every
coverage and on *both* screens — so PANC1, which carries the largest deficits and
every observed collapse, runs entirely on A375-selected settings. Each family
getting its own best setting is the arrangement most favourable to the baselines,
but only if that setting transfers. It does not: structured drew lr=0.01 and
unrestricted lr=0.03, and structured prefers 0.03 at coverage 0.10 and 0.003 at
0.70. §7.8 re-runs both families across the whole rate grid and reports the
contrast with each family at its own validation-selected rate; the conclusion is
unchanged, but two of the reported effect sizes are not.

**Fairness note on inputs:** drug identity embeddings only. No molecular
descriptors, no metadata, nothing available to one family and not another.

## 5. Limitations

Stated before the results, so they cannot be read as excuses.

* **The target is a modelled quantity, not an observation.** Every deposited
  value is a posterior mean from one joint 45,000-parameter Bayesian fit. A
  held-out pair's value is therefore *not statistically independent* of the
  training pairs — they share fitted single-agent curves. Phase 2 predicts the
  authors' model output, not a fresh experiment. This cannot be fixed without
  the raw nuclei counts, which are not deposited.
* **The protocol is order-asymmetric.** First drug at a fixed dose, second in
  dose response. Part of the measured "schedule effect" is an artifact of which
  drug occupies which position (see §2), and the additive component that
  captures it is handed to every family precisely so the comparison is not about
  it.
* **Identity embeddings only**, so there is no entity-level out-of-distribution
  test: nothing here says whether the structure generalises to a drug never seen
  in training.
* **Biological heterogeneity is sidestepped, not solved.** Two cell lines,
  modelled separately and never pooled. Nothing here transfers across cell lines.
* **Two cell lines is not a replication study.** Agreement between A375 and
  PANC1 is weak evidence of generality; disagreement is weak evidence of
  fragility.
* **Coverage is a proxy.** Real sparsity in a drug-interaction dataset is
  structured — some drugs are studied far more than others — while the coverage
  grid here removes pairs uniformly at random.
* **Ordering accuracy must be read against the additive null, never against
  0.5.** `synergy_measure`'s Bliss baseline is a function of the *second* drug
  alone, so a two-way additive fit already reaches ~0.80 held-out ordering
  accuracy at every coverage. A pair model scoring 0.80 has added nothing. The
  additive column appears in every table and in the directional figure for this
  reason.
* **Many comparisons.** The tables report 2 screens × 5 coverages × ~11 metrics
  × 4 contrasts. One pre-registered primary comparison is named in §1; every
  other cell is exploratory and an uncorrected `p < 0.05` among several hundred
  paired tests is expected by chance. No conclusion here rests on a non-primary
  p-value.
* **The ordering-accuracy threshold for A375 is transferred, not derived, and
  the transfer factor is genuinely uncertain.** The deposit supplies posterior
  SDs for PANC1 only. The code scales them by the ratio of overall spread
  (1.39×), giving τ ≈ 0.142. Two independent audits produced two different
  corrections and disagreed with each other: a within-PANC1 regression of
  `log sd` on `log |y|` implies ~1.10× (τ ≈ 0.113), and an argument from A375's
  own posterior implies ~0.74× (τ ≈ 0.076) — i.e. the shipped threshold may be
  up to 1.9× too large. **Nothing in the conclusion turns on this**: the full
  accuracy-vs-threshold curve is reported at every τ ∈ {0.00, 0.05, 0.10, 0.15,
  0.20, 0.30} for every family, the threshold is applied identically to all
  families so it cancels in the paired contrast, and the primary comparison is
  a correlation that uses no threshold at all. It is recorded because a reader
  should not take τ = 0.142 as measured when it is assumed.

## 6. Reproduction

```bash
python -m pip install -e ".[dev]"

python scripts/download_koplev.py          # digest-verified fetch + publication-stat check
python scripts/prepare_koplev.py           # dataset audit -> results/phase2/dataset_audit.json
python scripts/select_phase2_hparams.py    # validation-only tuning -> results/phase2/hparams.json
python scripts/run_phase2.py --part all    # the experiment -> results/phase2/runs.jsonl
python scripts/run_phase2.py --part parity # tuning-parity check (S7.8) -> tuning_parity.jsonl
python scripts/report_phase2.py            # tables and figures -> results/phase2/summary/
```

`--part all` writes 448 rows (320 `main`, 64 `direction_shuffle`, 64 `initvar`);
`--part parity` writes a further 192 to a separate file and is not pooled with
them. The parity block is the slower of the two -- it is dominated by the
coverage-0.70 cells -- and is not needed to reproduce the headline tables.

`python scripts/run_phase2.py --part smoke` is a two-minute pipeline check, not a
result. CI runs the Phase 2 invariant tests against a generated 12-drug fixture
and never downloads the deposit.

## 7. Results

448 runs, 0 failures, 83 min on 8 cores. 2 screens × 5 coverages × 4 families ×
8 split seeds, plus a direction-shuffle control and an init-variance block.
Generated by `scripts/report_phase2.py` from `results/phase2/runs.jsonl`. §7.8
adds a further 192 runs (`tuning_parity.jsonl`, 56 min) which are reported
separately and never pooled with the headline grid.

### 7.1 The sanity conditions, first

Both hold, so the comparison is not vacuous:

* **The screen contains learnable order information.** The order-insensitive
  family is worse than the unrestricted one at every coverage on both screens,
  and the gap widens with data — A375 held-out MSE 0.0177 vs 0.0070 at coverage
  0.70. Its ordering accuracy is exactly 0.500 everywhere, by construction.
* **The direction-shuffle control removes the signal.** Destroying the schedule
  direction in training only, and scoring on untouched test pairs, drops the
  directional correlation from +0.68 to **+0.011** (unrestricted) and +0.67 to
  **+0.014** (structured) at coverage 0.10, and to −0.02 / −0.03 at coverage
  0.40. Ordering accuracy falls to 0.48–0.51. The models were reading the
  schedule, not an artifact.

### 7.2 Held-out ordered MSE (lower is better)

**A375**

| coverage | additive null | order-insensitive | unrestricted | structured | structured − unrestricted |
|---|---:|---:|---:|---:|---:|
| 0.05 | **0.0221** | 0.0267 | 0.0230 | 0.0230 | +0.0000 (p=0.99, 4/8) |
| 0.10 | 0.0199 | 0.0223 | **0.0159** | 0.0161 | +0.0002 (p=0.36, 2/8) |
| 0.20 | 0.0190 | 0.0188 | **0.0112** | 0.0115 | +0.0003 (p=0.12, 2/8) |
| 0.40 | 0.0183 | 0.0180 | 0.0084 | **0.0083** | −0.0001 (p=0.60, 5/8) |
| 0.70 | 0.0180 | 0.0177 | **0.0070** | 0.0079 | **+0.0009 (p=7e-7, 0/8)** |

**PANC1**

| coverage | additive null | order-insensitive | unrestricted | structured | structured − unrestricted |
|---|---:|---:|---:|---:|---:|
| 0.05 | **0.0133** | 0.0157 | 0.0150 | 0.0153 | +0.0003 (p=0.35, 2/8) |
| 0.10 | 0.0121 | 0.0142 | **0.0126** | 0.0128 | +0.0002 (p=0.32, 2/8) |
| 0.20 | 0.0115 | 0.0109 | **0.0083** | 0.0086 | +0.0002 (p=0.07, 2/8) |
| 0.40 | 0.0110 | 0.0122 | 0.0057 | **0.0056** | −0.0001 (p=0.30, 5/8) |
| 0.70 | 0.0108 | 0.0130 | **0.0043** | 0.0050 | **+0.0006 (p=4.5e-5, 0/8)** |

### 7.3 Directional effect — the primary metric

**Pearson r against measured `D_ij`** (higher is better). The pre-registered
primary comparison is A375 at coverages 0.05 and 0.10.

| coverage | additive | unrestricted | structured | Δ (A375) | Δ (PANC1) |
|---|---:|---:|---:|---:|---:|
| 0.05 | 0.5666 | 0.5692 | 0.5714 | **+0.0022 (p=0.74, 5/8)** | −0.0493 (p=0.048, 1/8) |
| 0.10 | 0.6447 | 0.6847 | 0.6735 | **−0.0112 (p=0.074, 1/8)** | −0.0209 (p=0.21, 1/8) |
| 0.20 | 0.6866 | 0.7655 | 0.7509 | −0.0146 (p=0.058, 2/8) | −0.0266 (p=0.012, 1/8) |
| 0.40 | 0.7067 | 0.8180 | 0.8214 | +0.0034 (p=0.18, 5/8) | +0.0020 (p=0.65, 5/8) |
| 0.70 | 0.7159 | 0.8517 | 0.8339 | **−0.0178 (p=8.8e-6, 0/8)** | **−0.0246 (p=2.6e-5, 0/8)** |

**Ordering accuracy, A375** (threshold τ = 0.142; order-insensitive is 0.500 at
every rung): additive 0.800 / 0.845 / 0.865 / 0.863 / 0.868; unrestricted
0.802 / 0.850 / 0.888 / 0.916 / 0.939; structured 0.799 / 0.839 / 0.882 /
0.922 / 0.926. Δ = −0.002, −0.011, −0.006, +0.006, **−0.013 (p=0.007, 0/8)**.

### 7.4 Two findings larger than the family contrast

**At the sparsest coverage no pair model earns its keep, and on PANC1 the
additive null is outright better.** On PANC1 the null wins decisively: MSE 0.0133
against 0.0150 / 0.0153 (p=0.001 / 0.003, 8/8 split seeds), and on directional
correlation it beats unrestricted (0.4597 vs 0.4207) and structured (0.3714)
outright. On A375 the picture is weaker and the two-sided claim does not hold:
the null's MSE edge (0.0221 against 0.0230 for both) is p=0.14 / 0.27 with the
null ahead on only 5/8 and 4/8 seeds, and on directional r the null (0.5666) is
*behind* both pair models (0.5692 / 0.5714). The defensible statement is that
below coverage 0.10 the pair models buy nothing on A375 and actively cost
accuracy on PANC1. Ordering accuracy of ~0.80 at the sparse rung is the *null's*
number, not a pair model's.

**The pair models' advantage is real but arrives late, and is entirely about the
non-additive remainder.** By coverage 0.70 the unrestricted model more than
halves the null's error (0.0070 vs 0.0180 on A375) and lifts directional r from
0.716 to 0.852. The interaction structure in this screen is learnable — it just
needs a lot of pairs.

### 7.5 Antisymmetric collapse

Measured on the pair head alone (see §4), for the two families that have one.

| block | family | runs | collapsed runs | collapsed restarts |
|---|---|---:|---:|---:|
| main | structured | 80 | **0** | **13** |
| main | unrestricted | 80 | 0 | 0 |
| direction-shuffle | structured | 16 | **8** | **16** |
| direction-shuffle | unrestricted | 16 | 1 | 0 |

Three things worth stating plainly:

1. **No structured run in the headline grid collapsed** — but 13 of its 160
   restarts did, against 0 of 160 for the unrestricted family. The degenerate
   basin is real on real data and validation-loss restart selection caught every
   instance (worst restart-loss ratio observed: **53.9**). The pooled 8% rate is
   misleading, though, and the concentration matters: **all 13 are on PANC1, at
   coverages 0.40 (5 of 16 restarts) and 0.70 (8 of 16 — 50%)**, and none at all
   on A375. The rate is therefore highest in exactly the cell carrying the
   strongest negative claim. It is not the *mechanism* behind that claim —
   re-running PANC1 at coverage 0.70 with `n_restarts=6` leaves the deficit
   unmoved (directional r 0.8022 against the shipped 0.8013; still −0.024 versus
   unrestricted, p=1e-4) — but a reader should know the structured family is
   fighting its own optimisation hardest precisely where it loses by the most.

   The 13-versus-0 contrast is also structurally asymmetric and should not be
   read as a like-for-like comparison: the unrestricted family has no separate
   antisymmetric head, so the same statistic can only reach zero for it if `G`
   becomes exactly symmetric, which is a measure-zero event. See the caveat
   below.
2. **Under the direction-shuffle control the structured family collapses in half
   its runs (8 of 16), and in every restart of those runs (16 of 32).** That is the correct behaviour — there is no
   directional signal left to fit — and it is the strongest evidence the
   detector works, since the same detector reads a healthy 0.74–0.80 on the real
   grid.
3. The cost is paid asymmetrically: only the family carrying the inductive bias
   has a head that can die.

### 7.6 Variance sources

The init-variance block (A375, coverages 0.10/0.40, 2 split seeds × 4 init
seeds) is scored on the **same evaluation pool as the headline grid**. That is
worth stating because an earlier version of this block was not: it was run with
a two-rung coverage grid, which also redefines the nested split, so it scored on
2,970 pairs with nothing dropped for connectivity instead of the headline's
848–1,003. The numbers below replace those, and they are materially different —
the old figure understated initialisation noise by roughly a factor of four and
so overstated how much of it pairing removes.

Within-split SD across initialisations, in directional r:

| coverage | additive | unrestricted | structured | across-split SD (8 seeds) |
|---|---:|---:|---:|---:|
| 0.10 | 0.0000 / 0.0007 | **0.0214 / 0.0290** | 0.0064 / 0.0132 | 0.0355 (unr) / 0.0308 (str) |
| 0.40 | 0.0001 / 0.0002 | 0.0014 / 0.0016 | 0.0036 / 0.0017 | 0.0138 (unr) / 0.0165 (str) |

(Two entries per cell, one per split seed. The order-insensitive family is
omitted: its `D_pred` is identically zero, so directional r is undefined.)

Split-seed variance is still the larger term, but the margin depends sharply on
coverage and family. At coverage 0.40 it dominates by roughly 8×. **At coverage
0.10 — inside the pre-registered primary window — it leads the unrestricted
family's initialisation noise by only about 1.4×** (0.0355 against 0.0290).

The previously reported single figure of 0.0048 was an average that pooled in
the additive family, whose pair head does not exist and whose init SD is
therefore ~0; that alone halved it, and the inflated pool did the rest.

The consequence is a caveat on the primary comparison, not a change to it. Each
headline run uses one init seed, so initialisation noise is not averaged out
across the grid: at coverage 0.10 it contributes a standard error of roughly
0.0290/√8 ≈ 0.010 per family, which is the same size as the primary delta itself
(−0.011, and −0.002 once the learning rates are matched — see §7.8). Pairing on
split seed removes the larger variance component but not this one. Read together
with §7.8, the honest statement about coverage 0.10 is that the two families are
indistinguishable there, and that the experiment as run does not have the
resolution to claim otherwise.

### 7.7 Parameter counts

| family | total | pair | first-order | pair width | pair evals/row |
|---|---:|---:|---:|---:|---:|
| additive | 1,801 | 0 | 201 | — | 0 |
| order-insensitive | 34,224 | 32,423 | 201 | 86 | 2 |
| unrestricted | 34,224 | 32,423 | 201 | 86 | 1 |
| structured | 34,347 | 32,546 | 201 | 48 (×2 heads) | 4 |

Pair capacity matched to 0.4%; the structured family is the *larger* by that
margin, so nothing here favours it by parameter count. It does spend 4× the pair
forward passes.

### 7.8 Tuning parity — the one way the structured family *was* handicapped

The headline grid reuses a single hyperparameter choice, made on A375 at
coverage 0.10, at every coverage and on both screens (§4). Structured drew
lr=0.01, unrestricted lr=0.03. That is only fair if the choice transfers, and an
adversarial re-check found it does not: structured prefers **0.03** at coverage
0.10 and **0.003** at 0.70. At the primary cell the shipped pairing was the
single least favourable of the available comparisons for the family under test.

`run_phase2.py --part parity` re-runs both pair families across the full rate
grid {0.003, 0.01, 0.03} at four cells (192 runs, `tuning_parity.jsonl`). The
contrast is then recomputed with **each family at its own rate, selected per
split seed on validation loss** — never on test, and the shipped rate is in the
pool, so this removes a tuning artifact rather than shopping for a result.

| cell | metric | as reported | re-tuned |
|---|---|---:|---:|
| A375 0.05 | directional r | +0.0022 (p=0.74, 5/8) | −0.0017 (p=0.81, 6/8) |
| **A375 0.10** | **directional r** | **−0.0112 (p=0.074, 1/8)** | **+0.0026 (p=0.63, 4/8)** |
| A375 0.70 | directional r | −0.0178 (p=9e-6, 0/8) | **−0.0119 (p=8e-5, 0/8)** |
| PANC1 0.70 | directional r | −0.0246 (p=3e-5, 0/8) | **−0.0150 (p=4e-4, 0/8)** |
| A375 0.70 | MSE | +0.00091 (p=7e-7) | +0.00065 (p=2e-5) |
| PANC1 0.70 | MSE | +0.00064 (p=5e-5) | +0.00043 (p=9e-5) |
| A375 0.70 | ordering acc | −0.0125 (p=0.007, 0/8) | −0.0084 (**p=0.057**, 2/8) |
| PANC1 0.70 | ordering acc | −0.0119 (p=0.012, 0/8) | −0.0124 (p=0.002, 0/8) |

Two things change, and one does not.

1. **The coverage-0.10 deficit was largely a tuning artifact.** It reverses sign
   under matched tuning and is a clear null either way. Any reading of the
   original −0.0112 as "the baseline is pulling ahead as coverage grows" was
   reading the learning rate, not the parameterisation. The §1 primary
   comparison should be read as **tied at both sparse coverages**.
2. **The coverage-0.70 deficit is real but roughly a third smaller than
   reported** — −0.012 / −0.015 rather than −0.018 / −0.025 — and A375 ordering
   accuracy stops being significant (p=0.057). The MSE and directional-r results
   remain significant at p < 1e-4 with 0 of 8 seeds favouring structured.
3. **The verdict does not change.** Structured is never significantly *better*
   than unrestricted at any cell, under any tuning tried.

Things ruled out as handicaps, so the null is not fragile in the other
direction: the weight-decay grid edge (wd=0.03 is catastrophic for *both*
families, val loss 0.69 against 0.22–0.25); the learning-rate top edge
(structured at lr=0.03 at coverage 0.70 is far worse, r 0.760 against 0.834);
restart starvation from collapse (PANC1 0.70 with `n_restarts=6` moves the
deficit not at all); and capacity, initialisation, schedule and epoch budget,
which are identical across families by construction (§7.7).

## 8. Verdict

**Did the Phase 1 structural bias transfer to real sequential interventions?**

**No transfer — shading into evidence against at high coverage.**

On the pre-registered primary comparison (A375, directional Pearson r,
structured − unrestricted, sparse coverages), the two are **tied**: +0.0022
(p=0.74) at coverage 0.05 and −0.0112 (p=0.074) at 0.10 as run. The second of
those does *not* survive a tuning-parity check — the learning rate was selected
once at coverage 0.10 on A375 and does not transfer, and with each family at its
own validation-selected rate the same contrast is +0.0026 (p=0.63) — so it
should not be read as favouring the baseline (§7.8). Either way it is a null.
PANC1 agrees in direction and is slightly worse for the structured model.
Nothing in the sparse regime supports the hypothesis.

At coverage 0.70 the structured model is **consistently and significantly
worse** on both screens. As run: held-out MSE +0.0009 / +0.0006, directional r
−0.018 / −0.025, ordering accuracy −0.013 / −0.012. Roughly a third of that gap
is hyperparameter transfer rather than parameterisation; with each family tuned
at that coverage the deficits are MSE +0.00065 / +0.00043 and directional r
−0.012 / −0.015, still p < 1e-3 with **0 of 8 split seeds** favouring structured.
A375 ordering accuracy is the one metric that loses significance under
re-tuning (−0.008, p=0.057); PANC1 ordering accuracy does not (−0.012, p=0.002).
Read as exploratory (it is not the primary cell), but the sign is unambiguous
and consistent across seeds and screens, and it does not depend on the tuning
choice.

**The Phase 1 shape is absent, and if anything inverted.** Phase 1 found the
structured model helping when pairs were sparse and hurting when they were
dense. Here it is level when sparse and hurt when dense. The one coverage where
it edges ahead (0.40, +0.003 / +0.002) is not significant on either screen. Note
the sparse end is level *after* correcting for the tuning artifact; as run, the
curve looked mildly downward-sloping from 0.05 onward, and that slope was partly
an artifact.

**What this does and does not license.** It does not refute the Phase 1 result,
which was about a benchmark generated from the model's own assumptions and stays
frozen. It says the bias did not carry to the first real sequential-intervention
dataset it was pointed at, under a design where the same bias had a fair chance:
capacity matched, hyperparameters tuned per family on validation, the sanity
conditions met, and the control clean. One qualification on "fair chance": the
tuning was done at a single coverage on a single screen and did not transfer, so
the *headline* grid did handicap the structured family. §7.8 removes that
handicap and re-reports; the conclusion is unchanged, two effect sizes shrink by
about a third, and one becomes non-significant.

The most likely reason is visible in the data rather than the models. Roughly
half the antisymmetric variance in this screen is a per-drug "better first than
second" potential `g_i − g_j`, which every family gets for free from the
first-order term, and which plausibly reflects the protocol's own asymmetry
(first drug at a fixed dose, second titrated) rather than biology. The residual
that a pair-specific antisymmetric head could capture is what is left over, and
on this screen it is small enough that constraining it costs more than it buys.

**Classification: no transfer.**
