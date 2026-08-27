# The scientific record

This is what the project asked, what happened to the question, and what survived.
It is written to be read by someone who was not here, and it is deliberately
honest about the parts that did not work, because those are the parts that
redirected the work.

The private research notebook this repository was cut from holds the
chronological version — every dated session, every wrong turn, every internal
review. That record is not published. What is published is the science: the
findings, the failures that mattered, the corrections, the pre-registrations, the
code and the evidence.

---

## The original hypothesis

The project began as **intervention algebra**. When two interventions `i` and `j`
are applied together, the outcome decomposes as

```
y(i, j) = f(i) + f(j) + I(i, j)
```

— two independent contributions and a term that belongs to the pair. The
hypothesis was architectural: that imposing explicit algebraic structure on
`I` — splitting it into a symmetric part `S(i,j) = S(j,i)` and an antisymmetric
part `A(i,j) = −A(j,i)` — would be a useful inductive bias when pair
observations are sparse, because it forces the model to reuse what it learns
about one pair on every other pair sharing an entity.

**Phase 1 tested this on synthetic data and the answer was a qualified yes.**
563 runs, five model families, five pair coverages, held out at the level of whole
unordered pairs. At pair coverage 0.10 the constrained model beat a
capacity-matched unconstrained pair model by −0.137 in held-out MSE
(p = 0.0007, 14/17 seeds), surviving Holm correction over nine tests and holding
at a genuinely matched 2× capacity.

**And the qualification was severe.** The advantage is not universal: it reverses
by coverage 0.20 (+0.055, p = 0.011) and has decayed to +0.012 by 0.40
(p = 0.12). More importantly, at the coverage where it wins, the structured model
merely *ties the additive null* — 0.841 against 0.843 — so what it demonstrates
is that structure prevents a pair model from hurting, not that pair structure
helps. And the benchmark's own generator installs the constrained model's
inductive bias by construction, which is the standard failure of synthetic
evidence: it can show that a prior is *learnable*, never that it is *true*.

The ablation ladder also does not isolate antisymmetry. The step from
`shared_pair` to `algebra` imposes antisymmetry, symmetry, a trunk split and a
width change at once. The evidence is consistent with antisymmetry being
load-bearing and does not establish it.

Phase 1 is frozen and reported in [`PHASE1_SYNTHETIC.md`](PHASE1_SYNTHETIC.md).

---

## What failed

**Phase 2 took the same architecture to real data, and it did not transfer.**

The data is the Koplev et al. 2017 sequential anticancer screen: two 100×100
ordered drug matrices (A375 melanoma, PANC1 pancreatic), 9,900 usable ordered
rows per screen, every one of the 4,950 unordered pairs measured in both
directions. The order matters — drug `i` then drug `j` is a different experiment
from `j` then `i` — which is what makes an antisymmetric component meaningful at
all. 38–39% of the outcome variance is antisymmetric.

Against a capacity-matched unrestricted model, the structured parameterisation
bought nothing at sparse coverage and cost something when data was plentiful:

| screen | coverage | directional Pearson, structured − unrestricted | p |
|---|---:|---:|---:|
| A375 | 0.05 | +0.0022 | 0.74 |
| A375 | 0.10 | +0.0026 (under tuning parity) | 0.63 |
| A375 | 0.70 | **−0.012** | 8e-5 |
| PANC1 | 0.70 | **−0.015** | 4e-4 |

At coverage 0.70 the structured model lost on 0 of 8 seeds in both screens. This
is a clean negative and it is not rescued by re-tuning: the numbers above are
already the re-tuned ones, produced after an audit found the original comparison
had given the two arms unequal tuning.

**The architectural hypothesis, as stated, did not survive contact with real
data.** What happened next is the actual content of this project.

---

## What survived

The negative result was informative in a specific way: it said the *architecture*
was wrong, not that there was no pair structure to find. So Phase 2R asked a
narrower question about the directional effect

```
D(i, j) = y(i → j) − y(j → i)
```

which is antisymmetric by construction. `D` splits **exactly** — a discrete Hodge
decomposition — into a gradient part and a cyclic part:

```
D(i, j) = (g_i − g_j) + C(i, j)
```

`g_i − g_j` is a per-drug *ordering potential*: some drugs simply tend to work
better first, others better second, and that tendency alone explains a lot of the
matrix. `C` is what is left: genuinely pair-specific circulation, the part that
cannot be explained by ranking the drugs on a line. The two parts are orthogonal
(measured inner product ≈ 5e-19).

| screen | gradient share | cyclic share |
|---|---:|---:|
| A375 | 53.8% | 46.2% |
| PANC1 | 39.8% | 60.2% |

Nearly half of the directional signal, and on PANC1 the majority, is
irreducibly pair-specific. The question then becomes whether that part is
**predictable for a pair never observed in either direction**.

It is — given enough pairs:

| screen | coverage | training pairs | residual skill on unseen pairs | 95% CI | seeds |
|---|---:|---:|---:|---|---:|
| A375 | 0.40 | 1,683 | **+0.229** | [+0.189, +0.269] | 8/8 |
| A375 | 0.70 | 2,945 | **+0.353** | [+0.315, +0.391] | 8/8 |
| PANC1 | 0.40 | 1,683 | **+0.198** | [+0.147, +0.248] | 8/8 |
| PANC1 | 0.70 | 2,945 | **+0.366** | [+0.323, +0.410] | 8/8 |

and it is not, below that:

| screen | coverage | training pairs | residual skill | 95% CI |
|---|---:|---:|---:|---|
| A375 | 0.20 | 842 | +0.022 | [−0.011, +0.055] |
| A375 | 0.10 | 421 | −0.009 | [−0.038, +0.020] |
| PANC1 | 0.20 | 842 | +0.019 | [−0.020, +0.058] |

A permutation control sits at −0.004 / −0.005 where the real model reaches +0.23
and +0.37. A rank-2 model with **204 parameters and no hyperparameter search at
all** reaches +0.197 / +0.250 — so the structure is genuinely low-rank, not an
artefact of a large model finding something to fit.

The project's question had changed. It was no longer *is this the right
architecture*; it was **when does pair-specific interaction structure generalise
to combinations you have not observed** — and the answer already had a shape: it
depends on how much of the interaction graph you have seen.

---

## Artifact falsification

There is a serious objection to everything above, and it has to be dealt with
before anything else is worth saying.

The Koplev "synergy" values are not measurements. They are posterior means from a
single joint Bayesian fit — the **d-chain** model, roughly 45,000 parameters
estimated over the whole screen at once. Every pair's number depends on
single-agent curves shared with every other pair. So a held-out pair is not
statistically independent of the training pairs, and worse: a low-rank cyclic
structure is exactly the kind of thing a shared-parameter estimator could
*manufacture* out of nothing. Error in drug `a`'s shared single-agent curve is
pushed into the combination parameters for every partner `b`, and the product of
a per-first-drug error with a per-second-drug sensitivity is a rank-1 bilinear
term whose antisymmetric part looks precisely like the signature Phase 2R
measures.

Phase 2N tested this directly, and the only honest way to test it was to run **the
published sampler itself** rather than a model of it. `scripts/prepare_dchain_null.py`
fetches the authors' C++ at a pinned commit, verifies every digest, applies the
patches needed to build it, and then **refuses the build unless the patched
program reproduces the unpatched one byte for byte** on the authors' own example
data at the published seed. The synergy scoring step is ported line by line from
the authors' R.

Then: simulate screens in which the true pair-specific sequential interaction is
**exactly zero**, run the real estimator over them, and measure whether Phase 2R's
signature appears anyway.

It did not. Across the null screens the best held-out rank-2 skill is **+0.0003**,
against **+0.16 to +0.25** on the real data. Every decision statistic hit the
design's one-sided p floor of 0.048 (1/21). The registered rule returned
**LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT** on a complete 116/116 ensemble.

**What this does and does not establish.** It falsifies the *tested* estimator
artifact under the *tested* nulls — sampling error in a correctly specified
d-chain fit. It does not falsify every possible source of misspecification, and
the comparison only bounds any artifact below the detector's sensitivity floor of
roughly 5%. The full statement of scope is in
[`dchain_null_falsification.md`](dchain_null_falsification.md).

---

## Entity extrapolation

Predicting an unseen *pair* is one thing. Predicting anything about a drug that
has never appeared anywhere in training is a different and much harder thing, and
it is the question that matters if this is ever to be useful.

Phase 3 held out **drugs**, not pairs. 100 drugs mapped to structures via PubChem
and to mechanisms via ChEMBL, ECFP4 fingerprints, 30 entity-disjoint folds, with
leakage guards that fail if a test drug reaches any fitted quantity and
hyperparameter selection that cannot see a test entity.

There is a real signal, and it needs the held-out drug's own structure:

| screen | incremental pair skill over the additive baseline | p |
|---|---:|---:|
| A375 | +0.0273 | 0.0134 |
| PANC1 | +0.0536 | 0.00031 |

**And it is almost entirely analogue interpolation.** Stratifying the held-out
drugs by their maximum chemical similarity to any training drug:

| screen | stratum | drugs | incremental skill | p |
|---|---|---:|---:|---:|
| A375 | high similarity | 31 | **+0.090** | 0.0024 |
| A375 | low similarity | 40 | +0.016 | 0.13 |
| PANC1 | high similarity | 31 | **+0.138** | 0.00029 |
| PANC1 | low similarity | 40 | +0.015 | 0.22 |

High-minus-low is significant in both screens (p = 0.0089 / 0.0011), and
per-drug skill correlates with similarity at Spearman +0.23 / +0.32. **For a drug
with no close training analogue, the transfer is not detectable.** The honest
label for Phase 3 is *analogue-only transfer*, and that is a limit on the claim,
not a footnote to it.

Phase 3 could not answer the both-unseen question at all. Every E2 cell is
negative and none is significant, and the design would have needed an effect of
+0.11 to +0.14 to detect one — with only 45 pairs per fold available in that
regime, it was never powered for it.

---

## External validation

Everything so far rests on one dataset, from one lab, with an estimator-derived
target. Phase 4 went to a different scientific domain, a different measurement
type, and a directly measured endpoint: the **ChemLex** acid–amine coupling
high-throughput screen — 11,669 wet-lab reactions over 272 carboxylic acids and
230 amines.

The structure is bipartite, so the decomposition changes shape and loses its
antisymmetry entirely:

```
y(a, n, c) = μ + f_A(a) + f_N(n) + f_C(c) + I_AN(a, n)
I_AN(a, n) = z_A(a)ᵀ W z_N(n)
```

An acid and an amine are different kinds of thing. `I(n, a)` does not even
typecheck. **There is no antisymmetry here and there should not be** — what
carries over from Koplev is the low-rank bilinear interaction term, not the
algebra.

The result is reported in three registers, and they do not agree, which is the
most important thing about it.

**The frozen, pre-registered verdict is `INCONCLUSIVE`.** The rule was registered
before any authoritative fold was fitted, implemented literally, and evaluated on
the registered statistic. No cell clears every criterion and the two screens fail
different ones.

**The same rule with one statistic corrected returns `ANALOGUE-ONLY CHEMICAL
TRANSFER`**, and that is the reported reading. The single change: the per-entity
statistic's denominator becomes the fold's baseline MSE rather than the entity's
own, which is not bounded away from zero. A reactant that fails with every
partner is predicted correctly at ≈0 by both models, so a tiny absolute
worsening becomes an enormous negative ratio — the worst case scores −13.20 on
eight rows against a baseline 55× smaller than the fold's. The tell that this is
a denominator problem rather than a vanishing effect is the Wilcoxon statistic,
which is insensitive to that tail and is significant in **every** cell under
**both** statistics. Only the t-test moves.

The frozen verdict is reported and not believed. Both are published, the
registered one is labelled registered, and the correction names its single
change. That protocol was fixed in advance, and Phase 3 had already exercised it:
its registered rule also returned `INCONCLUSIVE`, for the same class of reason,
and its corrected reading is `PAIR-SPECIFIC ENTITY TRANSFER`.

**The project's first detectable both-unseen result.** On the single-condition
HATU screen, holding out both the acid and the amine:

| quantity | value |
|---|---|
| incremental pair skill, E2 (both reactants unseen) | **+0.0344** |
| 95% CI | [+0.0127, +0.0561] |
| paired t | **p = 0.0043** |
| folds favouring | 11 / 15 |
| same quantity with the pair blinded to the training marginal | −0.0039 |
| blind drop | +0.0383, CI [+0.0175, +0.0591], p = 0.0015 |

The blind diagnostic is the load-bearing one: the effect disappears when the pair
term is denied the unseen reactants' structures, which is what distinguishes
"the model learned something about these molecules" from "the model got lucky
with the marginals". Phase 3 could not power this regime at all — 45 pairs per
fold there against 476 rows here.

**And the analogue boundary is not where Phase 3's was.** On the pooled screen it
reproduces Phase 3 exactly: the effect lives in the high-similarity stratum
(+0.097 for acids, +0.077 for amines) and the low stratum is null. On the
single-condition screen it is *absent for acids* — the low-similarity stratum is
itself significant at +0.071 (family-bootstrap p = 0.011) and high-minus-low is
−0.0006 (p = 1.00). This survives leave-one-out and restriction to acids with
≥20 test rows. The honest statement is that **the analogue gradient appears when
reaction conditions are pooled and disappears when they are not.** The obvious
reading — that some of what looks like analogue dependence is condition
structure, because close analogues tend to have been run under the same
conditions — is a hypothesis the data suggests and does not establish. The two
screens are nested, so their disagreement is not two independent measurements.

---

## Important corrections

This section exists because the corrections are load-bearing. A record that
reported only the surviving claims would be a less trustworthy record.

### The flexible comparator never fitted, and the claim resting on it is withdrawn

An earlier version of this work claimed that *low rank is the useful inductive
bias rather than capacity*, on the strength of a flexible MLP comparator with
roughly twice the parameters that found nothing.

An adversarial review refitted the models by hand and found the flexible rung's
interaction term is **numerically zero at the selected fit** — a fitted standard
deviation of order 1e-19 to 1e-43 against 0.5–0.6 for the low-rank term on the
same folds. It never leaves its initialisation. Its reported incremental skill of
≈0.000 is what an *untrained* term scores, not what a flexible model found.

**So the claim is withdrawn. Nothing in this project supports "low rank is
superior to flexible capacity", and nothing in it should be read as supporting
it.** The proximate cause is diagnosable and the fix was not: the projections read
a 2,048-bit fingerprint with ~34 bits on, so 94% of the MLP's output at
initialisation is a constant that `μ` absorbs, leaving the zero-initialised
output layer a gradient of order 1e-3 along the only direction that matters.
Adding LayerNorm and removing hidden biases changed nothing measurable.

`scripts/measure_pair_terms.py` now measures the fitted interaction term of every
rung, so a dead term can no longer report itself as a finding of "no benefit".

### The detection floor was stated at roughly half its value, with its conclusion inverted

An earlier statement put the detection floor at "roughly a quarter of the
outcome's standard deviation, and the real effect is just above it". The
registered gate is met only at planted scale 1.0, which carries **48.8%** of the
target's standard deviation, and the observed effect sits **below** that, not
above. The positive control does detect, but only at the largest planted size;
at half that size the shuffled control arm scores numerically *higher* than the
real one.

The real effect is resolved through 15 folds and several hundred entities, not
through the planted-control gate. That is a weaker form of evidence than the
original wording implied, and the original wording was wrong about its direction.

### Three defects invalidated the results and forced a complete corrected re-run

All three had been asserted not to exist in a docstring written by the same
hand:

1. **The per-condition seed was Python's `hash()`**, which is salted per process.
   Three interpreters returned three different seeds for the same specification.
   No conclusion rested on a single initialisation, but the committed results
   could not be regenerated from the committed code — which is the whole point of
   committing them. Replaced with a `blake2b` digest of the condition key.
2. **The control representations were re-permuted on every fold.** The real arm
   saw one representation across 15 folds; the control arm saw five. That
   inflated the control's fold-to-fold variance and made "the control collapses"
   easier to satisfy than registered.
3. **Two acids were the same compound in different split groups** —
   Fmoc-Lys(Dde)-OH drawn as the imine and the enaminone, and valsartan with its
   tetrazole drawn 1H and 2H. Standard InChI does not equate either pair. Both
   landed test-versus-train in several authoritative folds. Fixed by adding a
   fourth grouping relation: the canonical tautomer of the already
   stereo-stripped structure.

### Two findings changed a verdict

The decision-rule classifier **never read two of the seven registered
criteria**. Mutation testing confirmed it: forcing either true or false in every
cell left the output unchanged. Two of seven registered criteria were decorative
while the document tabulated them. Separately, one validity gate's threshold
existed, drew a dashed line on a figure, and decided nothing. This is why the
superseded frozen verdict read `TRANSDUCTIVE-ONLY PAIR STRUCTURE` and the final
one reads `INCONCLUSIVE`. There are now seven mutation tests, one per criterion.

### The review that found these was stopped early

Eighteen findings were upheld out of the twenty-five adjudicated before the review
was halted to free the machine for the final run — from eighty raw findings, of
which forty-nine were claimed material. **Twenty-four material findings were never
adjudicated.** The corrected re-run is the better artifact, and it is not the
product of a completed review.

### One registered analysis was never run

The fifth registered Phase 4 sensitivity — incremental pair skill on rows with
`Conversion > 0` only — was registered and never implemented. It was registered
as outcome-conditioned and therefore as a labelled diagnostic that could never be
a headline, so its absence changes no verdict. That is a reason it was not
missed, not a reason it is fine.

---

## Current conclusion

Stated as carefully as the evidence permits:

**There is reusable pair-specific interaction structure in real scientific
systems, it is low-rank, and it transfers to combinations that were never
observed — provided enough of the interaction graph has been.** This is the
best-supported claim here. It holds on the Koplev screen for unseen pairs above
roughly 1,700 training pairs, it survives a falsification test against the
estimator that produced the target, and it reproduces in an independent wet-lab
chemical system with a directly measured endpoint.

**Extrapolation to genuinely novel entities is much weaker, and mostly looks like
interpolation between analogues.** On Koplev, transfer to an unseen drug is
detectable only when a close structural analogue is in training. On ChemLex the
same boundary appears on the pooled screen — and does not appear on the
single-condition screen, which is either a genuine limit on the analogue story or
evidence that part of it was condition structure. That question is open.

**Two entities unseen at once is detectable, once, in one system.** The HATU E2
result is the project's first, it is properly blinded, and it is a single result
in a single screen that is a subset of another screen. It is a beginning, not a
capability.

**And the architectural hypothesis the project is named after did not survive.**
The symmetric/antisymmetric decomposition helped on synthetic data at one
coverage, did not transfer to the Koplev screen, and does not even typecheck on
the bipartite chemistry problem. What survived the journey from Phase 1 to
Phase 4 is not the algebra. It is the much plainer observation that the
interaction term is low-rank and that low-rank things can be predicted from the
entities that compose them — sometimes.

For what this does **not** license, see [`LIMITATIONS.md`](LIMITATIONS.md).
