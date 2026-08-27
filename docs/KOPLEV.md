# The Koplev sequential drug study

Four experiments on one dataset, in the order they were run. The dataset is the
same throughout; what changes is what is held out, and each change makes the
question harder.

## The data

Koplev et al. 2017, *Dynamic Rearrangement of Cell States Detected by Systematic
Screening of Sequential Anticancer Treatments*, Cell Reports 20(12):2784–2791,
doi:10.1016/j.celrep.2017.08.095. Data deposit: Mendeley
doi:10.17632/wgybvcvjwf.1, CC BY 4.0. Fetched, not vendored — see
[`../THIRD_PARTY_DATA.md`](../THIRD_PARTY_DATA.md).

| | |
|---|---|
| drugs | 100 (Approved Oncology Drug Set IV) |
| screens | two, A375 melanoma and PANC1 pancreatic |
| ordered rows per screen | 9,900 (10,000 minus 100 self-combinations) |
| unordered pairs | 4,950, **all** measured in both directions |
| endpoint | posterior mean of an area-based synergy index |
| antisymmetric share of variance | 38.2% (A375) / 39.1% (PANC1) |

Full ingestion and audit: [`phase2_dataset.md`](phase2_dataset.md).

**The endpoint is estimator-derived, and this is the single most important fact
about it.** The synergy values are not measurements. They are posterior means
from one joint Bayesian fit — the d-chain model, roughly 45,000 parameters
estimated over the whole screen at once. A held-out pair shares fitted
single-agent curves with the training pairs, so it is **not statistically
independent of them**. Everything in Phases 2, 2R and 3 inherits that caveat, and
Phase 2N exists to test whether it is fatal.

## Phase 2 — the original architecture, and its failure

The question: does imposing an explicit symmetric/antisymmetric decomposition on
the interaction term beat a capacity-matched unrestricted model on real data?

It does not.

| screen | coverage | directional Pearson, structured − unrestricted | p | seeds favouring |
|---|---:|---:|---:|---:|
| A375 | 0.05 | +0.0022 | 0.74 | 5/8 |
| A375 | 0.10 | +0.0026 | 0.63 | 4/8 |
| A375 | 0.70 | **−0.012** | 8e-5 | 0/8 |
| PANC1 | 0.70 | **−0.015** | 4e-4 | 0/8 |

Nothing at sparse coverage, and a significant deficit when data is plentiful,
losing on every seed. The capacity match is real: the structured model carries
32,546 pair parameters against the unrestricted model's 32,423 — it is the
*larger* of the two by 0.4%, so this is not a capacity story.

These numbers are the **re-tuned** ones. An audit found the original comparison
had given the two families unequal tuning, and the corrected contrast is what is
reported. A direction-shuffle control confirms the models are reading real
directional signal and not an artefact: destroying schedule direction in training
collapses directional Pearson from 0.68 to 0.011.

Full analysis: [`phase2_koplev.md`](phase2_koplev.md).

## Phase 2R — what survived

The negative said the architecture was wrong, not that there was no structure.
So: strip out the part of the directional effect that any per-drug ranking could
explain, and ask whether what remains is predictable for an unseen pair.

`D(i, j) = y(i→j) − y(j→i)` decomposes exactly into a gradient part and a cyclic
part, orthogonal to machine precision (measured inner product ≈ 5e-19):

| screen | mean D² | gradient (ordering potential) | cyclic (pair-specific) |
|---|---:|---:|---:|
| A375 | 0.0491 | 53.8% | 46.2% |
| PANC1 | 0.0261 | 39.8% | 60.2% |

The cyclic part is the interesting one. Predicting it for a pair observed in
neither direction, using a low-rank antisymmetric model with the additive
baseline fitted on training pairs only:

| screen | coverage | training pairs | skill | 95% CI | seeds > 0 | Pearson | sign acc. |
|---|---:|---:|---:|---|---:|---:|---:|
| A375 | 0.05 | 211 | −0.012 | [−0.027, +0.003] | 0/8 | 0.005 | 0.479 |
| A375 | 0.10 | 421 | −0.009 | [−0.038, +0.020] | 1/8 | 0.061 | 0.544 |
| A375 | 0.20 | 842 | +0.022 | [−0.011, +0.055] | 4/8 | 0.176 | 0.598 |
| A375 | 0.40 | 1,683 | **+0.229** | [+0.189, +0.269] | 8/8 | 0.479 | 0.787 |
| A375 | 0.70 | 2,945 | **+0.353** | [+0.315, +0.391] | 8/8 | 0.594 | 0.852 |
| PANC1 | 0.40 | 1,683 | **+0.198** | [+0.147, +0.248] | 8/8 | 0.456 | 0.733 |
| PANC1 | 0.70 | 2,945 | **+0.366** | [+0.323, +0.410] | 8/8 | 0.608 | 0.845 |

**There is a threshold and it is sharp.** Below roughly 1,700 training pairs the
model has nothing; above it, skill is large and every seed agrees. A permutation
control sits at −0.004 / −0.005 where the real model reaches +0.23 and +0.37.

The structure is genuinely low-rank: a rank-2 model with **204 parameters and no
hyperparameter search at all** reaches +0.197 (A375) and +0.161 (PANC1) at
coverage 0.40. The `potential` rung — which can express `c_i − c_j` exactly —
never exceeds |0.009| anywhere, so beating it is evidence of pair-specific
structure rather than of a better ranking.

Full analysis: [`phase2_residual_directionality.md`](phase2_residual_directionality.md).
Pre-registered decision rule: [`PREREGISTRATIONS.md`](PREREGISTRATIONS.md).

**The independence caveat.** The eight split seeds are repeated random pair-level
splits, not k-fold CV, and their evaluation pools overlap at mean pairwise
Jaccard 0.105. The unit of inference is the split seed, n = 8, which is why the
Wilcoxon p-values bottom out at 0.0078 — that is the floor for n = 8, not a
strong result.

## Phase 2N — is the signature an artifact of the estimator?

The objection that has to be answered: a shared-parameter estimator could
manufacture a low-rank cyclic signature out of nothing. Error in drug `a`'s
shared single-agent curve is pushed into the combination parameters for every
partner `b`; the product of a per-first-drug error and a per-second-drug
sensitivity is a rank-1 bilinear term whose antisymmetric part looks exactly like
what Phase 2R measures.

The only honest test runs **the published sampler itself**, not a model of it.
`scripts/prepare_dchain_null.py` fetches the authors' C++ at pinned commit
`72b2445`, verifies every file digest, applies the patches needed to build it,
and **refuses the build unless the patched program reproduces the unpatched one
byte for byte** on the authors' own example data at the published seed. The
synergy scoring step is ported line by line from the authors' R and is pinned to
an algebraically independent second implementation in the tests.

Then: simulate screens where the true pair-specific sequential interaction is
**exactly zero**, run the real estimator over them, and look for the signature.

| | null screens | real data |
|---|---:|---:|
| best held-out rank-2 skill | **+0.0003** | +0.16 to +0.25 |

Every decision statistic hit the design's one-sided p floor of 0.048 (1/21). The
registered rule returned **LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT** on a complete
116/116 ensemble.

**Scope.** This falsifies the *tested* artifact — sampling error in a correctly
specified d-chain fit — under the *tested* nulls. It does not falsify every
possible misspecification, and the comparison only bounds any artifact below the
detector's sensitivity floor of roughly 5%. The document states its own limits at
length.

Cost: the sampler runs at its compiled-in defaults, roughly **100 minutes per
simulated screen on one core**; the ensemble is on the order of ten hours. CI
runs the null's logic and never its MCMC.

Full analysis: [`dchain_null_falsification.md`](dchain_null_falsification.md).
Source reconstruction: [`dchain_reconstruction.md`](dchain_reconstruction.md).

## Phase 3 — an unseen drug

Hold out **drugs**, not pairs. A test drug appears nowhere: not in training rows,
not in any fitted quantity, not in hyperparameter selection. Leakage guards fail
the run if it does.

100 drugs mapped to structures through PubChem and to mechanisms through ChEMBL,
100/100 resolved, ECFP4 fingerprints, 30 entity-disjoint folds. The mapping
carries a SHA-256 provenance digest over its cached API responses and its audit
found three cases needing manual override — carboplatin most consequentially,
where rejecting PubChem's parent relation had otherwise admitted an unrelated
small diacid as the fingerprint. Full mapping audit:
[`phase3_drug_mapping.md`](phase3_drug_mapping.md).

**The result.** There is a real pair-specific signal and it requires the held-out
drug's own fingerprint:

| screen | model | zero skill | incremental pair skill | Pearson | sign acc. |
|---|---|---:|---:|---:|---:|
| A375 | `potential` | +0.1931 | — | 0.464 | 0.734 |
| A375 | `lowrank` | +0.2169 | **+0.0273** | 0.491 | 0.753 |
| A375 | `antisym_mlp` | +0.2302 | **+0.0431** | 0.493 | 0.746 |
| PANC1 | `potential` | +0.1414 | — | 0.400 | 0.713 |
| PANC1 | `lowrank` | +0.1855 | **+0.0536** | 0.448 | 0.727 |
| PANC1 | `antisym_mlp` | +0.1779 | +0.0441 | 0.439 | 0.724 |

**And it is analogue interpolation.** Stratifying held-out drugs by maximum
Tanimoto similarity to any training drug:

| screen | stratum | drugs | incremental skill | 95% CI | p |
|---|---|---:|---:|---|---:|
| A375 | low | 40 | +0.016 | [−0.005, +0.037] | 0.13 |
| A375 | medium | 29 | −0.013 | [−0.064, +0.037] | 0.59 |
| A375 | high | 31 | **+0.090** | [+0.035, +0.146] | 0.0024 |
| PANC1 | low | 40 | +0.015 | [−0.009, +0.038] | 0.22 |
| PANC1 | medium | 29 | −0.001 | [−0.034, +0.033] | 0.98 |
| PANC1 | high | 31 | **+0.138** | [+0.070, +0.207] | 0.00029 |

High-minus-low: +0.074 (p = 0.0089) and +0.124 (p = 0.0011). Per-drug skill
against similarity: Spearman +0.232 (p = 0.02) and +0.316 (p = 0.0013).

**For a drug with no close training analogue, transfer is not detectable.**

**Both unseen (E2) was not achievable here.** Every E2 cell is negative and none
is significant; the design would have needed an effect of +0.11 to +0.14 to
detect one, with only 45 pairs per fold in that regime. Phase 3 is a **one**-unseen
result.

**The verdicts.** The registered rule returned **INCONCLUSIVE** — two validity
gates fired because the random and shuffled representations posted mean E1 skill
of +0.204 and +0.118 against a threshold of 0.05. That gate measured
*skill-against-zero*, which a control can achieve by fitting the additive
structure alone; measured as *incremental* pair skill the same controls sit at
−0.0007 and −0.0016. The same rule with that one statistic corrected returns
**PAIR-SPECIFIC ENTITY TRANSFER**. Both are published, and the frozen one is the
registered one.

Full analysis: [`phase3_entity_ood.md`](phase3_entity_ood.md).

## What the drug study establishes

- Pair-specific directional structure exists, is roughly half the directional
  signal, and is low-rank.
- It is predictable for entirely unseen pairs, above a coverage threshold of
  roughly 1,700 training pairs, and not below it.
- It is not manufactured by the estimator that produced the target — under the
  nulls that were tested.
- It transfers to an unseen drug only when a close structural analogue is in
  training.
- It does not transfer to two unseen drugs at once in this dataset, and the
  design could not have detected it if it did.

For the external test of these claims, see [`CHEMLEX.md`](CHEMLEX.md).
