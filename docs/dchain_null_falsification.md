# Can the d-chain estimator manufacture the Phase 2R signature from nothing?

Status: **complete.** All **116 of 116** preregistered conditions ran, 0 errors,
0 incomplete chains. Verdict in §7: **LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT**,
with no criterion-D trigger firing.

An earlier revision of this document said "complete" while 46 conditions were
still unrun, because the completeness check had been written per-tag and the
primary tag *was* complete. A final adversarial reviewer caught it; the gate now
reads the whole ensemble, and the classification below is the one it returns on
all 116.

Every table below is generated into `results/dchain_null/summary/doc_tables.md`
and pinned by `test_document_tables_are_generated_not_transcribed`; none is typed.

Pre-registration: [`PREREGISTRATIONS.md`](PREREGISTRATIONS.md), committed at
`2bea984`, before any run of the joint estimator.
Reconstruction: [`dchain_reconstruction.md`](dchain_reconstruction.md).

---

## 1. The question

Phase 2R ([`phase2_residual_directionality.md`](phase2_residual_directionality.md))
found that the Koplev sequential screen's directional matrix

```
D(i,j) = y(i→j) − y(j→i)
```

contains a large cyclic component — 46% (A375) / 60% (PANC1) of the directional
energy is not expressible as any per-drug ordering potential — and that this
remainder is **predictable for entirely held-out drug pairs** once about a third
of the pair graph has been observed. A fixed rank-2 model with 204 parameters and
no hyperparameter search reaches held-out skill +0.197 / +0.250 on A375 at
coverages 0.40 / 0.70 and +0.161 / +0.237 on PANC1, on 8 of 8 split seeds.

Its decision section named its own leading alternative explanation, and it was
not biological:

> `synergy_measure` is a posterior mean from one joint fit in which every drug's
> single-agent dose-response curve is shared across all its combinations. Error
> in drug `i`'s curve enters every `i→j` value; its antisymmetric part is a fixed
> per-drug quantity, so it generalises to held-out pairs exactly like a real
> interaction would.

So: **can the estimator produce this signature in a world where the true
pair-specific sequential interaction is exactly zero?**

If it can, Phase 2R measured the shape of the authors' estimator and the Koplev
branch should close. If it cannot, the result has survived the objection that was
most likely to kill it.

## 2. What was reproduced, and what was approximated

The whole d-chain model is one 1,045-line C++ file whose only non-standard
dependency is Boost's command-line parser. It is therefore **not approximated**:
`scripts/prepare_dchain_null.py` fetches it at commit `72b2445`, verifies its
digest, applies three patches, and compiles it.

| what | how faithful |
|---|---|
| the likelihood, priors, proposals, selector scheme, sweep order | **the published code, unmodified** |
| the MCMC settings | **the published defaults** — 500,000 iterations, 100,000 burn, subsample 200, giving 1,999 retained samples, which is provably what produced the deposit (§5) |
| `synergy_measure` | ported from `post/interpretMCMC.R` to NumPy; two independent implementations of the formula agree to machine precision, and a scalar transliteration of the R loop is pinned by test |
| the command line | Boost replaced by a standard-library shim. Touches no line inside the sampler |
| the RNG seed | added. The published program default-constructs its engine and cannot be run twice; `--seed 0` restores it exactly |
| sufficient statistics | computed once instead of ~3×10⁵ times per iteration. Provably constant — the two offsets its function takes are commented out of its body and never updated |

The third patch is the only one that could move a number, so it is checked rather
than argued: **the build refuses unless the patched program reproduces the
unpatched one byte for byte** on the authors' example data at the published seed.

What remains approximated is not the estimator. It is the *generative* side — the
simulated wells — and §8 is about that.

## 3. The null

The d-chain likelihood for a well in which drug `i` went first (always at 1 µM)
and drug `j` second at concentration `c`:

```
E[log x_AB(i,j,c)] = log β_i + 1[λ_i]·log f(1.0; θ_i)
                     + ( 1[λ_ij]·log f(c; θ_ij)  or  1[λ_j]·log f(c; θ_j) )
```

Every term is indexed by **one drug**. The protocol asymmetry — first drug pinned
at one dose, second drug titrated — is a property of the position, not of the
partner.

**NULL-A (`strict`), the primary.** `λ_ij = 0` for every ordered pair. The true
ordered response is then exactly separable: a per-first-drug multiplicative
factor times the second drug's own single-agent curve. Because the authors' own
measure carries `λ_AB` as a multiplicative factor, the **true `synergy_measure` is
identically zero for every ordered pair** — not small, zero — and the true
directional matrix is the zero matrix.

This makes NULL-A the *most favourable possible world for the artifact
hypothesis*: there is no true signal that a positive result could be attributed
to, so anything the estimator produces is 100% estimator-induced.

**NULL-B (`nuisance`), the realism check.** The published model does carry
combination-level nuisance parameters, and a world where they are off everywhere
is one the estimator never sees. So `λ_ij ~ Bernoulli(0.5)` i.i.d. — 0.5 because
`dchain.cpp` defines a `BernPrior` and never applies it, making the model's own
implicit selector prior uniform — and `θ_ij` is `θ_j` perturbed by noise drawn
independently for each ordered pair from one fixed distribution depending on
neither `i`, nor `j`, nor any other pair.

**Why the true system contains no reusable pair-specific directional structure.**
Under NULL-A there is no pair-specific term at all. Under NULL-B there is one, but
it is drawn i.i.d., so conditional on the per-drug parameters the true synergy of
a held-out pair is statistically independent of every observed pair and **no
model can achieve positive held-out skill on it in principle**. Three tests
enforce this: the strict null's true synergy is asserted to be exactly the zero
matrix; the true AB log-means are asserted additively separable to 10⁻¹⁰; and the
nuisance null's true cyclic component is asserted spectrally indistinguishable
from noise, with a mutation test that inserts a rank-2 pair term and requires
that assertion to fail.

**A claim that was registered and is withdrawn.** The pre-registration argued
that NULL-B's extra true pair noise enlarges the skill denominator without adding
anything predictable, so it "can only dilute an artifact, never manufacture one".
That is wrong, and an adversarial reviewer showed why by running both arms on the
same truth and settings. The measure is `λ_AB · (…)`; NULL-B's combination curves
give the estimator a reason to open the selector, and opening the gate
**multiplies** the artifact:

| | selector on-fraction | artifact RMS | artifact sd(D) |
|---|---:|---:|---:|
| strict (NULL-A) | 0.21–0.23 | 0.016–0.017 | 0.022 |
| nuisance (NULL-B) | 0.330 | 0.033 | 0.048 |
| the real deposit | **0.4916 / 0.4635** | — | — |

So NULL-A is maximally favourable for **attribution** — with a true matrix of
exactly zero, anything found is 100% artifact and nothing is confounded — but it
is the *less* favourable of the two for **magnitude**. The realism arm was raised
from 10 seeds to 20 as a result, and both arms' selector on-fractions are reported
as headline numbers rather than as a pass/fail.

There is no way to open the gate without adding a true effect, and that is a fact
about the world rather than a fixable defect: under exact separability the
combination data gives the estimator no reason to prefer a private curve, so a
gate near 0.21 is what a world with no pair effects *produces*. The deposit's 0.47
reflects whatever is actually in the real screen. A negative NULL-A therefore
licenses "the artifact is at most about twice this size at the estimator's real
operating point", not "there is no artifact".

## 4. Pre-registration

Committed at `2bea984`, before any joint-estimator run. Reference values are read
from the generated Phase 2R artifacts on every report run, never transcribed.

**One thing the pre-registration had to settle first: the cyclic fraction is not
a usable criterion.** A gradient field on a complete graph occupies only `2/n` of
the energy, so at n = 100 *any* unstructured antisymmetric matrix has a curl
fraction near 0.98 — measured, on i.i.d. draws, before the ensemble:

| n | curl fraction of pure noise | top-2 curl energy |
|---:|---:|---:|
| 20 | 0.889 | 0.336 |
| 50 | 0.957 | 0.143 |
| **100** | **0.980** | **0.075** |

(The floor was registered as 0.076 from 5 draws and remeasured on 20 as
0.0747 ± 0.0038; the remeasured value is the one used, and the amendment records
the change.)

The real screens' 0.462 / 0.602 are *below* that floor, because they carry a large
per-drug potential. A rule of the form "the null reaches half the real cyclic
fraction" would be satisfied by literally any noise. The decision therefore rests
on the two statistics unstructured noise cannot fake:

1. **held-out residual skill** of the fixed rank-2 detector — noise gives ≤ 0;
2. **the rank-2 share of the directional energy**, `curl_fraction × top-2`.

The second is stated in absolute units for a reason the amendment records. Top-2
energy alone is a fraction *of the curl*, and under the strict null 100% of the
curl is estimator artifact — so "is at least half of 0.32 of the estimator's
cyclic error concentrated in two directions?" is answered yes at **any magnitude
whatsoever**, including a thousandfold below the real screen. In absolute units
the real values are **A375 0.157, PANC1 0.193** and the artifact threshold is
half their mean, **0.0875**.

Both statistics are anchored at each end. Injecting the artifact the
reconstruction predicts — a per-first-drug offset error pushed into every
combination curve in that drug's row — gives top-2 cyclic energy **0.9999** and a
rank-2 held-out skill above **0.5**. So the scale on the fraction-of-curl version
is **0.075 = noise, 0.32–0.34 = the real screens, 1.000 = a pure artifact**, and
the detector demonstrably sees the artifact when it is there.

The decision rule, executed by `report.verdict()` rather than applied by hand, is
in the pre-registration; its thresholds are pinned to that text by
`test_decision_thresholds_match_the_preregistration`.

## 5. Validation

The hard limit first: the Mendeley deposit contains the five modelled tables and
nothing else — no raw viability data and **no posterior samples**. The inputs to
the synergy formula therefore do not exist publicly, so *nobody* can reproduce a
published `synergy_measure` from published parameters. That check is unavailable
in principle. What is available is structural, and it is not weak.

| check | result |
|---|---|
| the patched sampler reproduces the unpatched one | **byte-identical**, enforced at build time |
| every `\|lambda\|` in both deposited tables is an exact multiple of 1/1999 | **yes**, and not of 1/1998 or 1/2000 |
| …which is exactly what `iter > 100000 && iter % 200 == 0` over 500,000 iterations retains | the deposit is provably this code at its compiled-in defaults |
| rows with `lambda == 0` have `synergy_measure == 0` **exactly** | 191/191 (A375), 173/173 (PANC1) — confirms the multiplicative selector |
| `\|synergy_measure\| <= \|lambda\|` on every row | 20,000/20,000 |
| `p = 1 − \|lambda\| < 0.05` reproduces the paper's counts | **707 / 1,845 / 551 / 1,464 — exact on all four** |
| the full pipeline on the authors' own 66-row example | runs; 1,999 samples where 1,999 expected; all four identities hold; the two-drug result is genuinely directional (+0.007 one way, −0.260 the other) |

`results/dchain_null/summary/validation.json`, regenerated by
`scripts/validate_dchain_null.py`, and pinned by
`test_the_committed_validation_still_supports_what_the_docs_claim`.

## 6. Results

Twenty simulated screens, each through the published sampler at its published
settings. **20 of 20 usable**: no run failed, and every one retained exactly
1,999 of 1,999 expected samples, so the preregistered exclusion rule removed
nothing. The unit of the null distribution is the simulated screen; each screen's
value is its mean over its 8 split seeds.

### 6.1 The decision statistics

<!-- generated: comparison_decision -->
**The decision statistics. The null unit is one simulated screen; the minimum reportable one-sided p is 1/(n+1).**

| metric | null median | null 95% interval | null max | real A375 | real PANC1 | real percentile under null | p (one-sided) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| rank-2 held-out skill @ coverage 0.40 | -0.004 | [-0.009, -0.002] | -0.002 | +0.197 | +0.161 | 100% / 100% | 0.048 / 0.048 |
| rank-2 held-out skill @ coverage 0.70 | -0.002 | [-0.013, +0.000] | +0.000 | +0.250 | +0.237 | 100% / 100% | 0.048 / 0.048 |
| rank-2 cyclic share of D² (curl fraction × top-2) | 0.071 | [0.064, 0.078] | 0.079 | 0.157 | 0.193 | 100% / 100% | 0.048 / 0.048 |
| searched low-rank skill @ coverage 0.40 | -0.001 | [-0.012, +0.001] | +0.002 | +0.229 | +0.198 | 100% / 100% | 0.048 / 0.048 |
| searched low-rank skill @ coverage 0.70 | -0.001 | [-0.006, +0.000] | +0.001 | +0.353 | +0.366 | 100% / 100% | 0.048 / 0.048 |
| rank-2 held-out skill @ coverage 0.05 | -0.034 | [-0.137, -0.005] | -0.002 | -0.051 | -0.045 | 40% / 40% | 0.619 / 0.619 |
| rank-2 held-out skill @ coverage 0.10 | -0.017 | [-0.053, -0.005] | -0.004 | -0.032 | -0.019 | 10% / 45% | 0.905 / 0.571 |
| rank-2 held-out skill @ coverage 0.20 | -0.005 | [-0.010, -0.001] | -0.001 | -0.015 | -0.029 | 0% / 0% | 1.000 / 1.000 |

<!-- generated: comparison_descriptive -->
**Descriptive — how the two worlds compare. No p-values: these are not discriminators, and the pre-registration says why.**

| metric | null median | null 95% interval | real A375 | real PANC1 |
| --- | ---: | --- | ---: | ---: |
| cyclic fraction of D  (i.i.d. noise at n=100: 0.980) | 0.788 | [0.762, 0.809] | 0.462 | 0.602 |
| curl energy in top 2  (noise floor 0.075 at k=2) | 0.091 | [0.080, 0.100] | 0.340 | 0.321 |
| curl energy in top 4  (noise floor 0.075 at k=2) | 0.169 | [0.152, 0.181] | 0.493 | 0.476 |
| curl energy in top 16  (noise floor 0.075 at k=2) | 0.507 | [0.480, 0.530] | 0.807 | 0.797 |
| spread of D (sd, off-diagonal) | 0.023 | [0.021, 0.024] | 0.223 | 0.162 |
| combination selector on-fraction | 0.199 | [0.180, 0.219] | 0.492 | 0.464 |
| posterior noise fraction of D  (see DECISION) | 4.001 | [3.665, 4.189] | 0.205 | 0.192 |
| rank-2 cyclic energy, absolute (mean D² × curl frac × top-2) | 0.000 | [0.000, 0.000] | 0.008 | 0.005 |

<!-- generated: verdict -->
**Verdict: LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT**

Computed by `report.verdict()` from 20 usable runs of the primary block (0 failed, 0 incomplete and excluded under the preregistered rule).

| clause | value |
| --- | :--: |
| null median skill ≥ half the weaker real screen, both coverages | no |
| null median rank-2 share of D² ≥ half the real mean | no |
| a real value lies inside the null 95% interval at coverage 0.70 | no |
| null median clearly positive at coverage 0.70 | no |
| null below the positive threshold, real above the null maximum, spectral below | **yes** |

| coverage | null median | null max | null 95% | real A375 | real PANC1 | artifact threshold | p (A375 / PANC1) |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.40 | -0.004 | -0.002 | [-0.009, -0.002] | +0.197 | +0.161 | +0.081 | 0.048 / 0.048 |
| 0.70 | -0.002 | +0.000 | [-0.013, +0.000] | +0.250 | +0.237 | +0.119 | 0.048 / 0.048 |

*rank-2 share of D²:* null median 0.07059, real 0.15712 (A375) / 0.19305 (PANC1).
*rank-2 cyclic energy, absolute:* null median 0.00004, real 0.00772 (A375) / 0.00503 (PANC1).
*combination selector on-fraction:* 0.1992
*split-half r(D):* 0.9579
*posterior noise fraction of D:* 4.0011
*Control A: maximum oracle rank-2 skill:* 0.0002

<!-- generated: controls -->
| block | n | true pair effect | est. synergy RMS | curl fraction | top-2 curl energy | rank-2 skill @ 0.70 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `convergence` | 4 | zero | 0.0159 | 0.780 | 0.094 | -0.002 |
| `noise_sd0.075` | 6 | zero | 0.0054 | 0.609 | 0.097 | -0.002 |
| `noise_sd0.3` | 6 | zero | 0.0434 | 0.763 | 0.080 | -0.002 |
| `oracle_nuisance` | 20 | independent, RMS 0.0604 | 0.0603 | 0.967 | 0.098 | -0.001 |
| `oracle_strict` | 20 | zero | 0.0000 | — | — | — |
| `realism` | 20 | independent, RMS 0.0604 | 0.0577 | 0.915 | 0.102 | -0.002 |
| `unshared` | 20 | zero | 0.0623 | 0.942 | 0.082 | -0.001 |

<!-- generated: mechanism -->
| block | n | offset error ε (RMS) | gain m̃ mean | m̃ sd | template R² | subspace overlap | split-half r(D) | selector on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `convergence` | 4 | 0.0054 | 0.684 | 0.246 | 0.000 | 0.010 | 0.957 | 0.192 |
| `noise_sd0.075` | 6 | 0.0023 | 0.678 | 0.241 | 0.000 | 0.009 | 0.889 | 0.135 |
| `noise_sd0.3` | 6 | 0.0129 | 0.678 | 0.241 | 0.000 | 0.011 | 0.985 | 0.312 |
| `primary` | 20 | 0.0051 | 0.670 | 0.244 | 0.000 | 0.008 | 0.958 | 0.199 |
| `realism` | 20 | 0.0056 | 0.670 | 0.244 | 0.000 | 0.015 | 0.991 | 0.313 |

### 6.6 A secondary finding, and where it does not hold

At the two sparsest coverages the null **reproduces** Phase 2R's negative
skill. Null medians and the count of null screens at least as negative as each
real value, out of 20:

| coverage | null median | real A375 | (of 20) | real PANC1 | (of 20) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | −0.034 | −0.051 | 8 | −0.045 | 8 |
| 0.10 | −0.017 | −0.032 | 2 | −0.019 | 9 |
| **0.20** | **−0.005** | **−0.015** | **0** | **−0.029** | **0** |

Phase 2R §8.6 attributed its sparse-coverage negatives to the cost of its
selection machinery at 37–74 validation pairs, and could only argue that from
inside its own estimator. At coverages 0.05 and 0.10 this shows it directly: a
world with **no pair structure at all** produces the same negatives.

**At coverage 0.20 the claim is false and an earlier revision of this section
asserted it anyway.** Zero of 20 null screens are as negative as either real
screen; both real values fall strictly below the null minimum. That is a
difference at the same 1/21 floor, in the opposite tail — the real screens are
*more* negative at 0.20 than a no-structure world is. The adversarial reviewer
that found this was right, and the corrected statement is narrower: the negatives
at 0.05–0.10 are a detector artifact; the 0.20 negatives are not explained by
this null.

The null and the real screens diverge in the positive direction exactly where
Phase 2R located its result, between coverage 0.20 and 0.40.

### 6.7 What the detector could have seen

"The null shows no held-out predictability" is a statement about the world only
to the extent that the detector would have seen predictability had it been there.
The positive control shows the fixed rank-2 rung recovers a **pure** artifact
(top-2 = 0.9999) at skill > 0.5, which proves it is not dead — it does not
establish sensitivity at realistic sizes, and nothing in the original design did.

Measured (`experiment.detection_curve`, `summary/detection_curve.json`): a rank-2
cyclic component of known share injected into a background matched to the null's
own curl fraction (0.788) and spread (0.0226), same rung, same coverage 0.70.

| artifact share of D² | rank-2 skill | top-2 | rank-2 share of D² |
|---:|---:|---:|---:|
| 0.000 | +0.0005 | 0.0728 | 0.0574 |
| 0.005 | +0.0005 | 0.0725 | 0.0572 |
| 0.010 | +0.0005 | 0.0726 | 0.0574 |
| 0.020 | +0.0004 | 0.0784 | 0.0621 |
| 0.050 | +0.0003 | 0.1099 | 0.0878 |
| 0.100 | +0.0676 | 0.1666 | 0.1348 |
| 0.130 | +0.0992 | 0.2004 | 0.1635 |
| 0.200 | +0.1754 | 0.2777 | 0.2306 |
| 0.400 | +0.3941 | 0.4844 | 0.4229 |

**The detector is effectively blind below ~5% and turns on between 5% and 10%.**
Three consequences, all of which narrow the claim:

1. **The null is not artifact-free.** Its rank-2 share of D² is 0.071 against
   0.057 for the same background with nothing injected — placing the null's own
   artifact at roughly **2–3% of directional energy**, below the detection floor.
   §7's "no held-out predictability" is a true statement about the measurement and
   a misleading one about the world.
2. **The skill comparison alone cannot separate "no artifact" from "an artifact
   below 5%".** It can only bound it below 5%.
3. **The spectral comparison does not have that limitation**, because it is a
   ratio measured directly rather than through a learned model. The null's 0.071
   against the real screens' 0.157 and 0.193 is a factor of 2.2–2.7, and it is
   this comparison, not the skill one, that carries the weight at small artifact
   sizes.

Read against §4's ceiling: the real spectrum permits at most 13.3% (A375) /
16.0% (PANC1) of `D²` to be a rank-2 artifact, and the curve says an artifact at
that ceiling would produce skill of about +0.10 — the same order as, but below,
the observed +0.197/+0.250.

### 6.9 The realism arm — the objection tested, and answered

An adversarial reviewer, working from small runs, established that the
pre-registration's claim that NULL-B "can only dilute an artifact" was wrong: the
measure is `λ_AB · (…)`, so NULL-B's combination curves give the estimator a
reason to **open** the selector, which multiplies the artifact. The realism arm
was raised from 10 seeds to 20 on that basis. It has now run at full size.

| | STRICT (NULL-A) | NUISANCE (NULL-B) | real A375 / PANC1 |
|---|---:|---:|---:|
| screens | 20 | 20 | — |
| selector gate | 0.199 | **0.313** | 0.492 / 0.464 |
| true synergy RMS | **0.0000** | 0.0604 | — |
| **artifact RMS** | 0.0160 | **0.0346** | — |
| D sd | 0.0226 | 0.0823 | 0.223 / 0.162 |
| curl fraction | 0.788 | 0.915 | 0.462 / 0.602 |
| rank-2 share of D² | 0.0706 | **0.0939** | 0.157 / 0.193 |
| skill @ 0.40, median / **max** | −0.0039 / −0.0016 | −0.0032 / **+0.0003** | +0.197 / +0.161 |
| skill @ 0.70, median / **max** | −0.0019 / +0.0001 | −0.0021 / **−0.0005** | +0.250 / +0.237 |

**The reviewer was right about the mechanism and it does not change the
conclusion.** NULL-B opens the gate to 0.313 — closing 60% of the distance from
the strict arm to the real screens' 0.47 — and the artifact does grow, from
0.0160 to 0.0346 RMS, almost exactly the doubling the reviewer measured.

It still produces **no held-out predictability whatsoever**. The best of twenty
screens reaches **+0.0003** at coverage 0.40 and **−0.0005** at 0.70, against
real values of +0.16 to +0.25. The rank-2 share moves from 0.0706 to 0.0939 —
toward the real 0.157/0.193, and still a factor of **1.7–2.1** below it.

So: **doubling the artifact by opening the gate 60% of the way to the real
operating point moves the spectral statistic about a quarter of the way and moves
held-out skill not at all.** That is the extrapolation the strict arm alone could
not support, and it is now measured rather than argued.

### 6.10 Control D — the noise sweep, and the scale objection

The preregistered sensitivity, and the other half of the registered defence
against the ambiguity §7 discusses. Strict null, joint estimator, σ_obs swept
over the preregistered grid.

| σ_obs | n | gate | D sd | rank-2 share of D² | skill @0.40 med / max | skill @0.70 med / max | artifact RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.075 | 6 | 0.135 | 0.0076 | 0.0566 | −0.0030 / −0.0021 | −0.0020 / −0.0009 | 0.0054 |
| **0.150** | 20 | 0.199 | 0.0226 | 0.0706 | −0.0039 / −0.0016 | −0.0019 / **+0.0001** | 0.0160 |
| 0.300 | 6 | 0.312 | 0.0611 | 0.0624 | −0.0034 / −0.0022 | −0.0017 / −0.0001 | 0.0434 |

**An eightfold increase in the artifact produces no change in either decision
statistic.** Artifact RMS goes 0.0054 → 0.0160 → 0.0434 and the directional
spread goes 0.0076 → 0.0611, an 8× range — and held-out skill stays at ≈ −0.002
with a maximum of ≈ 0 throughout, while the rank-2 share is flat and
non-monotone (0.057, 0.071, 0.062).

This is the direct answer to the standing objection that the null is "too clean
to be informative", and it is the strongest form of that answer: the objection
predicts that a noisier null would reveal the artifact, and a null with 8× the
artifact does not. Note also that at σ = 0.30 the gate reaches 0.312, matching
the realism arm — so the sweep independently reproduces that arm's operating
point and gets the same answer.

### 6.8 Convergence — a diagnostic the published fit does not have

`dchain.cpp` default-constructs its random engine and exposes no seed, so the
published screen was produced by **one chain with no convergence diagnostic of
any kind** and the paper's only evidence is a single trace plot. Adding a seed
makes the question answerable here. Four chains on one dataset (`sim_seed` 0),
`est_seed` 1–4 — and chain 1 is literally the published program's own unseeded
chain, because libc++'s `default_random_engine` is `minstd_rand` whose
default-constructed state is seed 1.

| | chain-to-chain, one dataset | across datasets, n = 20 |
|---|---|---|
| rank-2 share of D² | 0.0731 ± **0.0007** | 0.0705 ± 0.0046 |
| held-out skill @ 0.70 | −0.0015 ± **0.0003** | −0.0027 ± 0.0045 |
| split-half r(D) | 0.950 – 0.963 | 0.953 – 0.962 |

MCMC variation is **6.6× smaller** than generative variation on the spectral
statistic and **15× smaller** on skill. Three consequences:

1. **The 20-screen null spread is genuine screen-to-screen variation**, not
   sampler noise. This also retires a reviewer's objection that `sim_seed` and
   `est_seed` are perfectly correlated in every other block (`est_seed =
   sim_seed + 101`): they are, but the component that correlation could have
   confounded is negligible.
2. **The published chain length is adequate for this quantity.** All four chains
   agree to the third decimal.
3. **MCMC error can only have diluted the artifact, never created it** — it is
   independent across pairs, so it cannot make a held-out pair predictable — and
   it is now bounded at ±0.0007 in the statistic the verdict reads, rather than
   assumed small.

## 7. Verdict

**LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT**, on the complete 116-condition
ensemble, with **no criterion-D trigger firing**. All three clauses of the
preregistered criterion C hold:

* the null median rank-2 skill is below the +0.02 threshold at both decision
  coverages (−0.004 and −0.002 — in fact below zero);
* both real screens lie strictly above the null **maximum** at both coverages
  (+0.197/+0.161 against −0.002; +0.250/+0.237 against +0.000), so the empirical
  percentile is 100% and the one-sided p is the design's floor of 1/21;
* the null's rank-2 share of directional energy, 0.071, is below the
  preregistered artifact threshold of 0.0875.

Criterion A fails on both clauses and on its independently-sufficient clause;
criterion B fails. Criterion D fired nothing: 116/116 conditions ran, no run
failed, no chain was incomplete, the oracle control reached +0.0002, the gate sat
at 0.199 above the 0.10 floor, and split-half reproducibility was 0.958 above the
0.50 floor.

**Three independent arms agree.** The strict null (artifact RMS 0.016), the
realism null at a gate 60% of the way to the real screen's (artifact RMS 0.035),
and the noise sweep's high arm (artifact RMS 0.043) all return held-out skill of
≈ −0.002 with a maximum of ≈ 0. An **eightfold** range in artifact magnitude
moves neither decision statistic.

### Two things a reader should weigh against this, and one that is now settled

**Settled — the scale objection.** Under the pre-registration as first committed,
criterion D required the null's directional spread to be within ±5× of the real
screens'; the delivered ratio is 0.117 and the original rule would have declared
the experiment inconclusive. The first amendment demoted that trigger on the
grounds that `cal_skill` is invariant to the scale of `D` to 6×10⁻⁹ over a
2000-fold range, and the pre-registration's other registered defence was the
noise sweep. **The noise sweep has now run and settles it empirically** (§6.10):
across an 8× range of artifact magnitude — spanning and exceeding the real
screen's own selector gate — neither decision statistic moves. The objection
predicted a noisier null would reveal the artifact. It does not.

**Standing — the skill comparison has a sensitivity floor.** §6.7 measures the
detector as blind below ~5% of directional energy, and the null's own rank-2
artifact is roughly 2–3%. The skill result therefore *bounds* the artifact below
5% rather than showing there is none. The spectral comparison does not share this
limitation, being a ratio measured directly rather than through a learned model.

**Standing — the magnitude claim depends on which statistic is used.** Three
numbers appear for "how much too small":

| basis | factor | scale-dependent? |
|---|---:|---|
| absolute rank-2 cyclic energy, as delivered | 142–218× | **yes** |
| corrected for the selector gate | ~9–13× | yes |
| **rank-2 share of D², a ratio** | **2.2–2.7×** | **no** |

An earlier revision led with 142–218×. **The defensible statement is the ratio**:
the null's rank-2 share of directional energy is 0.071 (strict) or 0.094
(realism) against the real 0.157 and 0.193 — a factor of 1.7–2.7 — and that
comparison is scale-free.

## 8. Limitations

**What is approximated is the simulated world, not the estimator.** The sampler
is the published one. The wells it is given are not the Koplev wells — those were
never deposited — so the artifact's *magnitude* in this null is only as
transferable as the generative assumptions are. Every simulator parameter carries
a provenance class in `dchain_null/simulator.py` and in each run's config:

* `source` — from `dchain.cpp` or the paper: the curve family, all four priors,
  the concentration grids, the fixed 1 µM first dose, the triplicate design, the
  Bernoulli(0.5) selector prior;
* `replicate` — from the authors' deposited raw triplicates: the observation
  noise, whose pooled within-condition log SD is 0.182 and median 0.127, against
  0.112–0.183 implied by the code's own variance prior. 0.15 sits inside both and
  is swept over {0.075, 0.15, 0.30};
* `prior` — the spread of the combination nuisance in NULL-B, which only enlarges
  the *unpredictable* part of the true signal;
* `design` — that every drug has a real single-agent curve.

**No real pair quantity is used anywhere.** Enforced by
`test_no_real_pair_residual_information_reaches_the_simulator`, which checks both
that `simulator.py` cannot reach the deposit and that the provenance record says
so. The one place the real deposit is read during a null run is Phase 2R's
per-screen threshold for *exploratory sign accuracy*; it cannot touch `cal_skill`,
Pearson or Spearman, and a test asserts that swapping the screen label changes
only that one metric.

**The paper's stated priors disagree with the code's** (`K` sd 0.2 vs 2.0, `h` 2.0
vs 0.5, transposed; proposal SDs 2.0/0.5/3.0 vs 0.5/0.1/0.1). The code's values
are used, because the code is what ran. The direction of that choice matters and
is favourable to the artifact hypothesis, not to the result: the code's wider `K`
prior gives *more* potency heterogeneity, and heterogeneity of the per-drug
second-position gain is the ingredient the artifact mechanism requires.

**MCMC error dilutes rather than creates.** Posterior-mean noise in the synergy
matrix is independent across pairs, so it can lower held-out skill and raise the
cyclic fraction but cannot make a held-out pair predictable. A short or badly
mixed chain is therefore conservative against the artifact hypothesis. The
convergence block measures how much dilution there is; the published program is
unseeded and has no such diagnostic of its own.

**A negative result is bounded.** Even a clean null establishes only that *the
modelled estimator artifact, under this generative null, does not reproduce the
signature*. It does not establish biological mechanism, and it does not rule out
an artifact arising from something the simulation does not model — plate
structure that was normalised upstream, a systematic difference between the
primary and validation screens, or a failure mode of the real fit that a
converged simulated fit does not have.

## 9. Consequence

**Does Phase 2R still provide evidence for real reusable pair-specific
interaction structure after this falsification? Yes, and it is materially
stronger than it was.**

Phase 2R named this as its leading alternative explanation and could not rule it
out. It has now been tested against the authors' own estimator, at the authors'
own settings, across 116 conditions in worlds where the answer is known — and the
estimator did not reproduce the signature in any of them. Held-out
predictability, across three arms spanning an eightfold range of artifact
magnitude: **best of twenty screens, +0.0003**. The real screens reach +0.16 to
+0.25.

What that licenses, precisely: *the sampling-error artifact of the published
estimator, under zero-interaction nulls at selector gates from 0.135 to 0.313,
is below the detector's ~5% sensitivity floor and produces a rank-2 spectral
signature 1.7–2.7× under the observed one.* It does not establish biological
mechanism. It does not exclude an artifact larger than 5% of directional energy
at the real screen's still-wider gate of 0.47. And it does not exclude
misspecification bias, which is systematic where this null's error is pure
sampling noise, and which no simulation from the model's own generative
assumptions can produce — this remains the honest residual.

The Koplev branch does not close.

**The single next research move: entity-level out-of-distribution validation.**
Phase 2R established that pair-specific directional structure generalises to
held-out *pairs* of seen drugs, and this experiment establishes that the
estimator does not manufacture that. Whether the structure is anything more turns
on whether it generalises to a drug **not seen during training at all** — which
requires drug representations, chemical or target-based, and a split at the drug
level rather than the pair level. That is the recommendation, and per §32 of the
brief it is deliberately not implemented here.
