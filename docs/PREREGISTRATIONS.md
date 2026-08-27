# Pre-registrations

Four decision rules, each written down and committed to version control **before**
the experiment it decides was run, reproduced here verbatim from the private
research notebook this repository was cut from.

They are published because a frozen verdict is only worth something if the rule
that produced it can be read. Two of the four returned `INCONCLUSIVE`, and in
both cases a single corrected statistic changes the answer. That is exactly the
situation in which a reader needs the original text rather than a summary of it:
without the registered rule in front of you, there is no way to check whether the
correction is a repair or a rationalisation.

Nothing below has been edited for the public release. The wording, the
thresholds, the arithmetic and the hedges are as they were committed. Where a
rule was amended before its experiment ran, the amendment appears in place with
the reason it was made, and the original text it amends is still there.

| Rule | Registered at | Date | Verdict the rule returned |
|---|---|---|---|
| Phase 2R — residual directionality | `995a166` | 2026-08-18 | reported in full; the low-rank contrast is coverage-dependent |
| Phase 2N — d-chain estimator null | `2bea984` | 2026-08-20 | **LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT** |
| Phase 3 — entity-OOD on Koplev | `deadb46` | 2026-08-25 | **INCONCLUSIVE** (post-hoc: PAIR-SPECIFIC ENTITY TRANSFER) |
| Phase 4 — ChemLex external validation | `a1f396f` | 2026-08-26 | **INCONCLUSIVE** (post-hoc: ANALOGUE-ONLY CHEMICAL TRANSFER) |

The commit hashes are the private notebook's, and are recorded so that the
ordering claim — registered before run — is checkable by whoever holds it. They
do not resolve in this repository, which has its own history.

---

# Phase 2R — residual directionality on the Koplev screen

Registered at `995a166`, 2026-08-18. Registered before the main grid ran. The rule was executed as written; the result is reported in [`phase2_residual_directionality.md`](phase2_residual_directionality.md).

## Pre-registered decision rule — written and committed before the main grid ran

**Question.** After removing the per-drug ordering potential `g_i − g_j` with a
model fitted on training pairs only, is the residual directional effect
`D_res(i,j) = [y(i→j) − y(j→i)] − (g_i − g_j)` predictable for an entirely
unseen unordered pair?

**Design.** Same screens, same coverage grid `(0.05, 0.10, 0.20, 0.40, 0.70)`,
same 8 split seeds, same fixed evaluation pool as Phase 2. Additive baseline =
closed-form ridge on training rows only, penalty chosen on validation ordered
MSE. Five rungs: `zero`, `potential` (`c_i − c_j`), `lowrank` (`u_i^T K u_j`,
`K = −K^T`), `mlp` (antisymmetrised MLP on `D_res`), `mlp_ordered` (Phase 2's
unrestricted model pointed at the ordered residual). Hyperparameters selected
inside every run on that run's own validation pairs — there is no separate
tuning stage, because a separate tuning stage is what produced the Phase 2
fairness failure.

**Primary metric.** `cal_skill = 1 − MSE(α·D̂_res) / mean(D_res²)` on the
held-out pairs, where `α ∈ [0,1]` is a shrinkage coefficient chosen on
validation. Uncalibrated `heldout_skill` is reported alongside for every run and is
what distinguishes outcome C from outcome D.

**Primary rung.** `lowrank`. It is the capacity-controlled pair-specific
hypothesis. `mlp` and `mlp_ordered` are flexible diagnostics — upper bounds on
what is findable, not fair comparators.

**Primary contrast.** `lowrank − potential`, paired by split seed. This is the
one that matters and it is not the obvious one. `lowrank` *contains* `c_i − c_j`
as a special case (proved in `test_lowrank_can_express_a_pure_potential`), and
the residual necessarily retains whatever fraction of the potential the ridge
penalty over-shrank. A `lowrank` win over `zero` is therefore **not** evidence
of pair-specific structure on its own; only a win over `potential` is.

**Unit of replication.** The split seed. n = 8. Both a paired t-test and a
Wilcoxon signed-rank test are reported for every contrast, always both, decided
now rather than after seeing which is kinder. At n = 8 the Wilcoxon's smallest
attainable two-sided p is 0.0078.

### Evidence FOR learnable residual pair-specific structure

All five must hold, on at least one screen:

1. mean `cal_skill(lowrank)` > 0 with the 95% CI excluding 0;
2. paired `lowrank − potential` > 0 with `p_ttest < 0.05` **and**
   `p_wilcoxon < 0.05`, and ≥ 7 of 8 split seeds favouring `lowrank`;
3. mean held-out Pearson r(`D̂_res`, `D_res`) > 0.15;
4. replicated at two adjacent coverages on the same screen, **or** at the same
   coverage on both screens;
5. the permutation control at the same cell has mean `cal_skill` ≤ 0.02 with a
   CI containing 0.

### Weak / marginal

Criteria 1 and 3 hold somewhere but 2, 4 or 5 fails; or the effect exists only
at the single densest coverage.

### No residual learnability

No cell satisfies criterion 1 — **and** the positive control recovers mean
`cal_skill` ≥ 0.10 at those same cells for κ = 0.10, so the null is powered. If
the positive control fails to recover an injected signal at a coverage, a null
at that coverage is reported as *underpowered*, not as *no structure*. This
qualifier is registered now precisely so it cannot be added later to soften a
result.

### Evidence against practical learnability

Uncalibrated `heldout_skill` for the flexible rungs is negative with a CI excluding
0 across most cells.

### Registered in advance: the answer is allowed to be coverage-dependent

If the sparse regime (0.05, 0.10) and the dense regime (0.40, 0.70) disagree,
the headline is the coverage-resolved statement and not a single verdict. Phase
2 already showed this screen's pair structure "is learnable — it just needs a
lot of pairs", and it would be dishonest to force one label onto a curve.

### Disclosure: what had already been seen when this rule was written

Not blind, and pretending otherwise would be worse than saying so. While
building and debugging the pipeline, these **single-split-seed, A375-only**
pilot numbers were observed (they are what caught a dead-model bug in the
`lowrank` rung, where a doubly zero-initialised parameterisation had every
gradient vanish and returned exactly 0.0 skill — i.e. the null result — by
construction):

| rung | coverage | pilot `cal_skill` (seed 0) | pilot r |
|---|---:|---:|---:|
| potential | 0.10 | −0.002 | −0.05 |
| potential | 0.70 | +0.016 | +0.13 |
| lowrank | 0.10 | −0.024 | +0.20 |
| lowrank | 0.40 | +0.240 | +0.49 |
| lowrank | 0.70 | +0.339 | +0.58 |
| mlp | 0.10 | +0.000 | +0.02 |
| mlp | 0.40 | +0.210 | +0.46 |
| mlp_ordered | 0.10 | −0.093 | +0.11 |
| mlp_ordered | 0.40 | +0.159 | +0.40 |

So the thresholds above were **not** chosen blind at the dense end: it was
already visible that `lowrank` scores well above zero at coverages 0.40 and
0.70. What was *not* observed at registration time, and what criteria 2, 4 and 5
turn on, is the whole of: the `lowrank − potential` contrast, any across-seed
statistic or CI, the permutation control, and the power control. The criteria
that decide the outcome are therefore unobserved even though one input to them
is not.

Also registered now: a null at a coverage will **not** be reported as absence of
structure unless the positive control shows the same cell would have detected an
injected signal of comparable size.

---

# Phase 2N — the d-chain estimator-artifact null

Registered at `2bea984`, 2026-08-20. Registered after the source reconstruction and before any run of the joint estimator on 100 drugs. The rule returned **LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT** on a 116/116 ensemble; see [`dchain_null_falsification.md`](dchain_null_falsification.md). One amendment was made after independent statistical review and before the ensemble ran; it is included below, in place, with its reasons.

## Pre-registration — the d-chain null, written and committed before the main ensemble ran

Written after the source reconstruction (`docs/dchain_reconstruction.md`) and
after the simulator, adapter and control estimator were implemented and smoke
tested, and **before any run of the joint estimator on 100 drugs**. The commit
that carries this section carries no joint-null result. Nothing below may be
changed after the ensemble runs; if something has to change, the change is
recorded as a new entry with the reason, and the original stays.

### The question

> Can the Koplev/d-chain estimation procedure itself generate the observed
> low-rank cyclic directional structure in a simulated world where the true
> pair-specific sequential interaction is exactly zero?

### The null world, formally

The d-chain likelihood gives the expected log relative cell count of a well in
which drug `i` was applied first (always at 1 µM) and drug `j` second at
concentration `c`:

```
E[log x_AB(i,j,c)] = log β_i + 1[λ_i]·log f(1.0; θ_i)
                     + ( 1[λ_ij]·log f(c; θ_ij)   if the combination has its own
                         parameters,  else  1[λ_j]·log f(c; θ_j) )
E[log x_A(j,c)]    = 1[λ_j]·log f(c; θ_j)
E[log x_A0(i,c)]   = log β_i + 1[λ_i]·log f(c; θ_i)
f(c; K,h,α)        = (1-α)/(1+(Kc)^h) + α
```

Every term is indexed by **one drug**. The protocol asymmetry — first drug pinned
at one dose, second drug titrated — is a property of the position, not of the
partner.

**NULL-A (`strict`) — the primary.** `λ_ij = 0` for every ordered pair. The true
ordered response is then exactly separable: a per-first-drug multiplicative
factor times the second drug's own single-agent curve. Because the authors' own
measure carries `λ_AB` as a multiplicative factor, the **true `synergy_measure`
is identically zero for every ordered pair** — not small, zero — and the true
directional matrix is the zero matrix.

This makes NULL-A the **most favourable possible world for the artifact
hypothesis**: there is no true signal that a positive result could be attributed
to, so anything the estimator produces is 100% estimator-induced.

**NULL-B (`nuisance`) — the realism check.** The published model does carry
combination-level nuisance parameters, and a world where they are off everywhere
is one the estimator never sees. So: `λ_ij ~ Bernoulli(0.5)` i.i.d. over ordered
pairs (0.5 because `dchain.cpp` defines `BernPrior` and never applies it, making
the model's implicit selector prior uniform), and `θ_ij` = `θ_j` perturbed by
noise drawn independently for each ordered pair from one fixed distribution that
depends on neither `i`, nor `j`, nor any other pair.

**Forbidden, and asserted in tests:** any true term depending jointly on `{i,j}`
beyond independent noise — no `h_ij`, no `u_i' K u_j`, no coupled pair selector,
nothing whose parameters were chosen by looking at the real screen's residual
directional matrix. Under both variants the true synergy of a held-out pair is,
conditional on the per-drug parameters, statistically independent of every
observed pair, so **no model can achieve positive held-out skill on it in
principle**.

Because Phase 2R's primary metric `skill = 1 − MSE_model/MSE_zero` is a *ratio*,
NULL-B's extra true pair noise enlarges the denominator without adding anything
predictable. NULL-B can only **dilute** an artifact relative to NULL-A, never
manufacture one.

### The estimator

The published sampler, compiled and run: `dchain.cpp` at commit `72b2445`, at its
**own compiled-in defaults** — 500,000 iterations, 100,000 burn, subsample 200,
init phase 20,000, giving 1,999 retained samples. Those settings are not a
convenience: every `|lambda|` in the deposited tables is an exact multiple of
1/1999, which pins the deposit to exactly this configuration. Patched only to
remove the Boost CLI, to expose a seed, and to stop recomputing a constant; the
third is checked byte-for-byte against the unpatched program.

### Reference values — fixed, from generated artifacts, not tuning targets

Read from `results/phase2_residual/summary/{hodge_decomposition.json,
doc_tables.md, rank2.csv}`:

| quantity | A375 | PANC1 |
|---|---:|---:|
| cyclic (curl) fraction of `D` | 0.4618 | 0.6017 |
| curl energy in top 2 singular directions | 0.3403 | 0.3208 |
| curl energy in top 4 | 0.4932 | 0.4764 |
| curl energy in top 16 | 0.8071 | 0.7973 |
| **rank-2 `cal_skill` @ coverage 0.40** | **+0.1967** | **+0.1611** |
| **rank-2 `cal_skill` @ coverage 0.70** | **+0.2500** | **+0.2372** |
| rank-2 held-out Pearson @ 0.40 / 0.70 | 0.454 / 0.501 | 0.406 / 0.489 |
| searched low-rank, honest-α `cal_skill` @ 0.40 | +0.229 | +0.198 |
| searched low-rank, honest-α `cal_skill` @ 0.70 | +0.353 | +0.366 |
| searched low-rank, honest-α Pearson @ 0.70 | 0.594 | 0.608 |

### Why the cyclic fraction is NOT the criterion — and what is

Measured before the ensemble, on i.i.d. Gaussian matrices at the real geometry
(5 draws each, `residual.hodge_decomposition`):

| n | curl fraction of pure noise | top-2 curl energy |
|---:|---:|---:|
| 20 | 0.889 | 0.336 |
| 50 | 0.957 | 0.143 |
| **100** | **0.980** | **0.076** |

A gradient field on a complete graph occupies only `2/n` of the energy, so at
n = 100 **any** unstructured matrix has a curl fraction near 0.98. The real
screens' 0.46 / 0.60 are *below* the noise floor, because they carry a large
per-drug potential. A rule of the form "the null reaches ≥50% of the real cyclic
fraction" would therefore be satisfied by literally any noise and would be
vacuous. This is recorded here, before the ensemble, as the reason the decision
rule below is not the one the brief suggested.

The discriminating statistics are the two that unstructured noise cannot fake:

1. **held-out residual skill** — noise gives skill ≤ 0 by construction;
2. **spectral concentration of the curl** — noise gives top-2 energy ≈ 0.076 at
   n = 100, against the real 0.34 / 0.32.

The reconstruction (`docs/dchain_reconstruction.md` §3.3) predicts the artifact,
if present, is **exactly rank-2 cyclic** — the wedge of a per-drug first-position
offset error with a per-drug second-position gain. So the artifact hypothesis
makes a sharp prediction on statistic 2 as well as on statistic 1.

### Primary detector

The **fixed rank-2 low-rank rung** — 204 parameters, one 2×2 antisymmetric
bilinear form, one learning rate, **no hyperparameter search at all** — run
through `residual_experiment.run_residual_condition` with
`force_hparams = residual_sweep.RANK2_HPARAMS`, which is byte-for-byte the
configuration `residual_sweep.rank2_grid()` uses on the real screens. Primary
because it removes "a hyperparameter search found estimator noise" as an
explanation before the question is asked. The searched low-rank rung under the
honest-α shrinkage (the configuration the Phase 2R decision quotes) is run as a
secondary, with the `zero` and `potential` rungs it is contrasted against.

Coverages **0.40 and 0.70**, on the full five-coverage split grid so the splits
and the evaluation pool are identical to the real ones.

### Decision rule

Let `s_null(c)` be the ensemble distribution, over simulation seeds, of the
NULL-A rank-2 `cal_skill` at coverage `c` (each seed's value is its mean over its
split seeds). Let `s_real(c)` be `min(A375, PANC1)` at that coverage — the
weaker of the two real screens, so the comparison is against the *harder* target.
Let `t2_null` and `t2_real` be the corresponding top-2 curl energies.

**A — ESTIMATOR ARTIFACT REPRODUCES RESULT.** Both of:
* `median(s_null(0.70)) >= 0.5 * s_real(0.70)`, i.e. **≥ +0.119**, and
  `median(s_null(0.40)) >= 0.5 * s_real(0.40)`, i.e. **≥ +0.081**;
* `median(t2_null) >= 0.5 * t2_real`, i.e. **≥ 0.160**.

Or, independently sufficient: `s_real(0.70)` for *either* screen lies inside the
null ensemble's central 95% interval at that coverage.

**B — PARTIAL ESTIMATOR CONTRIBUTION.** The null produces a clearly positive
median rank-2 skill at coverage 0.70 — `median(s_null(0.70)) >= +0.02` with the
ensemble's 95% interval excluding 0 — but criterion A is not met.

**C — LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT.** All of:
* `median(s_null(0.70)) < +0.02` and `median(s_null(0.40)) < +0.02`;
* both real screens' values lie strictly above the null ensemble maximum at both
  coverages (empirical percentile 100%, one-sided p ≤ 1/(n_seeds+1));
* `median(t2_null) < 0.5 * t2_real`.

**D — INCONCLUSIVE RECONSTRUCTION.** Declared, overriding A/B/C, if any of:
* the byte-equivalence check on the patched sampler fails;
* more than 20% of ensemble runs fail to complete or fail the retained-sample
  check `n_samples == n_samples_expected`;
* the oracle control (Control A) shows non-negligible held-out skill on the true
  matrix, i.e. the null is malformed;
* the null screens' estimated `synergy_measure` spread is more than 5× or less
  than 1/5 of the real screens', so the simulated world is not on the same scale
  as the one being argued about.

`0.02` is chosen as "clearly positive" because it is roughly ten times the
largest `|cal_skill|` the real `potential` rung reaches at coverage ≥ 0.20
(≤ 0.002), which is this project's existing operational definition of a rung
finding nothing.

### The ensemble, as planned

| block | variant | estimator | seeds | MCMC |
|---|---|---|---:|---|
| **primary** | strict | joint (`dchain.cpp`) | 20 | published defaults |
| realism | nuisance | joint | 10 | published defaults |
| Control A — oracle | strict + nuisance | oracle | 20 + 10 | none |
| Control C — no sharing | strict | unshared | 20 | none (batched MAP) |
| Control D — noise sweep | strict, σ ∈ {0.075, 0.30} | joint | 6 each | published defaults |
| convergence | strict, sim_seed 0 | joint, est_seed 1–4 | 4 chains | published defaults |

Simulation seed and estimator seed are separate throughout, so "the data changed"
and "the sampler changed" are never confounded. Every run records its
configuration, both seeds, its retained-sample count against the expected count,
its split-half posterior agreement, its selector-on fraction, and the four
deposit identities.

**Exclusion criteria, fixed now:** a run is excluded only if the sampler exits
nonzero or if `n_samples != n_samples_expected`. No run is excluded on the basis
of the structure it produces. If compute forces the seed counts down, the actual
counts are reported next to the planned ones rather than the plan being edited.

### Registered predictions

**H_artifact-null.** If the estimator can manufacture the Phase 2R signature,
NULL-A will show rank-2 held-out skill at coverage 0.70 comparable to the real
screens' +0.24–0.25, top-2 curl energy well above the 0.076 noise floor, and a
coverage transition between 0.40 and 0.70 resembling the real one.

**H_real-structure.** If it cannot, NULL-A will show rank-2 skill at or below
zero at both coverages, top-2 curl energy at the noise floor, no stable held-out
Pearson, and the real values will lie far outside the null ensemble.

The reconstruction's algebra says the artifact is *real but possibly small*: it
is first order in the shared offset error, so its magnitude scales with how badly
`u_i = log β_i + log f(1; θ_i)` is determined, and that is set by the three A0
replicates and the eight-point single-agent curve. Predicting its size a priori
is exactly what this experiment declines to do.

### One thing that could still go wrong, named in advance

If the null's estimated synergy matrix turns out to be far *smaller* in absolute
spread than the real screens', a finding of "no skill" would be ambiguous between
"the estimator makes no artifact" and "this simulated world is too clean for the
artifact to be visible". That is why the scale check is a criterion-D trigger and
why the noise sweep is preregistered rather than optional.

### Amendment to the pre-registration, after independent statistical review and before the ensemble

An independent reviewer that did not write the code was given the design and the
decision rule *before any joint-estimator result existed*, and asked to find
inferential errors while they could still be fixed. It found several. They are
recorded here with their reasons; the original pre-registration above stands
unedited, and every change below was made with **zero joint-null results in
hand** — the primary block had produced no completed condition when this was
written, and was restarted from scratch afterwards.

Each finding was re-derived here before being acted on.

**1. The spectral criterion was scale-free, and therefore nearly vacuous.**
`top-2 curl energy` is a fraction *of the curl*. Under NULL-A the true
directional matrix is zero, so 100% of the null's curl is estimator artifact,
and "is at least half of 0.32 of the estimator's cyclic error concentrated in
two directions?" is answered **yes at any magnitude whatsoever** — including
magnitudes a thousandfold below the real screen. The artifact clause was
near-automatic and its negation near-impossible.

Fixed by moving to an **absolute** statistic: `curl_fraction × top2`, the share
of the *directional* energy that is rank-2 cyclic. Real values, computed from
`hodge_decomposition.json`: **A375 0.157, PANC1 0.193**; ensemble mean 0.175, so
the artifact threshold is **0.0875**. The fraction-of-curl version is still
reported, beside the noise floor, as a description.

**2. A ceiling on the artifact, derivable from the real data alone.** A pure
rank-2 artifact has top-2 curl energy 1.0 and unstructured noise has 0.0747
(remeasured on 20 draws; 0.076 was registered from 5). The real screens sit at
0.340 / 0.321, so **at most 28.8% (A375) / 26.7% (PANC1) of the real cyclic
energy can be a rank-2 artifact** — at most 13.3% / 16.1% of `D²`. This is a
bound on the artifact hypothesis that needs no simulation at all, and it is
reported whatever the ensemble says.

**3. "PARTIAL ESTIMATOR CONTRIBUTION" was the code's catch-all.** A null whose
skill was clearly *negative* at both coverages — the artifact hypothesis crushed
— was classified as a partial artifact contribution, because the `else` branch
caught it. A null that finds strictly less than nothing is evidence *against* an
artifact. Fixed: an explicitly named `null_skill_crushed` branch routes to
"little evidence", and `PARTIAL` no longer catches anything unlabelled.

**4. The criterion-D scale check policed a quantity the primary metric cannot
see.** `cal_skill` is standardised by the training RMS of `D_res`, so it is
**exactly** scale-invariant — verified to six decimals across a 250× range of
`D`. The ±5× band on `D_std_offdiag` therefore tested nothing, and worse, it
would have fired `INCONCLUSIVE` precisely in the "real but small artifact"
scenario the reconstruction says is likely, converting a true negative into a
non-answer. Demoted to descriptive, and replaced with two triggers that bear on
what skill actually depends on:

* **the selector gate.** The measure is `λ_AB · (…)`. If the posterior selector
  closes in the null, the artifact has no channel to express through and a
  negative result is a property of the simulated world, not of the estimator.
  Registered floor: median `selector_on_fraction ≥ 0.10`, against the deposit's
  mean `|λ|` of **0.4916 (A375) / 0.4635 (PANC1)**.
* **the reproducible fraction.** `2·mean(posterior sd²)/mean(D²)` — how much of
  the null's directional variance is posterior noise rather than structure. Above
  0.80 there is nothing for any model to find. The real screens are at **0.205 /
  0.192**, from the deposited `synergy_sd` column.

**5. A partial ensemble could produce a confident verdict.** `n_planned` counted
rows *present in the file*, so an interrupted run had a failure fraction of
exactly zero and criterion D never fired; a file with 1 of 20 seeds returned
"little evidence for estimator artifact" without complaint. Fixed: the planned
count is read from the grids, and fewer than 80% of the planned runs is a
criterion-D trigger.

**6. The oracle control cannot fail under the strict null, and that is a
property of the null rather than a defect.** With a true directional matrix of
exactly zero, held-out skill is `0/0` and comes back `NaN`; `NaN >= 0.02` is
false. The check that works under NULL-A is the one already recorded — that the
true matrix *is* identically zero — and it is now a criterion-D trigger in its
own right. The skill form of the control remains, and is the one that bites
under NULL-B, where the true matrix is nonzero but unpredictable.

**7. Two reporting bugs.** A row whose real value is undefined (the rank-2
Pearson rows) was being given percentile 0 and `p = 0.0476`, the most extreme
value the design can produce, for a comparison never made. And
`real_over_null_median` divided by a null median that is **exactly zero in the
modal case** — the shrinkage selects `α = 0` whenever the rung finds nothing, and
`cal_skill` is then exactly 0. Replaced by the standardised gap
`(real − median)/sd`.

**8. The coverage transition was predicted and not measured.** `H_artifact-null`
predicts "a coverage transition between 0.40 and 0.70 resembling the real one",
but only those two coverages were scored. The rank-2 detector is cheap, so it now
runs the **full five-coverage grid**, exactly as the real `rank2` block did. The
searched block stays at the two decision coverages.

**9. The mechanism test would have run on an ensemble of one.** The diagnostics
that can confirm or refute the predicted mechanism were gated behind
`keep_posterior`, which was set only in the convergence block — four chains on a
single dataset. They now run on **every joint condition**, because they need only
the posterior *means*, which are computed in-process and stored as a handful of
scalars rather than as the 500 MB of samples that motivated deleting them.

**The mechanism test itself is now a zero-free-parameter prediction.** The
reconstruction says the artifact is the wedge `ε ∧ m̃` of two per-drug vectors.
Both are computed, neither is fitted to the thing being explained:

* `ε_i = û_i − u_i` with `u_i = log β_i + 1[λ_i]·log f(1 µM; θ_i)` — the
  posterior's error in the shared first-position offset, truth known;
* `m̃_j` — how much of a uniform log-scale offset the second drug's curve converts
  back into apparent synergy, obtained by pushing the offset through the same
  fitter and taking a central difference. On a 100-drug panel from the model's own
  priors it lands in [0, 1] for every drug with sd ≈ 0.2, and that heterogeneity is
  the ingredient the mechanism requires.

The estimated cyclic component is then regressed on `curl(ε ∧ m̃)` with **one**
scale coefficient, and the principal angles between `span{ε, m̃}` and the leading
cyclic pair are reported. This is reported, never decided on: the verdict is about
whether an artifact of the observed *size* is present, and this is about whether
whatever is present has the predicted *shape*.

**What was considered and rejected.** The reviewer suggested that
`p_lambda_single = 1.0` removes the inert drugs the mechanism needs. Measured
before acting: on the model's own priors the panel already spans mean baseline
viability 0.22–1.00 with sd 0.16–0.21, and 13–23% of drugs are effectively inert
over the scoring grid. The heterogeneity is present; the setting stays.

**What is acknowledged and not fixed.** The reviewer is right that NULL-A is
maximally favourable to the artifact hypothesis for *attribution* (nothing is
confounded) but not demonstrably for *magnitude*: under NULL-A the shared offset
error `ε_i` is pure sampling error, while in the real screen it would also carry
misspecification bias, which is systematic and larger. A negative result therefore
licenses "the modelled sampling-error artifact does not reproduce the signature",
not "no estimator artifact of any kind could". §8 of the falsification document
says so, and the "maximally favourable" claim is restated there in the narrower
form it can support.

**10. One of the replacement criteria in item 4 was itself broken, and was
corrected before the ensemble.** A 60-drug, 60,000-iteration pipeline check —
run to verify the new code path, not as a result — showed the posterior noise
fraction `2·mean(sd²)/mean(D²)` coming back above 1 on the null. That is not a
sick simulation: under a correct null the true directional matrix is zero, so
the posterior *means* shrink toward zero while each retains its own posterior
uncertainty, and the ratio exceeds 1 **by construction**. Compared against the
real screens' 0.205 / 0.192 it would have fired `INCONCLUSIVE` on every
correctly-built null. A criterion that fires on a correct null is not a
criterion.

Replaced by the quantity that actually bears on whether there is anything to
find: the **split-half agreement of the posterior-mean directional matrix**
between the two halves of the chain, with a registered floor of `r ≥ 0.50`. The
pipeline check reached 0.90. Note that this is a diagnostic the real deposit
cannot report at all — the published fit was one unseeded chain with no
convergence diagnostics of any kind. The posterior noise fraction is still
recorded and reported beside the real values, with that caveat.

The pipeline check's own numbers are not results and are not reported as any: it
ran 60 drugs at an eighth of the published chain length, to prove the code path
executes end to end.

### Second amendment, after adversarial review of the null design

A second independent reviewer, given the brief "try to prove this experiment does
not test what it claims to test", was run against the code while the primary
ensemble was executing. It ran the compiled sampler itself on small strict and
nuisance screens to check its claims rather than reading. Its findings are below.
Everything acted on was acted on with **no primary-ensemble result in hand**; the
first joint condition had not completed.

**1. The preregistered exclusion rule was registered and never implemented, and
two bad runs out of twenty flip the verdict.** The pre-registration says a run is
excluded if the sampler exits nonzero or if `n_samples != n_samples_expected`.
The verdict *counted* incomplete runs and then used them anyway. The reviewer
demonstrated the cost: 18 clean seeds at skill 0.000 plus 2 truncated runs
carrying skill 0.9 is a 10% failure rate, inside the 20% allowance, so criterion
D stays silent — but the null's 97.5th percentile is dragged to 0.90, a real
value lands inside the null interval, and the verdict flips from **LITTLE
EVIDENCE** to **ESTIMATOR ARTIFACT REPRODUCES RESULT**. One outlier is nearly
enough on its own: 19 seeds at 0.01 and one at 0.30 gives q975 = 0.162, and
PANC1's +0.1611 is inside it.

Implemented as `report.is_usable`, applied before the null distribution is formed,
and pinned by a test that reproduces the reviewer's exact scenario. This was the
single most dangerous defect in the experiment: it converts a hardware hiccup
into the conclusion the project would have to act on.

**2. The "real result inside the null interval" clause was being evaluated at
both coverages; it is registered at 0.70 only.** It is *independently sufficient*
for criterion A, and a 95% interval from 20 draws is essentially min-to-max, so
it is the most outlier-sensitive part of the rule. Evaluating it at every coverage
doubled its chances — and in the reviewer's demonstration it was the coverage-0.40
cell that fired. Restricted to the registered coverage.

**3. The claim that NULL-B "can only dilute an artifact, never manufacture one"
is wrong, and NULL-B is the stronger arm.** The argument assumed NULL-B enlarges
the skill denominator with the numerator fixed. It does not. The measure is
`λ_AB · (…)`, and NULL-B's combination curves give the estimator a reason to open
the selector, which *multiplies* the artifact. Measured by the reviewer on the
same truth, seed and settings:

| | selector on-fraction | artifact RMS | artifact sd(D) | artifact top-2 |
|---|---:|---:|---:|---:|
| strict (NULL-A) | 0.21–0.23 | 0.016–0.017 | 0.022 | — |
| nuisance (NULL-B) | 0.330 | 0.0328 | 0.0476 | 0.294 |
| **the real deposit** | **0.4916 / 0.4635** | — | — | — |

So NULL-A is maximally favourable for *attribution* — with a true matrix of
exactly zero, anything found is 100% artifact and nothing is confounded — but it
is the **less** favourable of the two for *magnitude*, and it was the primary with
twice the seeds. **The realism arm is raised from 10 seeds to 20**, and the
"can only dilute" sentence is withdrawn.

There is no way to open the gate without adding a true effect, and that is worth
stating rather than working around: under exact separability the combination data
gives the estimator no reason to prefer a private curve, so a gate of ~0.21 is
what a world with no pair effects *produces*. The deposit's 0.47 reflects whatever
is actually in the real screen. `selector_on_fraction` is therefore reported as a
headline number for both arms, and a negative NULL-A licenses "the artifact is at
most about twice this size at the real gate", not "there is no artifact".

**4. Control C changes two things, not one.** Its docstring claimed the only
difference from the joint arm is whether per-drug error is reused. It is not: the
joint arm is an MCMC posterior mean over 1,999 samples with a fractional gate,
and the unshared arm is a MAP point estimate with a hard 0/1 BIC gate, whose
estimation-error scale is 3.3× larger. More importantly its held-out skill is
**≤ 0 by construction** — every per-drug quantity is redrawn per pair, so the
score is independent across pairs and no model can predict it. It is a pipeline
sanity check and a demonstration that per-pair-independent estimation error has
no spectral concentration (top-2 = 0.082 against a floor of 0.0747), not a clean
isolation of sharing. Restated as such.

**5. The artifact "ceiling" is a point estimate that errs low.**
`(top2 − floor)/(1 − floor)` assumes the non-artifact remainder is spectrally
i.i.d. and that energies add without cross terms. Tested by construction: a true
share of 0.10 recovers 0.070, 0.20 recovers 0.184, 0.30 recovers 0.266 — it
**under**-estimates by about 10% relative. Restated as "approximately 29% (A375)
/ 27% (PANC1)" rather than "at most", with the bias named. The direction favours
this project's headline, which is exactly why it should not be stated as a bound.

**6. The one scale statement that is directly comparable, and it matches.** The
deposit's PANC1 `synergy_sd` has median 0.0362; the null's posterior SD of the
same quantity is 0.0295–0.0310. A factor of **1.2**. The simulated world is on
the real world's scale in *estimator uncertainty* and differs from it in *signal*,
which is the point of the experiment and the right way to say it.

**7. Two things the reviewer checked and found clean, which are worth recording
because they are the assumptions everything else rests on.**

* *No real-data leakage into the decision metrics.* The Phase 2R path does read
  the real deposit, for `measurement_noise_sd`. The reviewer monkey-patched it to
  return absurd thresholds and re-ran an identical condition: exactly nine keys
  changed, all of them `*_sign_threshold`, `*_sign_n`, `*_sign_frac_of_pairs`.
  `cal_skill`, `cal_pearson`, `cal_spearman` and `heldout_skill` were bit-identical.
* *Phase 2R is run unchanged, on the same pairs.* Every field of the null's
  `ResidualConfig` matches `residual_sweep.rank2_grid()` and
  `honest_alpha_grid()` on every cell. Stronger: the train/val/test **pair index
  sets** for a 100-drug simulated screen are identical to the real A375 screen's
  at all five coverages (211/37/848, 421/74/848, 842/148/848, 1683/297/848,
  2945/520/848). The null is scored on exactly the pairs the real result was.

**8. Two smaller ones, recorded not fixed.** The deposited example data is not
uniformly triplicate (20 of 24 conditions are; two have n=2 and two n=1) while the
simulator emits a balanced design — which makes the null's per-drug parameters
*better* determined than the real screen's, so the artifact is understated. And
the searched block runs 4 split seeds against the real block's 8, so its
per-screen mean is noisier than the reference it is compared against; the fixed
rank-2 primary runs the full 8.

### Third review: source fidelity. The model is unmodified; three documentation defects

A third independent reviewer was given one question — *does the code that is
actually run faithfully reproduce the published estimator and its synergy
measure* — and told to be forensic. It diffed the patched source hunk by hunk,
re-ran the byte-equivalence check on four inputs the built-in check does not use,
transcribed `interpretMCMC.R` line by line into Python and compared numerically,
and designed a two-drug screen whose asymmetry it knew in advance to test the
index orientation end to end through the compiled sampler.

**On the model: clean.** All 14 diff hunks classify as command-line scaffolding,
the added seed, or the sufficient-statistic cache. **Nothing** inside `tQuotient`,
`logResponse`, any `qprior`, any acceptance ratio, any proposal, the four-case
selector logic, the three-block sweep order, or the storage rule differs. Four
further confirmations worth recording:

* the shipped binary is byte-identical to one compiled fresh from the audited
  source, so it is not stale;
* the equivalence check holds on the 66-row example at 30,000 and 120,000
  iterations and on a 6-drug simulated screen at two more settings — and the
  reference is built at `-O2` against the patched `-O3`, which makes the check
  stricter than advertised, not weaker;
* **the index orientation is right, proved by construction.** A two-drug screen
  in which only `AAA → ZZZ` kills, pushed through the real sampler, returns
  `synergy[AAA→ZZZ] = +0.768` and `synergy[ZZZ→AAA] = +0.0007`. A transpose would
  have put the effect in the other cell;
* end-to-end recovery on a 5-drug nuisance screen: `r = 0.961` between true and
  estimated synergy, and `beta` recovered to within 0.03 on all five drugs.

**Three defects, all in documentation or numerics, none in the model.**

1. **The patch documentation understated what is patched, and did so about the
   riskiest edits.** `PATCHES` was described as containing every edit. It
   contains eight; the other seven — the sufficient-statistic call sites, and the
   **only** edits that touch lines inside the Metropolis blocks — are done by a
   regex with its own assertion. A reader auditing `PATCHES` alone would miss
   exactly the seven that matter most. Corrected in both the module and
   `dchain_reconstruction.md` §5.
2. **`~3×10⁵` sufficient-statistic evaluations per iteration is wrong; it is
   ~1.6×10⁵** (40,100 + 80,900 + 40,000 at 100 drugs). Performance prose, but
   wrong is wrong.
3. **`synergy_posterior` accumulated the variance as `(Σx² − n·x̄²)/(n−1)`.** It
   agrees with the two-pass value to 8×10⁻¹⁷ on these inputs, so nothing that has
   been computed is affected — but that form cancels catastrophically when the
   mean is large relative to the spread, which is exactly the shape a posterior
   can have. Replaced with Welford.

**And one fact worth knowing rather than fixing: seed 0 and seed 1 are the same
chain.** libc++'s `default_random_engine` is `minstd_rand`, whose
default-constructed state is seed 1, so `--seed 1` *is* the published program's
unseeded chain. Verified: seeds 0 and 1 give byte-identical output, seeds 2 and 3
differ from both. That makes the convergence block's chain 1 literally the chain
the authors' own defaults produce — and makes adding 0 alongside it a way to get
two identical "independent" chains. `CONVERGENCE_CHAINS` now asserts against it.
The primary ensemble uses seeds 101–120 and is unaffected.

---

# Phase 3 — entity-level out-of-distribution transfer on Koplev

Registered at `deadb46`, 2026-08-25. Registered before a single real held-out-drug number existed. The rule returned **INCONCLUSIVE**; the same rule with one statistic corrected returns **PAIR-SPECIFIC ENTITY TRANSFER**. Both are reported, and the frozen one is the registered one; see [`phase3_entity_ood.md`](phase3_entity_ood.md).

## Pre-registration — Phase 3, written and committed before any real test outcome

Everything below is fixed before a single real held-out-drug number exists. What
has been looked at so far, and deliberately only this: the drug mapping, the
chemical-similarity geometry of the 100 compounds, the ChEMBL target coverage,
and one synthetic positive control whose target was generated by this repository
and carries no information about the Koplev screen.

### The question

> Can the interaction behaviour of a drug be predicted when that drug itself has
> never appeared anywhere in training?

### The target

`D(i, j) = y(i -> j) - y(j -> i)`, one row per unordered pair, canonical
orientation `i < j` by the screen's own drug index. Built by Phase 2R's
`residual.directional_pairs`, unchanged, so the two phases cannot come to
disagree about which way `D` points.

**No residualisation.** Phase 2R's `D_res` subtracts a per-drug potential fitted
by ridge from the drug's own training rows. For a drug with no training rows that
ridge returns `g_k = 0` silently — the normal equations reduce to `diag(λ)` with
`X'y = 0` — so the "residual" for a held-out drug would still contain that drug's
entire potential, and would be a different quantity from the residual for a
training drug. Phase 3 predicts raw `D` and makes the model earn the potential
from features, where it is measurable rather than assumed away.

### The split

Held out: **drugs**. Not pairs.

Three independent seeded partitions (seed 20260825) of the 100 drugs, each cut
into 10 disjoint folds of 10 test drugs, so every drug is a test entity exactly
three times: **30 folds**. Within each fold, 10 further drugs are drawn as
validation entities from that fold's own deterministic stream.

Every one of the 4,950 unordered pairs lands in exactly one bucket, decided only
by which drugs it touches:

| bucket | rule | pairs per fold |
|---|---|---:|
| `train` | neither endpoint in T or V | 3,160 |
| `val` | no endpoint in T, at least one in V | 845 |
| `test_e1` | exactly one endpoint in T | 900 |
| `test_e2` | both endpoints in T | 45 |

3,160 + 845 + 900 + 45 = 4,950. `assert_partition` checks the arithmetic per
fold rather than trusting the construction.

**Validation is entity-OOD too.** Tuning on held-out *pairs* among training drugs
and then reporting held-out *drugs* would select the hyperparameter that is best
for transductive completion and then report it on a different task. Every
validation pair has an endpoint the model never trained on.

**E1 and E2 are never pooled.** Different questions, and 900 against 45 rows —
pooling would bury the harder one inside the easier one.

### The representations

**Primary: ECFP4, radius 2, 2,048 bits, `useChirality=False`**, computed by RDKit
from the mapped structures. A fingerprint is a function of the molecular graph
alone; there is no path from it to a Koplev outcome. That is the point of
starting here rather than with a pretrained chemical language model whose
training corpus nobody in this repository can audit.

Preprocessing: bits with zero variance **across the fold's 80 training drugs** are
dropped. Fitted on training drugs only. It reads no outcome and could defensibly
have been fitted on all 100; it is not, so that "the preprocessing never saw a
test drug" needs no qualification.

**Secondary: ChEMBL curated mechanism targets**, 76 binary columns. Coverage
decided now, on the annotation and not on any result: 97/100 drugs annotated;
32 of 76 targets are shared by two or more drugs; **16 drugs (3 unannotated plus
13 whose every target is unique to them) have a target vector orthogonal to every
other drug** and cannot transfer by construction. That is a real limit of the
annotation's granularity, it is why this arm is secondary, and it is why a null
here would be much weaker evidence than a null from fingerprints.

### The model ladder

| rung | form | role |
|---|---|---|
| `zero` | `Dhat = 0` | mandatory null |
| `potential` | `g(x_i) - g(x_j)` | **the critical baseline** |
| `lowrank` | `g(x_i) - g(x_j) + z_i' K z_j`, `K = -K'` | the hypothesis |
| `antisym_mlp` | `F(x_i,x_j) - F(x_j,x_i)` | flexible upper bound |
| `pair_only` | `z_i' K z_j` alone | diagnostic |

Every rung is exactly antisymmetric by construction and starts at exactly
`Dhat = 0`. `lowrank` **nests** `potential`: `W` is zero-initialised, so at
initialisation the two are the same function and any incremental skill is
attributable to the pair term rather than to a better-fitted potential. Exactly
one tensor in the pair term starts at zero — zero-initialising both `W` and the
latent encoder would make every partial derivative vanish and produce a
permanent, silent 0.0, which is the failure Phase 2R already shipped once.

### Primary endpoint and primary contrast

**E1** (one unseen endpoint), **ECFP4**, **`lowrank` minus `potential`**, each
screen separately.

`incremental_skill = 1 - MSE_lowrank / MSE_potential`, computed **within a fold**
and then averaged over folds. A ratio of pooled means would be dominated by the
highest-variance folds.

### The unit of replication

The **fold**, n = 30 per screen. Not the pair: 900 E1 rows in a fold share ten
held-out drugs and are nowhere near independent. Reported alongside: a per-
held-out-drug summary, n = 100 drugs, each averaging its three folds.

### Similarity strata — cut points fixed now, from features only

Max ECFP4 Tanimoto from each drug to any of the other 99: q33 = **0.2340**,
q66 = **0.5161**. Strata are `low < 0.2340 <= medium < 0.5161 <= high`, applied
to each held-out drug's max similarity to its *fold's* training set. Outcome-free
and fixed before any run.

Context that makes the stratification worth doing: **no drug in this set has a
nearest neighbour above 0.883 Tanimoto, and only 12 of 100 are above 0.7**
(median 0.29). This is a chemically diverse deposit, so "analogue-only transfer"
is a hypothesis that can actually fail here.

### Decision rule, evaluated in this order

Let `S` range over {A375, PANC1}, statistics over the 30 folds of each.

**First, validity.** If any of these fires, the verdict is INCONCLUSIVE and no
category is claimed:

* the synthetic positive control's E1 incremental skill is <= 0.05 — the
  machinery cannot recover a planted rank-2 signal, so a real null means nothing;
* the `random` or `shuffled` representation posts mean E1 `lowrank` skill > 0.05
  — features that cannot contain the answer are predicting it, so something leaks;
* any fold fails `assert_no_drug_leakage`;
* more than 10% of planned conditions error out.

**Then, in order:**

1. **NO ENTITY TRANSFER** — `potential` mean E1 skill <= 0.02 in both screens.
2. **POTENTIAL-ONLY ENTITY TRANSFER** — `potential` mean E1 skill > 0.02 in both
   screens, and the incremental criteria below fail in both.
3. **PAIR-SPECIFIC ENTITY TRANSFER** — all of, in **both** screens:
   (a) `potential` and `lowrank` both have mean E1 skill > 0;
   (b) mean incremental skill > 0.01;
   (c) paired t-test p < 0.05 **and** Wilcoxon p < 0.05 over the 30 folds;
   (d) at least 20 of 30 folds favour `lowrank`;
   (e) mean `lowrank` E1 Pearson > 0;
   and, across the whole experiment,
   (f) `random` and `shuffled` both give mean E1 `lowrank` skill <= 0.02;
   (g) **not analogue-confined**: mean incremental skill > 0 among held-out drugs
   in the `low` similarity stratum, in at least one screen.
4. **ANALOGUE-ONLY TRANSFER** — 3(a)–(f) hold but (g) fails.
5. **WEAK/MARGINAL ENTITY TRANSFER** — anything else: significant in one screen
   only, or mean incremental skill in (0, 0.01], or fewer than 20/30 folds
   favouring `lowrank` in a screen.

No category is modified after seeing results. Requiring both screens is a
deliberate substitute for a multiplicity correction on two tests.

### Secondary, reported but not decisive

E2; the target representation and its shuffle; the coverage sweep among training
entities at 0.20 / 0.40 / 0.70 against the primary 1.00; the metal-excluded
re-scoring; `antisym_mlp` and `pair_only`; A375-vs-PANC1 replication.

The metal arm is a **re-scoring of the same fit**, not a refit: excluding the four
coordination complexes changes which test pairs are averaged, not what the model
learned. Refitting would have put optimiser noise between two numbers whose
difference is the entire comparison.

### Registered predictions

Written down so they can be wrong in public.

1. `potential` will have clearly positive E1 skill — the per-drug ordering
   tendency is the largest component of `D` and structural analogues share it.
2. Incremental skill will be **small and near zero**. Phase 2R's rank-2 structure
   was found with free per-drug embeddings, which can encode anything about a
   drug including things a fingerprint does not see.
3. E2 will be worse than E1 but not catastrophically so, because most of E2's
   error is potential too.
4. `antisym_mlp` will not beat `lowrank` — 3,160 training pairs is too few for an
   unrestricted interaction over 2,048-bit inputs.
5. The target representation will do worse than fingerprints, driven by the 16
   drugs whose target vectors are orthogonal to everything.

### One thing that could go wrong, named in advance

The single most likely way to get a **false positive** here is not leakage but
**the potential baseline being under-fitted**. If `potential` is handicapped —
too small a grid, too few epochs, wrong regularisation — then `lowrank` wins its
incremental comparison by fitting the potential better with its spare capacity,
and the result gets described as pair-specific interaction. Three defences are
registered: the models are nested and start identical; `pair_only` isolates
whether the bilinear term carries anything without a potential head; and
`potential` gets a grid member with a hidden layer, so it is not restricted to a
linear `g` while `lowrank` gets a nonlinear one.

### Amendment 1, after the independent mapping audit and still before any real result

Eighteen independent auditors reviewed the mapping — ten covering ten drugs each
from primary sources, eight applying specialist lenses across all 100 — and every
finding was then handed to a separate adjudicator instructed to refute it. 28 raw
findings, 14 distinct after merging, **4 upheld**.

**One is material and changes the features.** PubChem's name-indexed carboplatin
record, CID 426756 — the only hit for the literal name — mis-depicts the two
ammine (NH₃) ligands as azanide anions `[NH2-]` and the chelating
cyclobutane-1,1-dicarboxylate dianion as the neutral free diacid. The molecular
formula is coincidentally identical (C₆H₁₂N₂O₄Pt) so the formula check cannot see
it, and CHEMBL1351 carries no structure at all so the cross-check could not fire.
Formal charge and hydrogen count are both Morgan atom invariants, so it reaches
the features: ECFP4 Tanimoto between the two depictions is 0.50, and the
mis-depicted one is **0.875 similar to bare 1,1-cyclobutanedicarboxylic acid**.

That is exactly the failure the fragment rule was written to prevent, arriving by
a different route. Rejecting PubChem's parent relation stopped carboplatin from
entering as an unrelated small diacid; PubChem's *own name record* then delivered
a fingerprint 87.5% similar to that diacid anyway. The lesson is narrower than
"check the structures": the metal rows are the only ones with no second witness,
because ChEMBL deposits no structure for any of them, so the cross-check that
protects the other 96 drugs is silently absent for exactly the four compounds
whose depiction is hardest.

Overridden to CID 10339178, corroborated through UniChem by DrugBank DB00958,
ChEBI:31355 and FDA UNII BG3F62OND5.

**Two are minor and do not change the conclusion.** Pentostatin was deposited as
the 4H tautomer where the FDA label, CHEMBL1580 and PDB component DCF all give
the 3,6,7,8-tetrahydro form — invisible to every check in the pipeline, because
standard InChI treats that proton as mobile and both forms give the same key.
Vincristine's record is the C-15 epimer; the proof is internal rather than a
matter of database preference, since vincristine is N1′-desmethyl-N1′-formyl-
vinblastine and applying that one substitution to *this mapping's own vinblastine*
reproduces `OGWKCGZFUXNPDA-CFWMRBGOSA-N`, not the recorded key. The vincristine
correction does not reach the features at all, because the primary fingerprint
uses `useChirality=False`; it is applied so the stored structure is right.

**One is documentation.** The labels are the NCI DTP Approved Oncology Drugs Set
IV, not Selleck catalogue names — which `docs/phase2_dataset.md` already recorded
and this module's docstring contradicted.

#### The rule governing overrides, and the case that proves it bites

An override may only point at **another deposited record in an authoritative
database** — never at a hand-drawn structure, however good the reasoning. Without
that constraint this table becomes the place where the mapping gets adjusted
until it looks right.

The constraint bites immediately. The audit found the *same* defect in
oxaliplatin's record — CID 9887053 writes neutral oxalic acid plus two amide
anions — and **no database holds a corrected depiction**: the correct
(1R,2R)-DACH / oxalate(2−) / Pt(II) structure has no PubChem CID at all
(`ZROHGHOFXNOHSO-BNTLRKBRSA-L` returns no hit; the only near record, CID
15057790, is the wrong enantiomer). So oxaliplatin is **not** overridden. The
defect is recorded, and the pre-registered metal-excluded arm is what answers it.

#### One threshold that could have moved, and did not

Correcting carboplatin changes its fingerprint and therefore the empirical
similarity distribution: q33 moves from **0.2340** to **0.2222**. The registered
cut points are **left unchanged at 0.2340 / 0.5161**. The registration fixed
numbers, not a recipe to be re-run, and re-deriving a threshold after a change —
even a change made for reasons wholly unrelated to any outcome, and even before
any outcome has been looked at — is the habit that makes pre-registration
worthless.

The shift is 0.012. An earlier version of this paragraph said it "moves at most
one or two drugs", which was a guess and is wrong: re-deriving the cut would move
**four** drugs between the low and medium strata on the all-99 distribution and
**three** on the statistic actually stratified. Keeping the registered cut is the
conservative direction — it leaves those drugs in the *low* stratum, the one
whose non-significance the analogue verdict rests on.

The primary sweep was restarted from scratch against the corrected mapping. No
result computed under the uncorrected structures is reported anywhere.

---

# Phase 4 — external chemical validation on ChemLex

Registered at `a1f396f`, 2026-08-26. Registered before a single authoritative fold was fitted. The rule returned **INCONCLUSIVE**; the same rule with one statistic corrected returns **ANALOGUE-ONLY CHEMICAL TRANSFER**. Both are reported, and the frozen one is the registered one; see [`phase4_chemlex_interactions.md`](phase4_chemlex_interactions.md).

## Pre-registration — Phase 4, written and committed before any authoritative result

Everything below is fixed before a single authoritative fold is fitted. The
development work that preceded it -- timing the pipeline, setting the training
budget, checking that the hyperparameter grid brackets its optimum -- ran on
**fold seed 20260826**, and the authoritative folds use **seed 20260904**, a
value no model has ever been fitted on. That separation is the whole reason the
distinction is worth stating: a fold-0 pilot at the development seed *was* run
and its numbers *were* seen, and the honest way to handle that is to make it a
different fold rather than to claim it did not happen.

### The question

> Does reaction outcome contain reusable pair-specific interaction structure
> beyond the independent contributions of the two reactants, and can that
> structure be inferred from the molecular structures of reactants never observed
> during training?

Not "does antisymmetry work on reactions". An acid and an amine are different
entity types in different roles; `I(n, a)` does not typecheck, and imposing
`I(a,n) = -I(n,a)` would be meaningless rather than merely wrong. The interaction
here is bipartite:

    y(a, n, c) = mu + f_A(x_a) + f_N(x_n) + f_C(c) + I_AN(a, n) + residual
    I_AN(a, n) = z_A(x_a)^T W z_N(x_n)

### The usable dataset

Zenodo record 17596563, sha256 `a744e340...c9773`, all 11,669 measured rows. No
row is excluded for any reason: no outlier rule, no minimum-conversion filter, no
condition dropped for rarity. Two screens, and they are **nested, not independent
replicates**:

| screen | rows | acids | amines | conditions | pairs | observed fraction |
|---|---:|---:|---:|---:|---:|---:|
| `hatu` | 8,454 | 272 | 230 | 1 | 7,919 | 12.66 % |
| `all` | 11,669 | 272 | 230 | 7 | 8,064 | 12.89 % |

`hatu` is HATU/DIPEA/DMF, the first-pass campaign. It exists because condition
confounding **cannot occur inside it**: there is only one condition, so no
acid-by-condition or amine-by-condition term is identifiable and none can be
absorbed by a pair term. It is also the one condition whose membership is not
conditioned on a reaction having already failed elsewhere. `all` is the whole
table with the condition as a covariate, and is where the condition-expanded
contrast lives.

Condition encoding is `chemistry`: the 7 (reagent, base) combinations actually
run, after stripping the substrate counterions and merging the two HATU
depictions. The 8-level `protocol` encoding is a registered sensitivity and is
the *more conservative* of the two, since more condition levels means a strictly
more flexible additive baseline.

### The endpoint

**Primary: continuous.** `y = Conversion / 100`, in [0, 1], every measured row,
loss and metric MSE on that scale. It is the measured quantity, it is present
everywhere, it is not model-derived, and -- the reason the 60.8 % zero mass does
not disqualify it -- the load-bearing statistic is a *ratio* of two models'
errors on identical rows, in which a shared noise floor attenuates the effect
towards zero rather than manufacturing one. Binarising at 20 % would discard
exactly the information a pair term is most likely to carry: how well a pair
couples, not merely whether.

**Secondary: binary feasibility**, `Conversion >= 20`, the authors' own
documented rule (SI 2.2.1, Fig. 5a caption, `train.py:95`). Not re-tuned, not
re-derived. Fitted as a genuine classifier -- the same architectures with a
logistic output and a Bernoulli loss -- so that its incremental metric is against
the same additive baseline in the same function class.

Reported beside every number: the replicate ceilings, R2 ~ 0.49 (`all`) and
~ 0.54 (`hatu`), and binary accuracy ~ 0.89.

### The split

Held out is a **reactant**, and the unit of holdout is a *group*, not an entity.
Groups are the transitive closure, within a role, of three outcome-independent
relations: same stereo-stripped canonical SMILES, same neutralised stereo-stripped
SMILES, or byte-identical primary fingerprint. 272 acids collapse to 259 groups
and 230 amines to 225. Entities stay distinct everywhere else. This is not
congener clustering and it does not hold out chemical families -- two merely
similar entities stay separate.

Acid groups and amine groups are each cut into **k = 5** folds, stratified by the
group's total row count, over **3 independent partitions** -- 15 authoritative
folds, and every entity is a test entity 3 times. Within a fold, group `f` is
test and group `(f+1) mod 5` is validation, in both roles independently. A row's
bucket is decided by the pair of role memberships, and there are nine of them:

| acid role | amine role | bucket |
|---|---|---|
| train | train | `train` |
| val | train / val / test | `val_e1a` / `val_e2` / `test_e2_mixed` |
| train | val / test | `val_e1n` / `test_e1n` |
| test | train / val / test | `test_e1a` / `test_e2_mixed` / `test_e2` |

Naming all nine is the point. Phase 3's adversarial review found that its E1
bucket silently mixed two regimes -- a test drug meeting a training drug and a
test drug meeting a *validation* drug, which appears in no training pair either
-- while every document asserted E1 was homogeneous. Here the distinction is in
the type system.

Observed geometry of the authoritative folds on `all`, mean over 15 (min-max):

    train           4,215  (4,173 - 4,254)
    val_e1a         1,393  |  val_e1n  1,393  |  val_e2    476
    test_e1a        1,393  (1,346 - 1,434)
    test_e1n        1,393  (1,361 - 1,449)
    test_e2           476  (458 - 504)
    test_e2_mixed     929  (892 - 959)
    test acids 54, test amines 46 per fold

**E1-A** is `test_e1a`: unseen acid, trained amine. **E1-N** is `test_e1n`:
unseen amine, trained acid. **E2** is `test_e2`: neither seen. They are never
pooled -- three different questions with different sample sizes, and it is
chemically plausible that one role generalises far better than the other.
`test_e2_mixed` has both endpoints absent from every training row, so it answers
E2's question with twice the rows, but one endpoint is a validation entity whose
rows informed hyperparameter selection. Reported separately, never promoted.

k = 5 rather than 8 or 10 is a deliberate trade of training rows for E2 power.
The three are the same quantity: the val-by-test cross cells that make
`test_e2_mixed` are exactly the size of the E2 cell, so any k that shrinks one
shrinks the other. Phase 3's central limitation was an E2 of 45 pairs per fold;
choosing a dataset to fix that and then choosing a k that does not is not a
trade-off, it is a mistake.

**Validation is entity-OOD in both roles.** Selection sees `val_e1a`, `val_e1n`
and `val_e2` and nothing else. Tuning on held-out *pairs* among training entities
and reporting entity-OOD test numbers selects the best hyperparameter for
transductive completion and reports it as extrapolation; that is a fairness
mistake this project has already made once.

Conditions are never held out. Phase 4 asks about entity extrapolation, and
holding out a condition would confound the two.

### The representation

ECFP4, Morgan radius 2, 2,048 bits, RDKit, **chirality on**. Separate matrices
for the two roles, and separate encoders in every model: an acid and an amine do
not share a latent space any more than they share a role.

Chirality on is a change from Phase 3, which had it off. It matters here because
ignoring stereochemistry merges 11 acids and 4 amines into entries they are only
stereoisomers of, and the split-group rule already prevents those merges from
straddling a fold -- so the only thing turning chirality off would buy is a
weaker representation.

Deliberately no learned entity ids for held-out reactants, and deliberately no
pretrained chemical language model. ChemBERTa and MolFormer were trained on
corpora nobody in this repository can audit for overlap with a 2024 HTE deposit,
and a positive result from an unauditable representation is worth less than a
negative from an auditable one. If ECFP4 gives a negative for distant entities,
**that is the result to classify, not a cue to try a bigger encoder** -- the
point of Phase 4 is external validation, not a leaderboard.

### The model ladder

Small on purpose. Every rung that adds a term adds it with the term's output
tensor zero-initialised, so at initialisation the richer model *is* the simpler
one, exactly, and the incremental skill it earns is attributable to what was
added rather than to a luckier fit of what was already there.

| rung | form | role |
|---|---|---|
| `condition_only` | `mu + f_C(c)` | diagnostic reference |
| `additive` | `+ f_A(x_a) + f_N(x_n)` | **the baseline** |
| `lowrank` | `+ z_A^T W z_N` | **the hypothesis** |
| `flexible` | `+ G([p_A(x_a), p_N(x_n), e_C(c)])` | is low rank helping or merely limiting |
| `condition_expanded` | `+ u_A^T V_A[c] + u_N^T V_N[c]` | the condition-aware baseline |
| `condition_expanded_pair` | `+ z_A^T W z_N` on top of that | **the robustness contrast** |
| `transductive` | free per-entity embeddings, `+ u_i^T v_j` | a labelled ceiling, never a generalisation result |

Hyperparameters are chosen **per (model, fold) on the entity-OOD validation
buckets only**: weight decay in **{1e-3, 3e-3, 1e-2, 3e-2, 1e-1}** for every
rung, and rank in **{2, 4, 8}** for every rung that has one. Weight decay is
searched for the baseline too -- searching it for only the pair model would be an
asymmetry landing directly in the ratio they are compared by. On development
folds the entity-OOD validation optimum sat at 1e-2 for both rungs, interior to
the grid, with 1e-1 clearly worse; the grid brackets it.

Training is full-batch Adam, lr 3e-2, cosine-annealed over **800 epochs**, 2
restarts, best epoch chosen on validation. 800 because on development folds the
selected epoch averaged 186 for the additive rung and 71 for the pair rung with a
maximum of 595, and validation loss moved by 1e-4 between a 600- and a
1200-epoch budget. The same optimiser, schedule, budget, restart count and
early-stopping rule are used for every rung. The only thing that differs between
two rows of a contrast is the model class.

No shrinkage calibration. Phase 3 fitted one on validation and its own audit
found the coefficient biased upward, because the same validation rows had already
chosen the stopping epoch and the grid member. It existed there to cope with ~35
validation pairs; the smallest selection bucket here is ~476 rows.

### The primary contrast and the primary metric

    incremental pair skill = 1 - MSE(lowrank) / MSE(additive)

from **paired predictions on identical rows**, in identical folds, from models
fitted on identical training sets. Not a difference of two separately reported
skills, and emphatically not a skill against zero: Phase 3's registered validity
gate read skill-against-zero and fired on a control containing no chemistry,
which scored +0.204 that way and -0.0007 measured as an increment.

Reported for every regime: MSE, RMSE, MAE, Pearson, Spearman, R2, raw skill
against the fold's own additive baseline, and incremental pair skill. For the
binary endpoint: AUROC, AUPRC, balanced accuracy, log loss, Brier, and
incremental skill on log loss and on Brier against the same additive baseline.

The **robustness contrast** is `condition_expanded_pair` against
`condition_expanded`, on the `all` screen. It is the scientifically stronger of
the two, because it asks whether pair structure remains after each substrate has
been allowed to interact with the condition. On the `hatu` screen it does not
exist -- with one condition there is nothing for `AC` or `NC` to be -- and that
absence *is* the point: any pair gain there cannot be condition compatibility.

### The unit of inference

For **E1-A** the unit is a **held-out acid**; for **E1-N** a **held-out amine**;
for **E2** the fold. Reaction rows sharing one held-out acid are not independent
evidence -- they share a substrate and often a plate, and a single acid carries
up to 200 rows. Per-entity incremental skill is computed within the entity's own
test rows (minimum 3, entities below that reported and flagged), averaged across
that entity's 3 turns, and summarised with mean, SD, paired 95 % CI, paired t,
Wilcoxon, and the count of entities favouring the pair model. Fold-level pooled
numbers are reported alongside.

### Controls, all measured on incremental pair skill

Phase 3 established why the statistic matters more than the control: against a
weak baseline every control looks like a success.

* **A — shuffled acid features.** The real fingerprints permuted among acids.
* **B — shuffled amine features.**
* **C — both shuffled**, independently.
* **D — random features**, Bernoulli, per-entity density matched, drawn once
  before any split exists.

All four run through the identical pipeline, appear in the same tables as the
real representation, and are permuted once rather than per fold.

### The positive control

One synthetic target on the **exact ChemLex entity graph and the exact real
features**: additive terms that are linear functions of the real fingerprints,
plus a planted rank-3 acid-amine interaction that is also a linear function of
them, plus unit noise. The entity-OOD pipeline must recover positive incremental
pair skill on it, that recovery must depend on the real molecular features, and
it must collapse under `shuffled_both`. Without this, a negative chemistry result
cannot be distinguished from a broken evaluation geometry.

Run at **three planted sizes** -- interaction scale 0.25, 0.5 and 1.0 relative to
the additive part -- rather than one, because the number that bounds a negative
result is not "the pipeline can find a huge planted effect" but *the smallest
planted effect the pipeline can find*. The report states that floor and reads
every null against it. On development folds the scale-1.0 target (planted
interaction = 48 % of the target SD) was recovered at +0.117 / +0.093 / +0.067
incremental on E1-A / E1-N / E2 with real fingerprints and -0.003 / -0.008 /
-0.009 shuffled; the authoritative run repeats this on the authoritative folds.

One compact control, not another synthetic phase.

### The blind-entity diagnostic

Ported from Phase 3 in its **corrected** form. For E1-A, the held-out acid's
feature row is replaced by the **mean over the training acids** -- an
on-distribution, information-free stand-in -- and for E1-N the analogous
operation over training amines. The same blinding is applied to the baseline
*and* the pair model, and incremental pair skill is recomputed. The quantity of
interest is

    incremental(full) - incremental(blinded)

If the pair advantage does not fall, whatever it is using is not the unseen
reactant's structure.

**Not a zero vector.** Phase 3 used one and it manufactured a result: zeros
assert "this molecule has no substructures at all", a point no real molecule
occupies, and against that baseline a random-feature control containing no
chemistry scored a significant "attributable to the unseen drug" effect at
p = 0.049. That claim was withdrawn.

### The additive-projection diagnostic

Mandatory before any interaction claim. A bilinear form contains, as special
cases, functions of the acid alone and of the amine alone, so a pair model can
beat the additive baseline by fitting the *substrate* effects better rather than
by learning any pair structure.

The pair model's **predictions** are projected onto free per-entity additive
effects -- `acid + amine + condition` indicator columns, strictly more flexible
than the additive model's feature-derived heads -- using no outcome at all. Then
`incremental` is recomputed for the projected prediction. If projecting away the
non-additive component costs nothing, the pair term has demonstrated no
interaction. For the condition-expanded rungs the projection is onto
`acid + amine + condition + acid:condition + amine:condition`.

### Analogue dependence, pre-registered rather than post-hoc

Phase 3 ended at ANALOGUE-ONLY TRANSFER, so this is a primary question here, not
an appendix. For every held-out entity in every fold, `max Tanimoto` to the
**training entities of the same role**, computed from features only.

Cut points are frozen **now**, from the authoritative folds' geometry with no
outcome touched. Role-specific tertiles, because the two roles live on visibly
different scales -- the mean held-out acid has a 0.47 nearest training analogue
and the mean held-out amine only 0.29:

| role | low | medium | high |
|---|---|---|---|
| acid | < 0.3741 | 0.3741 - 0.5674 | > 0.5674 |
| amine | < 0.2143 | 0.2143 - 0.3056 | > 0.3056 |

A second, coarser stratification at **fixed thresholds 0.35 and 0.55** applied
identically to both roles is registered alongside, so the two roles can be read
on one scale at the cost of unbalanced strata.

### Congener dependence

Twenty near-identical analogues are not twenty independent demonstrations. Acids
and amines are clustered independently by single-linkage ECFP4 Tanimoto at a
**frozen threshold of 0.6**: 272 acids to 196 families (largest 29) and 230
amines to 220 families (largest 2). 0.5 chains the acids into a 69-member blob
and 0.7 leaves the amines essentially unclustered, so 0.6 is chosen here and not
revisited. Every similarity statistic is reported under both an entity bootstrap
and a **congener-family bootstrap**.

### Condition robustness and interaction reuse

Incremental pair skill is reported stratified by reaction condition on the `all`
screen. If it is carried by one reagent, the claim is much narrower and will be
stated that way.

2,076 of the 8,064 pairs were measured under 2 or more conditions and 848 under
3 or more, so the reuse question -- is `I_AN` a substrate property or a
condition-specific nuisance? -- is answerable. It is asked as a pair-level rank
correlation of the fitted interaction across conditions, and with one caveat
registered in advance: most of the apparent cross-condition agreement in this
screen is agreement that the reaction failed. The correlation is therefore also
reported on the both-nonzero subset, and that subset is outcome-conditioned and
will be labelled as such.

### Registered predictions

Written before the authoritative run, so they can be wrong in public.

1. **`additive` will do well on unseen acids.** Structure predicts substrate
   potential; a pilot on a *development* fold reached R2 0.42 and Pearson 0.68 on
   E1-A. This is not the claim under test.
2. **E1-N will be harder than E1-A**, and by a lot. Amine identity explains
   eta2 = 0.55 of the outcome against 0.13 for acid identity, and the mean
   held-out amine's nearest training analogue is 0.29 Tanimoto against 0.47 for
   acids. When the amine is unseen the model loses its largest predictor and has
   less to interpolate from.
3. **The transductive ceiling will show pair structure.** If it does not, the
   inductive question is empty and the phase reports outcome E early.
4. Incremental pair skill will be **small in absolute terms** whatever happens --
   single-digit percent at most -- because the replicate ceiling is R2 ~ 0.49 and
   the additive baseline already captures the two large main effects.
5. The most likely outcome, on Phase 3's precedent and on the acid set's
   nearest-neighbour distribution, is **analogue-confined transfer on the acid
   side and nothing on the amine side.**

### Decision rule, evaluated in this order

Validity gates first. If any fires, the registered verdict is INCONCLUSIVE
regardless of everything below, and the reason is printed.

* **V1 — no leakage.** Zero folds may fail `assert_no_entity_leakage` or
  `assert_partition`. Tolerance is exactly zero, not a fraction.
* **V2 — the sweep ran.** Fewer than 10 % of attempted conditions may error.
* **V3 — the positive control works.** The planted rank-3 interaction must be
  recovered at incremental pair skill > +0.05 on E1-A and E1-N, and that recovery
  must fall by more than half under `shuffled_both`. If the pipeline cannot find
  an interaction that is definitely there, it cannot report that one is not.
* **V4 — the controls do not.** Mean incremental pair skill for each of the four
  controls must be **below +0.02**, measured as an increment and not as a skill
  against zero. Above +0.05 for any control, the run is invalid.

Then, per screen and per regime (E1-A and E1-N separately, E2 reported but never
decisive on its own):

* **(a)** mean per-entity incremental pair skill > +0.01;
* **(b)** paired t and Wilcoxon both p < 0.05 against zero;
* **(c)** more than half of held-out entities favour the pair model;
* **(d)** the advantage survives the blind diagnostic: `incremental(full) -
  incremental(blinded)` > 0 with a 95 % CI excluding zero;
* **(e)** the advantage survives the projection diagnostic: the gain is not
  wholly recoverable by projecting the pair model's predictions onto free
  additive effects;
* **(f)** on the `all` screen, the robustness contrast
  (`condition_expanded_pair` vs `condition_expanded`) also satisfies (a)-(c);
* **(g)** the **low** similarity stratum's mean incremental pair skill is > 0
  **and** significant at 0.05 under the **congener-family** bootstrap.

Criterion (g) is deliberately stronger than Phase 3's. Phase 3 registered "the
low stratum's mean is > 0, in one screen, with no significance requirement",
which passed on +0.016 with a CI spanning zero -- a bar written too low before
the regime was understood, and one that would have let a badly drafted criterion
overrule the evidence it existed to test. It is not repeated.

Classification:

| condition | verdict |
|---|---|
| (a)-(g) hold in **both** screens, in **both** E1 regimes, and E2 also satisfies (a)-(c) | **BROAD CHEMICAL ENTITY TRANSFER** |
| (a)-(f) hold in at least one screen and one E1 regime, but (g) fails | **ANALOGUE-ONLY CHEMICAL TRANSFER** |
| the additive model transfers (R2 > 0 on held-out entities) but (a) or (e) fails everywhere | **SUBSTRATE/CONDITION-ONLY TRANSFER** |
| the transductive ceiling shows pair structure but no feature-based rung satisfies (a) in any regime | **TRANSDUCTIVE-ONLY PAIR STRUCTURE** |
| even the transductive ceiling gives no stable gain over its own additive baseline | **NO REUSABLE PAIR STRUCTURE** |
| a validity gate fires, or the criteria conflict irreconcilably between screens | **INCONCLUSIVE** |

If a registered gate is later shown to be **defective**, the frozen verdict stays
frozen and the corrected reading is reported beside it, labelled post-hoc, with
the single change named. That is exactly what Phase 3 did when its control gate
turned out to read the wrong statistic, and it is what keeps the registration
worth having.

### Registered sensitivities, decided now, reported whatever they show

1. `protocol` (8-level) condition encoding instead of `chemistry` (7-level).
2. Replicate cells aggregated to their mean (11,151 cells) instead of 11,669 rows.
3. Restricted to the 195 amine entities whose N-H is a classical amine
   (9,787 rows), excluding sulfonamides, amidines, hydroxylamines, thioamides,
   hydrazides, phosphoramides and amides.
4. Incremental pair skill among rows with Conversion > 0 only. **Outcome-
   conditioned** -- the rows are selected using the label -- so it is a labelled
   diagnostic and never a headline.
5. `k = 8` folds instead of 5, same seed stream, to show the conclusion is not an
   artefact of one fold geometry.

### Things that could go wrong, named in advance

* **The screen is not a random sample of acid-amine space.** ~48 % of it was
  chosen to fail. If pair structure exists only among reactions nobody thought to
  design against, this screen cannot see it.
* **The noise ceiling may swallow the effect.** With R2 capped near 0.49, an
  incremental pair skill of +0.02 on the observable scale is +0.04 of the
  recoverable signal, and the phase may be unable to distinguish a small real
  effect from none. The positive control is what bounds this: it fixes how large
  a planted effect has to be before the pipeline sees it.
* **The amine main effect may dominate everything.** eta2 = 0.55 for amine
  identity means the additive baseline starts from a strong position, and a pair
  term has little room above it.
* **The two screens are nested.** `hatu` is a subset of `all`; agreement between
  them is not replication. Phase 3's A375 and PANC1 were genuinely different cell
  lines and this is not that.
* **Selection may be misaligned with the test regime.** Hyperparameters are
  chosen on pooled entity-OOD validation and reported on three separate test
  regimes. A registered sensitivity with regime-matched selection is run if the
  pooled and per-regime winners disagree materially.

---
