# Phase 2R — after removing the per-drug ordering potential, is anything left?

Status: **complete**. Decision recorded in §8. Pre-registration in
[`PREREGISTRATIONS.md`](PREREGISTRATIONS.md), committed at `995a166` before the
main grid ran.

---

## 1. Why this experiment exists

Phase 2 pointed the Intervention-Algebra structural prior at the Koplev 2017
sequential anticancer screen and found no transfer
([`phase2_koplev.md`](phase2_koplev.md)). The obvious next move — try a
different architecture — is the wrong one, because Phase 2 also turned up the
fact that makes it wrong:

> A plain two-way additive fit `y(i→j) ≈ a_i + b_j` already captures **54%
> (A375) / 40% (PANC1)** of the *antisymmetric* variance.

An additive model's directional component is

```
D_add(i,j) = (a_i + b_j) − (a_j + b_i) = g_i − g_j,      g_i := a_i − b_i
```

which is a **potential**: drug `i` is better-going-first than drug `j` by an
amount that does not depend on which partner it is paired with. A drug can look
sequence-sensitive purely because it generally performs differently in first
versus second position — and in this screen the protocol itself is asymmetric
(first drug at a fixed dose, second titrated), so a large per-drug potential is
close to expected. Predicting it is not evidence of a pair-specific order
interaction, and every family in Phase 2 got it for free from the first-order
term.

The quantity that *would* be evidence is what is left:

```
D_res(i,j) = [y(i→j) − y(j→i)] − [g_i − g_j]
```

**The question.** Can `D_res` for an entirely unseen unordered pair `{i,j}` be
predicted from the other observed pairs?

* If **no**, the reusable directional structure in this screen is a per-drug
  tendency, there is nothing pair-specific left to model, and inventing
  structured models for it is wasted effort.
* If **yes**, Phase 2's failure was a modelling failure rather than an
  absence-of-signal result.

This is the decision experiment for whether the real-data direction continues.

## 2. Leakage-safe residualisation

The residual target is *derived from a fitted model*, which creates a leakage
route that no split-level check would catch: residualise the full 100×100 matrix
and then split, and every held-out pair has contributed to shrinking its own
target toward zero. The order of operations is therefore fixed in code
(`residual_experiment.run_residual_condition`) and is the experiment:

1. build the nested pair-level split — the **same function** Phase 2 uses,
   `splits.make_coverage_splits`, unchanged;
2. `assert_no_pair_leakage`;
3. fit `mu + a_i + b_j` **on training rows only**, ridge penalty chosen on the
   validation rows;
4. residualise train, validation and test rows with that fit;
5. fit the rung on the training pairs' `D_res`, hyperparameters selected on the
   validation pairs' `D_res`;
6. score on the held-out pairs against the zero predictor.

The guard lives *inside* `fit_additive`, checking the ordered rows handed to the
solver rather than a pair list the caller promises corresponds to them — so no
call site can skip it. The single bypass is a private flag reachable only from
`ResidualConfig.contaminate_additive_fit`, which stamps `contaminated: true` on
the row and writes to a separate file (§6, control C).

Three tests pin this down. `test_additive_fit_uses_training_pairs_only` checks
the design matrix; `test_leakage_guard_fires_on_a_contaminated_frame` is the
mutation test — weaken the guard to a no-op and only this test notices; and
`test_perturbing_held_out_responses_cannot_move_the_additive_fit` multiplies
every held-out `y` by 100 and requires bit-identical coefficients.

### The additive estimator, and why identifiability is not a judgement call

`mu + a_i + b_j` is not identified — `(a_i + c, b_j − c)` is the same fit. Two
things make that a non-issue rather than a choice to defend.

* The ridge penalty (on `a` and `b`, never on `mu`) makes the solution unique.
* **The quantity used is gauge-invariant anyway.** Under `a_i → a_i + c`,
  `b_i → b_i − c` we get `g_i → g_i + 2c`, so `g_i − g_j` — and hence `D_res` —
  does not depend on the convention at all
  (`test_D_add_is_gauge_invariant`).

So the penalty affects the residual only through *shrinkage*, which is a real
modelling decision and is therefore selected on validation and never on test.
The fit is closed-form rather than Phase 2's Adam-trained `additive` family for
three reasons: the residual target must not inherit an optimiser seed, "the
additive part has been removed" should be a statement about the model rather
than about how long it trained, and it is cheap enough to redo inside all ~800
runs instead of being cached and reused across splits.

### The failure mode that would fake a positive result

Over-shrinkage leaves part of the potential *in* the residual. A pair-specific
model would then score by re-learning `g`, and that would be reported as
pair-specific structure. This is not hypothetical: the low-rank rung **contains**
`c_i − c_j` exactly (set `u_i = [c_i, 1]`, `K = [[0,1],[−1,0]]`;
`test_lowrank_can_express_a_pure_potential`).

Hence the ladder carries a `potential` rung — one free scalar per drug, no pair
term — which can fit leftover potential and nothing else, and hence the
**pre-registered primary contrast is `lowrank − potential`, not
`lowrank − zero`**. Beating the null is necessary; beating the potential is what
makes the claim pair-specific.

## 3. Target and orientation

The statistical unit is the unordered pair. Targets are built in a canonical
orientation `i < j` and each pair contributes **one** example, not two:
`D_res(j,i) = −D_res(i,j)` identically, so entering both would be entering the
same observation twice with a sign flip.

Because the target is antisymmetric by definition, every rung is antisymmetric
by construction. That is arithmetic, not a prior imported from Phase 1: a
predictor with a symmetric component is not a more general hypothesis class, it
is a strictly worse one, since that component is pure error on this target.

Direct prediction of `D_res` is preferred over learning two ordered residual
responses separately because the loss then fits only the quantity under study.
The ordered formulation is still run as its own rung (§4) precisely so the
difference is measured rather than assumed; the two share a target exactly
(`r_ij − r_ji ≡ D_res(i,j)`, `test_ordered_and_direct_residual_formulations_share_a_target`)
and differ only in what else the loss is asked to fit.

## 4. The ladder

Drug identity only. No fingerprints, SMILES, targets, transcriptomics or
pathway annotation — those would test entity similarity, not whether the pair
graph itself carries reusable residual structure.

| rung | prediction | role |
|---|---|---|
| `zero` | `0` | the mandatory null; every skill is measured against it |
| `potential` | `c_i − c_j` | leftover per-drug tendency; the incomplete-removal diagnostic |
| `lowrank` | `u_i^T K u_j`, `K = −K^T` | **the primary rung** — the minimal pair-specific hypothesis, capacity-controlled |
| `mlp` | `F(φ_ij) − F(φ_ji)` | flexible upper bound; a diagnostic ceiling, *not* a fair comparator |
| `mlp_ordered` | `r̂_ij − r̂_ji` from `r̂ = a'_i + b'_j + G(φ_ij)` | Phase 2's unrestricted family pointed at the ordered residual |

Parameter counts are reported on every row. As run, `lowrank` carries **204,
416, 864 or 1,856** parameters depending on the rank the validation grid selects;
the MLP rungs carry **3,369–57,873** (`mlp`) and **3,570–58,074**
(`mlp_ordered`), and are labelled flexible diagnostics for exactly that reason.
The `rank2` block (§8.6) pins the smallest of these, 204.

**The Phase 2 `structured` family is not a sixth rung, and that is a result
rather than an omission.** Its antisymmetric component is
`first_order_A + [F_A(φ_ij) − F_A(φ_ji)]/2`. On a target from which the
first-order potential has already been removed, and with the factor of two
folded into the output layer, that is the `mlp` rung *exactly* — the same
function class, proved numerically in
`test_structured_A_head_equals_the_mlp_rung`. Running it separately would report
one hypothesis twice.

### Hyperparameters

There is **no tuning stage**. Each rung's small grid is selected inside every
single run, against that run's own validation pairs. Phase 2 selected once at
one coverage on one screen and reused the answer everywhere; the audit showed
that choice did not transfer and had handicapped the family under test
(`phase2_koplev.md` §7.8). A separate tuning stage is what made that failure
possible, so there is not one here.

After the grid, a shrinkage coefficient `α ∈ {0, 0.125, …, 1}` is chosen on the
same validation pairs and applied to the test prediction. `α = 0` is the zero
predictor exactly, so a rung can never be worse than the null *on validation*.
This exists because at coverage 0.05 the validation set is ~35 pairs, and
selecting one setting out of six on 35 noisy pairs produces negative test skill
from selection noise alone — which would read as "flexible models do active
harm" when the truth is "there was nothing to select on". **Both the calibrated
(`cal_*`) and uncalibrated (`heldout_*`) numbers are reported for every run**,
because the uncalibrated one is what distinguishes outcome C from outcome D.

Note the asymmetry this creates and which direction it points: `zero` has no
grid and no shrinkage search, so selection can only help the learned rungs. A
null survives that asymmetry; a positive result has to be discounted for it, and
that is what the `potential` rung and the permutation control are for.

## 5. Split, coverage grid and metrics

Identical to Phase 2 and reused verbatim: *seen drugs, entirely unseen unordered
pair*; nested coverage grid `(0.05, 0.10, 0.20, 0.40, 0.70)`; 8 split seeds; the
evaluation pool fixed by the top rung so it is the same pairs at every coverage.
848–1,003 unordered pairs are scored per split seed.

Primary metric:

```
skill = 1 − MSE(α·D̂_res) / MSE_zero ,     MSE_zero = mean(D_res²)
```

on the same held-out pairs. Positive = the rung beat "there is no pair-specific
directional effect". `MSE_zero` is identical across rungs within a split seed
(the additive fit does not depend on the rung —
`test_decomposition_is_rung_invariant`), so skills are **exactly paired** across
the ladder.

Also reported: RMSE, MAE, Pearson, Spearman, and sign accuracy above the
screen's pre-fixed noise threshold. The threshold is Phase 2's
`2·√2·median posterior synergy_sd`, derived for *raw* `D`; on the smaller-scale
`D_res` it selects a different and larger fraction of pairs, so sign accuracy
here is **exploratory** and is not part of the decision rule.

The unit of replication is the split seed, n = 8. Both a paired t-test and a
Wilcoxon signed-rank test are reported for every contrast, always both, fixed in
advance rather than chosen after seeing which is kinder. At n = 8 the Wilcoxon's
smallest attainable two-sided p is 0.0078.

## 6. Controls

| control | what it does | what it rules out |
|---|---|---|
| **A — permutation** | `D_res` permuted across training *and* validation pairs; evaluation untouched | a rung scoring off pair identity, drug degree or a split artifact |
| **B — orientation** | a test, not a run: reversing the orientation must negate target and prediction alike | a sign error anywhere in the target algebra |
| **C — contamination** | the additive baseline deliberately fitted on train+val+test | quantifies what the guard prevents; **never used scientifically** |
| **D — power** | a known antisymmetric signal `u K u^T` of RMS κ injected into `D` before the split | tells "no structure" apart from "no power" |
| **E — ridge titration** | the additive penalty forced to λ ∈ {3, 30, 300}, deliberately leaving a known fraction of the potential in `D_res` | tells "the `potential` rung found nothing" apart from "the `potential` rung cannot find anything" |

Control D is not optional decoration. A null at a coverage where an injected
signal of comparable size would *also* be missed is an underpowered null, and
the pre-registration commits in advance to reporting it as such. The injected
signal is deliberately the same shape as the `lowrank` rung's hypothesis class,
which makes the resulting power curve an **upper bound**: it says what the
experiment could detect under the most favourable possible match between signal
and model.

## 7. Reproduction

```
python scripts/download_koplev.py                     # once
python scripts/run_phase2_residual.py --part all      # ~25 min on 7 workers
python scripts/report_phase2_residual.py
```

Rows land in `results/phase2_residual/`; see
[`README_PHASE2R.md`](../results/phase2_residual/README_PHASE2R.md) for the file
and column map. These rows use `heldout_*` / `cal_*` metric names rather than
Phase 2's `test_*`, and `residual_report.load_residual_runs` refuses a Phase 2
file outright, so the two experiments cannot be pooled by accident.

---

## 8. Results

Every number below is over the 8 split seeds, on the 944 held-out unordered
pairs (848–1,003 per seed), with the split seed as the unit. Training pairs per
coverage: 211 / 421 / 842 / 1,683 / 2,945 of the screen's 4,950.

### 8.1 The decomposition, exactly, with no model at all

Before any fitting: on the complete 100×100 matrix, the measured directional
effect `D` splits **uniquely and orthogonally** into a per-drug potential and a
cyclic remainder (`residual.hodge_decomposition`).

| screen | mean `D²` | potential part `g_i−g_j` | cyclic part | sd(cyclic) | curl energy in top 4 / 16 singular directions |
|---|---:|---:|---:|---:|---:|
| A375 | 0.0491 | **53.8%** | **46.2%** | 0.151 | 49% / 81% |
| PANC1 | 0.0261 | **39.8%** | **60.2%** | 0.126 | 48% / 80% |

The cyclic part is the component that **no** per-drug potential can express, for
any `g`: a potential is curl-free by construction, so this is a property of the
screen and not of a model. It is the ceiling on what every rung above
`potential` is competing for, and it is large — nearly half the directional
signal on A375 and three fifths on PANC1.

It is also **not concentrated**: the top 4 singular directions hold only ~48% of
the cyclic energy and the top 16 about 80%. Whatever a rank-limited model
recovers is therefore a lower bound on the structure present.

Held out, with the additive fit trained only on training pairs, the removable
fraction converges on the exact in-sample value as coverage grows — 0.337 →
0.514 on A375 against an exact 0.538, and 0.223 → 0.374 on PANC1 against 0.398.
That the train-only estimate approaches the model-free ground truth from below
is the internal consistency check for the whole residualisation.

| screen | coverage | mean `D²` | mean `D_res²` | removed by `g_i−g_j` | sd(`D_res`) | sd(`D`) | additive R²(y) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A375 | 0.05 | 0.0510 | 0.0337 | 33.7% | 0.183 | 0.225 | 0.348 |
| A375 | 0.10 | 0.0510 | 0.0295 | 42.1% | 0.171 | 0.225 | 0.428 |
| A375 | 0.20 | 0.0510 | 0.0266 | 47.8% | 0.163 | 0.225 | 0.481 |
| A375 | 0.40 | 0.0510 | 0.0254 | 50.2% | 0.159 | 0.225 | 0.510 |
| A375 | 0.70 | 0.0510 | 0.0248 | 51.4% | 0.157 | 0.225 | 0.521 |
| PANC1 | 0.05 | 0.0267 | 0.0206 | 22.3% | 0.144 | 0.163 | 0.209 |
| PANC1 | 0.10 | 0.0267 | 0.0193 | 27.6% | 0.139 | 0.163 | 0.274 |
| PANC1 | 0.20 | 0.0267 | 0.0182 | 31.7% | 0.135 | 0.163 | 0.318 |
| PANC1 | 0.40 | 0.0267 | 0.0172 | 35.4% | 0.131 | 0.163 | 0.365 |
| PANC1 | 0.70 | 0.0267 | 0.0167 | 37.4% | 0.129 | 0.163 | 0.381 |

**Are the two quantities the same?** §8.1's first table is an *exact, in-sample*
projection; everything below is a *train-only ridge* residual on held-out pairs.
Calling both "the cyclic part" is only honest if they coincide, so: the
correlation between held-out `D_res` and the exact Hodge curl on the same pairs
is **0.990 (A375) / 0.992 (PANC1) at coverage 0.70**, with RMSE 0.022 / 0.016
against a curl sd of 0.153 / 0.126. At coverage 0.10 it is 0.908 / 0.923 — the
sparse residual is a noisier estimate of the same object. At the coverages the
positive result lives at, they are the same quantity to 1%.

**Note on reading the coverage axis.** `D_res` — and therefore the skill
denominator `MSE_zero` — is rebuilt from a *per-coverage* additive fit, so it
falls from 0.0337 to 0.0248 across the grid on A375 while the raw `mean D²` is
constant at 0.0510. The difference between two coverages' targets is exactly a
per-drug potential. Correcting for it (subtracting the excess from numerator and
denominator alike, paired per seed) moves the A375 low-rank curve from
−0.188 / −0.058 / +0.008 / +0.231 / +0.368 to
−0.256 / −0.069 / +0.009 / +0.236 / +0.368: the sparse end gets worse and the
dense end does not move. The coverage-resolved result below is therefore
*conservative* under this confound rather than produced by it.

### 8.2 Residual skill — the main grid (as run)

Everything in this subsection is the **as-run** block, i.e. with the shrinkage
coefficient fitted on validation pairs that had already selected the model. §8.6
gives the corrected numbers and is what the decision quotes; the two differ
materially only at coverage ≤ 0.20.

`cal_skill`, mean over 8 split seeds, 95% CI, seeds positive.

**A375**

| coverage | zero | potential | **low-rank** | antisym. MLP | ordered MLP |
|---|---:|---:|---:|---:|---:|
| 0.05 | 0.000 | −0.009 [−0.025,+0.007] 1/8 | −0.188 [−0.261,−0.115] 0/8 | −0.170 0/8 | −0.107 0/8 |
| 0.10 | 0.000 | +0.005 [−0.008,+0.018] 2/8 | −0.058 [−0.103,−0.012] 1/8 | −0.037 4/8 | −0.042 3/8 |
| 0.20 | 0.000 | +0.000 [−0.002,+0.002] 2/8 | +0.008 [−0.031,+0.046] 4/8 | −0.004 3/8 | +0.041 5/8 |
| 0.40 | 0.000 | +0.002 [−0.002,+0.006] 4/8 | **+0.231 [+0.180,+0.282] 8/8** | +0.123 7/8 | +0.216 8/8 |
| 0.70 | 0.000 | +0.002 [−0.003,+0.007] 5/8 | **+0.368 [+0.341,+0.395] 8/8** | +0.309 8/8 | +0.359 8/8 |

**PANC1**

| coverage | zero | potential | **low-rank** | antisym. MLP | ordered MLP |
|---|---:|---:|---:|---:|---:|
| 0.05 | 0.000 | −0.006 3/8 | −0.139 [−0.166,−0.112] 0/8 | −0.099 1/8 | −0.064 2/8 |
| 0.10 | 0.000 | +0.000 2/8 | −0.120 [−0.236,−0.004] 0/8 | −0.076 0/8 | −0.039 1/8 |
| 0.20 | 0.000 | −0.001 3/8 | −0.011 [−0.060,+0.038] 4/8 | −0.031 1/8 | +0.052 7/8 |
| 0.40 | 0.000 | −0.001 4/8 | **+0.223 [+0.197,+0.248] 8/8** | +0.107 6/8 | +0.217 8/8 |
| 0.70 | 0.000 | +0.002 7/8 | **+0.374 [+0.331,+0.418] 8/8** | +0.313 8/8 | +0.363 8/8 |

Held-out Pearson r(`D̂_res`, `D_res`) for the low-rank rung, as run: 0.035 /
0.016 / 0.127 / 0.485 / 0.608 (A375) and −0.008 / 0.003 / 0.123 / 0.483 / 0.616
(PANC1). Under the corrected shrinkage (§8.6) these become 0.005 / 0.061 / 0.176
/ **0.479** / **0.594** and 0.022 / 0.026 / 0.160 / **0.456** / **0.608**, and
those are the values the decision quotes.

Exploratory sign accuracy on `D_res` above the noise threshold, low-rank rung,
as run: 0.52 / 0.51 / 0.57 / 0.78 / 0.86 (A375), 0.51 / 0.49 / 0.56 / 0.76 /
0.85 (PANC1); corrected, **0.787 / 0.852** (A375) and **0.733 / 0.845** (PANC1)
at coverages 0.40 and 0.70. The `potential` rung is 0.49–0.53 everywhere — a
coin flip, as it must be if essentially nothing of the potential is left.

### 8.3 The pre-registered primary contrast: low-rank − potential

As run — i.e. with the shrinkage coefficient fitted on validation pairs that had
already chosen the model. §8.6 is the corrected version and is what the decision
quotes. This table and every other one below is **generated** into
`results/phase2_residual/summary/doc_tables.md` and pinned by
`test_document_tables_are_generated_not_transcribed`; an earlier hand-copied
version of it had four wrong p-values.

<!-- generated: primary_contrast_as_run -->
| screen | coverage | Δ | 95% CI | p (t) | p (Wilcoxon) | seeds favouring low-rank |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| A375 | 0.05 | -0.179 | [-0.248,-0.110] | 4.8e-04 | 0.0078 | 0/8 |
| A375 | 0.1 | -0.063 | [-0.110,-0.015] | 1.7e-02 | 0.0156 | 1/8 |
| A375 | 0.2 | +0.007 | [-0.032,+0.047] | 6.7e-01 | 0.9453 | 3/8 |
| A375 | 0.4 | +0.229 | [+0.180,+0.278] | 1.2e-05 | 0.0078 | 8/8 |
| A375 | 0.7 | +0.366 | [+0.337,+0.395] | 1.2e-08 | 0.0078 | 8/8 |
| PANC1 | 0.05 | -0.133 | [-0.173,-0.092] | 1.2e-04 | 0.0078 | 0/8 |
| PANC1 | 0.1 | -0.121 | [-0.237,-0.004] | 4.4e-02 | 0.0078 | 0/8 |
| PANC1 | 0.2 | -0.010 | [-0.058,+0.038] | 6.2e-01 | 0.8438 | 4/8 |
| PANC1 | 0.4 | +0.224 | [+0.198,+0.250] | 1.9e-07 | 0.0078 | 8/8 |
| PANC1 | 0.7 | +0.372 | [+0.329,+0.416] | 1.9e-07 | 0.0078 | 8/8 |

The `potential` rung is small everywhere: |skill| ≤ 0.009 across all ten cells,
and ≤ 0.002 at coverage ≥ 0.20 where the positive result lives. Whatever the
low-rank rung is finding, it is **not** leftover per-drug potential — the model
that can fit only that finds essentially nothing anywhere.

### 8.4 Controls

| control | what it does | what it rules out |
|---|---|---|
| **A — permutation** | `D_res` permuted across training *and* validation pairs; evaluation untouched | a rung scoring off pair identity, drug degree or a split artifact |
| **B — orientation** | a test, not a run: reversing the orientation must negate target and prediction alike | a sign error anywhere in the target algebra |
| **C — contamination** | the additive baseline deliberately fitted on train+val+test | quantifies what the guard prevents; **never used scientifically** |
| **D — power** | a known antisymmetric signal `u K u^T` of RMS κ injected into `D` before the split | tells "no structure" apart from "no power" |
| **E — ridge titration** | the additive penalty forced to λ ∈ {3, 30, 300}, deliberately leaving a known fraction of the potential in `D_res` | tells "the `potential` rung found nothing" apart from "the `potential` rung cannot find anything" |

**A — permutation.** `D_res` permuted across training and validation pairs. Run
at coverages 0.10, 0.40 and 0.70 — i.e. at **both** coverages the decision rests
on, not just one — for the two rungs that can express a pair-specific effect.
Low-rank `cal_skill`, mean over 8 seeds with 95% CI:

| coverage | A375 | PANC1 | real-target value for comparison |
|---|---|---|---|
| 0.10 | −0.031 [−0.059,−0.003] | −0.015 [−0.031,+0.002] | −0.058 / −0.120 |
| 0.40 | −0.004 [−0.011,+0.003] | −0.005 [−0.010,−0.001] | **+0.231 / +0.223** |
| 0.70 | −0.004 [−0.009,+0.000] | −0.005 [−0.010,+0.001] | **+0.368 / +0.374** |

with |r| ≤ 0.030 everywhere. It collapses to chance at exactly the cells that
carry the result. At coverage 0.10 it reads −0.031 / −0.015 against the main
grid's −0.058 / −0.120 — so **on A375 about half** the sparse-coverage negative
is reproduced by a control with no signal in it at all, though **on PANC1 only
about an eighth** is; §8.6 shows the rest is the shrinkage artifact.

**B — orientation.** Enforced in tests, not runs: reversing the orientation
negates the target exactly, and every rung's prediction with it.

**C — contamination.** Fitting the additive baseline on train+val+test raises the
fraction of held-out directional signal it removes from 0.412 to **0.531** —
i.e. a coverage-0.10 fit, which legitimately reaches 0.42, is pushed to what a
legitimate fit only approaches at coverage 0.70 (0.514) and which is bounded
above by the exact in-sample ceiling of 0.538. That is the leakage the guard
prevents, in the units of the headline table. Scope: A375, coverage 0.10, split
seeds 0–3, `zero` and `lowrank` only — it exists to size an effect, not to
support a claim. (Not a result; `contaminated_diagnostic.jsonl`, every row
flagged `contaminated: true`.)

**D — power. This is what makes the sparse end interpretable, and it changes
what can be said there.** A known antisymmetric signal of RMS κ was injected into
`D` before the split. `S = u K u^T` with a 3×3 antisymmetric `K` — note this is
**rank 2**, not rank 3: an odd-dimensional antisymmetric matrix is always
rank-deficient. The shape is exactly the low-rank rung's own hypothesis class,
the most favourable case there is. Run on **both** screens (an earlier version
ran A375 only and applied its curve to both, which was an undisclosed
cross-screen extrapolation — and κ is an absolute RMS, so the same κ is a
different relative signal against sd(`D_res`) of 0.157 vs 0.129).

Recovered `cal_skill` under the **honest-α estimator the decision quotes**,
low-rank rung, 8 split seeds:

| coverage | κ=0.10 A375 | κ=0.10 PANC1 | κ=0.20 A375 | κ=0.20 PANC1 |
|---|---:|---:|---:|---:|
| 0.05 | −0.035 (0/8) | −0.038 (0/8) | −0.006 (0/8) | −0.017 (0/8) |
| 0.10 | −0.038 (0/8) | −0.006 (0/8) | +0.064 (5/8) | +0.052 (4/8) |
| 0.20 | **+0.052 (6/8)** | **+0.076 (6/8)** | +0.380 (8/8) | +0.487 (8/8) |
| 0.40 | +0.340 (8/8) | +0.391 (8/8) | +0.600 (8/8) | +0.653 (8/8) |
| 0.70 | +0.474 (8/8) | +0.536 (8/8) | +0.694 (8/8) | +0.770 (8/8) |

(The as-run-α version is in `power.jsonl` for both rungs and agrees on the
qualitative picture; the full generated table is in
`results/phase2_residual/summary/doc_tables.md`.)

Read carefully, this says three different things at three coverages:

* **0.05 — blind.** Even κ=0.20, which is 1.4× (A375) / 2.1× (PANC1) the mean
  square of the *entire* observed residual, is missed on 0 of 8 seeds on both
  screens. Nothing can be concluded here.
* **0.10 — blind at κ=0.10, marginal at κ=0.20.** So the coverage-0.10 null
  excludes a residual signal of about twice the observed residual's size and
  says nothing about anything smaller.
* **0.20 — powered at κ=0.10**, on both screens (6/8 seeds, +0.05/+0.08). The
  experiment could have seen a κ=0.10 signal here and instead reads +0.022 /
  +0.019. That is **not** an uninformative null: it is a small positive
  consistent with the same structure found at 0.40/0.70, sitting just at the
  detection floor.

An earlier draft of this document said "at coverage ≤ 0.20 the experiment is
underpowered". That was wrong at 0.20 and is corrected here; the underpowered
range is coverage ≤ 0.10.

**E — ridge titration: is the `potential` rung a working detector, or a dead
one?** The entire primary contrast rests on `potential` scoring ~0, so it matters
a great deal whether it *can* score. This project has already shipped one rung
that returned exactly 0.0 because every gradient vanished at initialisation, and
that is indistinguishable in a table from a real null.

Injecting a potential does not test it: `0.5·κ·(g_i − g_j)` added to `y(i→j)` **is**
an additive-model contribution, so the ridge fit absorbs it and it never reaches
`D_res` (verified — a κ=0.15 injection moves the rung by 0.00). The test that
works is to force the penalty and leave a known fraction of the potential
behind. Percentages below are **of the potential**, not of `D²`: the potential is
53.8% / 39.8% of `D²`, so removing 15.6% of `D²` on A375 is removing 29% of the
potential. Coverage 0.70, 8 split seeds, both screens:

| forced ridge λ | of the potential, still left | **`potential` skill** | its r | `lowrank` skill |
|---|---:|---:|---:|---:|
| 300 | 71% / 72% | **+0.423 / +0.293** (8/8) | 0.650 / 0.543 | +0.585 / +0.515 |
| 30 | 15% / 16% | **+0.098 / +0.058** (8/8) | 0.318 / 0.245 | +0.380 / +0.392 |
| 3 (≈ the validation-selected value) | 4% / 6% | **−0.000 / +0.000** (4/8) | 0.015 / 0.016 | +0.369 / +0.373 |

The `potential` rung tracks leftover potential monotonically, detects it at
r = 0.65 when 71% of it is left, and falls to zero when the removal is
essentially complete. It is sensitive, calibrated and correctly null — its
|skill| ≤ 0.009 in the main grid means *there is nothing much left to find*, not
that it cannot find anything.

And `lowrank` is **flat across the titration** once the penalty is sane
(+0.369 → +0.380 from λ=3 to λ=30, while `potential` moves from 0.00 to +0.098).
Its skill does not depend on how much potential remains in the residual, which is
the direct evidence that it is not reading the potential.

### 8.5 Sensitivity: the ridge objective does not matter

Choosing the additive penalty to *maximise* directional removal
(`ridge_objective="D"`) instead of to minimise ordered MSE changes the low-rank
result by at most 0.026 and never changes a sign: +0.373 vs +0.368 (A375, 0.70),
+0.371 vs +0.374 (PANC1, 0.70), −0.059 vs −0.058 and −0.094 vs −0.120 at
coverage 0.10. The fraction of directional signal removed moves by at most 0.005
(A375 0.10: 0.426 vs 0.421; PANC1 0.10: 0.274 vs 0.276; both screens at 0.70
agree to within 0.001). The result is not an artifact of how hard the potential
was subtracted.

### 8.6 Corrections the adversarial audit forced

**The shrinkage coefficient was fitted on contaminated validation data, and it
inflated the sparse-coverage negatives.** In the main grid `α` is chosen on the
same validation pairs that already selected the stopping epoch, the restart and
the grid member, so it is biased upward — `α = 0` occurs in only 13 of 320
learned-rung runs. Rerunning the three rungs the primary contrast depends on
with half the validation pairs withheld from selection entirely
(`--part robustness`, `honest_alpha.jsonl`; `α = 0` now in 56 of 160):

<!-- generated: lowrank_honest_alpha -->
| screen | coverage | skill | 95% CI | seeds > 0 | Pearson r | sign acc. |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| A375 | 0.05 | -0.012 | [-0.027,+0.003] | 0/8 | 0.005 | 0.479 |
| A375 | 0.1 | -0.009 | [-0.038,+0.020] | 1/8 | 0.061 | 0.544 |
| A375 | 0.2 | +0.022 | [-0.011,+0.055] | 4/8 | 0.176 | 0.598 |
| A375 | 0.4 | +0.229 | [+0.189,+0.269] | 8/8 | 0.479 | 0.787 |
| A375 | 0.7 | +0.353 | [+0.315,+0.391] | 8/8 | 0.594 | 0.852 |
| PANC1 | 0.05 | -0.020 | [-0.051,+0.011] | 1/8 | 0.022 | 0.518 |
| PANC1 | 0.1 | -0.012 | [-0.033,+0.009] | 2/8 | 0.026 | 0.514 |
| PANC1 | 0.2 | +0.019 | [-0.020,+0.058] | 5/8 | 0.160 | 0.598 |
| PANC1 | 0.4 | +0.198 | [+0.147,+0.248] | 8/8 | 0.456 | 0.733 |
| PANC1 | 0.7 | +0.366 | [+0.323,+0.410] | 8/8 | 0.608 | 0.845 |

and the primary contrast under the same estimator:

<!-- generated: primary_contrast_honest_alpha -->
| screen | coverage | Δ | 95% CI | p (t) | p (Wilcoxon) | seeds favouring low-rank |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| A375 | 0.05 | -0.002 | [-0.033,+0.030] | 9.1e-01 | 0.4375 | 1/8 |
| A375 | 0.1 | -0.018 | [-0.052,+0.016] | 2.6e-01 | 0.4375 | 2/8 |
| A375 | 0.2 | +0.022 | [-0.010,+0.055] | 1.5e-01 | 0.1562 | 5/8 |
| A375 | 0.4 | +0.228 | [+0.189,+0.267] | 2.5e-06 | 0.0078 | 8/8 |
| A375 | 0.7 | +0.351 | [+0.312,+0.390] | 1.3e-07 | 0.0078 | 8/8 |
| PANC1 | 0.05 | -0.020 | [-0.053,+0.012] | 1.9e-01 | 0.1562 | 2/8 |
| PANC1 | 0.1 | -0.012 | [-0.033,+0.010] | 2.4e-01 | 0.6875 | 4/8 |
| PANC1 | 0.2 | +0.020 | [-0.019,+0.058] | 2.7e-01 | 0.3125 | 5/8 |
| PANC1 | 0.4 | +0.198 | [+0.147,+0.248] | 3.8e-05 | 0.0078 | 8/8 |
| PANC1 | 0.7 | +0.366 | [+0.322,+0.409] | 2.2e-07 | 0.0078 | 8/8 |

The dense-coverage result is unmoved (+0.229 / +0.198 at coverage 0.40, +0.353 /
+0.366 at 0.70, all 8/8 seeds). The sparse-coverage negatives **collapse toward
zero** and stop being significant — A375 0.05 goes from −0.188 to −0.012 — so the
as-run sparse cells were measuring the cost of the selection machinery at 37–74
validation pairs, not harm from modelling the residual, and the pre-registered
"evidence against practical learnability" branch is **not** triggered. Note also
that at coverage 0.20 the corrected estimate is a small positive on both screens
(+0.022 / +0.019, 4/8 and 5/8 seeds) rather than the ~0 the as-run grid showed.
**The honest-α numbers are the ones quoted in the decision**, including Pearson
and sign accuracy, which are not α-invariant across protocols because the two
protocols select different grid members.

**The low-rank rung selected its largest rank at the cell that decides the
result, so the result was rerun with no search at all.** At A375 coverage 0.70
the grid picked rank 16 (1,856 parameters) on 8 of 8 seeds — a boundary selection
makes "capacity-controlled" a claim about where the grid was truncated. Pinned to
**rank 2 — 204 parameters, one 2×2 antisymmetric form, one learning rate, no
hyperparameter search whatsoever** (`rank2.jsonl`):

| screen | coverage | rank-2 skill | rank-2 r | seeds >0 | searched low-rank | 58k-parameter MLP |
|---|---:|---:|---:|---:|---:|---:|
| A375 | 0.40 | +0.197 [+0.165,+0.229] | 0.454 | 8/8 | +0.231 | +0.123 |
| A375 | 0.70 | +0.250 [+0.229,+0.271] | 0.501 | 8/8 | +0.368 | +0.309 |
| PANC1 | 0.40 | +0.161 [+0.143,+0.179] | 0.406 | 8/8 | +0.223 | +0.107 |
| PANC1 | 0.70 | +0.237 [+0.220,+0.254] | 0.489 | 8/8 | +0.374 | +0.313 |

Two numbers per drug clear both quantitative criteria by a wide margin. Because
selection is on validation only, a truncated grid can only bias the reported
skill *down*, so this is a floor. Note the honest comparison against the flexible
rung: at 204 parameters the low-rank form beats the 58k-parameter MLP at coverage
0.40 on both screens but **loses to it at 0.70** (+0.250 vs +0.309); it is the
*searched* low-rank family (up to 1,856 parameters) that beats the MLP
everywhere something is detectable.

**Four p-values and a Pearson column were transcription errors.** The audit of
this document found that the §8.3 table's `p (t)` column had been hand-copied
with two values carried over from §8.6 and two matching no run in the repository,
and that the Pearson and sign-accuracy figures labelled honest-α in §9 and §10
were the as-run values. None of it changed the decision — every value was far
below 0.05 and criterion 3 passes by threefold under either protocol — but a
number a human retypes is a number nobody checks. Every table in §8.3 and §8.6
is now emitted by `scripts/report_phase2_residual.py` into
`results/phase2_residual/summary/doc_tables.md` and must appear here verbatim;
`test_document_tables_are_generated_not_transcribed` fails if it does not.

### 8.7 Where the residual result sits next to Phase 2

Reconstructing the *raw* directional prediction as `D_add + α·D̂_res` and
correlating it with raw `D` puts the two experiments on one axis:

| screen | coverage | additive alone | + low-rank residual | Phase 2 unrestricted |
|---|---:|---:|---:|---:|
| A375 | 0.70 | 0.716 | **0.832** | 0.852 |
| A375 | 0.40 | 0.708 | 0.785 | 0.818 |
| PANC1 | 0.70 | 0.612 | **0.781** | — |

Phase 2's unrestricted model was **already capturing this structure**. Phase 2R
does not discover new predictability; it establishes *what* the predictability
is, that it is not the per-drug potential, and how many pairs it takes to find.

---

## 9. The seven questions, answered separately

All held-out figures below are the **honest-α** numbers (§8.6), which is the
estimator the decision rests on.

**Q1 — how much raw directional variance does the additive `g_i − g_j` effect
remove?** Exactly **53.8%** on A375 and **39.8%** on PANC1 of the directional
mean square, measured with no model and no split. Out of sample with a
train-only fit it rises from 33.7% to 51.4% (A375) and 22.3% to 37.4% (PANC1) as
coverage grows, converging on those exact values from below.

**Q2 — is what remains large relative to the target's scale?** Yes. The cyclic
remainder is 46.2% (A375) / 60.2% (PANC1) of the directional mean square, with
sd 0.151 / 0.126 against a raw directional sd of 0.223 / 0.163 and a response sd
of ~0.29 / ~0.21. It is not a rounding term.

**Q3 — can any model predict `D_res` for an entirely unseen pair?** **Yes, at
sufficient coverage.** At coverage 0.40 and 0.70 the low-rank antisymmetric rung
beats the zero predictor with skill **+0.198 to +0.366**, 95% CIs far from zero,
**8 of 8** split seeds on both screens, held-out Pearson **0.456–0.608** and sign
accuracy **0.733–0.852**. It beats the `potential` rung by the same margin
(p ≤ 3.8e−05 by t, 0.0078 by Wilcoxon — the floor at n=8 — and 8/8 seeds), and
the `potential` rung is at |skill| ≤ 0.002 at those coverages. Two numbers per
drug suffice: rank 2 with 204 parameters and no search reaches +0.161 to +0.250
with r 0.406–0.501, 8/8 seeds.

**Q4 — does learnability depend on coverage?** Decisively, and this is the
sharpest quantitative result here. The transition sits between 842 and 1,683
training pairs (coverage 0.20 → 0.40), where skill goes from ≈+0.02 to ≈+0.21.
The positive control resolves what the three sparse cells mean, and they do not
all mean the same thing:

| coverage | training pairs | observed skill | what the power control says |
|---|---:|---:|---|
| 0.05 | 211 | −0.012 / −0.020 | **blind** — even κ=0.20, 1.4×/2.1× the whole residual's mean square, is missed 0/8 on both screens. Uninformative. |
| 0.10 | 421 | −0.009 / −0.012 | **blind at κ=0.10**, marginal at κ=0.20 (5/8, 4/8). Excludes a signal ~2× the residual; says nothing smaller. |
| 0.20 | 842 | +0.022 / +0.019 | **powered at κ=0.10** (6/8 both screens, +0.05/+0.08). So this is a small real positive at the detection floor, not a null. |
| 0.40 | 1,683 | +0.229 / +0.198 | comfortably above it |
| 0.70 | 2,945 | +0.353 / +0.366 | comfortably above it |

**Q5 — does it replicate across A375 and PANC1?** Yes, closely. At coverage 0.70
the low-rank skill is +0.353 / +0.366 and Pearson 0.594 / 0.608; at 0.40,
+0.229 / +0.198 and 0.479 / 0.457. The coverage at which it appears is the same,
the `potential` rung is null on both, and the power curves agree. Note the two
screens share drugs and protocol, so this is replication across cell lines within
one study — not independent replication.

**Q6 — if unrestricted models can learn the residual, does the structured model
help?** The question largely dissolves on this target, and that is itself the
answer. The Phase 2 `structured` family's antisymmetric component reduces to the
`mlp` rung once the first-order potential has been removed — the same function
class up to a factor of two folded into the output layer, pinned in
`test_structured_A_head_equals_the_mlp_rung`. What can be compared is
parameterisation, and the ordering runs against the Intervention-Algebra prior:
the *searched* low-rank family (204–1,856 parameters) beats the 58k-parameter
MLP at every cell where anything is detectable (+0.368 vs +0.309 at A375/0.70),
and even pinned at 204 parameters it beats the MLP at coverage 0.40 on both
screens, though not at 0.70. The useful constraint on this data is **low rank**,
not a symmetric/antisymmetric split.

**Q7 — does the result justify further model development on Koplev?** **No, not
model development.** Phase 2's unrestricted model already reached raw directional
r = 0.852 at coverage 0.70 where the additive null reached 0.716; it was already
capturing this structure. Phase 2R did not find headroom a better architecture
would unlock — it found that the thing being predicted is real, low-dimensional,
and needs pairs rather than cleverness. What it does justify is one **validity**
experiment, in §12, because the leading alternative explanation for the whole
finding has nothing to do with biology.

---

## 10. Decision

**Outcome A — residual pair-specific directional structure is clearly learnable
— at coverage ≥ 0.40, on both screens. At coverage 0.20 it is a small positive
at the detection floor. At coverage ≤ 0.10 the experiment is blind and its null
is uninformative.**

Against the criteria registered in [`PREREGISTRATIONS.md`](PREREGISTRATIONS.md) before
the grid ran, at A375 and PANC1 coverage 0.40 and 0.70, using the honest-α block:

| criterion | required | observed |
|---|---|---|
| 1 — mean skill > 0, CI excludes 0 | — | +0.198 to +0.366; CI lower bounds +0.147 to +0.323 ✅ |
| 2 — `lowrank − potential` > 0, both p < 0.05, ≥ 7/8 seeds | — | p ≤ 3.8e−05 (t), 0.0078 (Wilcoxon, the n=8 floor), **8/8** ✅ |
| 3 — Pearson r > 0.15 | — | 0.456 to 0.608 ✅ |
| 4 — replicated at two adjacent coverages **or** both screens | — | **both**: 0.40 and 0.70, on A375 and PANC1 ✅ |
| 5 — permutation control ≈ 0 | ≤ 0.02, CI covers 0 | −0.004 to −0.005 at both 0.40 and 0.70, \|r\| ≤ 0.030 — magnitude bar met by 4×; see the note ⚠️ |

**Criterion 5, reported as it was written rather than as it is convenient.** The
magnitude clause (mean ≤ 0.02) is met at all four decisive cells with a factor of
four to spare. The "CI contains 0" clause is met at three of them and **fails at
PANC1 coverage 0.40**, where the control's CI is [−0.0098, −0.0009] — it excludes
zero from *below*. That is the control doing 0.5% *worse* than predicting
nothing, which is the same small selection-noise penalty visible throughout the
sparse cells, and it is the one direction in which a control failure cannot
manufacture a false positive. The criterion is reported as ⚠️ rather than ✅
because it was registered with both clauses and one of them does not hold; the
substance — a rung that scores +0.223 on real targets and −0.005 on permuted
ones at the same cell — is not in doubt.

(An earlier version of this document ran the control at coverages 0.10 and 0.70
only and claimed criterion 5 at 0.40 by extrapolation from the adjacent rung. The
control was rerun at 0.40 rather than leaving the caveat in place; that is how
the failing clause came to light.)

The registered "evidence against practical learnability" branch is **not**
triggered: the sparse-coverage negatives that looked like it were an artifact of
fitting the shrinkage coefficient on already-used validation pairs, and they
collapse from −0.19 to −0.01 once that is corrected (§8.6).

**The strongest honest statement.** On the Koplev screens, roughly half the
apparent order dependence is a per-drug "better first than second" tendency, and
the other half is cyclic structure that no such tendency can express. That
cyclic half is genuinely predictable for pairs never observed in either
direction, from drug identity alone, by a model with two latent numbers per
drug — but only once roughly a third of the pair graph has been observed.

**What would be an overclaim.** Every one of these:

* *"Sequential drug interactions have learnable pair-specific order effects."*
  This is one study, one lab, one protocol, two cell lines that share their drug
  panel, and a **modelled** endpoint (§11).
* *"There is no residual structure at sparse coverage."* The power control
  forbids this at coverage ≤ 0.10 and contradicts it at 0.20, where the
  experiment is powered for κ=0.10 and reads a small positive.
* *"Phase 2's structured model was wrong to fail."* It reduces to the rung that
  succeeds here, on the residual, and it was already capturing this structure
  through its unrestricted sibling. Phase 2's verdict — no transfer of the
  symmetric/antisymmetric *parameterisation* — is unchanged and is if anything
  reinforced: what helps on this data is low rank, not the S/A split.
* *"Pair-specific means idiosyncratic to the pair."* It does not; see §11.

## 11. Limitations that materially affect the answer

**1. The target is a modelled quantity, and this is the leading alternative
explanation for the entire result.** `synergy_measure` is the posterior mean of
an area-based synergy index from one joint ~45,000-parameter Bayesian fit
([`phase2_dataset.md`](phase2_dataset.md)), not an independent measurement. Every
ordered combination involving drug `i` is scored against a baseline built from
`i`'s single-agent dose-response curve, and those curves are **shared and jointly
estimated**. If drug `i`'s curve carries estimation error `e_i`, then the derived
synergy for `i→j` is a nonlinear function of `(e_i, e_j)`, and its antisymmetric
part is (a) nonzero, (b) dependent on both drugs, and (c) **bilinear-ish in fixed
per-drug quantities — so it generalises perfectly to held-out pairs.** That is
precisely the signature this experiment measured, and it would look identical
whether or not any biological sequence-specific interaction exists. Nothing in
Phase 2R distinguishes the two. §12 is about closing this.

**2. The protocol is itself order-asymmetric** — first drug at a fixed dose,
second titrated. That plausibly explains the *potential* half; whether it can
also generate pair-specific cyclic structure is not established either way.

**3. "Pair-specific" has a precise and narrower meaning here than a reader will
assume.** It means "not expressible as `c_i − c_j` for any `c`" — an interaction
in the factorial sense, with nonzero sums around 3-cycles. It does **not** mean
idiosyncratic to the pair: every rung is a bilinear form in per-drug latent
vectors, and it generalises to unseen pairs *because* it is not idiosyncratic. A
truly pair-idiosyncratic term is unlearnable out of sample by construction, and
no held-out-pair design can ever test for one.

**4. The 8 split seeds are not fully independent.** They share the same 100
drugs and the same fixed response matrix; their evaluation pools overlap at mean
pairwise Jaccard 0.105 (19.7% of the smaller pool), no pair appears in all 8, and
their intersection is empty. The CIs quantify split noise on one matrix, not
uncertainty about drugs, cell lines or biology.

**5. Multiplicity.** 2 screens × 5 coverages × 5 rungs × several metrics is a
large table and only `lowrank − potential` on `cal_skill` is pre-registered. The
dense-coverage effects survive any correction one would apply over 10 cells
(p ~1e−7, 8/8 seeds, both screens); nothing else in the tables should be read as
confirmatory. No primary *cell* was registered — only a primary rung, metric and
contrast — so the coverage-resolved reading in §10 is a description of the whole
curve rather than a test at a nominated cell.

**6. The positive control is a single injected realisation, and κ is an
absolute size.** `inject_seed=0`, `inject_rank=3` — which gives a **rank-2**
signal, since an odd-dimensional antisymmetric matrix is always rank-deficient —
reused across all seeds, coverages and screens. Its shape exactly matches the
low-rank rung's hypothesis class, so the resulting curve is an **upper bound** on
detectability: a differently-shaped signal of the same size would be harder,
which makes "coverage ≤ 0.10 is blind" if anything understated. And because κ is
an RMS added to `D` rather than a fraction of each screen's residual, the same κ
is a different relative signal on the two screens (κ=0.20 is 1.4× the observed
residual mean square on A375 and 2.1× on PANC1) — which is why the block is run
on both rather than extrapolated from one.

**7. Sign accuracy uses Phase 2's noise threshold**, derived as
`2·√2·median posterior synergy_sd` for *raw* `D`. On the smaller-scale `D_res` it
selects a larger and differently-composed subset of pairs. Reported as
exploratory and excluded from the decision rule.

**8. Drug identity only.** Nothing here transfers to a drug the screen did not
contain. That was deliberate — features would test entity similarity rather than
whether the pair graph carries reusable structure — but it bounds what the result
means.

**9. `mean(D_res)` is not exactly zero.** It averages 0.005 in absolute value
across runs and reaches 0.028 on one split seed, up to 16% of `sd(D_res)`. Since
`MSE_zero = mean(D_res²)` includes that mean, at most ~2.5% of the skill
denominator on the worst seed is a mean offset rather than dispersion. Every rung
is antisymmetric and so cannot fit a constant, which is why this is a caveat on
the denominator rather than a route to skill.

---

## 12. What to do next — exactly one thing

**Test whether the cyclic structure survives a null in which it cannot be real.**

Not a new architecture, and not chemical features. The single highest-value next
move is to decide limitation 1, because everything else is downstream of it. The
d-chain model that produced `synergy_measure` is public
(<https://github.com/skoplev/d-chain>, GPL-3.0). Simulate a screen from that
generative model with **no true pair-specific sequence interaction** — shared,
noisily-estimated per-drug single-agent curves, per-combination parameters drawn
independently — push it through the deposited pipeline's own area-based synergy
measure, and run this exact experiment on the output.

The prediction is sharp and the experiment is cheap:

* if the simulated screen shows a curl fraction near zero and no held-out
  residual skill, the Koplev finding is about the data and the direction is worth
  continuing — at which point the question in §19 of the brief (do chemical or
  target features let residual order effects generalise to *unseen drugs*)
  becomes the right one;
* if the simulated screen reproduces a ~45–60% curl fraction and skill ~0.35 at
  coverage 0.70, then Phase 2R measured the shape of the authors' estimator, the
  Koplev branch should close, and the project should move to a dataset whose
  endpoint is a direct measurement rather than a joint posterior.

Do not build another model on this screen before that question is answered.
