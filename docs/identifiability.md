# Identifiability of the (v, S, A) decomposition

This note records *why* the Phase 1 evaluation is allowed to report "recovery of
S" and "recovery of A" at all, and under exactly which conditions those numbers
are meaningful. It is the analysis that governs which metrics appear in the
results tables.

## 1. The gauge problem, and why single-intervention rows fix it

Consider only simultaneous observations. With

```
y_sim(i,j) = v_i + v_j + S_ij
```

the reparameterisation

```
v_i  ->  v_i + c_i
S_ij ->  S_ij - c_i - c_j
```

leaves every simultaneous observation unchanged for any vector `c`. So from
simultaneous data alone the split between "first-order effect" and "symmetric
interaction" is **not identified**: there is an `N x d`-dimensional gauge
freedom, and a claim like "the model recovered S" would be meaningless.

Ordered observations do not fix this either, since

```
y_ord(i->j) - y_ord(j->i) = 2 A_ij
```

is invariant to the same reparameterisation, and

```
(y_ord(i->j) + y_ord(j->i)) / 2 = y_sim(i,j)
```

adds nothing new.

The gauge is fixed by including **all N single-intervention rows in the training
set, always** (`generate_observations(..., include_singles=True)` for the train
split, enforced by `assert_all_interventions_seen`). Then

```
y_single(i) = v_i
```

pins `v` outright, `S_ij = y_sim(i,j) - v_i - v_j` is determined, and
`A_ij = y_ord(i->j) - y_sim(i,j)` is determined. The decomposition is exactly
identified on every **observed** pair.

Consequences:

* Held-out interventions are *out of scope for Phase 1*. Every intervention must
  appear in training or its `v_i` is unidentified and everything downstream is
  gauge-dependent. Entity-level OOD is future work (see README roadmap).
* On **held-out pairs** `S_ij` and `A_ij` are *not* determined by the data. That
  is precisely the quantity the models must extrapolate, and it is the point of
  the benchmark. Extrapolation is only possible because the generator makes
  `S` and `A` structured functions of a shared per-intervention latent `u_i`
  rather than i.i.d. per pair.

## 2. A benchmark can be unlearnable on purpose

If `S_ij` were drawn i.i.d. per pair, a fully held-out pair would carry zero
information and *no* model could beat an additive predictor. Such a benchmark
would produce a null result for reasons that have nothing to do with the
hypothesis.

The generator therefore has three topology modes:

* `sparsity_mode="latent"` (**default**) — whether a pair interacts is a smooth
  function of the same hidden factors that drive `S` and `A`:
  `g_ij = sigma(beta * (<u_i,u_j> - tau))`, with `tau` set to the
  `1 - sparsity` quantile so a controlled fraction of pairs is "on". Because the
  gate is a property of the individual interventions, the topology of a pair
  never seen in training is predictable in principle. Used for the main
  experiments.
* `sparsity_mode="module"` — pairs interact iff their latent *module* labels
  interact. The module label is independent of `u`, so this is learnable only by
  inferring a discrete per-intervention label from a node's other training
  pairs. Harder; kept as a secondary condition.
* `sparsity_mode="random"` — the interaction mask is i.i.d. per pair. The
  *magnitudes* remain bilinear in `u`, but which pairs are switched on is
  unpredictable. This is an **identifiability control**: topology recovery on
  held-out pairs should be at chance for every model family. If a model scores
  above chance here, something is leaking.

### The soft gate, and what it means for topology labels

In `latent` mode the interaction magnitudes are multiplied by the *soft* gate
`g_ij in [0,1]`, while the binary topology label used for AUROC/AUPRC is the
*hard* threshold `mask = g_ij > 0.5`. Pairs whose gate sits near 0.5 therefore
carry an intermediate-strength interaction but receive a hard 0/1 label. At the
default `gate_beta=8.0` about 19.5% of pairs lie in the ambiguous band
`0.1 < g < 0.9`. This caps achievable topology AUPRC below 1.0 and is a property
of the benchmark, not of any model. Raising `gate_beta` sharpens the gate (at
`beta=20` the ambiguous band is ~8%) at the cost of a less smoothly predictable
topology.

## 3. What the recovery numbers must be compared against

Because `S_ij` on a held-out pair must be *extrapolated* from a sparse sample of
other pairs, recovery is not attainable at every coverage even in principle. The
ceiling is measured by an actual model in the actual pipeline —
`WellSpecifiedModel` (`models.py`), run by `phase1.ceiling_sweep()`:

```bash
python scripts/run_phase1.py --part ceiling --out results/phase1_ceiling_fixedlen.jsonl
```

It learns a per-intervention factor `u_i` and the bilinear forms directly, with
`B_c` constrained symmetric, `K_c` constrained antisymmetric, and a learnable
gate `sigmoid(beta*(<u_i,u_j> - tau))` — i.e. **the generator's own functional
form**, gate included. It is excluded from every headline family comparison,
because being handed the true functional form is not something an honest baseline
gets.

Measured on held-out pairs (source: `results/phase1_ceiling_fixedlen.jsonl`,
5 seeds, mean ± sd; regenerate with the command above). Note that
`ceiling_sweep` now defaults to `early_stopping=False`, so `--part ceiling`
emits `tag="ceiling_fixedlen"` rows; the committed
`results/phase1_ceiling.jsonl` holds the **superseded** early-stopping protocol
and has no CLI surface that regenerates it:

| pair coverage | S pearson | A pearson | test MSE | noise floor |
|---|---|---|---|---|
| 0.05 | 0.361 | 0.266 | 0.702 | 0.0025 |
| 0.10 | 0.810 | 0.764 | 0.293 | 0.0025 |
| 0.20 | 0.942 | 0.929 | 0.107 | 0.0025 |
| 0.40 | 0.985 | 0.979 | 0.035 | 0.0025 |
| 0.70 | 0.993 | 0.989 | 0.019 | 0.0025 |

Measured under the **fixed-length** protocol (`tag="ceiling_fixedlen"`), i.e.
the same protocol as every model it is used to normalise.

**The interpretation rule this licenses**, and its exact scope:

* At coverage **0.05 the benchmark is not identifiable**. A model given the true
  functional form reaches only r ≈ 0.36 for S and ≈ 0.27 for A. Nothing measured
  at this coverage is evidence about the hypothesis in either direction, and a
  win for the additive family there means "the latent factors cannot be
  recovered from ~4 pairs per intervention", not "the inductive bias fails when
  data is scarce".
* At coverage **0.10 and above the benchmark is identifiable** (r ≥ 0.80 rising
  to ≈ 0.99). Differences between families at these coverages are real
  differences between families.
* Because the ceiling is ≈ 0.99 from coverage 0.4 upward, recovery numbers there
  should be read against **1.0, not against a lowered bar**. A family scoring
  r = 0.7 at coverage 0.4 has captured roughly 70% of what is extractable, not
  "most of it".

The non-circular form of the argument is: *the correctly-specified estimator,
given the true functional form, cannot recover the structure at this coverage;
therefore no misspecified model can be expected to; therefore this coverage tests
identifiability rather than inductive bias.* That holds only where
`WellSpecifiedModel` really is well specified.

**Protocol caveat — raised, then measured, then closed.** The ceiling was
originally measured under `patience=600` (early stopping on) while every
comparison run trains fixed-length, so the "% of achievable" ratios spanned two
protocols. Worse, the committed code did not reproduce the committed ceiling:
`ceiling_sweep()` emitted `TRAIN_BASE` while every stored row carried
`patience=600`, so `--part ceiling` silently ran a different experiment.

Both are now fixed. `ceiling_sweep(early_stopping=True)` reproduces the original
artifact exactly, and the default reproduces the fixed-length one, which is what
the table above reports. The two were then compared directly:

| coverage | S, early stopping | S, fixed-length | A, early stopping | A, fixed-length |
| --- | --- | --- | --- | --- |
| 0.05 | 0.387 | 0.361 | 0.273 | 0.266 |
| 0.10 | 0.814 | 0.810 | 0.756 | 0.764 |
| 0.20 | 0.943 | 0.942 | 0.929 | 0.929 |
| 0.40 | 0.985 | 0.985 | 0.979 | 0.979 |
| 0.70 | 0.993 | 0.993 | 0.989 | 0.989 |

The ceiling is effectively protocol-invariant: identical to three decimals at
every coverage above 0.10, and different by at most 0.026 below it. So the
caveat was real as a reproducibility defect and empirically negligible as a
numerical one — which is worth stating in that order, because the first was not
knowable from the second.

**Scope limit — this ceiling applies to the default `latent` topology only.**
Under `sparsity_mode="random"` the gate is drawn i.i.d. per pair and is *not* a
function of `u`, so `WellSpecifiedModel` is itself misspecified there and its
score is an upper bound on the wrong model family. The `random` control must not
be declared "unidentifiable" on the strength of this table; its whole purpose is
that topology recovery should sit at chance for *everyone*.

> **History.** Earlier revisions of this document quoted a ceiling of r ≈ 0.69
> (S) and ≈ 0.48 (A) at coverage 0.2, from an ad-hoc regression that existed only
> in a scratch script and was never committed. That estimator was itself
> misspecified — it did not model the gate — and it understated the ceiling by a
> wide margin. Grading results against it would have inflated every family's
> apparent performance by roughly 30 percentage points. The numbers above come
> from committed code and a committed artifact.

## 4. Nonlinear observation maps break latent recovery

With `observation_map="tanh"` the observed quantity is `y = tanh(g * z)`. Two
things change:

1. The additive structure no longer lives in the observation space, so all model
   families are given the *same* learnable elementwise output head
   (`ElementwiseHead`) and predict `y = h(v_i + v_j + S_ij [+ A_ij])`.
2. Latent recovery is no longer identified. For any per-channel strictly
   monotone `phi`, replacing the latent by `phi(z)` and the head by `h . phi^-1`
   reproduces every observation, but does not preserve the additive
   decomposition — `S` under one parameterisation is not `S` under another.

Therefore in the nonlinear regime the results tables report **observable
prediction error only** as the primary metric. Topology AUROC/AUPRC is reported
as a secondary, descriptive number with the caveat that the ranking of
`||S_pred||` is only approximately preserved under the reparameterisation.
Pearson/Spearman correlations against the latent `S`/`A` are reported as `NaN`
by `structure_metrics(..., identifiable=False)` rather than being silently
computed — see `metrics.py`.

## 5. What the readouts mean, and which comparison is fair

Every family exposes `implied_S` and `implied_A`, and both are defined so that
they mean the *same thing* for every family:

* `implied_S(i,j)` is the term the model actually adds to `v_i + v_j` when
  predicting a **simultaneous** row.
* `implied_A(i,j)` is `(latent_ordered(i,j) - latent_ordered(j,i)) / 2`, which is
  exactly what `A` means.

If either of these were false for some family — if the `S` that gets *scored*
were not the `S` the model *uses* — the recovery comparison would be rigged, so
this is the single most load-bearing property in the evaluation. It has been
verified numerically for all five families (max deviation ≤ 2.6e-8, at
initialisation and after training).

> **History — a gap that was open and is now closed.** Earlier revisions of this
> document cited two tests by name as asserting the above. Those tests did not
> exist. The tests that existed at the time — `test_implied_S_is_symmetric` and
> `test_implied_A_is_antisymmetric` — asserted something *different* and, as
> `metrics.invariance_diagnostics` itself documents, tautological: `implied_S` is
> order-invariant by construction in every family (symmetrised in `algebra`,
> canonicalised in the others) and `implied_A` is *defined* as an antisymmetrised
> difference everywhere. They therefore proved nothing about the constrained
> model specifically.
>
> `tests/test_readout_honesty.py` now asserts the load-bearing identities
> directly, for every family, both at initialisation and after training:
> `implied_S(i,j) == latent_sim(i,j) - v(i) - v(j)` and
> `implied_A(i,j) == (latent_ordered(i,j) - latent_ordered(j,i))/2`, plus a guard
> that the readouts are not trivially zero for the interaction families. The
> guard was mutation-tested — deliberately breaking each family's readout makes
> it fail — so a regression in this property would now be caught by CI.
>
> That last clause was false when written: the repository had no CI of any kind,
> and the guard only fired when someone ran pytest by hand. It became true at
> Phase 1 closure, when `.github/workflows/ci.yml` was added — the suite runs on
> Python 3.11 and 3.12 on every push to `main` and on every pull request. Recorded rather than
> quietly amended, because "a test exists" and "a test runs" were being treated
> as the same claim.

Neither readout is privileged for the constrained model — for it these
definitions coincide with its own `S` and `A` heads by construction, and for the
baselines they are the honest analogue.

`implied_S_projected` additionally reports the symmetric projection
`(implied_S(i,j) + implied_S(j,i)) / 2`. For every family in the current model
set the simultaneous term is *already* order-invariant, so this equals
`implied_S`; it is retained and reported so that any future family whose
simultaneous term is not symmetric can be scored charitably as well as honestly.

### What the symmetry constraint can and cannot buy

Simultaneous observations are physically order-free, and the generator always
stores them canonically (`i < j`). The unconstrained baselines exploit this by
feeding the pair to their simultaneous head in canonical index order, so their
simultaneous term is symmetric *for free*. **No credit is claimed for symmetry of
the simultaneous term** — on that row alone the constrained model has no
advantage, and saying otherwise would be the single easiest way to manufacture a
fake win.

It goes further than "no credit is claimed": **symmetry of `S` has no isolable
content in this design at all.** Consider the family you would build to test it —
"symmetrise `S`, leave the ordered term unconstrained". Because canonicalisation
already makes the baseline's simultaneous term order-invariant, that family *is*
`UnconstrainedModel`. There is no contrast anywhere in the family set whose
difference is symmetry of `S`.

What the ablation ladder does isolate:

| contrast | isolates |
|---|---|
| `unconstrained` → `shared_pair` | **sharing**: the ordered rows are the simultaneous outcome plus a correction, rather than an independent function |
| `shared_pair` → `algebra` | **the cross-row identity**: the ordered rows are the simultaneous term plus an exactly antisymmetric correction, `A(i,j) = -A(j,i)`. Not antisymmetry alone — the same step also forces the ordered rows' symmetric part to equal the simultaneous term, splits the shared trunk into two independent MLPs, and changes `pair_hidden` 76 → 48. See `models.py`. |

Together these give the algebra model an exact constraint the unconstrained model
lacks: a pair's three rows (`sim`, `i->j`, `j->i`) are explained by two
quantities. But that constraint is **sharing plus the cross-row identity**, not symmetry
plus antisymmetry.

**Consequence for the hypothesis statement.** Phase 1 as built tests
*"sharing + antisymmetry improves compositional generalization"*. It does **not**
test the symmetry half of the informal claim, and any write-up that says it does
is overclaiming. Testing symmetry would require changing the benchmark — for
instance storing simultaneous pairs in a random index order, so that a baseline
has to *learn* order-invariance rather than being handed it by canonicalisation.
That is a defensible future variant, but it is not what the current numbers
speak to.
