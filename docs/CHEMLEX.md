# The ChemLex external chemistry validation

Everything in [`KOPLEV.md`](KOPLEV.md) rests on one dataset, from one lab, with a
target that is an estimator output rather than a measurement. Phase 4 is the
external test: a different scientific domain, a different measurement type, a
directly measured endpoint, and a structure that cannot support the original
architecture at all.

## The data

Zhong H, et al., *Towards global reaction feasibility and robustness prediction
with high throughput data and Bayesian deep learning*, Nature Communications 16,
4522 (2025), doi:10.1038/s41467-025-59812-0. Deposit: Zenodo
doi:10.5281/zenodo.17596563, **CC BY-NC 4.0**. Fetched and digest-verified,
never vendored — the non-commercial term means it is not ours to redistribute.

| | `all` | `hatu` |
|---|---:|---:|
| measured reactions | 11,669 | 8,454 |
| carboxylic acids | 272 | 272 |
| amines | 230 | 230 |
| coupling conditions | 7 | **1** |
| distinct pairs | 8,064 | 7,919 |

The endpoint is an uncalibrated LC-MS UV area ratio — a conversion, not a
calibrated yield. Two measurements of the same nominal reaction in this deposit
correlate at Pearson ≈ 0.6, so **roughly half the variance in this endpoint is
not available to any deterministic predictor**, and every R² below should be read
against that ceiling.

The deposit was audited before anything was designed, and two corrections were
needed: HATU is drawn two ways in the reagent column (O-uronium and guanidinium
N-oxide; on the 37 pairs measured under both, the paired difference is −0.05
points, p = 0.987, so they are merged), and two "reagents" are substrate
counterions rather than reagents and are stripped. The newest of the three
published deposit versions is used, after all three were compared cell by cell:
the modelling columns are byte-identical across versions, and the newest fixes a
hexafluorophosphate written as a cation, which RDKit cannot parse.

Full audit: [`phase4_chemlex_dataset.md`](phase4_chemlex_dataset.md). Molecular
identity and split grouping: [`phase4_chemlex_mapping.md`](phase4_chemlex_mapping.md).

## The model, and why the algebra does not come along

```
y(a, n, c) = μ + f_A(a) + f_N(n) + f_C(c) + I_AN(a, n)
I_AN(a, n) = z_A(a)ᵀ W z_N(n)
```

The interaction is **bipartite**. An acid and an amine are different kinds of
entity, so there is no `I(n, a)` to compare `I(a, n)` against — it does not
typecheck. Symmetry and antisymmetry are not merely unhelpful here; they are
undefined. What is being tested is the surviving claim from Koplev — that the
interaction term is low-rank and inferable from the entities' structures — in a
setting where the original architecture cannot even be stated.

## Two verdicts, and why both are published

**The frozen, pre-registered verdict is `INCONCLUSIVE`.**

The rule was registered at commit `a1f396f` before any authoritative fold was
fitted, on a fold seed no model had been fitted on. Implemented literally and
evaluated on the registered statistic:

| screen/regime | entities | mean incremental | (a) above floor | (b) both p<0.05 | (c) majority | (d) blind drop | (e) survives projection | (f) robust contrast | (g) low similarity |
|---|---:|---:|---|---|---|---|---|---|---|
| all/E1-A | 242 | +0.0355 | yes | no | yes | no | no | no | no |
| all/E1-N | 228 | +0.0426 | yes | no | yes | yes | yes | yes | no |
| hatu/E1-A | 233 | −0.0115 | no | no | yes | yes | yes | n/a | no |
| hatu/E1-N | 224 | +0.0205 | yes | no | yes | yes | yes | n/a | no |

`INCONCLUSIVE` is what the registered table yields because **no cell clears every
criterion and the two screens fail different ones**. That is a finding about the
registration, not a shrug.

**The same rule with one statistic corrected returns `ANALOGUE-ONLY CHEMICAL
TRANSFER`, and that is the reported reading.**

The single change: the per-entity statistic's denominator becomes the fold's
baseline MSE rather than the entity's own. The registered statistic is
`1 − MSE_pair(entity) / MSE_add(entity)`, and on this screen that denominator is
not bounded away from zero — a reactant that fails with every partner is
predicted correctly at ≈0 by both models, so a tiny absolute worsening becomes an
enormous negative ratio. The worst case scores **−13.20** on eight test rows,
against a fold-level baseline 55× larger, and drags a cell's mean to +0.0426 from
a median of +0.0538.

| screen/regime | entities | mean incremental | (a) | (b) | (c) | (d) | (e) | (f) | (g) |
|---|---:|---:|---|---|---|---|---|---|---|
| all/E1-A | 242 | +0.0419 | yes | yes | yes | no | no | no | no |
| all/E1-N | 228 | +0.0303 | yes | yes | yes | yes | yes | yes | no |
| hatu/E1-A | 233 | +0.0563 | yes | yes | yes | yes | yes | n/a | **yes** |
| hatu/E1-N | 224 | +0.0298 | yes | yes | yes | yes | yes | n/a | no |

**The tell that this is a denominator problem and not a disappearing effect is
the Wilcoxon statistic**, which is insensitive to that tail and is significant in
every screen × regime cell under *both* statistics. Only the t-test, which reads
the mean, moves.

The freeze protocol was fixed in advance: *if a registered gate is later shown to
be defective, the frozen verdict stays frozen and the corrected reading is
reported beside it, labelled post-hoc, with the single change named.* Phase 3 had
already exercised it. The frozen verdict is reported and not believed, and both
readings are in the repository.

## The HATU E2 result — both reactants unseen

The project's first detectable both-unseen result, on the single-condition
screen:

| quantity | value |
|---|---|
| incremental pair skill, E2 | **+0.0344** |
| 95% CI | [+0.0127, +0.0561] |
| paired t | **p = 0.0043** |
| folds favouring | 11 / 15 |
| pair blinded to the training marginal | −0.0039 |
| blind drop | **+0.0383**, CI [+0.0175, +0.0591], p = 0.0015 |

The blind diagnostic carries the weight. Replacing the pair term's view of the
two unseen reactants with the training marginal destroys the effect — so the
model is using *these molecules' structures*, not exploiting the marginals. The
full blind table for every block and regime is in
[`phase4_chemlex_interactions.md`](phase4_chemlex_interactions.md).

Phase 3 could not power this regime at all: 45 pairs per fold there against 476
rows here.

**What it is not.** One result, in one screen, which is a strict *subset* of the
other screen — agreement between them is not replication. It is a beginning, not
a capability.

## The analogue boundary is not where Phase 3's was

On the pooled screen, Phase 3's boundary reproduces exactly: the effect lives in
the high-similarity stratum (+0.097 for acids, p = 2e-6; +0.077 for amines,
p = 0.0016), the low stratum is null, high-minus-low survives a congener-family
bootstrap at p = 0.015 and 0.003, and Spearman of per-entity skill against
similarity is +0.186 and +0.168.

On the single-condition screen it is **absent for acids**. The low-similarity
stratum is itself significant at +0.071 (family-bootstrap p = 0.011) and
high-minus-low is −0.0006 (p = 1.00). This is not an influence artefact:
leave-one-out ranges over [+0.062, +0.083], dropping the three most influential
entities leaves +0.048, and restricting to the 58 acids with at least 20 test
rows leaves +0.047 — where the same restriction on the pooled screen takes its
low stratum to +0.012.

**The analogue gradient appears when reaction conditions are pooled and
disappears when they are not.** The obvious reading — that some of what looks
like analogue dependence is condition structure, because close analogues tend to
have been run under the same conditions — is a hypothesis the data suggests and
does not establish. The two screens are nested, so their disagreement is not two
independent measurements.

## The withdrawn claim

An earlier version of this work claimed that **low rank is the useful inductive
bias rather than capacity**, on the strength of a flexible MLP comparator with
roughly twice the parameters that found nothing.

An adversarial review refitted the models and found the flexible rung's
interaction term is numerically zero at the selected fit:

| screen | rung | median fitted pair-term sd | target sd | ratio | alive |
|---|---|---:|---:|---:|---|
| all | `condition_expanded_pair` | 2.523e-01 | 0.3031 | 8.3e-01 | yes |
| all | `lowrank` | 3.077e-01 | 0.3031 | 1.0e+00 | yes |
| all | `flexible` | 2.964e-05 | 0.3031 | 9.8e-05 | **no** |
| hatu | `lowrank` | 5.249e-01 | 0.3146 | 1.7e+00 | yes |
| hatu | `flexible` | 1.130e-19 | 0.3146 | 3.6e-19 | **no** |

**It never leaves its initialisation.** Its reported incremental skill of ≈0.000
is what an *untrained* term scores, not what a flexible model found. **The claim
is withdrawn and nothing in this project supports it.**

The cause is diagnosable and the fix was not: the projections read a 2,048-bit
fingerprint with ~34 bits on, so 94% of the MLP's output at initialisation is a
constant `μ` absorbs, leaving the zero-initialised output layer a gradient of
order 1e-3 along the only direction that matters. LayerNorm and bias removal
changed nothing measurable. `scripts/measure_pair_terms.py` now measures the
fitted term of every rung so this cannot recur silently.

## The positive control, and the floor it sets

A synthetic target on the exact ChemLex entity graph with the exact real
features: additive terms plus a planted **rank-3** acid–amine interaction, at
three sizes, against a both-shuffled arm.

| planted scale | interaction share of sd | E1-A | E1-N | E2 | shuffled E1-A |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 13.8% | −0.0015 | −0.0006 | +0.0030 | −0.0228 |
| 0.5 | 26.9% | +0.0119 | +0.0163 | −0.0042 | **+0.0139** |
| 1 | 48.8% | **+0.1679** | **+0.1155** | **+0.0567** | +0.0045 |

It detects, but **only at the largest planted size**. At half that size the
shuffled arm scores numerically *higher* than the real one. So the smallest
interaction this pipeline resolves through the planted-control gate carries 48.8%
of the target's standard deviation — and **the observed real effect (~+0.05) sits
below that, not above it.**

An earlier statement had this floor at roughly half its value with its conclusion
inverted. That statement is withdrawn. The real effect is resolved through 15
folds and several hundred entities, not through the planted-control gate, and
that is a weaker form of evidence than the original wording implied.

The control is also loose in a second way: a synthetic target's variance
structure is not the real one, so its power curve converts only approximately.

## Condition selection

Conditions were never held out — doing so would confound entity extrapolation
with condition extrapolation. But the deposit's condition assignment is
**adaptive**: a pair was retried under a second reagent *because* it failed under
the first.

| conditions the pair was eventually run under | HATU rows | mean HATU conversion | zero fraction |
|---:|---:|---:|---:|
| 1 | 6,043 | 20.51 | 0.571 |
| 2 | 1,364 | 26.78 | 0.490 |
| 3 | 772 | 3.10 | 0.804 |
| 4 | 275 | 4.99 | 0.709 |

Spearman −0.0784 (p = 5.3e-13, n = 8,454): negative, significant, and **not
monotone**. The point that survives is that condition membership is not
independent of the outcome — so the pooled table's condition covariate carries
outcome information that no causal reading licenses.

**The `hatu` screen exists precisely because of this.** It is the one screen in
which condition confounding cannot occur — one condition, so no
acid-by-condition or amine-by-condition term is identifiable — and it is the only
condition whose membership is not conditioned on a reaction having already failed
elsewhere. It is also the screen that produced the E2 result.

Two further selection structures in the row set itself: roughly 5,600 rows were
deliberately enriched for predicted failure and roughly 6,069 came from MaxMin
diversity down-sampling, with no column saying which is which.

## Where this leaves the external validation

- Transferable acid–amine interaction structure **is** detectable in an
  independent wet-lab chemical system with a directly measured endpoint.
- The registered rule says `INCONCLUSIVE`; the corrected reading says
  `ANALOGUE-ONLY CHEMICAL TRANSFER`; both are published and the difference is one
  named statistic.
- Two reactants unseen at once gave a properly blinded, significant result —
  once, on one screen.
- The analogue boundary reproduces when conditions are pooled and vanishes when
  they are not, and the project does not know which of those is the artefact.

For everything this does not establish, see [`LIMITATIONS.md`](LIMITATIONS.md).
