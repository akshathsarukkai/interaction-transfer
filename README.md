# Interaction Transfer to Unseen Scientific Entities

**Predicting pair-specific effects for unseen combinations, unseen interventions,
and pairs of entirely unseen entities in drug-response and chemical-reaction data.**

[![tests](https://github.com/akshathsarukkai/interaction-transfer/actions/workflows/ci.yml/badge.svg)](https://github.com/akshathsarukkai/interaction-transfer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

---

## Overview

Many scientific prediction problems involve interactions between two entities:
two drugs, two reactants, a substrate and a catalyst, or two perturbations.

A useful starting point is

```text
y(i, j) = f(i) + f(j) + I(i, j)
```

where `f(i)` and `f(j)` describe the independent contributions of the two
entities, and `I(i, j)` is the part of the outcome specific to the pair.

This project studies when `I(i, j)` can be predicted outside the combinations
used for training.

The distinction between different kinds of generalization is central. Predicting
a new combination of two familiar entities is a substantially easier problem
than predicting how an entity with no experimental history will interact with
anything else.

The experiments here therefore separate three settings:

1. a pair is unseen, but both entities have appeared elsewhere;
2. one entity is entirely unseen experimentally;
3. both entities are unseen.

The work is evaluated on two different scientific systems:

- the **Koplev sequential-drug screen**, where treatment order matters;
- the **ChemLex acid–amine reaction dataset**, where acids and amines occupy
  distinct chemical roles.

The project began as a test of explicit symmetric and antisymmetric
parameterizations. That hypothesis did not survive unchanged on real data. The
later phases instead focus on a narrower question: whether pair-specific
interaction structure is reusable, and how far that structure transfers to
unseen entities.

## The scientific question

> Given a partially observed set of combination outcomes and structural
> descriptions of the entities, when can pair-specific interaction effects be
> predicted for combinations — and for entities — that were never measured?

## The generalisation hierarchy

| Regime | Held out | What remains available |
|---|---|---|
| **Unseen pair** | the combination `(i, j)` | both entities appear in other combinations |
| **One unseen entity** | every observation involving `j` | `j` is available only through its structure |
| **Two unseen entities** | every observation involving `i` or `j` | neither entity has experimental training data |

The first regime is close to matrix completion. The latter two require the model
to infer interaction behavior from properties of entities it has never observed
experimentally.

## Main findings

| # | Finding | Result |
|---|---|---|
| 1 | The original symmetric/antisymmetric architecture did **not** improve prediction on the real sequential-drug data | negative result |
| 2 | Directional drug effects contain a substantial pair-specific component after generic per-drug ordering effects are removed | 46–60% of directional variance |
| 3 | That residual interaction component becomes predictable for unseen drug pairs once enough of the pair graph is observed | up to +0.37 held-out skill |
| 4 | A reconstruction of the estimator used to produce the Koplev target does not reproduce the observed structure under the tested zero-interaction nulls | best null skill +0.0003 |
| 5 | Molecular structure contains information about the interaction behavior of an unseen drug | positive overall, concentrated among close analogues |
| 6 | Transferable pair structure also appears in an independent wet-lab chemical system | ChemLex external validation |
| 7 | Under a fixed HATU condition, the pair model improves prediction when **both reactants are unseen** | +0.0344 incremental skill, p = 0.0043 |

The project does **not** establish that the original algebraic architecture is
correct, that low rank is uniquely optimal, or that broad molecular
extrapolation has been solved.

---

## The Koplev sequential-drug study

The Koplev dataset contains two ordered sequential-treatment screens over 100
anticancer drugs, measured in A375 and PANC1 cells. Each unordered drug pair was
evaluated in both treatment orders.

For drugs `i` and `j`, define

```text
D(i, j) = y(i → j) − y(j → i)
```

The directional matrix can be decomposed as

```text
D(i, j) = (g_i − g_j) + C(i, j)
```

where:

- `g_i − g_j` captures a generic per-drug tendency to perform better first or
  second;
- `C(i, j)` is the remaining pair-specific directional component.

This distinction matters because good prediction of `D(i, j)` alone does not
establish pair-specific interaction structure. A model could perform well simply
by learning that certain drugs generally prefer one position in the sequence.

| screen | generic ordering component | pair-specific cyclic component |
|---|---:|---:|
| A375 | 53.8% | 46.2% |
| PANC1 | 39.8% | 60.2% |

### Phase 2 — the original architecture did not transfer

The original model imposed the symmetric/antisymmetric structure that motivated
the project.

On the real Koplev data, that parameterization did not improve performance over
a capacity-matched unrestricted model. At sparse coverage the difference was
approximately zero; at higher coverage the structured model performed worse.

This result redirected the project away from asking whether the original
architecture was useful and toward asking what structure in the real interaction
matrix was actually predictable.

### Phase 2R — predicting the pair-specific residual

After removing the generic ordering component, the residual `C(i, j)` becomes
predictable once enough of the pair graph has been observed.

| screen | coverage | training pairs | skill on unseen pairs | 95% CI | seeds > 0 |
|---|---:|---:|---:|---|---:|
| A375 | 0.20 | 842 | +0.022 | [−0.011, +0.055] | 4/8 |
| A375 | 0.40 | 1,683 | **+0.229** | [+0.189, +0.269] | 8/8 |
| A375 | 0.70 | 2,945 | **+0.353** | [+0.315, +0.391] | 8/8 |
| PANC1 | 0.40 | 1,683 | **+0.198** | [+0.147, +0.248] | 8/8 |
| PANC1 | 0.70 | 2,945 | **+0.366** | [+0.323, +0.410] | 8/8 |

At 20% coverage the effect is weak. At 40–70% coverage, every reported seed is
positive.

A compact low-rank model is sufficient to recover much of this signal. That
shows that the observed structure can be represented economically; it does not
establish that low rank is uniquely preferable to a properly trained flexible
alternative.

→ [`docs/KOPLEV.md`](docs/KOPLEV.md)

## Estimator-artifact falsification

The Koplev synergy endpoint is not a raw experimental measurement. It is a
posterior mean produced by the authors' joint Bayesian d-chain model.

That creates an alternative explanation for the previous result: shared
parameters in the estimator itself might induce reusable low-dimensional
structure, even if no such structure exists in the underlying interactions.

To test this, the project reconstructs the published estimator and applies it to
simulated datasets in which the true reusable pair-specific interaction is zero.

Across 116 registered null conditions:

| | null screens | real data |
|---|---:|---:|
| best held-out rank-2 skill | **+0.0003** | +0.16 to +0.25 |

The registered conclusion is:

**LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT**

This conclusion is intentionally narrow. It addresses the tested null models and
the reconstructed estimator. It does not exclude every possible form of
estimator-induced structure or model misspecification.

→ [`docs/dchain_null_falsification.md`](docs/dchain_null_falsification.md)

## The entity-OOD drug study

Predicting an unseen pair still allows the model to learn both drugs from their
interactions with other partners.

The next experiment removes that advantage.

For each entity-OOD fold, every experimental observation involving a test drug is
excluded from training, validation, preprocessing, and model selection. The
model receives only the drug's molecular representation.

Using ECFP4 fingerprints, the pair model improves over the additive baseline by:

- **A375:** +0.0273
- **PANC1:** +0.0536

Blinding the unseen drug's molecular representation largely removes the
improvement, indicating that the result depends on information contained in the
held-out structure.

The effect is not uniform across chemical space:

| screen | similarity stratum | drugs | incremental skill | p |
|---|---|---:|---:|---:|
| A375 | high similarity | 31 | **+0.090** | 0.0024 |
| A375 | low similarity | 40 | +0.016 | 0.13 |
| PANC1 | high similarity | 31 | **+0.138** | 0.00029 |
| PANC1 | low similarity | 40 | +0.015 | 0.22 |

The high-similarity strata are significant; the low-similarity strata are not.

The most conservative interpretation is that Koplev supports molecularly
informed interpolation to unseen drugs with close analogues, rather than broad
extrapolation across chemical space.

The frozen registered decision rule returned **`INCONCLUSIVE`**. A post-hoc
correction to one statistic returns **`PAIR-SPECIFIC ENTITY TRANSFER`**. Both are
retained in the public record.

The both-unseen regime was not sufficiently powered in this dataset to establish
a result.

→ [`docs/phase3_entity_ood.md`](docs/phase3_entity_ood.md)

---

## ChemLex external chemistry validation

ChemLex provides a substantially different test of the same question.

The dataset contains 11,669 measured acid–amine coupling reactions over 272
acids and 230 amines. Unlike the Koplev system, the interaction is bipartite:
acids and amines are different entity types, so there is no symmetry or
antisymmetry assumption.

The model is

```text
y(a, n, c) = μ + f_A(a) + f_N(n) + f_C(c) + I_AN(a, n)

I_AN(a, n) = z_A(a)ᵀ W z_N(n)
```

The question is whether the pair term improves prediction beyond independent
acid, amine, and condition effects when one or both reactants are absent from
training.

### Registered and corrected analyses

The frozen preregistered Phase 4 decision rule returns:

**`INCONCLUSIVE`**

No screen/regime combination satisfies every registered criterion.

A post-hoc analysis corrects one unstable per-entity statistic. The original
statistic divides each entity's pair-model error by that same entity's baseline
MSE. For entities whose baseline error is close to zero, the ratio can become
arbitrarily large despite a small absolute difference between models.

The corrected statistic instead uses the fold-level baseline MSE as the
denominator.

Under that correction, the same decision framework gives:

**`ANALOGUE-ONLY CHEMICAL TRANSFER`**

The registered and corrected conclusions are reported separately.

### Both reactants unseen

The strongest Phase 4 result occurs on the single-condition HATU screen.

When both the acid and amine are absent from training:

| quantity | value |
|---|---|
| incremental pair skill | **+0.0344** |
| 95% CI | [+0.0127, +0.0561] |
| paired t | **p = 0.0043** |
| folds favouring | 11 / 15 |
| pair blinded to the training marginal | −0.0039 |
| blind drop | **+0.0383**, p = 0.0015 |

The blinding control replaces information about the unseen reactants with the
training marginal. Its collapse toward zero indicates that the improvement
depends on the structures of the held-out reactants rather than only on global
pair statistics.

The ChemLex results also complicate the analogue-only interpretation. Similarity
dependence is strong when all reaction conditions are pooled, but the
low-similarity acid stratum remains positive on the single-condition HATU screen.

One possible explanation is that molecular similarity and condition assignment
are partly entangled in the pooled data. The current experiments do not establish
that interpretation causally.

→ [`docs/CHEMLEX.md`](docs/CHEMLEX.md)

---

## Key limitations and withdrawn claims

Several limitations materially affect the interpretation of the results.

- **The frozen Phase 4 verdict is `INCONCLUSIVE`.** The corrected post-hoc
  interpretation is reported separately and does not replace the registered
  result.
- **The project does not establish that low rank is the preferred inductive
  bias.** The flexible Phase 4 comparator failed to train: its interaction term
  remained effectively at initialization, so its performance cannot be used as
  evidence against flexible models.
- **The positive-control sensitivity is larger than the observed real-data
  effect.** The registered detection gate is reached only at a planted signal
  corresponding to roughly 48.8% of the target standard deviation.
- **Three Phase 4 defects required a corrected re-run:** non-deterministic
  process-salted seeds, control representations re-permuted by fold, and
  chemically equivalent acids appearing in different split groups.
- **Repeated cross-validation folds are not statistically independent.** The
  per-entity analyses are therefore preferable where fold-level and entity-level
  inference disagree.
- **Only two external scientific systems are studied.**
- **The Koplev target is estimator-derived**, although the dedicated null study
  does not reproduce the observed signature under the tested artifact models.
- **Koplev entity-OOD transfer is concentrated among close structural
  analogues.**
- **ChemLex outcomes are noisy**, limiting attainable deterministic prediction.
- **Condition assignment in ChemLex is adaptive**, which may contribute to the
  similarity dependence observed in pooled-condition analyses.
- **Latent-axis correlations are exploratory.** They are not presented as
  evidence of chemical mechanism.

The project does not claim that the original algebraic hypothesis was validated,
that low rank is uniquely correct, or that broad chemical extrapolation has been
solved.

→ [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) ·
[`docs/SCIENTIFIC_RECORD.md`](docs/SCIENTIFIC_RECORD.md)

---

## Reproduction

Install the package and run the test suite:

```bash
git clone https://github.com/akshathsarukkai/interaction-transfer
cd interaction-transfer
python -m pip install -e ".[dev]"
pytest
```

The test suite uses generated fixtures for experiments that depend on external
datasets, so a clean checkout does not require the Koplev or ChemLex deposits.

Basic pipeline checks:

```bash
python scripts/run_phase1.py --smoke --workers 2

python scripts/run_phase4_chemlex.py \
  --part smoke \
  --workers 2 \
  --raw-dir tests/fixtures/chemlex_tiny
```

External data can be acquired with:

```bash
python scripts/download_koplev.py
python scripts/download_chemlex.py
python scripts/prepare_dchain_null.py
```

Full experiment commands, computational requirements, and the distinction
between authoritative and superseded artifacts are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Repository layout

```text
src/intervention_algebra/   models, training and evaluation
scripts/                    acquisition, experiments and report generation
tests/                      invariants, leakage guards and pipeline tests
results/                    committed result artifacts
configs/                    selected experiment configurations
data/external/              derived public metadata, not raw deposits
docs/                       detailed analyses and scientific record
```

The repository is named `interaction-transfer`, while the Python distribution
and import package remain `intervention-algebra` / `intervention_algebra` for
compatibility with the original research code.

For Phase 1, do not aggregate every JSONL file under `results/` directly.
Several files are subsets of the authoritative result table and would
double-count runs if combined naively.

[`results/README_RESULTS.md`](results/README_RESULTS.md) documents which files are
authoritative and which are retained as superseded evidence.

---

## Data provenance

Raw third-party datasets are not distributed in this repository. Acquisition
scripts retrieve the original resources and verify recorded digests where
applicable.

| Resource | Licence | In repo? |
|---|---|---|
| Koplev sequential drug screen (`10.17632/wgybvcvjwf.1`) | CC BY 4.0 | No — fetched |
| ChemLex acid–amine screen (`10.5281/zenodo.17596563`) | CC BY-NC 4.0 | No — fetched |
| d-chain sampler (`skoplev/d-chain`) | GPL-3.0 | No — fetched from upstream |
| PubChem / ChEMBL | see original sources | derived mapping tables only |

Full provenance, acquisition instructions, and licensing details are in
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).

---

## Citation

```bibtex
@software{sarukkai_interaction_transfer_2026,
  author  = {Sarukkai, Akshath},
  title   = {Interaction Transfer to Unseen Scientific Entities},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/akshathsarukkai/interaction-transfer}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

Please also cite the original datasets used in derivative work.

This repository is a research artifact and has not been peer reviewed.

---

## License

Original project code is licensed under the **Apache License 2.0**
([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).

External datasets and third-party code retain their original licences and are
not relicensed by this repository.

See [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).
