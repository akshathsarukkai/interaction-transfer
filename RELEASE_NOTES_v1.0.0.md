First public research release, cut from an audited Phase 4 scientific checkpoint.

**This is a research artifact. It has not been peer reviewed.**

## The problem

Most of what science measures is a combination, and the grid is almost always
mostly empty. Writing the outcome as `y(i, j) = f(i) + f(j) + I(i, j)`, the pair
term `I` is the only part that cannot be obtained by measuring the entities
separately — so it is the part worth predicting. This release asks when `I`
transfers: to an unseen combination of seen entities, to a combination involving
one entity never observed, and to one in which neither entity was observed.

## Koplev — unseen pairs

Two 100×100 ordered sequential-drug matrices. The directional effect
`D(i,j) = y(i→j) − y(j→i)` splits exactly into a generic ordering potential
`g_i − g_j` and genuinely pair-specific cyclic structure `C(i,j)` — 46% of the
directional variance on A375 and 60% on PANC1.

That cyclic part is predictable for entirely unseen pairs, **above a coverage
threshold and not below it**: +0.229 and +0.353 skill on A375 at coverages 0.40
and 0.70 (8/8 seeds), against +0.022 at 0.20 with a CI crossing zero. A rank-2
model with 204 parameters and no hyperparameter search reaches +0.197.

The original symmetric/antisymmetric architecture the project started from **did
not transfer** — nothing at sparse coverage, and a significant deficit at high
coverage on 0 of 8 seeds. That negative is what redirected the work.

## d-chain null — the signature is not an estimator artifact

The Koplev endpoint is a posterior mean from one joint ~45,000-parameter Bayesian
fit, so a shared-parameter estimator could in principle manufacture a low-rank
cyclic signature from nothing. Testing that honestly meant running **the
published sampler itself**: fetched at a pinned commit, digest-verified, and the
build refused unless the patched program reproduces the unpatched one byte for
byte on the authors' own data.

On simulated screens where the true pair-specific interaction is exactly zero,
best held-out rank-2 skill is **+0.0003**, against +0.16 to +0.25 on real data.
Registered verdict on a complete 116/116 ensemble: **LITTLE EVIDENCE FOR
ESTIMATOR ARTIFACT** — bounded to the artifacts and nulls that were tested, and
only below the detector's ~5% sensitivity floor.

## Koplev — one unseen drug

Holding out drugs rather than pairs gives a real signal that needs the held-out
drug's own structure (+0.0273 A375, +0.0536 PANC1), and it is **largely analogue
interpolation**: +0.090 / +0.138 for the 31 drugs with a close training analogue,
and not detectable for the 40 chemically distant ones (p = 0.13, p = 0.22).
Frozen verdict **INCONCLUSIVE**; corrected reading **PAIR-SPECIFIC ENTITY
TRANSFER**.

## ChemLex — external chemical validation

11,669 wet-lab acid–amine coupling reactions, 272 acids × 230 amines, a directly
measured endpoint, and a **bipartite** interaction in which antisymmetry is not
merely unhelpful but undefined.

**The frozen, pre-registered verdict is `INCONCLUSIVE`** — no cell clears every
registered criterion and the two screens fail different ones.

**The same rule with one statistic corrected returns `ANALOGUE-ONLY CHEMICAL
TRANSFER`, and that is the reported reading.** The single change: the per-entity
denominator becomes the fold's baseline MSE rather than the entity's own, which
is not bounded away from zero. The tell that this is a denominator problem rather
than a vanishing effect is the Wilcoxon statistic, significant in every cell
under both statistics; only the t-test moves.

## HATU — both reactants unseen

The project's first detectable both-entities-unseen result, on the
single-condition screen where condition confounding cannot occur:

| quantity | value |
|---|---|
| incremental pair skill | **+0.0344** |
| 95% CI | [+0.0127, +0.0561] |
| paired t | **p = 0.0043** |
| folds favouring | 11 / 15 |
| blinded to the training marginal | −0.0039 |
| blind drop | **+0.0383**, p = 0.0015 |

One result, one screen — and that screen is a strict subset of the other, so
their agreement is not replication.

## Withdrawn claims

- **"Low rank is the useful inductive bias rather than capacity" is withdrawn.**
  The flexible comparator it rested on **never fitted**: its interaction term has
  a fitted standard deviation of order 1e-19 against 0.5–0.6 for the low-rank
  term on the same folds. An incremental skill of ~0.000 from an untrained term
  is a training failure that prints a number, not a finding.
- **The stated detection floor is withdrawn.** It was given at roughly half its
  value with its conclusion inverted. The positive control detects only at a
  planted size carrying 48.8% of the target's standard deviation, and the
  observed effect sits *below* that.

## Limitations

Two external systems, not a sample. The Koplev target is estimator-derived.
Entity extrapolation is largely analogue interpolation, and the analogue boundary
reproduces on the pooled ChemLex screen but vanishes on the single-condition
one — the project does not know which of those is the artefact. Phase 4's 15
folds are 3×5 repeated CV treated as independent, which understates the standard
error. The endpoint's replicate ceiling is Pearson ≈ 0.6. Three defects forced a
complete corrected re-run, and the adversarial review that found them was stopped
early with 24 material findings never adjudicated.

Full list: `docs/LIMITATIONS.md`.

## Reproduction

```bash
git clone https://github.com/akshathsarukkai/interaction-transfer
cd interaction-transfer
python -m pip install -e ".[dev]"
pytest
```

No third-party dataset is vendored; each acquisition script digest-verifies what
it fetches. Full commands and costs in `REPRODUCIBILITY.md`; licences and
provenance in `THIRD_PARTY_DATA.md`; the four pre-registered decision rules,
verbatim, in `docs/PREREGISTRATIONS.md`.
