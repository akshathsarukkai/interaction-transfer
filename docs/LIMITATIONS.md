# Limitations

What this project does not establish. Read this next to
[`SCIENTIFIC_RECORD.md`](SCIENTIFIC_RECORD.md), which says what it does.

## Scope

**Two external scientific systems, and they are not a sample.** One sequential
drug screen and one acid–amine coupling screen. Two systems can show that a
finding is not unique to one dataset; they cannot show it is general. Nothing
here supports a claim about interaction structure in biology or chemistry at
large.

**The two ChemLex screens are nested.** `hatu` is a strict subset of `all`.
Where they agree, that is not replication. Where they disagree — and on the
analogue boundary they do — that is not two independent measurements either.

**One representation.** ECFP4 fingerprints throughout, deliberately chosen as the
simplest defensible option. A negative result for chemically distant entities is
a result about ECFP4 at this sample size, not about chemistry.

## The Koplev target

**It is estimator-derived, not measured.** The synergy values are posterior means
from a single joint Bayesian fit over the whole screen. Held-out pairs share
fitted single-agent curves with training pairs and are therefore **not
statistically independent of them**. Every Koplev result inherits this.

**Phase 2N falsifies tested estimator artifacts, not every possible source of
misspecification.** The null tests sampling error in a *correctly specified*
d-chain fit. A misspecified likelihood, a systematic bias in the deposit, or an
artifact the simulator does not generate would all pass this test. The
falsification also only bounds any artifact *below the detector's sensitivity
floor* of roughly 5% — it does not show the artifact is zero.

## Generalisation limits

**Unseen-pair prediction has a coverage threshold, and below it there is
nothing.** On Koplev the low-rank residual model is null at coverages 0.05, 0.10
and 0.20 (842 training pairs), and strong at 0.40 (1,683) and 0.70. The finding is
"predictable given enough of the graph", not "predictable".

**Entity extrapolation is weak, and mostly analogue interpolation.** On Koplev,
transfer to an unseen drug is significant only for the 31 drugs with a close
training analogue (+0.090 / +0.138) and is **not detectable** for the 40
chemically distant ones (+0.016, p = 0.13; +0.015, p = 0.22). The honest label
is analogue-only transfer.

**The analogue boundary does not reproduce consistently.** It reproduces on the
pooled ChemLex screen and disappears on the single-condition one, where the
low-similarity stratum is itself significant. The offered explanation — that part
of the apparent analogue dependence is condition structure — is a hypothesis the
data suggests and does not establish. **This is an open question, not a resolved
one.**

**Two unseen entities: one result, one screen.** The HATU E2 finding (+0.0344,
p = 0.0043, 11/15 folds, blind drop p = 0.0015) is the only both-unseen result
in the project. Phase 3 could not power that regime at all. And **strict E2 does
not survive the registered sensitivities** — the unseen-amine effect roughly
halves under replicate-cell aggregation.

## Statistical limitations

**The frozen Phase 4 verdict is `INCONCLUSIVE`.** The registered rule,
implemented literally on the registered statistic, does not conclude. The
corrected post-hoc reading is `ANALOGUE-ONLY CHEMICAL TRANSFER`. Both are
published. A reader who wants only the registered answer must take
`INCONCLUSIVE`.

**Phase 3's frozen verdict is also `INCONCLUSIVE`**, for the same class of
reason — a validity gate reading skill-against-zero rather than incremental
skill. Its corrected reading is `PAIR-SPECIFIC ENTITY TRANSFER`.

**Repeated CV is treated as independent, and it is not.** Phase 4's 15 folds are
3 × 5 repeated cross-validation over the same 11,669 rows. Each entity is a test
entity exactly three times. Every fold-level confidence interval and p-value in
the Phase 4 documents treats them as independent, which **understates the
standard error** — and registered criteria (d) and (e) are decided by exactly
those t-tests. This is a limitation of the registered design, recorded rather
than corrected: no variance estimator for repeated CV was registered, and
substituting one afterwards would be a second undeclared change. **Prefer the
per-entity analysis where the two disagree.**

Phase 2R's eight split seeds are likewise not independent: their evaluation pools
overlap at mean pairwise Jaccard 0.105, and with n = 8 the Wilcoxon p floor is
0.0078 — which several reported results sit exactly at.

**The endpoint has a hard noise ceiling.** Replicate ChemLex measurements
correlate at Pearson ≈ 0.6, so roughly half the endpoint variance is unavailable
to any deterministic predictor. Every R² must be read against that.

**Positive-control sensitivity is poor, and the real effect is below it.** The
planted-interaction control detects only at scale 1, carrying 48.8% of the
target's standard deviation. At 26.9% the shuffled arm scores numerically higher
than the real one. The observed real effect sits **below** the detection floor,
and is resolved through folds and entities rather than through the control gate.
The control is also loose: a synthetic target's variance structure is not the
real one.

**Controls are weak on the pooled ChemLex screen.** Control separation is
significant on the single-condition screen and not on the pooled one, where on
unseen amines the shuffled control is numerically *higher* than the real
representation on the five control folds. Eight of the twelve control cells are
off-role — they shuffle the endpoint the model has already trained on — and are
not really controls; the registration did not distinguish them and the frozen
rule is evaluated as written.

**The secondary endpoint contradicts the primary in one cell.** On the binary
feasibility endpoint at the authors' own threshold, `all`/E1-N shows the pair
term significantly *degrading* performance (−0.0233, p = 0.0175, 5/15 folds) —
in the one cell whose core criteria passed on the continuous endpoint.

**Condition selection is adaptive and not adjusted for.** A pair was retried
under a second reagent because it failed under the first, so the pooled screen's
condition covariate carries outcome information no causal reading licenses.

## Claims explicitly not made

**No claim that low rank is uniquely optimal, or optimal at all.** The comparator
built to support that claim never trained; the claim is withdrawn. Nothing here
compares low-rank structure against a *working* flexible model.

**No claim of mechanistic causality from latent-axis correlations.** 66 of 94
correlations between learned interaction axes and pre-specified descriptors
survive Benjamini–Hochberg at q < 0.05, which is as much a caution as a result.
Nothing in that analysis feeds any decision rule, and no mechanism is claimed
from a latent coordinate.

**No claim that broad chemical extrapolation is solved, or approached.** The
transfer that exists is largely to analogues. The one both-unseen result is a
single result in a single screen.

**No claim that this constitutes a general chemical or biological world model.**

**No claim that the original symmetric/antisymmetric hypothesis was validated.**
It helped on synthetic data at one coverage, did not transfer to Koplev, and does
not typecheck on bipartite chemistry.

**The transductive ceiling is not an entity-generalisation result.**
`results/phase4_chemlex/transductive.jsonl` holds out pairs, not entities. The
report refuses to place it in an entity-OOD table and so should any reader.

## Process limitations

**The adversarial review was stopped early.** Eighteen findings upheld of the
twenty-five adjudicated, from eighty raw findings of which forty-nine were
claimed material. **Twenty-four material findings were never adjudicated.** The
corrected re-run is the better artifact and is not the product of a completed
review.

**One registered analysis was never run.** The fifth registered Phase 4
sensitivity — incremental pair skill on rows with `Conversion > 0` only — was
registered and never implemented. It was outcome-conditioned and could never have
been a headline, so it changes no verdict.

**Some Phase 1 provenance is irrecoverable.** `results/hparam_search_stage1.log`
is the only surviving evidence for stage-1 hyperparameter selection; its raw
JSONL was overwritten by an accidental re-launch.

**Prose can stop matching its data without anything failing.** The repository
checks that every cited path resolves, that every result file is indexed, and
that every generated table matches its results. It cannot check that a sentence
still describes the number beside it. Four defects of exactly that kind were
found by review rather than by CI, and that remains the residual risk.
