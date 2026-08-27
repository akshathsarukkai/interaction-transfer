# Changelog

## 1.0.0 — 2026-08-27

First public research release. Prepared as a curated snapshot of a private
research notebook at its audited Phase 4 checkpoint, with new Git history: the
private repository's history contains working documents that are not part of this
artifact, and deleting a file in a later commit does not remove it from a clone.

### What this release contains

- **The code** for all five phases: the synthetic benchmark, the Koplev
  sequential-drug study, the estimator-artifact falsification, the entity-level
  out-of-distribution study, and the ChemLex external chemical validation.
- **The committed evidence** — every claim in the documents is checkable against
  a file in `results/`, and CI fails if a document has drifted from the results it
  reports.
- **Four pre-registered decision rules**, verbatim, in
  [`docs/PREREGISTRATIONS.md`](docs/PREREGISTRATIONS.md).
- **A scientific record** that states what failed, what survived and what was
  withdrawn: [`docs/SCIENTIFIC_RECORD.md`](docs/SCIENTIFIC_RECORD.md).

### Scientific status at this release

- Phase 1 (synthetic): a real but narrow effect, confined to sparse coverage,
  where the structured model only ties the additive null.
- Phase 2 (Koplev): the original symmetric/antisymmetric architecture **did not
  transfer**. Clean negative.
- Phase 2R: pair-specific cyclic structure **is** predictable for unseen pairs
  above ~1,700 training pairs, and not below.
- Phase 2N: registered verdict **LITTLE EVIDENCE FOR ESTIMATOR ARTIFACT** on a
  116/116 ensemble, bounded to the artifacts and nulls that were tested.
- Phase 3: frozen verdict **INCONCLUSIVE**; corrected reading **PAIR-SPECIFIC
  ENTITY TRANSFER**. Transfer is largely analogue interpolation.
- Phase 4: frozen verdict **INCONCLUSIVE**; corrected reading **ANALOGUE-ONLY
  CHEMICAL TRANSFER**. First detectable both-entities-unseen result
  (+0.0344, p = 0.0043) on the single-condition HATU screen.

### Withdrawn in the run-up to this release

- **"Low rank is the useful inductive bias rather than capacity."** The flexible
  comparator that claim rested on never fitted — its interaction term never left
  its initialisation. Withdrawn; nothing in this project supports it.
- **The stated detection floor.** It was given at roughly half its value with its
  conclusion inverted; the observed effect sits below the floor, not above it.

### Changes made while preparing the public snapshot

These were made to the released code and documents, not to any result:

- **Removed the ChemLex substrate inventory from committed results.** Five
  per-entity CSVs carried a `smiles` column holding 497 of the deposit's 503
  distinct reactant structures. The deposit is CC BY-NC 4.0. The column is gone,
  both writers drop it, and CI now fails if it returns. No numeric result changed;
  `role` + `entity` was already the join key.
- **Fixed a duplicate-aggregation hazard.** The Phase 4 results index counted
  every `.jsonl` in its directory, so two regenerable diagnostics would have
  silently rewritten its headline from 173 conditions to 285. Replaced with an
  explicit allow-list of the five decision-rule blocks.
- **Recorded the flexible-comparator withdrawal in the document that reports the
  rung.** The withdrawal existed in the research record and not in the generated
  Phase 4 document, which tabulated the dead rung 45 times without a note.
- **Replaced a hand-typed number with its generated table.** The Phase 4 document
  stated a condition-adaptivity gradient of "21.7 → 3.3 → 0.0" that matched no
  computation and reached monotonicity by omitting the stratum that reverses it,
  in a document whose own header says no number is typed. It now renders the
  generated table, which is not monotone.
- **Corrected the split-grouping prose.** It described three relations and
  disclaimed the tautomer relation; the code applies four and has since the defect
  fix that added it. The document now describes what the code does and says why
  the relation was added.
- **Disclosed an unrun registered analysis.** The fifth registered Phase 4
  sensitivity was never implemented. Named rather than left silent.
- **Removed a personal email address** from two scripts' outbound User-Agent
  strings, now read from `INTERACTION_TRANSFER_CONTACT`.
- **Published the pre-registrations.** Extracted verbatim from the private
  notebook so that a frozen verdict can be checked against the rule that produced
  it.
- **Added licensing metadata**: `LICENSE` (Apache-2.0), `NOTICE`,
  `THIRD_PARTY_DATA.md`, `CITATION.cff`, and explicit upstream GPL-3.0 notices in
  the two files that quote d-chain source.

### Not included

The private research notebook's chronological record, its raw internal review,
and the fetched third-party deposits. See
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for how to obtain the data.
