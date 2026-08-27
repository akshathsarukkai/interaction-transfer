# Overview

## The problem

Most of what science measures is a *combination*. Two drugs given in sequence.
An acid coupled to an amine. A gene knocked out in a particular cell line. The
number of possible combinations grows as the product of the entities involved,
and the number anyone can actually measure grows linearly with money and time —
so the interesting regime is always the one where most of the grid is empty.

Write the outcome of combining two entities as

```
y(i, j) = f(i) + f(j) + I(i, j)
```

`f(i)` and `f(j)` are what each entity does on its own. `I(i, j)` is the part
that belongs to *the pair* — the surprise, the synergy, the interaction. It is
also the only part that cannot be predicted by measuring the entities
separately, which is exactly why it is the part worth predicting.

This project asks when `I` generalises, and it asks it in a specific,
increasingly demanding order.

## The generalisation hierarchy

Three regimes, in strictly increasing difficulty. They are not the same question
and a result in one says almost nothing about the next.

| Regime | What is held out | What the model has seen |
|---|---|---|
| **Unseen pair** | the combination `(i, j)` | both `i` and `j`, in other combinations |
| **One unseen entity** | every combination involving `j` | `i` and its combinations; `j` only as a structure |
| **Two unseen entities** | every combination involving `i` *or* `j` | neither entity, only their structures |

The first regime is a matrix-completion problem: the model has direct evidence
about both entities and only has to fill in a hole. The second requires the model
to have learned a *map* from an entity's description — a molecular
fingerprint — to how that entity interacts, and to apply it to a description it
has never been trained on. The third requires that map to work at both ends
simultaneously, with no anchor at all.

Most published work on combination prediction lives in the first regime. This
project starts there and pushes into the second and third.

## The two systems

Deliberately different in domain, measurement type, and structure.

### Koplev — sequential drug pairs

Two 100×100 ordered matrices of sequential anticancer treatments (A375 melanoma,
PANC1 pancreatic), every one of the 4,950 unordered pairs measured in both
directions. **Order matters here**: drug `i` then drug `j` is a different
experiment from `j` then `i`. That makes the *directional* effect

```
D(i, j) = y(i → j) − y(j → i)
```

a meaningful object, and it is antisymmetric by construction. It decomposes
exactly — a discrete Hodge decomposition — into

```
D(i, j) = (g_i − g_j) + C(i, j)
```

where `g_i − g_j` is a **generic ordering tendency** (some drugs simply work
better first) and `C(i, j)` is **genuinely pair-specific circulation** — the part
that cannot be produced by ranking the drugs on a line. `C` is 46% of the
directional variance on A375 and 60% on PANC1.

The catch, and it is a serious one: the Koplev endpoint is **not a
measurement**. It is a posterior mean from a single joint Bayesian fit over the
whole screen, so held-out pairs are not statistically independent of training
pairs. That objection gets its own experiment; see
[`KOPLEV.md`](KOPLEV.md).

### ChemLex — acid–amine coupling

11,669 wet-lab reactions over 272 carboxylic acids and 230 amines under seven
coupling conditions, with a directly measured endpoint. The structure is
**bipartite**: an acid and an amine are different kinds of thing, so

```
y(a, n, c) = μ + f_A(a) + f_N(n) + f_C(c) + I_AN(a, n)
I_AN(a, n) = z_A(a)ᵀ W z_N(n)
```

**There is no antisymmetry here and there should not be.** `I(n, a)` does not
typecheck — you cannot swap an acid for an amine. What carries over from Koplev
is not the algebra; it is the low-rank bilinear interaction term and the question
of whether it transfers. See [`CHEMLEX.md`](CHEMLEX.md).

## How a claim gets made here

Three habits do most of the work, and they are the reason the results are worth
the space they take:

**Pre-registration.** Each phase's decision rule was written down and committed
before the experiment ran. All four are published verbatim in
[`PREREGISTRATIONS.md`](PREREGISTRATIONS.md). Two of the four returned
`INCONCLUSIVE`, which is precisely when having the original text matters.

**Generated documents.** No number in a phase document is typed by hand. The
report scripts write the tables and CI fails if a committed document has drifted
from the committed results. This was not a design principle; it was installed
after an audit found four hand-copied p-values that matched no run in the
repository.

**Controls that can invalidate.** Every phase runs shuffled-feature and
random-feature controls, and a planted positive control that establishes the
smallest interaction the pipeline can resolve — which is the number that bounds
a negative result. On Phase 4 the positive control detects only at the largest
planted size, and the real effect sits *below* that. That is reported.

## Where to read next

| Document | What it holds |
|---|---|
| [`SCIENTIFIC_RECORD.md`](SCIENTIFIC_RECORD.md) | the intellectual history: what failed, what survived, what was withdrawn |
| [`KOPLEV.md`](KOPLEV.md) | the drug study end to end — Phases 2, 2R, 2N, 3 |
| [`CHEMLEX.md`](CHEMLEX.md) | the chemistry study — Phase 4, both verdicts, the E2 result |
| [`LIMITATIONS.md`](LIMITATIONS.md) | what none of this establishes |
| [`PREREGISTRATIONS.md`](PREREGISTRATIONS.md) | the four registered decision rules, verbatim |
| [`PHASE1_SYNTHETIC.md`](PHASE1_SYNTHETIC.md) | the synthetic benchmark the project started from |
| [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) | exact commands, costs, and which artifacts are authoritative |
