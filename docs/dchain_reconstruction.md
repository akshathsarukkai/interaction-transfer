# The d-chain estimator, reconstructed from source

This document exists so that the null experiment in
[`dchain_null_falsification.md`](dchain_null_falsification.md) can be read
without trusting a summary. Everything below is either quoted from the primary
source or derived from it in one step, and each claim says which.

**Status of the reconstruction: the published sampler is not approximated. It is
compiled and run.** `dchain.cpp` is a single self-contained C++11 file with no
dependency on anything but the standard library and Boost's command-line
parser; `src/intervention_algebra/real_data/dchain_null/dchain.py` fetches it at
a pinned commit, verifies its digest, replaces the Boost CLI with a standard
shim, and compiles it. The measurement layer (`post/interpretMCMC.R`) is ported
to NumPy in `dchain_null/synergy.py`. §6 gives the fidelity evidence for both.

---

## 1. Sources

| what | where | verified |
|---|---|---|
| paper | Koplev S, Longden J, Ferkinghoff-Borg J, Blicher Bjerregard M, Cox TR, Erler JT, Pedersen JT, Voellmy F, Sommer MOA, **Linding R**. "Dynamic Rearrangement of Cell States Detected by Systematic Screening of Sequential Anticancer Treatments." *Cell Reports* 20(12):2784–2791, 2017. doi:10.1016/j.celrep.2017.08.095 | read (main text + `mmc1.pdf` Supplemental Experimental Procedures) |
| code | <https://github.com/skoplev/d-chain>, GPL-3.0, commit `72b2445786daa13c3df41aa4b2312b84a7f79266` (2022-08-21) | all four files read in full; digests pinned in `dchain_null/dchain.py` |
| data | Mendeley Data doi:10.17632/wgybvcvjwf.1, five modelled tables | on disk, digests in `data/raw/koplev2017/PROVENANCE.json` |
| tutorial site | `https://dchain.lindinglab.org`, cited by both the paper and the repo README | **dead** (NXDOMAIN); any documentation there is lost |

The repository is four files: `README.md`, `dchain.cpp` (1,045 lines — the whole
model), `post/interpretMCMC.R` (227 lines — the whole scoring step), and
`data/viability_data.csv` (66 rows — a two-drug example, **not** the screen).

The lab is Linding's (BRIC Copenhagen / DTU), not Mount Sinai; Koplev moved to
Mount Sinai later.

Two independent reconstruction agents were run against these sources with no
contact between them, and their reports were compared against a third reading
done directly in this session. All three agree on every structural claim in §2
and §3. Where they disagree with each other or with the paper, it is recorded in
§7.

---

## 2. The experimental model

### 2.1 The curve

`dchain.cpp:312-320` (`logResponse`) and `interpretMCMC.R:40-46` (`response`) are
the same function:

```
f(c; K, h, alpha) = (1 - alpha) / (1 + (K*c)^h) + alpha
```

Three parameters, no separate baseline or E_max: `f(0) = 1` and `f(inf) = alpha`
by construction. The paper: *"a parameter, K, represents the inverse of half of
the maximal inhibitory concentration (IC50), h determines slope, and α
represents the maximum effect on cell viability."* Concentrations are in µM.

**`f(0) = 1` is load-bearing** and is easy to skip past: the family is *not*
closed under multiplication by a constant, so a constant offset in log viability
cannot be absorbed by a change of `(K, h, alpha)` uniformly across
concentrations. §3.3 is entirely about the consequences.

### 2.2 Three experiment types, and where the ordering lives

The header comment of `dchain.cpp` states the model in one line:

```
// x_AB = beta * f_A * f_B, assuming Bliss independence in the two time intervals.
// x_A = f_A, x_A0 = beta * f_A.
```

On the log scale, with `i` the first (pretreatment) drug and `j` the second:

| type | what it is | `E[log x]` |
|---|---|---|
| `A` | drug `j` alone, 8-point titration 0.03–100 µM | `1[λ_j] · log f(c; θ_j)` |
| `A0` | drug `i` at 1 µM, then vehicle — the **residual/carryover** effect | `log β_i + 1[λ_i] · log f(c; θ_i)` |
| `AB` | drug `i` at 1 µM, then drug `j` titrated at 0.01/0.1/1/10 µM | `log β_i + 1[λ_i] · log f(1.0; θ_i) + ( 1[λ_ij] · log f(c; θ_ij)  or  1[λ_j] · log f(c; θ_j) )` |

The literal `logResponse(1.0, theta[a])` appears at five places in the AB
likelihood blocks. **The first drug enters every combination well as a single
scalar evaluated at one fixed dose; the second drug enters as a four-point
titration curve.** That is the protocol asymmetry, and it is one line of code.

### 2.3 What is indexed by what

| symbol | index | count at 100 drugs |
|---|---|---|
| `theta[a] = (K, h, alpha)` | **drug** | 300 |
| `lambda[a]` (bool) | **drug** | 100 |
| `beta_residual[a]` | **drug**, first position only | 100 |
| `theta_AB[a][b]` | **ordered pair** | 30,000 |
| `lambda_AB[a][b]` (bool) | **ordered pair** | 10,000 |
| `var_prior` (Gamma) | global, fixed | — |
| `beta_plate`, `beta_run` | — | **dead code** |

**Nothing is indexed by unordered pair.** There is no `{i,j}`-level parameter
anywhere in the model, and no hierarchical pooling: `theta[a]` for different
drugs are a priori independent draws from one *fixed* prior. There are no
random effects and no shrinkage toward a learned mean.

Total sampled: **40,500**. The paper says 45,000, twice. Neither reconstruction
could produce 45,000 from the code, and neither can this one. See §7.

`beta_plate` and `beta_run` are allocated, passed into `calcSufficientStat`, and
every line that would use them is commented out; no proposal ever updates them.
There are no plate or batch effects in the fitted model — normalisation happened
upstream in R.

### 2.4 The selectors

Two Boolean families, both `vector<bool>`:

* **`lambda[a]`**, per drug. False ⇒ the drug contributes `log(1) = 0`, a flat
  curve at viability 1. True ⇒ `theta[a]` is used.
* **`lambda_AB[a][b]`**, per **ordered** pair, initialised true. True ⇒ the
  second drug's curve in this ordered combination is its own private
  `theta_AB[a][b]`; false ⇒ it falls back on the shared single-agent `theta[b]`.
  The first-position factor is present in both branches.

The paper describes this as *"a scheme where curve parameters were only explored
if a selector variable remains on."* Four proposal cases per sweep; when a
selector is off its `theta` is frozen rather than resampled, and both families
are **forced on** during the first `init_phase = 20,000` iterations.

**There is no prior on either selector.** `struct BernPrior` and its
`qprior(bool, bool, BernPrior)` overload are defined at `dchain.cpp:199` and
`:346` and **never called**; the acceptance ratio for a `(theta, lambda)`
proposal multiplies only the K, h and alpha prior quotients. With a symmetric
flip proposal, the implicit prior on `lambda` and `lambda_AB` is therefore
**Bernoulli(0.5)** — no complexity penalty beyond what the curve priors impose.
This is why the simulator uses `p_lambda_ab = 0.5`: it is the model's own prior,
not a choice.

### 2.5 Noise, replicates, priors, inference

Log relative nuclei count, Gaussian, with the per-condition precision integrated
out against a `Gamma(a=0.6, b=0.02)` prior — a Student-t marginal computed from
the sufficient statistics `(n, mean, var)` only (`tQuotient`,
`dchain.cpp:206-217`). The `0.6 / 0.02` is commented *"worst case from statistics
covering PANC1 and A375"*. Replicates enter **only** through `(n, mean, var)`;
the screen was in triplicate. Variance is not pooled across conditions.

Priors, from `dchain.cpp`'s `main()`: `K ~ LogNormal(0.1, 2.0)`,
`h ~ LogNormal(1.5, 0.5)`, `alpha ~ Beta(1, 3)`,
`beta_residual ~ LogNormal(1.0, 0.05)`. The `qprior` overload for LogNormal
divides by `2*sd`, not `2*sd^2`, so the struct's `sd` field is a **variance** and
the log-scale SD is its square root — which is what `simulator.K_PRIOR_LOGSD`
holds.

Inference is a hand-written componentwise Metropolis–Hastings sweep in C++11
`<random>`, three blocks per iteration (`beta_residual`, then per-drug
`(theta, lambda)`, then per-ordered-pair `(theta_AB, lambda_AB)`), with defaults
`iterations 500000, burn 100000, subsample 200, init 20000`.

**One chain, unseeded.** `default_random_engine generator;` is default
constructed and there is no seed option, so the published program is
deterministic and cannot be run twice. There are no convergence diagnostics
anywhere in the repository; the paper's only convergence evidence is a single
trace plot (Fig. S1B).

---

## 3. The synergy computation

### 3.1 The formula

`interpretMCMC.R`, `summaryStatisticsMCMC()`, quoted in full in
`dchain_null/synergy.py`. Per retained MCMC sample `t`:

```
baseline_j^t(s) = (1 - lambda_j^t) + lambda_j^t * f(s; theta_j^t)
S_ij^t          = lambda_AB^t(i,j) * mean_s [ baseline_j^t(s) - f(s; theta_AB^t(i,j)) ]
```

over `s = seq(0.01, 10, length.out=10)` — **ten linearly spaced points**, and

```
synergy_measure(i->j) = mean_t S_ij^t          synergy_sd(i->j) = sd_t S_ij^t
```

The `lambda_AB` factor is exact, not an approximation: substituting `resp` into
`baseline - resp` collapses the R expression to the line above, so a combination
whose selector is off contributes **exactly zero**.

### 3.2 What the measure does and does not depend on

Depends on exactly four things: `lambda_j`, `theta_j` (the **second** drug's
shared single-agent parameters) and `lambda_AB(i,j)`, `theta_AB(i,j)` (this
ordered pair's private parameters).

**`beta_residual[i]` and `theta[i]` do not appear at all.** The dataset note in
[`phase2_dataset.md`](phase2_dataset.md) says the first drug's effect "enters as
a multiplicative factor that cancels in the difference"; the effect is right but
the mechanism is worth stating exactly, because the difference matters in §3.3:
the first-position factor is **never formed**. The score is expressed in units
of viability relative to the post-pretreatment cell count. It is divided out, not
subtracted.

Two consequences worth recording:

* Nine of the ten integration points lie in [1.12, 10] µM, a range containing
  exactly **one** of the four measured doses. The measure is dominated by model
  interpolation.
* The score is computed per ordered combination over the full double loop with
  no symmetrisation, which is what makes `y(i→j)` and `y(j→i)` separate
  quantities and the Phase 2R directional target legitimate.

### 3.3 Can shared estimation error manufacture *cyclic* structure?

Yes, at **first** order, and the structure it makes is exactly rank-2 cyclic.
This is the mechanism the null exists to size. Both independent reconstructions
derived it; the version below is the sharper one.

Write the shared first-position offset and its error

```
u_i = log beta_i + log f(1; theta_i)          eps_i = u_hat_i - u_i
```

**Step 1.** The AB wells for pair `(i,j)` constrain only the *sum*
`u_i + log f(c; theta_ij)`. So conditional on the fit's `u_hat_i`, the
combination curve is fitted to the data shifted by `-eps_i`, uniformly across
concentrations and identically for every `j`. To first order, with `J_j` the
Jacobian of `log f` at `theta_j` and `P_j = (J_j' W J_j)^-1 J_j' W`,

```
theta_hat_ij - theta_j  =  P_j ( -eps_i * 1 + noise_ij )
```

Because `f(0) = 1` pins the family (§2.1), that uniform offset cannot be absorbed
exactly — it is *projected* onto a three-dimensional tangent space, and with four
measured doses against three free parameters the projection is nearly exact.

**Step 2.** Push it through the score. With `m_j` the mean baseline viability of
drug `j` over the integration grid and `g_j` its gradient,

```
S_hat(i,j)  ~=  eps_i * mtilde_j  +  c_j  +  xi_ij
mtilde_j := g_j' P_j 1        c_j := g_j' (theta_hat_j - theta_j)       xi = noise
```

`mtilde_j` is a **per-drug second-position gain**. Its two limits are checkable:
an inert drug `j` (`f = 1`) absorbs the whole offset into `alpha`, giving
`mtilde_j ~= 1`; a fully potent drug (`f ~= 0` over the grid) has nothing left to
lose, giving `mtilde_j ~= 0`. So `mtilde` spans roughly [0, 1] and is strongly
heterogeneous across a real drug panel. In words: *the error in how much
carryover the model attributes to the first drug is reassigned to "synergy", in
proportion to how much viability the second drug leaves available to be lost.*

**Step 3.** Antisymmetrise. `D = S_hat - S_hat'` decomposes into

```
D  =  (eps ^ mtilde)  +  (1 ^ c)  +  antisymmetric noise
```

writing `u ^ v` for `u_i v_j - u_j v_i`. For any wedge, the circulation around a
triangle is `det[1; u; v]`, so:

* **`(1 ^ c)` is curl-free.** A column-only error — error in the second drug's
  own curve — antisymmetrises to `c_j - c_i`, a **pure per-drug potential**.
  Phase 2R's additive fit removes exactly this, and the `potential` rung is
  exactly the class that can express it. So error in the shared *baseline* is
  not a threat to the Phase 2R result: it lands in the part that was already
  subtracted.
* **`(eps ^ mtilde)` is cyclic** whenever `1`, `eps` and `mtilde` are linearly
  independent, and it is a **rank-2 antisymmetric matrix — exactly one cyclic
  mode**. Both factors are per-drug vectors, so a rank-2 model fitted on
  observed pairs predicts held-out pairs. It looks precisely like reusable
  pair-specific interaction and is not.
* Antisymmetric noise is full-rank and does **not** generalise.

Equivalently and more generally: expand the score around truth in the first- and
second-position error blocks. Every additively separable term antisymmetrises to
a potential; the *only* cyclic source is the mixed term `e_i' (B - B') e_j`. So
cyclic structure exists iff `B` is not symmetric — iff the score's sensitivity to
first-position error differs from its sensitivity to second-position error. It
does, because the first drug is a scalar at one dose that is then discarded and
the second drug is a full curve that *is* the baseline.

A second amplification channel, noted by one reconstruction: the `lambda_AB`
gate multiplies the column term, giving `p_ij * eps_j` with a **pair-specific**
weight, and `p_ij != p_ji` breaks separability again.

**This makes a sharp, falsifiable prediction, and it is why the experiment is
worth running rather than arguing about.** The artifact is rank 2. The real
screens' cyclic component is not: only 34% (A375) / 32% (PANC1) of its energy
sits in the top two singular directions, against 8% for pure noise and 100% for
a pure rank-2 artifact. Whether the observed 34% is "mostly artifact plus noise"
or "mostly something else" is a quantitative question, and the null answers it by
measuring what this estimator actually produces rather than what it could.

### 3.4 `lambda`

The paper's Fig. 2D legend: *"Distribution of the posterior MCMC frequencies of
selector variables for all 10,000 sequential combinations… These 'λ scores' were
multiplied by −1 for antagonistic combinations yielding a range of [−1, 1]."*
So `|lambda| = P(lambda_AB = 1)` and `p = 1 - |lambda|`. It is an
effect-*existence* probability, not an effect size.

The signing step is **not in the public repository**, and it is demonstrably not
`sign(synergy_measure)`: the two disagree on 583 of 10,000 A375 rows and 861 of
10,000 PANC1 rows, concentrated at small `|synergy|`. Phase 2 never used
`lambda` for anything but reproducing the paper's counts, so nothing in this
project depends on the rule.

---

## 4. Dependency classification for the null

| component | class | why |
|---|---|---|
| shared per-drug first-position offset `u_i` in every AB row | **REQUIRED** | it is the only per-drug error shared across a whole row; without it there is nothing to reuse |
| the score's omission of `u_i` | **REQUIRED** | the misallocation between `u_i` and `theta_ij` is transported whole into the score |
| heterogeneity of `theta_j` across drugs | **REQUIRED** | if `mtilde` were constant the artifact is additively separable and its curl is **zero** |
| position asymmetry (first at one fixed dose, second as a curve) | **REQUIRED** | it is what makes the mixed sensitivity `B` non-symmetric |
| joint fit: AB observations feed back into per-drug parameters | **REQUIRED** | otherwise `eps` is per-pair, and the residual is unpredictable noise |
| `f(0) = 1`, i.e. the log/linear scale mismatch | **REQUIRED** | this is the nonlinearity that turns a row error into a row × column product |
| `lambda_AB` selector | **useful** | gates and amplifies the artifact; does not create it. Needed to match score magnitudes and the `lambda` column |
| Student-t marginal vs fixed-variance Gaussian | **useful** | reweighting only |
| full MCMC vs a MAP fit | **useful** | the artifact is propagation of estimation error, present in both; MCMC is needed for posterior means and the `lambda` column |
| Hill form specifically | **useful** | any smooth bounded 3-parameter family with `f(0)=1` gives the same structure |
| `lambda_i` single-drug selectors | **useful** | can be pinned to 1 |
| `beta_plate`, `beta_run` | **irrelevant** | dead code |
| MCMC tuning constants | **irrelevant** | not model structure |
| upstream plate normalisation, imaging, drug annotation, elastic net, the four extra pancreatic lines | **irrelevant** | not in the estimator |

Everything in the REQUIRED column is present in the null by construction, because
the null runs the published sampler rather than a model of it.

---

## 5. What is patched before compiling, and why none of it is the model

Listed exactly, as `(old, new)` pairs, in `dchain_null.dchain.PATCHES`; each is
asserted to match exactly once, so the diff is auditable without reading the
upstream file.

1. **Boost removed.** `dchain.cpp` uses Boost only for `program_options` and
   `filesystem`. Replaced by a standard-library shim. No line inside the sampler
   is touched, and without it nothing compiles in this environment at all.
2. **`--seed` added.** The published program default-constructs its engine, so
   every run of it is the same chain; there is no way to run a second one. An
   ensemble needs an estimator seed and the convergence question needs more than
   one chain. `--seed 0` restores the published behaviour exactly.
3. **Sufficient statistics computed once instead of ~1.6×10⁵ times per
   iteration** — at 100 drugs, 40,100 evaluations in the `beta_residual` sweep,
   80,900 in the per-drug sweep and 40,000 in the per-pair sweep.
   `calcSufficientStat` takes `beta_plate` and `beta_run` but every line that
   would use them is commented out, and neither vector is ever updated, so its
   return value is constant for the whole run.

Edit 3 is the only one that could change a number, so it is checked rather than
argued: `dchain.verify_equivalence` runs the patched and unpatched programs on
the same input at `--seed 0` and requires **byte-identical** output. The build
refuses if they differ. It runs on every `scripts/prepare_dchain_null.py`.

Not patched, and therefore inherited exactly: the likelihood, the Student-t
marginal and its variance prior, the `(K, h, alpha)` priors, the proposal
distributions, the four-case selector scheme, the absence of a selector prior,
the sweep order, and the storage rule `iter > burn && iter % subsample == 0`.

---

## 6. Fidelity evidence

**The hard limit first.** The Mendeley deposit contains the five modelled tables
and nothing else: no raw viability data (~250,000 wells) and **no posterior
samples**. The inputs to the synergy formula therefore do not exist publicly, so
*nobody* — not this project — can reproduce a deposited `synergy_measure` value
numerically from deposited parameters. That check is unavailable in principle,
not merely unattempted. What is available is structural, and it is not weak.

### 6.1 The deposit is provably the output of this code at its default settings

Every `|lambda|` value in Data Tables 1 and 2 is an **exact multiple of 1/1999**,
and not of 1/1998 or 1/2000. The storage rule `iter > 100000 && iter % 200 == 0`
over 500,000 iterations retains exactly

```
|{100200, 100400, ..., 499800}| = 1999
```

samples. Two independent reconstructions found this separately. It ties the
deposited tables to `dchain.cpp` at its compiled-in defaults, and it is the
reason this null runs at exactly those settings rather than at convenient ones.
(The paper says 2,000.)

### 6.2 The measurement layer reproduces the deposit's structural identities

`dchain.deposit_identities` checks four consequences of the formula in §3.1.
On the real deposit, all four hold:

| identity | A375 | PANC1 |
|---|---|---|
| `\|lambda\|` an exact multiple of 1/1999 | yes | yes |
| rows with `lambda == 0` | 191 | 173 |
| …of which `synergy_measure == 0` **exactly** | 191 | 173 |
| `\|synergy_measure\| <= \|lambda\|` on every row | yes | yes |

The second and third rows are a direct check of the multiplicative selector: a
formula without the `lambda_AB` factor cannot produce exactly-zero scores on
exactly the zero-selector rows. The fourth follows because the per-sample
difference lies in [−1, 1]. The same four are computed on every null screen and
recorded with it.

### 6.3 The paper's own counts reproduce exactly

Using `p = 1 - |lambda| < 0.05` and `sign(lambda)`: **707** synergistic /
**1,845** antagonistic in Table 1 and **551** / **1,464** in Table 2, against the
paper's *"1,258 synergistic (551 in PANC1 and 707 in A375) and 3,309 antagonistic
(1,464 PANC1, 1,845 A375)"*. Exact on all four. This is also what pins Table 1 =
A375 and Table 2 = PANC1, independently of the deposit's labels.

### 6.4 The two synergy implementations agree

`synergy.synergy_index` transcribes the R expression as written;
`synergy.synergy_index_collapsed` computes the algebraically reduced form. They
agree to machine precision and a test refuses to let them drift.

### 6.5 The sampler runs, and reproduces itself

The compiled binary runs on the deposited 66-row example and on simulated
screens, and the patched build is byte-identical to the unpatched one (§5).

### 6.6 The published chain is reproducible here, exactly

`default_random_engine` is implementation-defined, so a chain realised on another
standard library would differ. On this one (libc++) it is `minstd_rand`, whose
default-constructed state is seed 1 — **so `--seed 1` reproduces the published
program's unseeded chain byte for byte, and `--seed 0` is the same chain again.**
Verified: seed 0 and seed 1 give identical output; seeds 2 and 3 differ from both
and from each other. The convergence block's chain 1 is therefore literally the
chain the authors' own defaults would produce, and `CONVERGENCE_CHAINS` carries
an assertion that 0 is never added alongside it — two identical "independent"
chains would inflate the convergence result silently.

**What none of this establishes:** that the numbers in a null screen would match
numbers the original authors would get on *their* build. A different standard
library realises a different chain from the same seed. That affects one chain's
realisation, not the posterior it targets, and it is what the multi-chain block
measures.

---

## 7. Discrepancies and open gaps

Recorded rather than reconciled, because each is a place where a reader could
reasonably reach a different reconstruction.

1. **Paper priors ≠ code priors.** The paper states `K ~ logN(0.1, 0.2)` and
   `h ~ logN(1.5, 2.0)`; the code has `K_prior{0.1, 2.0}` and `h_prior{1.5, 0.5}`
   — the second hyperparameters appear transposed. Proposal SDs: `{0.5, 0.1,
   0.1}` in code, *"2.0, 0.5, and 3.0"* in the paper. The `beta_residual` prior
   is in the code and not in the paper. **This simulation uses the code's
   values**, because the code is what ran. The choice matters: the code's wider
   `K` prior gives *more* potency heterogeneity, and heterogeneity of `mtilde` is
   the REQUIRED ingredient of the artifact (§3.3, §4). Using the code's values is
   therefore the choice that is **more favourable to detecting an artifact**, not
   less.
2. **45,000 vs 40,500 parameters.** Stated twice in the paper; not reproducible
   from the code by either reconstruction or by this one. Unexplained. Nothing
   here depends on it.
3. **1,999 vs 2,000 retained samples.** The data says 1,999 (§6.1); the paper
   says 2,000.
4. **The `lambda` signing rule is not public** and is not `sign(synergy_measure)`
   (§3.4).
5. **No raw data and no posterior samples were deposited** (§6). The published
   scores are not independently reproducible by anyone.
6. **Tables 4 and 5 are re-exports of Table 2 (PANC1)**, not independent
   validation screens, and their `lambda` sign convention differs from Tables 1
   and 2 on 860 rows. Already derived and enforced in `koplev.verify_raw`; both
   reconstructions rediscovered it independently.
7. **No convergence diagnostics exist for the published fit.** One unseeded
   chain, one trace plot. Whatever MCMC error the deposited `synergy_measure`
   carries is unquantified and unquantifiable from the deposit.
8. **`dchain.lindinglab.org` is gone.**

---

## 8. What this buys the null

The null needs an estimator that (a) shares per-drug parameters across pairs,
(b) is asymmetric in position, (c) is nonlinear in the way that converts a row
error into a row × column product, and (d) feeds combination observations back
into per-drug parameters. All four are properties of `dchain.cpp`, and
`dchain.cpp` is what runs. The reconstruction risk that remains is not in the
model — it is in the *generative* side: whether the simulated wells resemble the
real screen closely enough for the artifact's magnitude to transfer. That is
addressed by the noise sweep and by the parameter-provenance table in
`dchain_null/simulator.py`, and it is the honest limitation in
[`dchain_null_falsification.md`](dchain_null_falsification.md).
