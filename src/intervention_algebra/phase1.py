"""The Phase 1 experiment grid.

This module *is* the experiment definition. Everything the headline result rests
on -- which regimes, which coverages, which seeds, which controls -- is declared
here rather than being buried in a shell script, so that a reader can audit the
design in one place and reproduce it with one command.

Design notes that affect interpretation
---------------------------------------
* **Report seeds are disjoint from the seeds used to choose hyperparameters.**
  Hyperparameters were selected on ``DEV_SEEDS`` by validation MSE, with the same
  search grid offered to every family (see ``results/README_RESULTS.md``). The numbers
  in the README come from ``REPORT_SEEDS``, which that search never saw.
* **Every family was offered the same search and could have taken different
  hyperparameters; in the event the search returned the same setting for all
  four.** ``configs/hparams.json`` ships identical values per family. Forcing one
  setting *a priori* would handicap whichever family it suited least; arriving at
  one by an identical search does not.
* **Disclosed weakness in that selection (review finding R5).** The optimiser
  settings (``lr``, ``weight_decay``, ``emb_weight_decay``) were chosen while the
  architecture was ``pair_mode="concat_outer"``. The feature map was then
  re-selected to ``"outer"`` on validation MSE, and the optimiser search was
  *not* re-run underneath it -- no artifact in ``results/`` varies weight decay
  at ``pair_mode="outer"``. This is a real gap and it is not repaired here,
  because re-searching after seeing the headline is the tuning-until-it-wins
  pattern this study is trying to avoid. What limits the damage is that all four
  families ship the *same* optimiser settings, so any mis-tuning is shared rather
  than differential, and the feature-map selection recorded in
  ``configs/hparams.json`` found the family ordering unchanged between the two
  product maps.
* **Pair capacity is matched** between the constrained and unconstrained families
  (``capacity_match=True``); a deliberate un-matched control is included.
* Controls are not optional extras -- ``random`` topology and the no-interaction
  ``independent`` regime are the two conditions where an apparent win would
  indicate a bug rather than a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------- seeds
DEV_SEEDS = (100, 101)              # hyperparameter selection only
REPORT_SEEDS = (0, 1, 2, 3, 4)      # everything in the README

# ------------------------------------------------------------------ families
FAMILIES = ("additive", "unconstrained", "shared_pair", "algebra")

# ----------------------------------------------------------------- coverages
COVERAGES = (0.05, 0.1, 0.2, 0.4, 0.7)
# a couple of representative coverages for the secondary conditions, so the
# regime/control sweeps stay cheap
# Controls run at the sparse and the first above-threshold operating point.
# 0.4 was dropped from the secondary sweeps for compute reasons on this machine,
# not for a scientific one; 0.1 and 0.2 bracket the identifiability boundary
# (ceiling S-recovery 0.80 and 0.94 respectively), which is where the controls
# are informative.
SECONDARY_COVERAGES = (0.1,)

# ------------------------------------------------------------------- system
# Calibrated so that the benchmark has dynamic range: pairs-per-intervention is
# ``coverage * (N - 1)``, so a larger N keeps low *fractional* coverage while
# leaving enough pairs per intervention for the latent factors to be recoverable
# at all. The sizing argument is recorded in the private research notebook this repository was cut from.
N_INTERVENTIONS = 64

# ``s_scale``/``a_scale`` are calibrated so that interactions carry ~30% of the
# variance of the observed target (``interaction_variance_share``, recorded on
# every run). At the original 0.6 they carried ~7%, which meant the additive null
# model was already near-optimal and every interaction-modelling family was mostly
# fitting noise -- a setting in which a null result would say nothing about the
# hypothesis. A weak-interaction condition is retained deliberately as a control.
INTERACTION_SCALE = 1.5

BASE_SYSTEM = dict(
    n_interventions=N_INTERVENTIONS,
    latent_dim=4,
    n_factors=6,
    sparsity=0.25,
    sparsity_mode="latent",
    noise_std=0.05,
    s_scale=INTERACTION_SCALE,
    a_scale=INTERACTION_SCALE,
)

# --------------------------------------------------- selected hyperparameters
# Filled in by scripts/select_hparams.py; see results/README_RESULTS.md for
# the protocol. Kept explicit rather than re-searched at run time so that the
# headline experiment is a single deterministic sweep.
#: Fallback used only if ``configs/hparams.json`` is absent. The committed JSON
#: is the authoritative record of what the reported sweep actually ran with.
_HPARAMS_FALLBACK: dict[str, dict] = {
    f: dict(emb_dim=8, pair_mode="outer", lr=3e-3,
            emb_weight_decay=1e-2, weight_decay=1e-3)
    for f in FAMILIES
}

HPARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "hparams.json"


def _load_hparams() -> dict[str, dict]:
    """Read the selected hyperparameters from the committed JSON.

    Selection protocol (scripts/select_hparams.py): validation MSE only, on
    DEV_SEEDS, with the same grid offered to every family, and a single shared
    architecture chosen across families before per-family optimiser settings.
    REPORT_SEEDS never enter the search.

    Loaded from a file rather than hard-coded so that the config the sweep runs
    under and the config recorded in the repository cannot drift apart -- an
    earlier version of this module carried hand-transcribed values that no
    longer matched the search that produced them.
    """
    if not HPARAMS_PATH.exists():
        return dict(_HPARAMS_FALLBACK)
    raw = json.loads(HPARAMS_PATH.read_text())
    out = {}
    for family in FAMILIES:
        entry = raw.get(family)
        if entry is None:
            out[family] = dict(_HPARAMS_FALLBACK[family])
            continue
        out[family] = {k: v for k, v in entry.items() if not k.startswith("_")}
    return out


HPARAMS: dict[str, dict] = _load_hparams()

# Budget set from measured convergence: across 31 hyperparameter-search runs
# the median best epoch was 570 and 91% of runs peaked below 1500. 2000 epochs
# with patience 300 therefore costs essentially nothing in fit quality while
# keeping the full suite tractable on one CPU machine. `epochs_run` and
# `best_epoch` are recorded per run so any truncation stays visible in the data.
# EARLY STOPPING IS DISABLED (patience == max_epochs). Checkpoint selection
# still uses validation -- the best-validation state is restored at the end --
# but training no longer *terminates* on a validation plateau.
#
# Why: with only 66 validation pairs the plateau signal is noisy, and a spurious
# plateau killed runs 2-4x short of the fit they would have reached. The damage
# was not cosmetic. Across the interaction families, run length correlated with
# final training MSE at r = -0.76 (cov 0.10) and -0.77 (cov 0.20), and with
# held-out MSE at r = -0.52 and -0.55. Regressing held-out MSE on run length vs
# on model family over coverages 0.20/0.40 gave R^2 = 0.195 for run length
# against 0.042 for family -- the stopping artifact explained roughly five times
# as much of the outcome as the architecture under test, and which family it
# struck was a coin flip. Any family comparison under that regime measures when
# validation happened to plateau.
#
# Fixed-length training costs more but makes run length constant by
# construction, so the confound cannot exist. The diagnostic that gates the
# headline is corr(epochs_run, test_mse) ~ 0 at every coverage.
#
# Budget sized from the measured convergence of the *deployed* configuration
# (not of the search, which included configurations that never converge): across
# the first 51 main-sweep runs the best epoch was 1280 at the median, 3170 at p90
# and 4590 at p95, with 1/51 reaching the cap. n_restarts=2 rather than 3 because
# restarts were originally believed to be pure insurance under this
# configuration. They are not. Over all 538 recorded runs the worst
# restart-to-best training-MSE ratios are 8.75 and 8.50, both `algebra` at
# coverage 0.40 (seeds 10 and 0) -- squarely in the range the degenerate A == 0
# basin produces, and both runs *were* rescued by their second restart. A third
# run (seed 4) had both restarts fail and was not rescued. So n_restarts=2 is
# load-bearing, not decorative, and it is not sufficient. This comment has now
# said "worst ratio 2.8" and "worst ratio 4.79" at earlier points in the
# project; both were true when written and neither was updated as runs were
# added, which is why the number is now derived in the text above from the
# committed file rather than remembered. Restarts are retained as insurance and any residual
# collapse remains visible via restart_train_mse_worst and the per-metric n.
#
# The epoch budget must be large enough that EARLY STOPPING, not the cap, ends
# every run -- otherwise the families are compared at an arbitrary truncation
# point rather than each at its best. This bit twice: at max_epochs=2500 the
# best hyperparameter-search runs all stopped at epoch ~2490, and at 2000 a
# quarter of the main sweep ran to the cap. run_phase1.py now reports how many
# runs ended at their cap. NOTE: with early stopping disabled (patience ==
# max_epochs) *every* run ends at the cap by construction, so that count is no
# longer a usability gate -- it only means something when early stopping is on.
TRAIN_BASE = dict(max_epochs=5000, patience=5000, eval_every=10, n_restarts=2)


#: Pair-net widths for the capacity-**matched 2x** control. ``pair_hidden`` is
#: not comparable across families -- ``algebra`` runs two pair MLPs while the
#: unconstrained families run one with 2d outputs -- so equal width is roughly
#: double the parameters. At emb_dim=8 / pair_mode="outer": algebra@78 = 23 096
#: pair params against unconstrained@120 = 23 288, a 0.8% gap. Using 120 for
#: everyone would compare 4x against 2x.
DOUBLE_WIDTH = {"algebra": 78, "additive": 78,
                "unconstrained": 120, "shared_pair": 120}


def _spec(family: str, seed: int, coverage: float, *, tag: str,
          system: dict | None = None, capacity_match: bool = True,
          hparams: dict | None = None) -> dict:
    hp = dict(hparams or HPARAMS[family])
    model = dict(emb_dim=hp["emb_dim"], pair_mode=hp["pair_mode"])
    train = dict(TRAIN_BASE, lr=hp["lr"],
                 emb_weight_decay=hp["emb_weight_decay"],
                 weight_decay=hp.get("weight_decay", 0.0))
    return dict(
        family=family,
        seed=seed,
        system=dict(BASE_SYSTEM, **(system or {})),
        split=dict(pair_coverage=coverage),
        model=model,
        train=train,
        capacity_match=capacity_match,
        tag=tag,
        meta=dict(tag=tag),
    )


def main_sweep(seeds: Iterable[int] = REPORT_SEEDS) -> list[dict]:
    """The headline experiment: full pair-coverage curve in the combined regime."""
    return [_spec(f, s, c, tag="main", system=dict(regime="both"))
            for c in COVERAGES for f in FAMILIES for s in seeds]


def regime_sweep(seeds: Iterable[int] = REPORT_SEEDS) -> list[dict]:
    """Does behaviour stay sensible when only S, only A, or neither is present?"""
    return [_spec(f, s, c, tag=f"regime_{r}", system=dict(regime=r))
            for r in ("independent", "symmetric", "antisymmetric")
            for c in SECONDARY_COVERAGES for f in FAMILIES for s in seeds]


def control_sweep(seeds: Iterable[int] = REPORT_SEEDS) -> list[dict]:
    """Conditions where a *win* would mean a bug, plus two robustness checks."""
    specs: list[dict] = []

    # (1) i.i.d. interaction topology: which pairs interact is unpredictable from
    #     the latent factors, so held-out topology recovery must sit at chance for
    #     every family. Above-chance here means leakage.
    specs += [_spec(f, s, c, tag="control_random_topology",
                    system=dict(regime="both", sparsity_mode="random"))
              for c in SECONDARY_COVERAGES for f in FAMILIES for s in seeds]

    # (2) nonlinear observation map: latent recovery is not identifiable, so only
    #     observable prediction is scored (metrics.py returns NaN correlations).
    specs += [_spec(f, s, c, tag="control_tanh_observation",
                    system=dict(regime="both", observation_map="tanh", obs_gain=1.0))
              for c in SECONDARY_COVERAGES for f in FAMILIES for s in seeds]

    # (3b) DOUBLE capacity, in the direction that can actually hurt the
    #      hypothesis. `capacity_match=False` (below) turns out to *halve* the
    #      baseline rather than widen it -- matching widens the unconstrained
    #      pair net from 48 to 76 to reach parity with the algebra model's two
    #      nets, so disabling it leaves 5864 pair params against 11336. This
    #      runs every family at pair_hidden=120 (23288 pair params, ~2.05x),
    #      including `algebra`: widening only the baseline would confound "the
    #      baseline got wider" with "the advantage was capacity all along".
    #      `pair_hidden` is NOT comparable across families: algebra runs two
    #      pair MLPs while the unconstrained families run one with 2d outputs,
    #      so equal width is roughly double the parameters. At emb_dim=8 /
    #      pair_mode="outer": algebra@78 = 23096 params, unconstrained@120 =
    #      23288 (0.8% gap). Those are the widths that make this a matched 2x
    #      control; using 120 for everyone would give algebra 45608 and compare
    #      4x against 2x.
    specs += [matched2x_spec(f, s, c, tag="control_double_capacity")
              for c in SECONDARY_COVERAGES for f in FAMILIES for s in seeds]

    # (3) capacity control: the unconstrained families at their *natural* width,
    #     i.e. roughly twice the constrained model's pair parameters. If the
    #     constrained model only wins when the baseline is capacity-matched, the
    #     result is about capacity, not about algebra.
    specs += [_spec(f, s, c, tag="control_unmatched_capacity",
                    system=dict(regime="both"), capacity_match=False)
              for c in SECONDARY_COVERAGES
              for f in ("unconstrained", "shared_pair") for s in seeds]

    # (4) weak interactions: the original operating point, where interactions
    #     carry only ~7% of target variance. The additive model should be close
    #     to optimal here, and no interaction family should gain much. This makes
    #     the dependence of the result on signal strength explicit rather than an
    #     unstated property of the chosen constants.
    specs += [_spec(f, s, c, tag="control_weak_interactions",
                    system=dict(regime="both", s_scale=0.6, a_scale=0.6))
              for c in SECONDARY_COVERAGES for f in FAMILIES for s in seeds]

    return specs


def ceiling_sweep(seeds: Iterable[int] = REPORT_SEEDS, *,
                  early_stopping: bool = False) -> list[dict]:
    """The identifiability frontier.

    ``wellspecified`` is given the generator's own functional form (learned
    factors, a symmetric and an antisymmetric bilinear form, and the interaction
    gate). It is not a competitor -- it answers "is this coverage level
    recoverable at all?". Where this curve is flat on the floor, no family's
    failure can be read as evidence about the hypothesis.

    ``early_stopping=True`` reproduces the *original* ceiling artifact
    (``tag="ceiling"``, ``patience=600``), which is what every ceiling number
    quoted in the docs was originally measured under. It is retained because for
    a while the committed code could not reproduce the committed ceiling at all:
    the sweep emitted ``TRAIN_BASE`` while every stored row carried
    ``patience=600``, so ``--part ceiling`` silently produced a different
    experiment. Since the ceiling normalises the "% of achievable" figures and
    fixes where the identifiability threshold is said to lie, that was a live
    reproducibility hole in an interpretive load-bearing quantity.

    The default is now the fixed-length protocol (``tag="ceiling_fixedlen"``),
    matching every comparison run, so the ratios no longer span two protocols.
    """

    specs = []
    for c in COVERAGES:
        for s in seeds:
            tag = "ceiling" if early_stopping else "ceiling_fixedlen"
            spec = _spec("algebra", s, c, tag=tag, system=dict(regime="both"))
            spec["family"] = "wellspecified"
            spec["model"] = dict(emb_dim=8)
            spec["capacity_match"] = False
            if early_stopping:
                spec["train"] = dict(spec["train"], patience=600)
            specs.append(spec)
    return specs


#: Misspecification levels for ``misspecification_sweep`` (fraction of s_scale).
DEFECT_LEVELS = (0.3, 0.8)


def misspecification_sweep(seeds: Iterable[int] = REPORT_SEEDS) -> list[dict]:
    """How far can reality depart from the algebra before the constraint hurts?

    Without this axis a win for the constrained family is close to tautological:
    the benchmark's interactions would decompose *exactly* into a symmetric and
    an antisymmetric part, which is the algebra model's own functional form.

    The axis has to be chosen carefully. Adding a term that is "neither
    symmetric nor antisymmetric" to the ordered rows would not be a
    misspecification at all -- every pair function decomposes uniquely into
    those two parts, so the algebra model can already represent any ordered pair
    function. Its actual constraint is the *cross-row* identity
    ``z_sim = (z_ord(i,j) + z_ord(j,i)) / 2``. ``simultaneity_defect`` adds a
    symmetric term to the simultaneous row only, which breaks exactly that tie
    and nothing else.
    """
    seeds = tuple(seeds)[:3]        # cheaper: this is a sensitivity curve
    return [_spec(f, s, c, tag=f"misspec_{eps}",
                  system=dict(regime="both", simultaneity_defect=eps))
            for eps in DEFECT_LEVELS
            for c in SECONDARY_COVERAGES for f in FAMILIES for s in seeds]


POWER_SEEDS = tuple(range(5, 17))          # extra seeds, disjoint from REPORT_SEEDS
# 0.7 is included even though the pilot effect was seen at 0.2/0.4. Running extra
# seeds only where an underpowered pilot happened to show something is a
# garden-of-forking-paths pattern regardless of intent, and 0.7 is not excluded
# by any stated criterion -- unlike 0.05/0.1, which sit below the identifiability
# threshold where even a correctly-specified reference model fails.
POWER_COVERAGES = (0.2, 0.4, 0.7)


def power_sweep(seeds: Iterable[int] = POWER_SEEDS) -> list[dict]:
    """Extra seeds for the headline comparison, at the coverages where it lives.

    The primary constrained-vs-unconstrained effect on held-out MSE is roughly
    0.05 against a seed-to-seed spread of 0.06--0.10, so five seeds cannot resolve
    it: the first run returned p = 0.17 and 0.31 with the difference in the
    predicted direction at 4/5 and 3/5 seeds. That is an underpowered design, not
    a null result, and reporting it as a null would be wrong.

    **This sweep was never run, and its stated exclusion criterion was wrong.**
    Both facts are left here rather than tidied away, because the docstring is
    part of the record of how the powering actually happened.

    It planned to add seeds at coverages 0.2/0.4/0.7 only, excluding 0.05 and
    0.10 on the grounds that "a reference model given the correct functional form
    also fails there". That is false at 0.10: the measured ceiling is S r = 0.814
    there, the second-best cell on the whole curve. And coverage 0.10 -- where
    the headline lives, and where the pilot had returned p = 0.055 -- *was*
    subsequently powered to 17 seeds, via ``rep10_matched1x`` and
    ``power10_algebra``, i.e. exactly the coverage this criterion said to leave
    alone.

    That is a garden-of-forking-paths exposure on the headline and it is named as
    one. What limits it: the replication block was seeds 5-16, fixed at twelve in
    advance, disjoint from the dev and report seeds, and it agrees with the pilot
    in direction and magnitude on its own (-0.143, p = 0.0069, n = 12). The
    effect is genuinely out of sample. But the decision to run those seeds was
    taken after seeing a promising pilot at that coverage, and no reading of this
    docstring should suggest otherwise.

    ``closure_sweep`` is the sweep that actually ran at coverage 0.40, and it was
    run against a prediction registered beforehand. Coverage 0.70 was never run.
    """
    return [_spec(f, s, c, tag="power", system=dict(regime="both"))
            for c in POWER_COVERAGES for f in FAMILIES for s in seeds]


def all_specs(seeds: Iterable[int] = REPORT_SEEDS) -> list[dict]:
    seeds = tuple(seeds)
    return (main_sweep(seeds) + ceiling_sweep(seeds)
            + regime_sweep(seeds) + control_sweep(seeds)
            + misspecification_sweep(seeds))


def smoke_specs() -> list[dict]:
    """A tiny grid used to check the pipeline end-to-end in seconds, not minutes.

    Uses a smaller system and a short epoch budget. Not scientifically meaningful
    -- it exists so that a broken pipeline fails in under a minute rather than
    after an hour of sweeping.

    NOTE: this grid deliberately re-enables early stopping with a tiny budget
    (200 epochs, patience 100), so `run_phase1.py` will print its
    "runs ended at their epoch cap" warning. That is expected here and is not a
    problem with your installation -- the warning is doing its job. The reported
    sweeps run with early stopping disabled entirely; see TRAIN_BASE.
    """
    specs = []
    for c in (0.1, 0.4):
        for f in FAMILIES:
            s = _spec(f, 0, c, tag="smoke",
                      system=dict(regime="both", n_interventions=40))
            s["train"] = dict(s["train"], max_epochs=200, patience=100)
            specs.append(s)
    return specs


#: Seeds for the Phase 1 closure cell: the union of REPORT_SEEDS and POWER_SEEDS,
#: i.e. exactly the 17 seeds the powered 0.10 and 0.20 comparisons already use.
CLOSURE_SEEDS = REPORT_SEEDS + POWER_SEEDS
CLOSURE_COVERAGE = 0.4


def closure_sweep(seeds: Iterable[int] = CLOSURE_SEEDS) -> list[dict]:
    """The one cell Phase 1 was missing: coverage 0.40 under the clean protocol.

    ``main_sweep`` declares coverage 0.40, but the only 0.40 rows that were ever
    produced ran with early stopping *on* and live in
    ``results/SUPERSEDED_main_confounded.jsonl``. They are not comparable with
    the powered 0.10/0.20 cells, so the coverage trajectory had a hole exactly
    where the crossover interpretation makes its prediction.

    This is ``main_sweep`` restricted to coverage 0.40 and widened to the 17
    seeds the other powered cells use. Nothing else differs: same
    ``BASE_SYSTEM``, same ``regime="both"``, same committed hyperparameters,
    same ``TRAIN_BASE``, same capacity matching. It is given its own tag purely
    so the provenance stays legible; the pooling logic in ``analysis`` keys on
    the run configuration, not on the tag, so a separate tag cannot create or
    hide a difference.

    Seed-major ordering: an interrupted sweep then leaves whole seeds finished
    rather than a ragged set of families, and a ragged family is the one shape
    that would bias a paired comparison.
    """
    return [_spec(f, s, CLOSURE_COVERAGE, tag="cov040", system=dict(regime="both"))
            for s in seeds for f in FAMILIES]



# ------------------------------------------------- the report's cell definitions
#
# The report must not decide what belongs in a headline cell by looking at sweep
# *tags*. A tag records which batch a run was launched in; it says nothing about
# what experiment was run, and a tag-keyed report either shows only the original
# five seeds or depends on a hand-maintained allow-list that silently goes stale
# every time a replication is added. The functions below name the *conditions*
# instead, and `analysis.spec_condition_key` turns each into a hash of the full
# resolved configuration. A run enters a cell iff it was configured identically
# up to its seed -- which is the definition of a replicate, and is exactly the
# property pooling requires.

#: Coverages the headline is reported at.
#:
#: 0.05 is included but is **not** powered, and deliberately so: it sits below
#: the identifiability threshold, where a reference model given the generator's
#: own functional form only reaches S r = 0.39, so no family's failure there can
#: speak to the hypothesis. Adding seeds would measure the noise floor more
#: precisely and nothing else. The criterion was stated before the outcome was
#: known, and the cell reports its own n = 5.
#:
#: 0.70 is absent because it was never run for the comparison families under any
#: protocol -- see README section 8, which names it alongside the coverage-0.40
#: prediction as unfinished rather than omitting it.
HEADLINE_COVERAGES = (0.05, 0.1, 0.2, 0.4)


def headline_spec(family: str, coverage: float, seed: int = 0) -> dict:
    """The spec defining one cell of the powered headline comparison.

    Identical to what ``main_sweep`` emits; ``seed`` is irrelevant to the
    condition and defaults to 0 only because a spec needs one.
    """
    return _spec(family, seed, coverage, tag="<headline>",
                 system=dict(regime="both"))


def headline_conditions(coverages: Iterable[float] = HEADLINE_COVERAGES,
                        families: Iterable[str] = FAMILIES) -> dict:
    """``{(family, coverage): spec}`` for the whole powered headline."""
    return {(f, c): headline_spec(f, c)
            for c in coverages for f in families}


def matched2x_spec(family: str, seed: int = 0, coverage: float = 0.1, *,
                   tag: str = "<matched2x>") -> dict:
    """The capacity-matched 2x arm: every family at ~2x its headline pair params."""
    return dict(_spec(family, seed, coverage, tag=tag,
                      system=dict(regime="both"), capacity_match=False),
                model=dict(emb_dim=HPARAMS[family]["emb_dim"],
                           pair_mode=HPARAMS[family]["pair_mode"],
                           pair_hidden=DOUBLE_WIDTH[family]))


def matched2x_conditions(families: Iterable[str] = ("algebra", "unconstrained"),
                         coverage: float = 0.1) -> dict:
    """``{(family, coverage): spec}`` for the matched-2x capacity control."""
    return {(f, coverage): matched2x_spec(f, coverage=coverage) for f in families}
