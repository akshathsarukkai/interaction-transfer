"""The preregistered ensemble, as code.

Every block below corresponds to a row of the ensemble table in the
pre-registration (``docs/PREREGISTRATIONS.md``, "Pre-registration -- the d-chain
null"), which was committed before any of them ran. The seed counts are the
planned ones; :func:`part_counts` derives what each ``--part`` costs so no
document has to quote a number that can go stale, which is the lesson the Phase
2R run-count drift taught this repository three days ago.
"""

from __future__ import annotations

from dataclasses import replace

from .experiment import DECISION_COVERAGES, NullRunConfig
from .simulator import NOISE_GRID, NUISANCE, SIGMA_OBS, STRICT, NullConfig

#: Primary ensemble size. 20 independent simulated screens, each with its own
#: generative seed and its own MCMC seed.
PRIMARY_SEEDS = tuple(range(20))
#: The realism variant, at the same size as the primary. It was registered at
#: half, on the argument that it can only dilute the artifact -- and that
#: argument is wrong. An independent reviewer ran both arms on the same truth
#: and the same settings: the strict null's posterior selector sits at 0.21-0.23
#: and the nuisance null's at 0.33, and because the measure is
#: ``lambda_AB * (...)``, opening the gate multiplies the artifact rather than
#: diluting it. Measured artifact RMS 0.016-0.017 (strict) against 0.033
#: (nuisance). NULL-B is the arm where the artifact is LARGER, not smaller, so
#: it gets the same 20 seeds.
REALISM_SEEDS = tuple(range(20))
#: Control D. Small because it is a sensitivity, not a headline.
NOISE_SEEDS = tuple(range(6))
#: Convergence block: four chains on one dataset.
#:
#: **Seed 0 and seed 1 are the same chain.** libc++'s ``default_random_engine``
#: is ``minstd_rand``, whose default-constructed state is seed 1, so the
#: published program's unseeded chain *is* ``--seed 1``. That is a useful fact --
#: chain 1 below is literally the chain the authors would have got -- and a trap:
#: adding 0 here would silently produce two identical "independent" chains and
#: inflate the convergence result. Asserted rather than remembered.
CONVERGENCE_CHAINS = (1, 2, 3, 4)
assert 0 not in CONVERGENCE_CHAINS, (
    "seed 0 and seed 1 are the same chain (libc++ default_random_engine is "
    "minstd_rand, default state = 1); including both would double-count one")


def _cfg(variant: str, sim_seed: int, estimator: str, tag: str,
         sigma: float = SIGMA_OBS, est_seed: int | None = None,
         **kw) -> NullRunConfig:
    return NullRunConfig(
        null=NullConfig(variant=variant, sim_seed=sim_seed, sigma_obs=sigma,
                        tag=tag),
        estimator=estimator,
        # Offset so the MCMC seed is never equal to the generative seed, and
        # never 0 -- 0 is the published default-constructed engine and is
        # reserved for the equivalence check.
        est_seed=(sim_seed + 101) if est_seed is None else est_seed,
        coverages=DECISION_COVERAGES, **kw)


def primary_grid() -> list[NullRunConfig]:
    """NULL-A through the published sampler. The experiment."""
    return [_cfg(STRICT, s, "joint", "primary") for s in PRIMARY_SEEDS]


def realism_grid() -> list[NullRunConfig]:
    """NULL-B through the published sampler."""
    return [_cfg(NUISANCE, s, "joint", "realism") for s in REALISM_SEEDS]


#: The searched-grid block is six times the cost of the fixed rank-2 one and is
#: preregistered as *secondary*. It is run on the two ensembles that have a
#: matched real reference (`primary`, `realism`) and omitted from the controls,
#: all of which are read off the rank-2 detector: Control A's criterion is
#: rank-2 skill on the true matrix, Control C's is whether rank-2 skill survives
#: the removal of sharing, and Control D's is how rank-2 skill moves with noise.
_CONTROL = {"run_honest_block": False}


def oracle_grid() -> list[NullRunConfig]:
    """Control A. The measure at the true parameters, both variants.

    Mandatory: if this shows the signature, the generative null secretly
    contains the structure being hunted and nothing downstream means anything.
    """
    return ([_cfg(STRICT, s, "oracle", "oracle_strict", **_CONTROL)
             for s in PRIMARY_SEEDS]
            + [_cfg(NUISANCE, s, "oracle", "oracle_nuisance", **_CONTROL)
               for s in REALISM_SEEDS])


def unshared_grid() -> list[NullRunConfig]:
    """Control C. The same estimation problem with the sharing removed."""
    return [_cfg(STRICT, s, "unshared", "unshared", **_CONTROL)
            for s in PRIMARY_SEEDS]


def noise_grid() -> list[NullRunConfig]:
    """Control D. The preregistered noise sweep, around the deposited scale."""
    return [_cfg(STRICT, s, "joint", f"noise_sd{sd:g}", sigma=sd, **_CONTROL)
            for sd in NOISE_GRID if sd != SIGMA_OBS
            for s in NOISE_SEEDS]


def convergence_grid() -> list[NullRunConfig]:
    """Four chains on one dataset, to size the MCMC error in the target.

    The published program is unseeded and cannot be run twice, so this is a
    diagnostic that does not exist for the real fit. It matters here for one
    specific reason: MCMC error in the posterior-mean synergy matrix is
    independent across pairs, so it can only *dilute* held-out skill, never
    create it -- which makes a short or badly mixed chain conservative against
    the artifact hypothesis rather than favourable to it. This block measures
    how much dilution there is.
    """
    # keep_posterior is on here and nowhere else: this block is already paying
    # for four chains on one dataset, so the mechanism probe (the log-scale
    # score, which needs the samples) rides along for free.
    return [replace(_cfg(STRICT, 0, "joint", "convergence", **_CONTROL),
                    est_seed=c, keep_posterior=True)
            for c in CONVERGENCE_CHAINS]


def smoke_grid() -> list[NullRunConfig]:
    """A pipeline check, not a result: 60 drugs, one split seed, no MCMC.

    Exercises simulate -> estimate -> decompose -> adapt -> Phase 2R end to end
    in about a minute and with no network, no compiler and no deposit, which is
    what makes it runnable in CI. It cannot say anything about the artifact --
    the ``unshared`` estimator is the control, not the experiment -- and its rows
    are written to their own file so they cannot be mistaken for one.
    """
    return [NullRunConfig(
        null=NullConfig(variant=v, n_drugs=60, sim_seed=0, tag="smoke"),
        estimator="unshared", est_seed=1, coverages=(0.70,),
        split_seeds=(0,), run_honest_block=False)
        for v in (STRICT, NUISANCE)]


PART_GRIDS: dict[str, str] = {
    "oracle": "oracle_grid",
    "unshared": "unshared_grid",
    "convergence": "convergence_grid",
    "primary": "primary_grid",
    "realism": "realism_grid",
    "noise": "noise_grid",
}

#: ``--part all`` runs every part above, in this order: the two free controls
#: first so a malformed null is caught before nine hours of MCMC, then the
#: convergence block, then the ensembles.
ALL_PARTS: tuple[str, ...] = tuple(PART_GRIDS)


def part_jobs(part: str) -> list[NullRunConfig]:
    """Conditions for one part, ``"all"``, or a comma-separated list of parts.

    The comma form exists for scheduling, not for science. Every joint condition
    costs about 110 minutes of one core, so running four blocks one after another
    on seven workers wastes most of a batch at each block's tail -- the
    convergence block alone is four conditions and would leave three cores idle
    for an hour and a half. Running the remaining blocks as one pool packs them.
    Which conditions run, and what each records, is unchanged.
    """
    if part == "smoke":
        return smoke_grid()
    if "," in part:
        out: list[NullRunConfig] = []
        seen: set[tuple] = set()
        for one in part.split(","):
            for cfg in part_jobs(one.strip()):
                key = (cfg.null.tag, cfg.estimator, cfg.null.sim_seed, cfg.est_seed)
                if key not in seen:
                    seen.add(key)
                    out.append(cfg)
        return out
    if part == "all":
        names: tuple[str, ...] = ALL_PARTS
    elif part in PART_GRIDS:
        names = (part,)
    else:
        raise ValueError(f"unknown part {part!r}; expected one of "
                         f"{sorted(PART_GRIDS) + ['all']}, or a comma-separated "
                         f"list of them")
    out: list[NullRunConfig] = []
    for name in names:
        out.extend(globals()[PART_GRIDS[name]]())
    return out


def part_counts() -> dict[str, int]:
    counts = {p: len(part_jobs(p)) for p in ALL_PARTS}
    counts["all"] = sum(counts[p] for p in ALL_PARTS)
    return counts


#: Planned size per tag, derived from the grids. Used by the tests to build a
#: complete ensemble when exercising the classification logic, so the
#: completeness gate can stay strict in production rather than being weakened
#: for the convenience of a fixture.
_PLANNED_FOR_TESTS: dict[str, int] = {}
for _c in part_jobs("all"):
    _PLANNED_FOR_TESTS[_c.null.tag] = _PLANNED_FOR_TESTS.get(_c.null.tag, 0) + 1
del _c
