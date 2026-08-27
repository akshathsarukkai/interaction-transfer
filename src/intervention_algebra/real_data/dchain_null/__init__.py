"""Can the d-chain estimator manufacture the Phase 2R signature from nothing?

Phase 2R found that the Koplev screen's directional matrix carries a substantial
cyclic component that no per-drug ordering potential can express, and that it is
predictable for entirely held-out drug pairs once about a third of the pair graph
has been observed. Its leading alternative explanation was never biological:
``synergy_measure`` is a posterior mean from one joint 40,500-parameter fit in
which every drug's single-agent curve is shared across all its combinations, so
correlated estimation error could produce the same signature with no true
pair-specific interaction anywhere.

This package builds a world in which the true pair-specific sequential
interaction is **zero**, pushes it through the authors' own compiled sampler, and
runs the existing Phase 2R pipeline on the output without changing it.

    simulator   the generative null, and what "zero" means precisely
    synergy     the authors' synergy measure, ported from interpretMCMC.R
    dchain      fetch, patch, build and run dchain.cpp itself
    estimator   the three rungs of the fidelity ladder
    adapter     the only code between the null and Phase 2R
    experiment  one null condition, end to end
    report      the ensemble, and the comparison against the real screens

Read ``docs/dchain_reconstruction.md`` before changing anything here.
"""

from __future__ import annotations

from .adapter import as_screen
from .simulator import NUISANCE, STRICT, NullConfig, simulate_truth, simulate_wells
from .synergy import synergy_index, synergy_posterior

__all__ = ["NullConfig", "STRICT", "NUISANCE", "simulate_truth",
           "simulate_wells", "synergy_index", "synergy_posterior", "as_screen"]
