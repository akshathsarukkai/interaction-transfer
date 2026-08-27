"""Intervention Algebra -- Phase 1 synthetic benchmark."""

from .generator import (ORD, SIM, SINGLE, ObservationTable, SyntheticSystem,
                        SystemConfig, generate_observations, make_system)
from .experiment import RunConfig, run_experiment, seeded
from .models import (AdditiveModel, AlgebraModel, ModelConfig, SharedPairModel,
                     UnconstrainedModel, build_model, match_pair_capacity)
from .splits import SplitConfig, make_pair_split
from .sweep import load_jsonl, run_sweep
from .train import TrainConfig, train_model

__all__ = [
    "SINGLE", "SIM", "ORD",
    "SystemConfig", "SyntheticSystem", "ObservationTable",
    "make_system", "generate_observations",
    "SplitConfig", "make_pair_split",
    "ModelConfig", "build_model", "match_pair_capacity",
    "AdditiveModel", "UnconstrainedModel", "SharedPairModel", "AlgebraModel",
    "TrainConfig", "train_model",
    "RunConfig", "run_experiment", "seeded",
    "run_sweep", "load_jsonl",
]
