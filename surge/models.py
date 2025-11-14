"""Backward compatible shim for the new :mod:`surge.model` package."""
from __future__ import annotations

import warnings

from .model import (  # noqa: F401,F403 - re-export
    BaseModelAdapter,
    EnsemblePrediction,
    FNNEnsemble,
    GPRModel,
    GPflowGPRAdapter,
    GPflowMultiKernelAdapter,
    MLPModel,
    MODEL_REGISTRY,
    PyTorchMLPAdapter,
    RandomForestModel,
    SklearnRegressorAdapter,
    create_model,
    list_models,
    register_model,
)

warnings.warn(
    "surge.models is deprecated; use surge.model package instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BaseModelAdapter",
    "SklearnRegressorAdapter",
    "RandomForestModel",
    "MLPModel",
    "GPRModel",
    "PyTorchMLPAdapter",
    "GPflowGPRAdapter",
    "GPflowMultiKernelAdapter",
    "FNNEnsemble",
    "EnsemblePrediction",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    "register_model",
]
