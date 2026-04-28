"""scikit-learn based model adapters."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from ..hpc import ResourceProfile
from .base import SklearnRegressorAdapter


_RF_PROFILE = ResourceProfile(
    name="sklearn.random_forest",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="n_jobs",
    notes="RandomForestRegressor uses joblib threads; num_workers -> n_jobs.",
)

_SKMLP_PROFILE = ResourceProfile(
    name="sklearn.mlp",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="sklearn.MLPRegressor is single-threaded; num_workers is ignored.",
)

_SKGPR_PROFILE = ResourceProfile(
    name="sklearn.gpr",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="sklearn GaussianProcessRegressor is single-threaded.",
)


class RandomForestModel(SklearnRegressorAdapter):
    """Random Forest model wrapper compatible with the SURGE API."""

    name = "sklearn.random_forest"
    estimator_cls = RandomForestRegressor
    resource_profile = _RF_PROFILE
    default_params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    @property
    def feature_importances_(self) -> Any:  # noqa: D401 - proxy property
        return self._model.feature_importances_  # type: ignore[attr-defined]

    def fit(self, X: Any, y: Any) -> Any:
        # If the user declared a ResourceSpec with a specific num_workers,
        # honor it by overriding the estimator's n_jobs right before fit.
        res = self._last_fit_resources or {}
        n_jobs = res.get("concrete", {}).get("n_jobs")
        if n_jobs is not None and self._model is not None:
            try:
                self._model.set_params(n_jobs=int(n_jobs))
            except Exception:  # pragma: no cover - defensive
                pass
        return super().fit(X, y)


class MLPModel(SklearnRegressorAdapter):
    """Scikit-learn based MLP model."""

    name = "sklearn.mlp"
    estimator_cls = MLPRegressor
    resource_profile = _SKMLP_PROFILE
    default_params = {
        "hidden_layer_sizes": (100, 50),
        "max_iter": 800,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        params = dict(self.default_params)
        params.update(kwargs)
        self.ensemble_n = max(1, int(params.pop("ensemble_n", 1)))
        if self.ensemble_n == 1:
            return MLPRegressor(**params)

        base_seed = params.get("random_state", 42)
        models = []
        for idx in range(self.ensemble_n):
            member_params = dict(params)
            if base_seed is not None:
                member_params["random_state"] = int(base_seed) + idx
            models.append(MLPRegressor(**member_params))
        return models

    def fit(self, X: Any, y: Any) -> Any:
        if isinstance(self._model, list):
            for member in self._model:
                member.fit(X, y)
            return self
        return super().fit(X, y)

    def predict(self, X: Any) -> Any:
        if isinstance(self._model, list):
            preds = np.stack([np.asarray(member.predict(X)) for member in self._model], axis=0)
            return preds.mean(axis=0)
        return super().predict(X)

    def predict_with_uncertainty(self, X: Any) -> Any:
        if not isinstance(self._model, list):
            raise NotImplementedError("Set ensemble_n > 1 for sklearn.mlp uncertainty.")
        preds = np.stack([np.asarray(member.predict(X)) for member in self._model], axis=0)
        return {
            "mean": preds.mean(axis=0),
            "std": preds.std(axis=0, ddof=0),
            "variance": preds.var(axis=0, ddof=0),
            "n_members": self.ensemble_n,
        }


class GPRModel(SklearnRegressorAdapter):
    """Gaussian Process regressor using scikit-learn backend."""

    name = "sklearn.gpr"
    estimator_cls = GaussianProcessRegressor
    resource_profile = _SKGPR_PROFILE
    default_params = {}

    def predict_with_uncertainty(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X, return_std=True)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        if self._model is None or not hasattr(self._model, "sample_y"):
            raise NotImplementedError("Posterior sampling not supported for this model")
        return self._model.sample_y(X, n_samples=num_samples, random_state=self.params.get("random_state"))
