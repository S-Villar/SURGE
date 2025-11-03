"""Model abstractions and registry for SURGE backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

# Import enhanced model implementations
try:
    from .pytorch_models import PyTorchMLPModel
    PYTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYTORCH_AVAILABLE = False

# Lazy import for gpflow to avoid loading on module import
# Only import gpflow_models if gpflow is actually available and compatible
GPFLOW_AVAILABLE = False
_GPflowGPRModel = None
_GPflowMultiKernelGPR = None

try:
    # Check if gpflow module exists without importing it
    import importlib.util
    spec = importlib.util.find_spec("gpflow")
    if spec is not None:
        # Try importing gpflow_models (which handles its own gpflow import safely)
        from .gpflow_models import GPflowGPRModel, GPflowMultiKernelGPR
        GPFLOW_AVAILABLE = True
        _GPflowGPRModel = GPflowGPRModel
        _GPflowMultiKernelGPR = GPflowMultiKernelGPR
except (ImportError, Exception):  # pragma: no cover - optional dependency
    GPFLOW_AVAILABLE = False


class BaseModelAdapter(ABC):
    """Unified interface shared by all SURGE model implementations."""

    name: str = "base"
    backend: str = "generic"
    uses_internal_preprocessing: bool = False
    handles_output_scaling: bool = False
    supports_sklearn_interface: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = dict(kwargs)
        self._model: Any = None
        self._initialize()

    def _initialize(self) -> None:
        self._model = self._build_model(**self.params)

    @abstractmethod
    def _build_model(self, **kwargs: Any) -> Any:
        """Construct the underlying model object."""

    def fit(self, X: Any, y: Any) -> Any:
        if self._model is None:
            raise ValueError("Model has not been initialized")
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any) -> Any:
        if self._model is None or not hasattr(self._model, "predict"):
            raise NotImplementedError("Predict with uncertainty not supported for this model")
        return self._model.predict(X, return_std=True)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        if self._model is None or not hasattr(self._model, "sample_posterior"):
            raise NotImplementedError("Posterior sampling not supported for this model")
        return self._model.sample_posterior(X, num_samples)

    def score(self, X: Any, y: Any) -> float:
        if self._model is None or not hasattr(self._model, "score"):
            raise NotImplementedError("score not implemented for this model")
        return self._model.score(X, y)

    def save(self, filepath: str) -> Any:
        if self._model is None or not hasattr(self._model, "save"):
            raise NotImplementedError("save not implemented for this model")
        return self._model.save(filepath)

    def load(self, filepath: str) -> Any:
        if self._model is None or not hasattr(self._model, "load"):
            raise NotImplementedError("load not implemented for this model")
        return self._model.load(filepath)

    def get_estimator(self) -> Any:
        return self._model

    def update_estimator(self, estimator: Any) -> "BaseModelAdapter":
        self._model = estimator
        return self


class SklearnRegressorAdapter(BaseModelAdapter):
    """Adapter that wraps a scikit-learn regressor."""

    backend = "sklearn"
    supports_sklearn_interface = True
    estimator_cls: Optional[Type[Any]] = None
    default_params: Dict[str, Any] = {}

    def _build_model(self, **kwargs: Any) -> Any:
        if self.estimator_cls is None:
            raise ValueError("estimator_cls must be defined for SklearnRegressorAdapter subclasses")
        params = dict(self.default_params)
        params.update(kwargs)
        return self.estimator_cls(**params)


class RandomForestModel(SklearnRegressorAdapter):
    """Random Forest model wrapper compatible with SURGE API."""

    name = "sklearn.random_forest"
    estimator_cls = RandomForestRegressor
    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    @property
    def feature_importances_(self) -> Any:  # noqa: D401 - proxy property
        return self._model.feature_importances_  # type: ignore[attr-defined]


class RFRModel(RandomForestModel):
    """Alias kept for backward compatibility."""


class MLPModel(SklearnRegressorAdapter):
    """Scikit-learn based MLP model."""

    name = "sklearn.mlp"
    estimator_cls = MLPRegressor
    default_params = {
        "hidden_layer_sizes": (100, 50),
        "max_iter": 500,
        "random_state": 42,
    }


class GPRModel(SklearnRegressorAdapter):
    """Scikit-learn Gaussian Process regressor."""

    name = "sklearn.gpr"
    estimator_cls = GaussianProcessRegressor
    default_params = {}

    def predict_with_uncertainty(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X, return_std=True)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        if self._model is None or not hasattr(self._model, "sample_y"):
            raise NotImplementedError("Posterior sampling not supported for this model")
        return self._model.sample_y(X, n_samples=num_samples, random_state=self.params.get("random_state", None))


class PyTorchMLPAdapter(BaseModelAdapter):
    """PyTorch-based MLP model with enhanced capabilities."""

    name = "pytorch.mlp"
    backend = "pytorch"
    uses_internal_preprocessing = True
    handles_output_scaling = True

    def __init__(self, **kwargs: Any) -> None:
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch models not available. Install torch: pip install torch")
        super().__init__(**kwargs)

    def _initialize(self) -> None:
        # Defer actual model construction until fit so input/output sizes are known
        self._model = None

    def _build_model(self, **kwargs: Any) -> PyTorchMLPModel:
        return PyTorchMLPModel(**kwargs)

    def fit(self, X: Any, y: Any) -> Any:
        if self._model is None:
            self._model = self._build_model(**self.params)
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def save(self, filepath: str) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before saving")
        return self._model.save(filepath)

    def load(self, filepath: str) -> "PyTorchMLPAdapter":
        if self._model is None:
            self._model = self._build_model(**self.params)
        self._model.load(filepath)
        return self


PyTorchMLPModelWrapper = PyTorchMLPAdapter  # Backward compatibility alias


class GPflowGPRAdapter(BaseModelAdapter):
    """GPflow-based Gaussian Process model with enhanced capabilities."""

    name = "gpflow.gpr"
    backend = "gpflow"
    uses_internal_preprocessing = True
    handles_output_scaling = True

    def __init__(self, **kwargs: Any) -> None:
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow models not available. Install gpflow: pip install gpflow")
        super().__init__(**kwargs)

    def _build_model(self, **kwargs: Any) -> GPflowGPRModel:
        return GPflowGPRModel(**kwargs)

    def fit(self, X: Any, y: Any) -> Any:
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any) -> Any:
        return self._model.predict_with_uncertainty(X)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        return self._model.sample_posterior(X, num_samples)


class GPflowMultiKernelGPRAdapter(GPflowGPRAdapter):
    """Advanced GPflow GP model that tests multiple kernels automatically."""

    name = "gpflow.multi_kernel_gpr"

    def _build_model(self, **kwargs: Any) -> GPflowMultiKernelGPR:
        return GPflowMultiKernelGPR(**kwargs)

    def get_best_kernel_info(self) -> Any:
        return self._model.get_best_kernel_info()

    def get_all_results(self) -> Any:
        return self._model.get_all_results()


GPflowGPRModelWrapper = GPflowGPRAdapter  # Backward compatibility alias
GPflowMultiKernelGPRWrapper = GPflowMultiKernelGPRAdapter  # Alias


@dataclass(frozen=True)
class RegisteredModel:
    key: str
    adapter_cls: Type[BaseModelAdapter]


class ModelRegistry:
    """Central registry for all available SURGE model adapters."""

    def __init__(self) -> None:
        self._registry: Dict[str, RegisteredModel] = {}

    def register(
        self,
        key: str,
        adapter_cls: Type[BaseModelAdapter],
        *,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        if not issubclass(adapter_cls, BaseModelAdapter):
            raise TypeError("adapter_cls must inherit from BaseModelAdapter")
        self._registry[key] = RegisteredModel(key=key, adapter_cls=adapter_cls)
        if aliases:
            for alias in aliases:
                self._registry[alias] = RegisteredModel(key=key, adapter_cls=adapter_cls)

    def create(self, key: str, **kwargs: Any) -> BaseModelAdapter:
        registered = self._registry.get(key)
        if registered is None:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Model '{key}' is not registered. Available: {available}")
        return registered.adapter_cls(**kwargs)

    def keys(self) -> Iterable[str]:
        """Return all registered model keys."""
        return sorted(self._registry)
    
    def list_models(self) -> Dict[str, str]:
        """
        List all registered models with their adapter classes.
        
        Returns
        -------
        dict
            Mapping of model keys to adapter class names.
        """
        return {
            key: registered.adapter_cls.__name__
            for key, registered in self._registry.items()
        }

    def get(self, key: str) -> RegisteredModel:
        registered = self._registry.get(key)
        if registered is None:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Model '{key}' is not registered. Available: {available}")
        return registered


MODEL_REGISTRY = ModelRegistry()


MODEL_REGISTRY.register(
    RandomForestModel.name,
    RandomForestModel,
    aliases=[
        "random_forest",
        "RandomForestRegressor",
        "sklearn.RandomForestRegressor",
    ],
)
MODEL_REGISTRY.register(
    MLPModel.name,
    MLPModel,
    aliases=["mlp", "sklearn.MLPRegressor"],
)
MODEL_REGISTRY.register(
    GPRModel.name,
    GPRModel,
    aliases=["gpr", "sklearn.GaussianProcessRegressor"],
)

if PYTORCH_AVAILABLE:
    MODEL_REGISTRY.register(
        PyTorchMLPAdapter.name,
        PyTorchMLPAdapter,
        aliases=["pytorch.mlp_model", "torch.mlp"],
    )

if GPFLOW_AVAILABLE:
    MODEL_REGISTRY.register(
        GPflowGPRAdapter.name,
        GPflowGPRAdapter,
        aliases=["gpflow.gpr_model"],
    )
    MODEL_REGISTRY.register(
        GPflowMultiKernelGPRAdapter.name,
        GPflowMultiKernelGPRAdapter,
        aliases=["gpflow.multi_kernel"],
    )


__all__ = [
    "BaseModelAdapter",
    "SklearnRegressorAdapter",
    "RandomForestModel",
    "RFRModel",
    "MLPModel",
    "GPRModel",
    "PyTorchMLPAdapter",
    "PyTorchMLPModelWrapper",
    "GPflowGPRAdapter",
    "GPflowGPRModelWrapper",
    "GPflowMultiKernelGPRAdapter",
    "GPflowMultiKernelGPRWrapper",
    "MODEL_REGISTRY",
    "PYTORCH_AVAILABLE",
    "GPFLOW_AVAILABLE",
]
