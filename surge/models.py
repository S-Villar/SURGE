from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

# Import enhanced model implementations
try:
    from .pytorch_models import PyTorchMLPModel
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from .gpflow_models import GPflowGPRModel, GPflowMultiKernelGPR
    GPFLOW_AVAILABLE = True
except ImportError:
    GPFLOW_AVAILABLE = False


class RandomForestModel:
    """
    Random Forest model wrapper compatible with SURGE API.
    """
    def __init__(self, **kwargs):
        self._rf = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        return self._rf.fit(X, y)

    def predict(self, X):
        return self._rf.predict(X)

    def score(self, X, y):
        return self._rf.score(X, y)

    @property
    def feature_importances_(self):
        return self._rf.feature_importances_


class RFRModel:
    def __init__(self, **kwargs):
        self._rf = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        return self._rf.fit(X, y)

    def predict(self, X):
        return self._rf.predict(X)


class MLPModel:
    """
    Scikit-learn based MLP model (original implementation).
    """
    def __init__(self, **kwargs):
        self._mlp = MLPRegressor(**kwargs)

    def fit(self, X, y):
        return self._mlp.fit(X, y)

    def predict(self, X):
        return self._mlp.predict(X)


class PyTorchMLPModelWrapper:
    """
    PyTorch-based MLP model with enhanced capabilities.
    """
    def __init__(self, **kwargs):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch models not available. Install torch: pip install torch")
        self._mlp = PyTorchMLPModel(**kwargs)

    def fit(self, X, y):
        return self._mlp.fit(X, y)

    def predict(self, X):
        return self._mlp.predict(X)

    def save(self, filepath):
        return self._mlp.save(filepath)

    def load(self, filepath):
        return self._mlp.load(filepath)


class GPRModel:
    """
    Scikit-learn based Gaussian Process model (original implementation).
    """
    def __init__(self, **kwargs):
        self._gpr = GaussianProcessRegressor(**kwargs)

    def fit(self, X, y):
        return self._gpr.fit(X, y)

    def predict(self, X):
        return self._gpr.predict(X)


class GPflowGPRModelWrapper:
    """
    GPflow-based Gaussian Process model with enhanced capabilities.
    """
    def __init__(self, **kwargs):
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow models not available. Install gpflow: pip install gpflow")
        self._gpr = GPflowGPRModel(**kwargs)

    def fit(self, X, y):
        return self._gpr.fit(X, y)

    def predict(self, X):
        return self._gpr.predict(X)

    def predict_with_uncertainty(self, X):
        return self._gpr.predict_with_uncertainty(X)

    def sample_posterior(self, X, num_samples=10):
        return self._gpr.sample_posterior(X, num_samples)


class GPflowMultiKernelGPRWrapper:
    """
    Advanced GPflow GP model that tests multiple kernels automatically.
    """
    def __init__(self, **kwargs):
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow models not available. Install gpflow: pip install gpflow")
        self._gpr = GPflowMultiKernelGPR(**kwargs)

    def fit(self, X, y):
        return self._gpr.fit(X, y)

    def predict(self, X):
        return self._gpr.predict(X)

    def predict_with_uncertainty(self, X):
        return self._gpr.predict(X, return_std=True)

    def get_best_kernel_info(self):
        return self._gpr.get_best_kernel_info()

    def get_all_results(self):
        return self._gpr.get_all_results()
