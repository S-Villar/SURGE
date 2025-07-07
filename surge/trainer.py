from .models import (
    MLPModel, GPRModel, RFRModel, 
    PyTorchMLPModelWrapper, GPflowGPRModelWrapper, GPflowMultiKernelGPRWrapper
)
from .preprocessing import train_test_split_data, make_cv_splits
from .metrics import evaluate, summarize
from .utils import get_device, save_model, save_json


class SurrogateTrainer:
    def __init__(self, model_type="rfr", **kwargs):
        self.device = get_device()
        model_map = {
            "mlp": MLPModel,              # Scikit-learn MLP
            "pytorch_mlp": PyTorchMLPModelWrapper,  # PyTorch MLP
            "gpr": GPRModel,              # Scikit-learn GP
            "gpflow_gpr": GPflowGPRModelWrapper,    # GPflow GP
            "gpflow_multi": GPflowMultiKernelGPRWrapper,  # GPflow Multi-kernel GP
            "rfr": RFRModel,              # Random Forest
        }
        
        if model_type not in model_map:
            available_types = list(model_map.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")
            
        try:
            self.model = model_map[model_type](**kwargs)
            self.model_type = model_type
        except ImportError as e:
            raise ImportError(f"Required dependencies for '{model_type}' not available. {e}")

    def fit(self, X, y, test_size=0.2):
        self.X_tv, self.X_test, self.y_tv, self.y_test = train_test_split_data(
            X, y, test_size
        )
        return self

    def cross_validate(self, n_splits=5):
        results = {"mse_list": [], "r2_list": [], "time_list": []}
        for tr_idx, vl_idx in make_cv_splits(self.X_tv, self.y_tv, n_splits):
            X_tr, X_vl = self.X_tv[tr_idx], self.X_tv[vl_idx]
            y_tr, y_vl = self.y_tv[tr_idx], self.y_tv[vl_idx]
            self.model.fit(X_tr, y_tr)
            mse, r2, t = evaluate(self.model, X_vl, y_vl)
            results["mse_list"].append(mse)
            results["r2_list"].append(r2)
            results["time_list"].append(t)
        self.results = summarize(results)
        return self.results

    def save(self, out_dir="models"):
        save_model(self.model, f"{out_dir}/model.joblib")
        save_json(self.results, f"{out_dir}/results.json")
        
    def predict(self, X):
        """Make predictions with the trained model."""
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty quantification (for GP models).
        """
        if hasattr(self.model, 'predict_with_uncertainty'):
            return self.model.predict_with_uncertainty(X)
        else:
            # For non-GP models, return predictions with None for uncertainty
            predictions = self.model.predict(X)
            return predictions, None
    
    def get_model_info(self):
        """Get information about the fitted model."""
        info = {
            "model_type": self.model_type,
            "device": str(self.device) if hasattr(self, 'device') else None,
        }
        
        # Add model-specific information
        if hasattr(self.model, 'get_best_kernel_info'):
            info["best_kernel"] = self.model.get_best_kernel_info()
        
        if hasattr(self, 'results'):
            info["cv_results"] = self.results
            
        return info
