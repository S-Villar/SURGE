"""
SURGE API wrapper for the visualizer.

Provides a clean interface to train, load, and run inference with SURGE models.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Import SURGE components
try:
    from surge import MLTrainer, MODEL_REGISTRY
    from surge.helpers import quick_train
    SURGE_AVAILABLE = True
except ImportError:
    SURGE_AVAILABLE = False
    print("⚠️ SURGE not available. Using mock implementation.")


class ModelHandle:
    """Handle to a loaded SURGE model."""
    
    def __init__(self, model_id: str, trainer: Any, metadata: Dict):
        self.model_id = model_id
        self.trainer = trainer
        self.metadata = metadata
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Run inference with the model.
        
        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or dict
            Input parameters. If dict, should map parameter names to values.
        
        Returns
        -------
        dict
            Prediction results with keys:
            - 'type': 'point' | 'profile' | 'image'
            - 'yhat': predicted values
            - 'yhat_std': uncertainty (if available)
            - 'meta': metadata
        """
        # Convert dict to array if needed
        if isinstance(X, dict):
            # Assume trainer has input_feature_names or use metadata
            if hasattr(self.trainer, 'input_feature_names'):
                feature_names = self.trainer.input_feature_names
            else:
                feature_names = list(X.keys())
            
            X_array = np.array([[X[name] for name in feature_names]])
        elif isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        # Ensure 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # Scale if scaler available
        if hasattr(self.trainer, 'x_scaler') and self.trainer.x_scaler is not None:
            X_scaled = self.trainer.x_scaler.transform(X_array)
        else:
            X_scaled = X_array
        
        # Get model (assuming model_index 0 for now)
        if len(self.trainer.models) > 0:
            adapter = self.trainer.models[0]
            y_pred_scaled = adapter.predict(X_scaled)
            
            # Inverse scale if needed
            if hasattr(self.trainer, 'y_scaler') and self.trainer.y_scaler is not None:
                y_pred = self.trainer.y_scaler.inverse_transform(y_pred_scaled)
            else:
                y_pred = y_pred_scaled
            
            # Ensure 2D
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(1, -1)
            
            # Determine output type (simplified)
            n_outputs = y_pred.shape[1]
            if n_outputs == 1:
                pred_type = 'point'
            elif n_outputs > 1:
                # Could be profile or image - check metadata
                if 'profile_length' in self.metadata:
                    pred_type = 'profile'
                elif 'image_shape' in self.metadata:
                    pred_type = 'image'
                else:
                    pred_type = 'profile'  # Default assumption
            else:
                pred_type = 'point'
            
            result = {
                'type': pred_type,
                'yhat': y_pred,
                'meta': self.metadata.copy(),
            }
            
            # Try to get uncertainty if model supports it
            if hasattr(adapter, 'predict_with_uncertainty'):
                try:
                    y_pred_std_scaled, std_scaled = adapter.predict_with_uncertainty(X_scaled)
                    if hasattr(self.trainer, 'y_scaler'):
                        y_pred_std = self.trainer.y_scaler.inverse_transform(std_scaled)
                    else:
                        y_pred_std = std_scaled
                    result['yhat_std'] = y_pred_std
                except:
                    pass
            
            return result
        else:
            raise ValueError("No models available in trainer")


def train_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    model_type: str = 'random_forest',
    model_kwargs: Optional[Dict] = None,
    test_split: float = 0.2,
    save_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Train a SURGE model.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features.
    y : np.ndarray or pd.DataFrame
        Target values.
    model_type : str, default='random_forest'
        Model type from registry (e.g., 'random_forest', 'mlp', 'pytorch.mlp_model').
    model_kwargs : dict, optional
        Additional keyword arguments for model.
    test_split : float, default=0.2
        Test split ratio.
    save_dir : str or Path, optional
        Directory to save model. If None, uses default location.
    
    Returns
    -------
    str
        Model ID (for loading later).
    """
    if not SURGE_AVAILABLE:
        # Mock training
        model_id = f"mock_{int(time.time())}"
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        return model_id
    
    model_kwargs = model_kwargs or {}
    save_dir = Path(save_dir) if save_dir else Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Use quick_train for simplicity
    trainer = quick_train(
        X, y,
        model_type=model_type,
        test_split=test_split,
        model_kwargs=model_kwargs,
    )
    
    # Generate model ID
    model_id = f"{model_type}_{int(time.time())}"
    model_path = save_dir / model_id
    model_path.mkdir(exist_ok=True)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'model_kwargs': model_kwargs,
        'test_split': test_split,
        'input_shape': X.shape if hasattr(X, 'shape') else (len(X), len(X[0]) if hasattr(X[0], '__len__') else 1),
        'output_shape': y.shape if hasattr(y, 'shape') else (len(y),),
        'R2': trainer.R2,
        'MSE': trainer.MSE,
    }
    
    if hasattr(trainer, 'input_feature_names'):
        metadata['input_feature_names'] = trainer.input_feature_names
    if hasattr(trainer, 'output_feature_names'):
        metadata['output_feature_names'] = trainer.output_feature_names
    
    # Save trainer and metadata
    import joblib
    trainer_package = {
        'trainer': trainer,
        'metadata': metadata,
    }
    joblib.dump(trainer_package, model_path / 'trainer.joblib')
    
    with open(model_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return model_id


def load_model(model_id: str, model_dir: Optional[Union[str, Path]] = None) -> ModelHandle:
    """
    Load a trained SURGE model.
    
    Parameters
    ----------
    model_id : str
        Model identifier.
    model_dir : str or Path, optional
        Directory containing models. If None, uses default location.
    
    Returns
    -------
    ModelHandle
        Loaded model handle.
    """
    if not SURGE_AVAILABLE:
        # Return mock handle
        return ModelHandle(
            model_id=model_id,
            trainer=None,
            metadata={'mock': True}
        )
    
    model_dir = Path(model_dir) if model_dir else Path("models")
    model_path = model_dir / model_id
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load trainer
    import joblib
    trainer_package = joblib.load(model_path / 'trainer.joblib')
    
    trainer = trainer_package['trainer']
    metadata = trainer_package.get('metadata', {})
    
    return ModelHandle(model_id=model_id, trainer=trainer, metadata=metadata)


def infer(model: ModelHandle, params: Dict[str, float]) -> Dict[str, Any]:
    """
    Run inference with a model.
    
    Parameters
    ----------
    model : ModelHandle
        Loaded model handle.
    params : dict
        Parameter dictionary mapping feature names to values.
    
    Returns
    -------
    dict
        Prediction results (see ModelHandle.predict).
    """
    return model.predict(params)

