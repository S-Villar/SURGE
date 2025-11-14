"""
Helper functions for common SURGE workflows.

Provides high-level convenience functions for typical machine learning tasks,
making SURGE easier to use for common workflows.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .engine import SurrogateEngine

# Backward compatibility alias
MLTrainer = SurrogateEngine


def quick_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    model_type: Union[int, str] = 0,
    test_split: float = 0.2,
    model_kwargs: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> SurrogateEngine:
    """
    Quick training function for rapid prototyping.
    
    Creates and trains a model with minimal setup, handling all preprocessing
    steps automatically.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features. Shape: (n_samples, n_features)
    y : np.ndarray or pd.DataFrame
        Target values. Shape: (n_samples,) or (n_samples, n_outputs)
    model_type : int or str, default=0
        Model type (0='sklearn.random_forest', 1='sklearn.mlp', etc.)
        or model registry key (e.g., 'random_forest', 'sklearn.random_forest')
    test_split : float, default=0.2
        Fraction of data to use for testing.
    model_kwargs : dict, optional
        Additional keyword arguments for model initialization.
    input_names : list of str, optional
        Names of input features. Auto-generated if None.
    output_names : list of str, optional
        Names of output variables. Auto-generated if None.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    SurrogateEngine
        Trained SurrogateEngine instance ready for predictions.
    
    Examples
    --------
    >>> import numpy as np
    >>> from surge.helpers import quick_train
    >>> 
    >>> X = np.random.random((100, 5))
    >>> y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
    >>> 
    >>> trainer = quick_train(X, y, model_type=0)  # Random Forest
    >>> predictions = trainer.predict_output(0)
    >>> print(f"R²: {trainer.R2:.4f}")
    """
    model_kwargs = model_kwargs or {}
    
    # Determine dimensions
    if isinstance(X, pd.DataFrame):
        n_features = X.shape[1]
        if input_names is None:
            input_names = list(X.columns)
        X_data = X
    else:
        n_features = X.shape[1]
        if input_names is None:
            input_names = [f'input_{i+1}' for i in range(n_features)]
        X_data = pd.DataFrame(X, columns=input_names)
    
    if isinstance(y, pd.DataFrame):
        n_outputs = y.shape[1]
        if output_names is None:
            output_names = list(y.columns)
        y_data = y
    else:
        if y.ndim == 1:
            n_outputs = 1
            y = y.reshape(-1, 1)
        else:
            n_outputs = y.shape[1]
        
        if output_names is None:
            output_names = [f'output_{i+1}' for i in range(n_outputs)]
        
        y_data = pd.DataFrame(y, columns=output_names)
    
    # Create trainer
    trainer = SurrogateEngine(n_features=n_features, n_outputs=n_outputs)
    
    # Load data
    if isinstance(X_data, pd.DataFrame) and isinstance(y_data, pd.DataFrame):
        df = pd.concat([X_data, y_data], axis=1)
        trainer.load_df_dataset(df, input_names, output_names)
    else:
        raise ValueError("X and y must both be pandas DataFrames for this helper")
    
    # Split data
    trainer.train_test_split(test_split=test_split, random_state=random_state)
    
    # Standardize
    trainer.standardize_data()
    
    # Initialize model
    # Handle model_type: can be int (legacy) or str (registry key)
    if isinstance(model_type, str):
        # Use model registry
        from .model import MODEL_REGISTRY
        model_adapter = MODEL_REGISTRY.create(model_type, **model_kwargs)
        # Find model index - add to trainer
        trainer.models.append(model_adapter)
        # Create a dummy spec for now
        from .engine import ModelSpec
        spec = ModelSpec(
            registry_key=model_type,
            display_name=model_type.replace('_', ' ').title(),
            default_params=model_kwargs
        )
        trainer.model_types.append(spec)
        # Initialize performance tracking for this model
        performance = trainer._build_performance_stub(spec.display_name)
        trainer.model_performance.append(performance)
        model_index = len(trainer.models) - 1
    else:
        # Legacy integer-based model type
        trainer.init_model(model_type, **model_kwargs)
        model_index = 0
    
    # Train
    trainer.train(model_index)
    
    # Evaluate
    trainer.predict_output(model_index)
    
    return trainer


def train_and_compare(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    model_types: List[Union[int, str]],
    test_split: float = 0.2,
    model_kwargs_list: Optional[List[Dict[str, Any]]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    random_state: int = 42,
    save_comparison_plots: bool = False,
    plots_dir: Optional[Union[str, Path]] = None,
) -> SurrogateEngine:
    """
    Train multiple models and compare them side-by-side.
    
    Convenience function for training multiple models, evaluating them,
    and optionally creating comparison plots.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features. Shape: (n_samples, n_features)
    y : np.ndarray or pd.DataFrame
        Target values. Shape: (n_samples,) or (n_samples, n_outputs)
    model_types : list of int or str
        List of model types to train and compare.
        Each can be an int (legacy) or str (registry key).
    test_split : float, default=0.2
        Fraction of data to use for testing.
    model_kwargs_list : list of dict, optional
        Additional keyword arguments for each model. If None, uses defaults.
    input_names : list of str, optional
        Names of input features. Auto-generated if None.
    output_names : list of str, optional
        Names of output variables. Auto-generated if None.
    random_state : int, default=42
        Random state for reproducibility.
    save_comparison_plots : bool, default=False
        Whether to save comparison plots.
    plots_dir : str or Path, optional
        Directory to save plots. If None, uses current directory.
    
    Returns
    -------
    SurrogateEngine
        Trained SurrogateEngine instance with all models ready.
        Use trainer.compare_models() to visualize comparisons.
    
    Examples
    --------
    >>> import numpy as np
    >>> from surge.helpers import train_and_compare
    >>> 
    >>> X = np.random.random((200, 5))
    >>> y = np.sum(X, axis=1) + 0.1 * np.random.randn(200)
    >>> 
    >>> # Compare Random Forest, MLP, and PyTorch MLP
    >>> trainer = train_and_compare(
    ...     X, y,
    ...     model_types=[0, 1, 2],  # RFR, sklearn MLP, PyTorch MLP
    ...     save_comparison_plots=True
    ... )
    >>> 
    >>> # View comparison plots
    >>> trainer.compare_models(model_indices=[0, 1, 2], save_path='comparison.png')
    """
    if model_kwargs_list is None:
        model_kwargs_list = [{}] * len(model_types)
    elif len(model_kwargs_list) != len(model_types):
        raise ValueError("model_kwargs_list must have same length as model_types")
    
    # Prepare data
    if isinstance(X, pd.DataFrame):
        n_features = X.shape[1]
        if input_names is None:
            input_names = list(X.columns)
        X_data = X
    else:
        n_features = X.shape[1]
        if input_names is None:
            input_names = [f'input_{i+1}' for i in range(n_features)]
        X_data = pd.DataFrame(X, columns=input_names)
    
    if isinstance(y, pd.DataFrame):
        n_outputs = y.shape[1]
        if output_names is None:
            output_names = list(y.columns)
        y_data = y
    else:
        if y.ndim == 1:
            n_outputs = 1
            y = y.reshape(-1, 1)
        else:
            n_outputs = y.shape[1]
        
        if output_names is None:
            output_names = [f'output_{i+1}' for i in range(n_outputs)]
        
        y_data = pd.DataFrame(y, columns=output_names)
    
    # Create trainer
        trainer = SurrogateEngine(n_features=n_features, n_outputs=n_outputs)
    
    # Load data
    df = pd.concat([X_data, y_data], axis=1)
    trainer.load_df_dataset(df, input_names, output_names)
    
    # Split data
    trainer.train_test_split(test_split=test_split, random_state=random_state)
    
    # Standardize
    trainer.standardize_data()
    
    # Train all models
    model_indices = []
    for i, (model_type, model_kwargs) in enumerate(zip(model_types, model_kwargs_list)):
        if isinstance(model_type, str):
            # Use model registry
            from .model import MODEL_REGISTRY
            model_adapter = MODEL_REGISTRY.create(model_type, **model_kwargs)
            trainer.models.append(model_adapter)
            from .engine import ModelSpec
            spec = ModelSpec(
                registry_key=model_type,
                display_name=model_type.replace('_', ' ').title(),
                default_params=model_kwargs
            )
            trainer.model_types.append(spec)
            perf = trainer._build_performance_stub(spec.display_name)
            trainer.model_performance.append(perf)
            model_idx = len(trainer.models) - 1
        else:
            # Legacy integer-based
            trainer.init_model(model_type, **model_kwargs)
            model_idx = i
        
        # Train
        trainer.train(model_idx)
        trainer.predict_output(model_idx)
        model_indices.append(model_idx)
    
    # Create comparison plots if requested
    if save_comparison_plots:
        if plots_dir is None:
            plots_dir = Path.cwd() / "plots"
        else:
            plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            trainer.compare_models(
                model_indices=model_indices,
                output_index=0 if n_outputs > 0 else None,
                dataset='test',
                save_path=str(plots_dir / 'model_comparison_test.png')
            )
        except Exception as e:
            print(f"⚠️ Could not create comparison plots: {e}")
    
    return trainer


def load_and_train(
    data_path: Union[str, Path],
    input_cols: List[str],
    output_cols: List[str],
    model_type: Union[int, str] = 0,
    test_split: float = 0.2,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SurrogateEngine:
    """
    Load data from file and train a model.
    
    Convenience function that loads data, prepares it, and trains a model
    in one go.
    
    Parameters
    ----------
    data_path : str or Path
        Path to data file. Supports CSV, pickle, parquet, Excel files.
    input_cols : list of str
        Column names for input features.
    output_cols : list of str
        Column names for output variables.
    model_type : int or str, default=0
        Model type to train.
    test_split : float, default=0.2
        Fraction of data to use for testing.
    model_kwargs : dict, optional
        Additional keyword arguments for model initialization.
    **kwargs
        Additional arguments passed to pandas read function.
    
    Returns
    -------
    SurrogateEngine
        Trained SurrogateEngine instance.
    
    Examples
    --------
    >>> from surge.helpers import load_and_train
    >>> 
    >>> trainer = load_and_train(
    ...     'data.csv',
    ...     input_cols=['x1', 'x2', 'x3'],
    ...     output_cols=['y1', 'y2'],
    ...     model_type='random_forest'
    ... )
    """
    data_path = Path(data_path)
    
    # Load data based on file extension
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path, **kwargs)
    elif data_path.suffix == '.pkl' or data_path.suffix == '.pickle':
        df = pd.read_pickle(data_path, **kwargs)
    elif data_path.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(data_path, **kwargs)
    elif data_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Use quick_train with the loaded data
    return quick_train(
        df[input_cols],
        df[output_cols],
        model_type=model_type,
        test_split=test_split,
        model_kwargs=model_kwargs,
        input_names=input_cols,
        output_names=output_cols,
    )

