"""
Basic tests for SURGE package
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from surge import SurrogateTrainer, MLTrainer


def test_import():
    """Test that we can import the main classes."""
    assert SurrogateTrainer is not None
    assert MLTrainer is not None


def test_surrogate_trainer_init():
    """Test SurrogateTrainer initialization."""
    trainer = SurrogateTrainer()
    assert trainer is not None
    assert trainer.df is None
    assert trainer.input_variables is None
    assert trainer.output_variables is None


def test_ml_trainer_init():
    """Test MLTrainer initialization."""
    trainer = MLTrainer(n_features=4, n_outputs=10)
    assert trainer is not None
    assert trainer.F == 4
    assert trainer.T == 10
    assert len(trainer.models) == 0


def test_basic_workflow():
    """Test basic training workflow."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)
    
    # Create DataFrame for testing
    df = pd.DataFrame(X, columns=['input1', 'input2', 'input3'])
    df['output1'] = y
    
    # Test with MLTrainer directly
    trainer = MLTrainer(n_features=3, n_outputs=1)
    trainer.load_df_dataset(df, ['input1', 'input2', 'input3'], ['output1'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    trainer.train(0)
    trainer.predict_output(0)
    
    # Check that we got reasonable results
    assert trainer.R2 > 0.5  # Should be high for this simple additive relationship
    assert trainer.MSE < 1.0


def test_all_model_types():
    """Test all supported model types."""
    np.random.seed(42)
    X = np.random.random((30, 2))
    y = np.sum(X, axis=1)
    
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['y'] = y
    
    # Test Random Forest
    trainer = MLTrainer(n_features=2, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    trainer.train(0)
    trainer.predict_output(0)
    assert trainer.R2 > 0.8  # Should be very high for this simple relationship
    
    # Test MLP (sklearn)
    trainer2 = MLTrainer(n_features=2, n_outputs=1)
    trainer2.load_df_dataset(df, ['x1', 'x2'], ['y'])
    trainer2.train_test_split(test_split=0.3)
    trainer2.standardize_data()
    trainer2.init_model(1)  # MLP
    trainer2.train(0)
    trainer2.predict_output(0)
    assert trainer2.R2 > 0.5  # May be lower due to limited training


def test_cross_validation():
    """Test cross-validation functionality."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)
    
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y
    
    trainer = MLTrainer(n_features=3, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    
    # Run cross-validation
    cv_results = trainer.cross_validate(model_idx=0, cv_folds=3)
    
    assert 'mse_mean' in cv_results
    assert 'r2_mean' in cv_results
    assert cv_results['r2_mean'] > 0.5
    assert cv_results['mse_mean'] < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
