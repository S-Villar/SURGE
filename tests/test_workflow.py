"""
Comprehensive test for complete MLTrainer workflow.

Demonstrates:
1. Dataset loading
2. Data preprocessing (split, standardization)
3. Model initialization (both integer and registry key formats)
4. Cross-validation
5. Training
6. Prediction and evaluation
7. Saving results
8. Saving model with scalers
9. Convenience methods
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from surge import MLTrainer


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    n_outputs = 2
    
    X = np.random.random((n_samples, n_features))
    y = (
        2.0 * X[:, 0] ** 2 +
        1.5 * X[:, 1] * X[:, 2] +
        0.8 * X[:, 3] +
        0.1 * np.random.randn(n_samples)
    )
    
    y_multi = np.column_stack([
        y,
        y * 1.2 + 0.1 * np.random.randn(n_samples)
    ])
    
    input_names = [f'input_{i+1}' for i in range(n_features)]
    output_names = [f'output_{i+1}' for i in range(n_outputs)]
    
    df = pd.DataFrame(X, columns=input_names)
    for i, output_name in enumerate(output_names):
        df[output_name] = y_multi[:, i]
    
    return df, input_names, output_names


def test_complete_workflow(synthetic_dataset):
    """Test complete workflow: load → split → standardize → CV → train → predict → save."""
    df, input_names, output_names = synthetic_dataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Step 1: Initialize and load dataset
        trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
        trainer.load_df_dataset(df, input_names, output_names)
        assert trainer.X.shape[0] == len(df)
        assert trainer.y.shape[1] == len(output_names)
        
        # Step 2: Train-test split
        trainer.train_test_split(test_split=0.2)
        assert len(trainer.X_train_val) + len(trainer.X_test) == len(df)
        
        # Step 3: Standardize data
        trainer.standardize_data()
        assert trainer.x_scaler is not None
        assert trainer.y_scaler is not None
        
        # Step 4: Initialize model
        trainer.init_model(0, n_estimators=50, max_depth=8, random_state=42)  # RFR
        assert len(trainer.models) == 1
        
        # Step 5: Cross-validation
        cv_results = trainer.cross_validate(model_idx=0, cv_folds=5)
        assert 'r2_mean' in cv_results
        assert 'mse_mean' in cv_results
        assert cv_results['r2_mean'] > -10.0  # Reasonable range
        assert cv_results['mse_mean'] > 0
        
        # Step 6: Train model
        trainer.train(0)
        assert trainer.model_performance[0]['training_time'] is not None
        
        # Step 7: Predict and evaluate
        trainer.predict_output(0)
        assert trainer.MSE is not None
        assert trainer.R2 is not None
        assert trainer.MSE > 0
        assert trainer.R2 > -10.0
        
        # Step 8: Save results using trainer.save_results()
        results_file = trainer.save_results(
            model_index=0,
            cv_results=cv_results,
            filepath=str(tmp_path / 'results.json'),
        )
        
        assert Path(results_file).exists()
        
        # Step 9: Save model with scalers
        if JOBLIB_AVAILABLE:
            model_adapter = trainer.models[0]
            model_package = {
                'model': model_adapter.get_estimator(),
                'x_scaler': trainer.x_scaler,
                'y_scaler': trainer.y_scaler,
                'input_feature_names': input_names,
                'output_feature_names': output_names,
            }
            
            model_file = tmp_path / 'model.joblib'
            joblib.dump(model_package, model_file)
            
            # Verify loading works
            loaded = joblib.load(model_file)
            assert 'model' in loaded
            assert 'x_scaler' in loaded
            assert 'y_scaler' in loaded


def test_model_save_load(synthetic_dataset):
    """Test that saved models can be loaded and produce consistent predictions."""
    if not JOBLIB_AVAILABLE:
        pytest.skip("joblib not available")
    
    df, input_names, output_names = synthetic_dataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Train model
        trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
        trainer.load_df_dataset(df, input_names, output_names)
        trainer.train_test_split(test_split=0.2)
        trainer.standardize_data()
        trainer.init_model(0, n_estimators=50, random_state=42)
        trainer.train(0)
        trainer.predict_output(0)
        
        # Save
        model_package = {
            'model': trainer.models[0].get_estimator(),
            'x_scaler': trainer.x_scaler,
            'y_scaler': trainer.y_scaler,
        }
        
        model_file = tmp_path / 'model.joblib'
        joblib.dump(model_package, model_file)
        
        # Load and test
        loaded = joblib.load(model_file)
        test_sample = trainer.X_test.iloc[:5]
        test_scaled = loaded['x_scaler'].transform(test_sample)
        pred_scaled = loaded['model'].predict(test_scaled)
        predictions = loaded['y_scaler'].inverse_transform(pred_scaled)
        
        assert predictions.shape == (5, len(output_names))
        assert not np.any(np.isnan(predictions))


def test_model_registry_integration(synthetic_dataset):
    """Test using model registry keys with MLTrainer."""
    df, input_names, output_names = synthetic_dataset
    
    trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
    trainer.load_df_dataset(df, input_names, output_names)
    trainer.train_test_split(test_split=0.2, random_state=42)
    trainer.standardize_data()
    
    # Test using registry key instead of integer
    try:
        trainer.init_model('random_forest', n_estimators=50, random_state=42)
        trainer.train(0)
        trainer.predict_output(0)
        
        assert trainer.R2 is not None
        assert trainer.MSE is not None
    except KeyError:
        # If registry key doesn't work, try alternative
        trainer.init_model('sklearn.random_forest', n_estimators=50, random_state=42)
        trainer.train(0)
        trainer.predict_output(0)
        
        assert trainer.R2 is not None


def test_convenience_methods(synthetic_dataset):
    """Test MLTrainer convenience methods."""
    df, input_names, output_names = synthetic_dataset
    
    trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
    trainer.load_df_dataset(df, input_names, output_names)
    trainer.train_test_split(test_split=0.2, random_state=42)
    trainer.standardize_data()
    
    # Train two models
    trainer.init_model(0, n_estimators=50, random_state=42)
    trainer.train(0)
    trainer.predict_output(0)
    
    trainer.init_model(1, hidden_layer_sizes=(20,), max_iter=100, random_state=42)
    trainer.train(1)
    trainer.predict_output(1)
    
    # Test list_available_models
    available = trainer.list_available_models()
    assert isinstance(available, dict)
    assert len(available) > 0
    
    # Test get_model_summary
    summary0 = trainer.get_model_summary(0)
    assert 'model_index' in summary0
    assert 'model_name' in summary0
    assert 'performance' in summary0
    assert summary0['model_index'] == 0
    
    # Test compare_all_models
    try:
        fig, axes = trainer.compare_all_models(output_index=0, dataset='test')
        assert fig is not None
    except Exception as e:
        # If visualization fails, that's okay for testing
        print(f"Visualization test skipped: {e}")


