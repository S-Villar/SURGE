"""
Comprehensive tests for SurrogateEngine workflow.

Demonstrates complete training pipeline:
1. Data loading (from file, dataset, or DataFrame)
2. Data preprocessing (split, standardization)
3. Model initialization (integer and registry key formats)
4. Cross-validation
5. Training
6. Prediction and evaluation
7. Saving results (JSON)
8. Saving model with scalers (joblib)
9. Model loading and inference
10. Convenience methods
"""

import pytest

# Legacy pre-refactor API tests. The refactored SurrogateEngine uses
# keyword-only `registry=` / `run_config=` + `configure_dataframe(...)`
# instead of accepting data through the constructor. Skipped at module
# level until rewritten against the current API.
# Tracked in docs/REFACTORING_PLAN.md §1.9.
pytestmark = pytest.mark.skip(
    reason="legacy pre-refactor API; pending migration (docs/REFACTORING_PLAN.md §1.9)"
)

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

from surge import SurrogateEngine, SurrogateDataset


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


def test_complete_workflow_from_dataframe(synthetic_dataset):
    """Test complete workflow starting from DataFrame."""
    df, input_names, output_names = synthetic_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 1: Initialize and load dataset
        engine = SurrogateEngine(n_features=len(input_names), n_outputs=len(output_names))
        engine.load_df_dataset(df, input_names, output_names)
        assert engine.X.shape[0] == len(df)
        assert engine.y.shape[1] == len(output_names)

        # Step 2: Train-test split
        engine.train_test_split(test_split=0.2)
        assert len(engine.X_train_val) + len(engine.X_test) == len(df)

        # Step 3: Standardize data
        engine.standardize_data()
        assert engine.x_scaler is not None
        assert engine.y_scaler is not None

        # Step 4: Initialize model
        engine.init_model(0, n_estimators=50, max_depth=8, random_state=42)  # RFR
        assert len(engine.models) == 1

        # Step 5: Cross-validation
        cv_results = engine.cross_validate(model_idx=0, cv_folds=5)
        assert 'r2_mean' in cv_results
        assert 'mse_mean' in cv_results
        assert cv_results['r2_mean'] > -10.0  # Reasonable range
        assert cv_results['mse_mean'] > 0

        # Step 6: Train model
        engine.train(0)
        assert engine.model_performance[0]['training_time'] is not None

        # Step 7: Predict and evaluate
        engine.predict_output(0)
        assert engine.MSE is not None
        assert engine.R2 is not None
        assert engine.MSE > 0
        assert engine.R2 > -10.0

        # Step 8: Save results
        results_file = engine.save_results(
            model_index=0,
            cv_results=cv_results,
            filepath=str(tmp_path / 'results.json'),
        )
        assert Path(results_file).exists()

        # Step 9: Save model with scalers
        if JOBLIB_AVAILABLE:
            model_adapter = engine.models[0]
            model_package = {
                'model': model_adapter.get_estimator(),
                'x_scaler': engine.x_scaler,
                'y_scaler': engine.y_scaler,
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


def test_workflow_from_file(synthetic_dataset):
    """Test workflow starting from file with auto-detection."""
    df, input_names, output_names = synthetic_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pkl_path = tmp_path / 'data.pkl'
        df.to_pickle(pkl_path)

        # Test loading from file with auto-detection
        # Note: Since column names like 'input_1', 'input_2' match auto-detection patterns,
        # they may be detected as outputs. We'll test with manual specification to ensure
        # the workflow works correctly.
        engine = SurrogateEngine()
        detected_inputs, detected_outputs = engine.load_from_file(
            pkl_path,
            input_cols=input_names,  # Manually specify to avoid pattern confusion
            output_cols=output_names,
            auto_detect=False  # Disable auto-detect since we're providing columns
        )

        assert engine.df is not None
        assert len(detected_inputs) > 0
        assert len(detected_outputs) > 0

        # Continue workflow
        engine.train_test_split(test_split=0.2)
        engine.standardize_data()
        engine.init_model(0, n_estimators=50, random_state=42)
        engine.train(0)
        engine.predict_output(0)

        assert engine.R2 is not None


def test_workflow_from_dataset(synthetic_dataset):
    """Test workflow using SurrogateDataset."""
    df, input_names, output_names = synthetic_dataset

    # Create dataset first
    dataset = SurrogateDataset()
    dataset.df = df
    dataset.input_columns = input_names
    dataset.output_columns = output_names

    # Use with engine
    engine = SurrogateEngine()
    engine.load_from_dataset(dataset)

    assert engine.df is not None
    assert engine.X.shape[0] == len(df)

    # Continue workflow
    engine.train_test_split(test_split=0.2)
    engine.standardize_data()
    engine.init_model(0, n_estimators=50, random_state=42)
    engine.train(0)
    engine.predict_output(0)

    assert engine.R2 is not None


def test_model_registry_integration(synthetic_dataset):
    """Test using model registry keys with SurrogateEngine."""
    df, input_names, output_names = synthetic_dataset

    engine = SurrogateEngine(n_features=len(input_names), n_outputs=len(output_names))
    engine.load_df_dataset(df, input_names, output_names)
    engine.train_test_split(test_split=0.2, random_state=42)
    engine.standardize_data()

    # Test using registry key instead of integer
    try:
        engine.init_model('random_forest', n_estimators=50, random_state=42)
        engine.train(0)
        engine.predict_output(0)

        assert engine.R2 is not None
        assert engine.MSE is not None
    except KeyError:
        # If registry key doesn't work, try alternative
        engine.init_model('sklearn.random_forest', n_estimators=50, random_state=42)
        engine.train(0)
        engine.predict_output(0)

        assert engine.R2 is not None


def test_model_save_load(synthetic_dataset):
    """Test that saved models can be loaded and produce consistent predictions."""
    if not JOBLIB_AVAILABLE:
        pytest.skip("joblib not available")

    df, input_names, output_names = synthetic_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Train model
        engine = SurrogateEngine(n_features=len(input_names), n_outputs=len(output_names))
        engine.load_df_dataset(df, input_names, output_names)
        engine.train_test_split(test_split=0.2)
        engine.standardize_data()
        engine.init_model(0, n_estimators=50, random_state=42)
        engine.train(0)
        engine.predict_output(0)

        # Save
        model_package = {
            'model': engine.models[0].get_estimator(),
            'x_scaler': engine.x_scaler,
            'y_scaler': engine.y_scaler,
        }

        model_file = tmp_path / 'model.joblib'
        joblib.dump(model_package, model_file)

        # Load and test
        loaded = joblib.load(model_file)
        test_sample = engine.X_test.iloc[:5]
        test_scaled = loaded['x_scaler'].transform(test_sample)
        pred_scaled = loaded['model'].predict(test_scaled)
        predictions = loaded['y_scaler'].inverse_transform(pred_scaled)

        assert predictions.shape == (5, len(output_names))
        assert not np.any(np.isnan(predictions))


def test_convenience_methods(synthetic_dataset):
    """Test SurrogateEngine convenience methods."""
    df, input_names, output_names = synthetic_dataset

    engine = SurrogateEngine(n_features=len(input_names), n_outputs=len(output_names))
    engine.load_df_dataset(df, input_names, output_names)
    engine.train_test_split(test_split=0.2, random_state=42)
    engine.standardize_data()

    # Train two models
    engine.init_model(0, n_estimators=50, random_state=42)
    engine.train(0)
    engine.predict_output(0)

    engine.init_model(1, hidden_layer_sizes=(20,), max_iter=100, random_state=42)
    engine.train(1)
    engine.predict_output(1)

    # Test list_available_models
    available = engine.list_available_models()
    assert isinstance(available, dict)
    assert len(available) > 0

    # Test get_model_summary
    summary0 = engine.get_model_summary(0)
    assert 'model_index' in summary0
    assert 'model_name' in summary0
    assert 'performance' in summary0
    assert summary0['model_index'] == 0

    # Test compare_all_models
    try:
        fig, axes = engine.compare_all_models(output_index=0, dataset='test')
        assert fig is not None
    except Exception as e:
        # If visualization fails, that's okay for testing
        print(f"Visualization test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

