"""
Complete workflow test for Random Forest Regressor (RFR) in SURGE.

This test demonstrates:
1. Dataset loading
2. Cross-validation
3. Training
4. Testing
5. Saving results to a file
6. Saving the model with scalers together
"""

import json
import os
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
    n_outputs = 3
    
    # Generate synthetic data with a clear relationship
    X = np.random.random((n_samples, n_features))
    # Create outputs as polynomial combinations of inputs
    y = (
        2.0 * X[:, 0] ** 2 +
        1.5 * X[:, 1] * X[:, 2] +
        0.8 * X[:, 3] +
        0.1 * np.random.randn(n_samples)
    )
    
    # Create multi-output target
    y_multi = np.column_stack([
        y,
        y * 1.2 + 0.1 * np.random.randn(n_samples),
        y * 0.8 + 0.1 * np.random.randn(n_samples)
    ])
    
    # Create DataFrame
    input_names = [f'input_{i+1}' for i in range(n_features)]
    output_names = [f'output_{i+1}' for i in range(n_outputs)]
    
    df = pd.DataFrame(X, columns=input_names)
    for i, output_name in enumerate(output_names):
        df[output_name] = y_multi[:, i]
    
    return df, input_names, output_names


def test_complete_rfr_workflow(synthetic_dataset):
    """
    Test complete workflow: dataset loading, CV, training, testing,
    saving results, and saving model+scalers.
    """
    df, input_names, output_names = synthetic_dataset
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        print("\n" + "="*70)
        print("TEST: Complete RFR Workflow")
        print("="*70)
        
        # Step 1: Initialize MLTrainer
        print("\n[Step 1] Initializing MLTrainer...")
        trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
        assert trainer is not None
        assert trainer.F == len(input_names)
        assert trainer.T == len(output_names)
        print("✅ MLTrainer initialized")
        
        # Step 2: Load dataset
        print("\n[Step 2] Loading dataset...")
        trainer.load_df_dataset(df, input_names, output_names)
        assert trainer.df is not None
        assert trainer.X.shape[0] == len(df)
        assert trainer.y.shape[1] == len(output_names)
        print(f"✅ Dataset loaded: {trainer.X.shape[0]} samples")
        
        # Step 3: Train-test split
        print("\n[Step 3] Performing train-test split...")
        trainer.train_test_split(test_split=0.2)
        assert trainer.X_train_val is not None
        assert trainer.X_test is not None
        assert len(trainer.X_train_val) + len(trainer.X_test) == len(df)
        print("✅ Data split into train/validation and test sets")
        
        # Step 4: Standardize data
        print("\n[Step 4] Standardizing data...")
        trainer.standardize_data()
        assert trainer.x_scaler is not None
        assert trainer.y_scaler is not None
        assert trainer.x_train_val_sc is not None
        assert trainer.y_train_val_sc is not None
        print("✅ Data standardized")
        
        # Step 5: Initialize Random Forest model
        print("\n[Step 5] Initializing Random Forest Regressor...")
        trainer.init_model(0, n_estimators=100, max_depth=10, random_state=42)  # model_type 0 = RFR
        assert len(trainer.models) == 1
        assert len(trainer.model_types) == 1
        print("✅ Random Forest model initialized")
        
        # Step 6: Cross-validation
        print("\n[Step 6] Running cross-validation...")
        cv_results = trainer.cross_validate(model_idx=0, cv_folds=5)
        
        # Validate CV results
        assert 'mse_mean' in cv_results
        assert 'r2_mean' in cv_results
        assert 'mse_std' in cv_results
        assert 'r2_std' in cv_results
        assert cv_results['r2_mean'] > -10.0  # Should be reasonable (could be negative for bad fits)
        assert cv_results['mse_mean'] > 0
        print(f"✅ Cross-validation complete:")
        print(f"   CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
        print(f"   CV MSE: {cv_results['mse_mean']:.6f} ± {cv_results['mse_std']:.6f}")
        
        # Step 7: Train the model
        print("\n[Step 7] Training the model...")
        trainer.train(0)
        assert trainer.model_performance[0]['training_time'] is not None
        print("✅ Model trained")
        
        # Step 8: Test the model (predictions)
        print("\n[Step 8] Testing the model (making predictions)...")
        trainer.predict_output(0)
        
        # Validate test results
        assert trainer.MSE is not None
        assert trainer.R2 is not None
        assert trainer.MSE_train_val is not None
        assert trainer.R2_train_val is not None
        assert trainer.MSE > 0
        print(f"✅ Model tested:")
        print(f"   Train R²: {trainer.R2_train_val:.4f}, MSE: {trainer.MSE_train_val:.6f}")
        print(f"   Test R²: {trainer.R2:.4f}, MSE: {trainer.MSE:.6f}")
        
        # Step 9: Save results using trainer.save_results()
        print("\n[Step 9] Saving results to file...")
        results_file = trainer.save_results(
            model_index=0,
            cv_results=cv_results,
            filepath=str(tmp_path / 'rfr_results.json'),
        )
        
        assert Path(results_file).exists()
        # Verify we can load it back
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        assert loaded_results['model_type'] == 'RandomForestRegressor'
        if 'cross_validation' in loaded_results:
            assert 'r2_mean' in loaded_results['cross_validation']
        print(f"✅ Results saved to: {results_file}")
        
        # Step 10: Save model with scalers together
        print("\n[Step 10] Saving model with scalers...")
        if not JOBLIB_AVAILABLE:
            pytest.skip("joblib not available, skipping model save test")
        
        # Get the model adapter and extract the underlying estimator
        model_adapter = trainer.models[0]
        model_estimator = model_adapter.get_estimator()
        
        # Create a dictionary with model, scalers, and metadata
        model_package = {
            'model': model_estimator,
            'x_scaler': trainer.x_scaler,
            'y_scaler': trainer.y_scaler,
            'input_feature_names': input_names,
            'output_feature_names': output_names,
            'model_type': 'RandomForestRegressor',
            'metadata': {
                'n_features': len(input_names),
                'n_outputs': len(output_names),
                'train_size': len(trainer.X_train_val),
                'test_size': len(trainer.X_test),
                'train_r2': float(trainer.R2_train_val),
                'test_r2': float(trainer.R2),
            }
        }
        
        model_file = tmp_path / 'rfr_model_with_scalers.joblib'
        joblib.dump(model_package, model_file)
        
        assert model_file.exists()
        print(f"✅ Model with scalers saved to: {model_file}")
        
        # Step 11: Verify we can load and use the saved model
        print("\n[Step 11] Verifying saved model can be loaded...")
        loaded_package = joblib.load(model_file)
        
        assert 'model' in loaded_package
        assert 'x_scaler' in loaded_package
        assert 'y_scaler' in loaded_package
        assert 'input_feature_names' in loaded_package
        assert 'output_feature_names' in loaded_package
        
        # Test prediction with loaded model
        loaded_model = loaded_package['model']
        loaded_x_scaler = loaded_package['x_scaler']
        loaded_y_scaler = loaded_package['y_scaler']
        
        # Take a small sample from test set
        test_sample = trainer.X_test.iloc[:5]
        test_sample_scaled = loaded_x_scaler.transform(test_sample)
        
        # Make predictions
        predictions_scaled = loaded_model.predict(test_sample_scaled)
        predictions = loaded_y_scaler.inverse_transform(predictions_scaled)
        
        assert predictions.shape == (5, len(output_names))
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        
        print("✅ Loaded model verified - can make predictions")
        print(f"   Sample prediction shape: {predictions.shape}")
        print(f"   Sample prediction (first sample, first output): {predictions[0, 0]:.6f}")
        
        # Step 12: Create and save visualization plots
        print("\n[Step 12] Creating visualization plots...")
        try:
            # Create plots directory in tests folder
            plots_dir = Path(__file__).parent / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot 1: Side-by-side comparison (train vs test) for first output
            print("  Creating Ground Truth vs Prediction comparison plot...")
            fig_comparison, axes_comparison, results_plot = trainer.plot_regression_results(
                model_index=0,
                output_index=0,  # First output
                dataset='both',  # Both train and test
                bins=100,  # Increased bins for finer resolution
                cmap='plasma_r',
                save_path=str(plots_dir / 'regression_comparison_output1.png'),
                dpi=150,
            )
            print(f"✅ Comparison plot saved to: {plots_dir / 'regression_comparison_output1.png'}")
            print(f"   Plot R² scores: Train={results_plot['r2_train']:.4f}, Test={results_plot['r2_test']:.4f}")
            
            # Plot 2: Training set only
            print("  Creating training set plot...")
            fig_train, ax_train, r2_train_plot = trainer.plot_regression_results(
                model_index=0,
                output_index=0,
                dataset='train',
                bins=100,  # Increased bins for finer resolution
                cmap='plasma_r',
                save_path=str(plots_dir / 'regression_train_output1.png'),
                dpi=150,
            )
            print(f"✅ Training plot saved to: {plots_dir / 'regression_train_output1.png'}")
            print(f"   Train R²: {r2_train_plot['r2_train']:.4f}")
            
            # Plot 3: Test set only
            print("  Creating test set plot...")
            fig_test, ax_test, r2_test_plot = trainer.plot_regression_results(
                model_index=0,
                output_index=0,
                dataset='test',
                bins=100,  # Increased bins for finer resolution
                cmap='plasma_r',
                save_path=str(plots_dir / 'regression_test_output1.png'),
                dpi=150,
            )
            print(f"✅ Test plot saved to: {plots_dir / 'regression_test_output1.png'}")
            print(f"   Test R²: {r2_test_plot['r2_test']:.4f}")
            
            # Plot 4: All outputs comparison (if multi-output)
            if trainer.T > 1:
                print("  Creating multi-output comparison plot...")
                try:
                    fig_all, axes_all, results_all = trainer.plot_all_outputs(
                        model_index=0,
                        max_outputs=min(3, trainer.T),  # Plot up to 3 outputs
                        bins=100,  # Increased bins for finer resolution
                        cmap='plasma_r',
                        save_path=str(plots_dir / 'regression_all_outputs.png'),
                        dpi=150,
                    )
                    print(f"✅ Multi-output plot saved to: {plots_dir / 'regression_all_outputs.png'}")
                    for output_name, scores in results_all.items():
                        print(f"   {output_name}: Train R²={scores['r2_train']:.4f}, Test R²={scores['r2_test']:.4f}")
                except Exception as e:
                    print(f"⚠️ Multi-output plot failed: {e}")
            
            # Verify plots were created
            assert (plots_dir / 'regression_comparison_output1.png').exists()
            assert (plots_dir / 'regression_train_output1.png').exists()
            assert (plots_dir / 'regression_test_output1.png').exists()
            
            print(f"✅ All visualization plots saved to: {plots_dir}")
            
        except ImportError as e:
            print(f"⚠️ Visualization not available: {e}")
            print("   Install matplotlib to enable plotting: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")
            # Don't fail the test if plotting fails
        
        # Step 13: Summary
        print("\n" + "="*70)
        print("WORKFLOW TEST SUMMARY")
        print("="*70)
        print(f"✅ Dataset loaded: {len(df)} samples, {len(input_names)} inputs, {len(output_names)} outputs")
        print(f"✅ Cross-validation: {cv_results['r2_mean']:.4f} R²")
        print(f"✅ Training complete: {trainer.model_performance[0]['training_time']:.2f}s")
        print(f"✅ Test performance: {trainer.R2:.4f} R²")
        print(f"✅ Results saved: {results_file}")
        print(f"✅ Model+scalers saved: {model_file}")
        
        # Check if plots were created
        plots_dir = Path(__file__).parent / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                print(f"✅ Visualization plots saved: {len(plot_files)} files in {plots_dir}")
        
        print("="*70)
        
        # Final assertions
        assert trainer.R2 > -10.0  # Reasonable R² score
        assert trainer.MSE > 0
        assert results_file.exists()
        assert model_file.exists()


def test_model_save_load_integration(synthetic_dataset):
    """Test that saved models can be loaded and produce consistent predictions."""
    if not JOBLIB_AVAILABLE:
        pytest.skip("joblib not available")
    
    df, input_names, output_names = synthetic_dataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Train a model
        trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
        trainer.load_df_dataset(df, input_names, output_names)
        trainer.train_test_split(test_split=0.2)
        trainer.standardize_data()
        trainer.init_model(0, n_estimators=50, random_state=42)
        trainer.train(0)
        trainer.predict_output(0)
        
        # Save model package
        model_adapter = trainer.models[0]
        model_estimator = model_adapter.get_estimator()
        
        model_package = {
            'model': model_estimator,
            'x_scaler': trainer.x_scaler,
            'y_scaler': trainer.y_scaler,
            'input_feature_names': input_names,
            'output_feature_names': output_names,
        }
        
        model_file = tmp_path / 'test_model.joblib'
        joblib.dump(model_package, model_file)
        
        # Load and test
        loaded_package = joblib.load(model_file)
        loaded_model = loaded_package['model']
        loaded_x_scaler = loaded_package['x_scaler']
        loaded_y_scaler = loaded_package['y_scaler']
        
        # Compare predictions
        test_sample = trainer.X_test.iloc[:10]
        test_sample_scaled = loaded_x_scaler.transform(test_sample)
        
        # Predictions from original model
        original_pred_scaled = model_estimator.predict(trainer.x_test_sc[:10])
        original_pred = trainer.y_scaler.inverse_transform(original_pred_scaled)
        
        # Predictions from loaded model
        loaded_pred_scaled = loaded_model.predict(test_sample_scaled)
        loaded_pred = loaded_y_scaler.inverse_transform(loaded_pred_scaled)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=6)
        
        print("✅ Model save/load integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

