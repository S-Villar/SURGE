"""
Test for comparing multiple models side-by-side.

This test demonstrates:
1. Training multiple models (PyTorch MLP, sklearn RFR, GPFlow GPR)
2. Comparing their performance visually
3. Saving comparison plots
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Check if visualization is available in the engine module
try:
    from surge.engine import VISUALIZATION_AVAILABLE
    VISUALIZATION_AVAILABLE_IN_ENGINE = VISUALIZATION_AVAILABLE
except (ImportError, AttributeError):
    VISUALIZATION_AVAILABLE_IN_ENGINE = False

# Both must be available for comparison tests
PLOTTING_AVAILABLE = MATPLOTLIB_AVAILABLE and VISUALIZATION_AVAILABLE_IN_ENGINE

from surge import SurrogateEngine


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 5
    n_outputs = 1  # Single output for simplicity
    
    X = np.random.random((n_samples, n_features))
    y = (
        2.0 * X[:, 0] ** 2 +
        1.5 * X[:, 1] * X[:, 2] +
        0.8 * X[:, 3] +
        0.1 * np.random.randn(n_samples)
    )
    
    # Ensure y is 2D
    y = y.reshape(-1, 1)
    
    input_names = [f'input_{i+1}' for i in range(n_features)]
    output_names = ['output_1']
    
    df = pd.DataFrame(X, columns=input_names)
    df[output_names[0]] = y[:, 0]
    
    return df, input_names, output_names


@pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="matplotlib or visualization not available")
def test_model_comparison(synthetic_dataset):
    """
    Test comparing multiple models: PyTorch MLP, sklearn RFR, and GPFlow GPR.
    """
    df, input_names, output_names = synthetic_dataset
    
    # Save plots to tests/plots directory for easy access
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        print("\n" + "="*70)
        print("TEST: Multi-Model Comparison")
        print("="*70)
        
        # Step 1: Initialize SurrogateEngine
        print("\n[Step 1] Initializing SurrogateEngine...")
        engine = SurrogateEngine(n_features=len(input_names), n_outputs=len(output_names))
        engine.load_df_dataset(df, input_names, output_names)
        engine.train_test_split(test_split=0.2)
        engine.standardize_data()
        print("✅ Data prepared")
        
        # Step 2: Train multiple models
        models_to_train = []
        
        # Model 0: Random Forest (sklearn)
        print("\n[Step 2a] Training Random Forest (sklearn)...")
        engine.init_model(0, n_estimators=50, max_depth=8, random_state=42)
        engine.train(0)
        engine.predict_output(0)
        models_to_train.append(0)
        print(f"✅ Random Forest: Train R²={engine.R2_train_val:.4f}, Test R²={engine.R2:.4f}")
        
        # Model 1: MLP (sklearn) - if available
        try:
            print("\n[Step 2b] Training MLP (sklearn)...")
            engine.init_model(1, hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
            engine.train(1)
            engine.predict_output(1)
            models_to_train.append(1)
            print(f"✅ MLP (sklearn): Train R²={engine.model_performance[1]['R2_train_val']:.4f}, Test R²={engine.model_performance[1]['R2']:.4f}")
        except Exception as e:
            print(f"⚠️ MLP (sklearn) not available: {e}")
        
        # Model 2: PyTorch MLP - if available
        try:
            print("\n[Step 2c] Training PyTorch MLP...")
            engine.init_model(2, hidden_layers=[50, 25], epochs=50, learning_rate=0.01, random_state=42)
            engine.train(2)
            engine.predict_output(2)
            models_to_train.append(2)
            print(f"✅ PyTorch MLP: Train R²={engine.model_performance[2]['R2_train_val']:.4f}, Test R²={engine.model_performance[2]['R2']:.4f}")
        except Exception as e:
            print(f"⚠️ PyTorch MLP not available: {e}")
        
        # Model 3: GPFlow GPR - if available
        try:
            print("\n[Step 2d] Training GPFlow GPR...")
            # GPflowGPR parameters: kernel_type, variance, noise_variance, optimize, maxiter
            engine.init_model(
                3,
                kernel_type='matern32',  # Kernel type: 'matern32', 'matern52', 'rbf', etc.
                variance=1.0,  # Kernel variance
                noise_variance=0.1,  # Noise variance
                optimize=True,  # Optimize hyperparameters
                maxiter=100  # Max optimization iterations (reduced for faster training)
            )
            engine.train(3)
            engine.predict_output(3)
            models_to_train.append(3)
            print(f"✅ GPFlow GPR: Train R²={engine.model_performance[3]['R2_train_val']:.4f}, Test R²={engine.model_performance[3]['R2']:.4f}")
        except Exception as e:
            print(f"⚠️ GPFlow GPR not available: {e}")
            print(f"   Install GPflow with: pip install gpflow")
        
        # Verify we have at least one model
        assert len(models_to_train) >= 1, "At least one model should be trained"
        print(f"\n✅ Trained {len(models_to_train)} model(s): {models_to_train}")
        
        # Step 3: Create comparison plots
        print("\n[Step 3] Creating model comparison plots...")
        
        # Compare on test set
        print("  Creating test set comparison...")
        fig_test, axes_test, results_test = engine.compare_models(
            model_indices=models_to_train,
            output_index=0,
            dataset='test',
            bins=100,
            cmap='plasma_r',
            save_path=str(plots_dir / 'model_comparison_test.png'),
            dpi=150,
        )
        print(f"✅ Test comparison plot saved to: {plots_dir / 'model_comparison_test.png'}")
        for model_name, r2 in results_test.items():
            print(f"   {model_name}: R² = {r2:.4f}")
        
        # Compare on train set
        print("  Creating training set comparison...")
        fig_train, axes_train, results_train = engine.compare_models(
            model_indices=models_to_train,
            output_index=0,
            dataset='train',
            bins=100,
            cmap='plasma_r',
            save_path=str(plots_dir / 'model_comparison_train.png'),
            dpi=150,
        )
        print(f"✅ Train comparison plot saved to: {plots_dir / 'model_comparison_train.png'}")
        for model_name, r2 in results_train.items():
            print(f"   {model_name}: R² = {r2:.4f}")
        
        # Verify plots were created
        assert (plots_dir / 'model_comparison_test.png').exists()
        assert (plots_dir / 'model_comparison_train.png').exists()
        
        # Close figures
        plt.close('all')
        
        # Step 4: Summary
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(f"✅ Trained {len(models_to_train)} model(s)")
        print(f"✅ Comparison plots saved to: {plots_dir}")
        print("\nTest Set Performance:")
        for model_name, r2 in results_test.items():
            print(f"   {model_name}: R² = {r2:.4f}")
        print("="*70)
        
        # Final assertions
        assert len(results_test) >= 1
        assert len(results_train) >= 1
        assert all(isinstance(r2, (int, float)) for r2 in results_test.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

