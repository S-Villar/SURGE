"""
Demonstration of SURGE visualization functions.

This script shows how to create publication-quality plots of Ground Truth
vs Prediction using density heatmaps with the 'plasma' colormap.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from surge import MLTrainer
from surge.visualization import plot_gt_vs_prediction, plot_regression_comparison


def create_sample_data():
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    n_outputs = 3
    
    # Generate features
    X = np.random.random((n_samples, n_features))
    
    # Create outputs with polynomial relationships
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
    output_names = [f'PwE_{i+1}' for i in range(n_outputs)]
    
    df = pd.DataFrame(X, columns=input_names)
    for i, output_name in enumerate(output_names):
        df[output_name] = y_multi[:, i]
    
    return df, input_names, output_names


def main():
    """Run visualization demonstration."""
    
    print("\n" + "="*70)
    print("SURGE Visualization Demonstration")
    print("="*70)
    
    # Create sample data
    print("\n[Step 1] Creating sample dataset...")
    df, input_names, output_names = create_sample_data()
    print(f"✅ Dataset created: {len(df)} samples, {len(input_names)} inputs, {len(output_names)} outputs")
    
    # Initialize trainer
    print("\n[Step 2] Initializing MLTrainer...")
    trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
    
    # Load dataset
    print("\n[Step 3] Loading dataset...")
    trainer.load_df_dataset(df, input_names, output_names)
    
    # Train-test split
    print("\n[Step 4] Performing train-test split...")
    trainer.train_test_split(test_split=0.2)
    
    # Standardize
    print("\n[Step 5] Standardizing data...")
    trainer.standardize_data()
    
    # Initialize and train model
    print("\n[Step 6] Training Random Forest model...")
    trainer.init_model(0, n_estimators=100, max_depth=10, random_state=42)
    trainer.train(0)
    
    # Make predictions
    print("\n[Step 7] Making predictions...")
    trainer.predict_output(0)
    print(f"✅ Train R²: {trainer.R2_train_val:.4f}, Test R²: {trainer.R2:.4f}")
    
    # Create visualization plots
    print("\n[Step 8] Creating visualization plots...")
    
    # Option 1: Plot single output (first output)
    print("\n  Creating Ground Truth vs Prediction plot for first output...")
    fig, ax, r2 = trainer.plot_regression_results(
        model_index=0,
        output_index=0,
        dataset='both',
        bins=50,
        cmap='plasma_r',
        save_path='test_outputs/regression_comparison.png',
    )
    print(f"✅ Plot saved to: test_outputs/regression_comparison.png")
    
    # Option 2: Plot all outputs
    print("\n  Creating plots for all outputs...")
    try:
        fig_all, axes_all, results_all = trainer.plot_all_outputs(
            model_index=0,
            max_outputs=3,
            bins=50,
            cmap='plasma_r',
            save_path='test_outputs/multi_output_comparison.png',
        )
        print(f"✅ Multi-output plot saved to: test_outputs/multi_output_comparison.png")
        print(f"   R² scores:")
        for output_name, scores in results_all.items():
            print(f"     {output_name}: Train R²={scores['r2_train']:.4f}, Test R²={scores['r2_test']:.4f}")
    except Exception as e:
        print(f"⚠️ Multi-output plot failed: {e}")
    
    # Option 3: Direct use of visualization functions
    print("\n  Creating direct visualization plot...")
    from surge.visualization import plot_gt_vs_prediction
    
    y_true_train = trainer.y_train_val.values if hasattr(trainer.y_train_val, 'values') else trainer.y_train_val
    y_pred_train = trainer.y_pred_train_val
    y_true_test = trainer.y_test.values if hasattr(trainer.y_test, 'values') else trainer.y_test
    y_pred_test = trainer.y_pred_test
    
    # Plot training set only
    if y_true_train.ndim > 1:
        y_true_train_1 = y_true_train[:, 0]
        y_pred_train_1 = y_pred_train[:, 0]
    else:
        y_true_train_1 = y_true_train
        y_pred_train_1 = y_pred_train
    
    fig_direct, ax_direct, r2_direct = plot_gt_vs_prediction(
        y_true_train_1, y_pred_train_1,
        dataset_name="Training Set",
        title="Direct Visualization Example",
        bins=50,
        cmap='plasma_r',
        save_path='test_outputs/direct_visualization.png',
    )
    print(f"✅ Direct visualization plot saved to: test_outputs/direct_visualization.png")
    print(f"   R² score: {r2_direct:.4f}")
    
    print("\n" + "="*70)
    print("✨ Visualization demonstration complete!")
    print("="*70)
    print("\nPlots saved to test_outputs/ directory:")
    print("  - regression_comparison.png: Train vs Test comparison")
    print("  - multi_output_comparison.png: All outputs comparison")
    print("  - direct_visualization.png: Direct function usage example")
    print("\nTo display plots, uncomment plt.show() in the functions or use:")
    print("  import matplotlib.pyplot as plt")
    print("  plt.show()")


if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    main()

