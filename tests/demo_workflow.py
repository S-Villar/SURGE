"""
Minimal standalone demo script showing complete MLTrainer workflow.

Demonstrates:
1. Dataset loading
2. Data preprocessing
3. Model training
4. Evaluation
5. Visualization
6. Saving results and model

Run with: python tests/demo_workflow.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from surge import MLTrainer


def create_dataset():
    """Create synthetic dataset."""
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


def main():
    """Run complete workflow demonstration."""
    print("\n" + "="*70)
    print("SURGE MLTrainer Workflow Demonstration")
    print("="*70)
    
    output_dir = Path(__file__).parent / "demo_outputs"
    output_dir.mkdir(exist_ok=True)
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Step 1: Create dataset
    print("\n[Step 1] Creating dataset...")
    df, input_names, output_names = create_dataset()
    print(f"✅ Created: {len(df)} samples, {len(input_names)} inputs, {len(output_names)} outputs")
    
    # Step 2: Initialize and load
    print("\n[Step 2] Initializing MLTrainer...")
    trainer = MLTrainer(n_features=len(input_names), n_outputs=len(output_names))
    trainer.load_df_dataset(df, input_names, output_names)
    print("✅ Dataset loaded")
    
    # Step 3: Preprocess
    print("\n[Step 3] Preprocessing data...")
    trainer.train_test_split(test_split=0.2)
    trainer.standardize_data()
    print("✅ Data split and standardized")
    
    # Step 4: Initialize model
    print("\n[Step 4] Initializing Random Forest model...")
    trainer.init_model(0, n_estimators=100, max_depth=10, random_state=42)
    print("✅ Model initialized")
    
    # Step 5: Cross-validation
    print("\n[Step 5] Running cross-validation...")
    cv_results = trainer.cross_validate(model_idx=0, cv_folds=5)
    print(f"✅ CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    
    # Step 6: Train
    print("\n[Step 6] Training model...")
    trainer.train(0)
    print("✅ Model trained")
    
    # Step 7: Evaluate
    print("\n[Step 7] Evaluating model...")
    trainer.predict_output(0)
    print(f"✅ Train R²: {trainer.R2_train_val:.4f}, Test R²: {trainer.R2:.4f}")
    
    # Step 8: Visualize
    if MATPLOTLIB_AVAILABLE:
        print("\n[Step 8] Creating visualizations...")
        trainer.plot_regression_results(
            model_index=0,
            output_index=0,
            dataset='both',
            save_path=str(plots_dir / 'demo_comparison.png'),
        )
        trainer.plot_all_outputs(
            model_index=0,
            max_outputs=2,
            save_path=str(plots_dir / 'demo_all_outputs.png'),
        )
        plt.close('all')
        print("✅ Plots saved to tests/plots/")
    else:
        print("\n[Step 8] Skipping visualization (matplotlib not available)")
    
    # Step 9: Save results using trainer.save_results()
    print("\n[Step 9] Saving results...")
    results_file = trainer.save_results(
        model_index=0,
        cv_results=cv_results,
        filepath=str(output_dir / 'results.json'),
    )
    print(f"✅ Results saved to {results_file}")
    
    # Step 10: Save model
    if JOBLIB_AVAILABLE:
        print("\n[Step 10] Saving model with scalers...")
        model_package = {
            'model': trainer.models[0].get_estimator(),
            'x_scaler': trainer.x_scaler,
            'y_scaler': trainer.y_scaler,
            'input_feature_names': input_names,
            'output_feature_names': output_names,
        }
        
        model_file = output_dir / 'model.joblib'
        joblib.dump(model_package, model_file)
        print(f"✅ Model saved to {model_file}")
    else:
        print("\n[Step 10] Skipping model save (joblib not available)")
    
    # Summary
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"📁 Outputs saved to: {output_dir}")
    if MATPLOTLIB_AVAILABLE:
        print(f"📊 Plots saved to: {plots_dir}")
    print()


if __name__ == "__main__":
    main()

