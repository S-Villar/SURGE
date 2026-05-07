#!/usr/bin/env python
"""
Simple test script for HHFW dataset - avoids optional dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly what we need, avoiding optional deps
from surge import SurrogateEngine, SurrogateDataset
import pandas as pd
import numpy as np

# Dataset path
DATASET_PATH = "data/datasets/HHFW-NSTX/PwE_.pkl"

print("=" * 70)
print("🚀 SURGE Test on HHFW-NSTX Dataset")
print("=" * 70)

# Check if dataset exists
dataset_file = Path(DATASET_PATH)
if not dataset_file.exists():
    print(f"❌ Error: Dataset not found at {DATASET_PATH}")
    print(f"   Current directory: {Path.cwd()}")
    print(f"   Looking for: {dataset_file.absolute()}")
    sys.exit(1)

print(f"\n📂 Loading dataset: {DATASET_PATH}")

# Use SurrogateTrainer to load dataset
try:
    surge_trainer = SurrogateTrainer()
    input_cols, output_cols = surge_trainer.load_dataset_pickle(DATASET_PATH)
    df = surge_trainer.df
    trainer = surge_trainer.get_trainer()
    
    print(f"\n📊 Dataset Analysis:")
    print(f"   Input features: {len(input_cols)}")
    print(f"   Output profiles: {len(output_cols)}")
    print(f"\n   Sample input columns: {input_cols[:10]}")
    print(f"   Sample output columns: {output_cols[:5]}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("🤖 Training Model")
print("=" * 70)

try:
    print("\n📋 Setting up MLTrainer...")
    
    # Load data
    trainer.load_df_dataset(df, input_cols, output_cols)
    
    # Split data
    print("   Splitting data (80% train, 20% test)...")
    trainer.train_test_split(test_split=0.2, random_state=42)
    
    # Standardize
    print("   Standardizing data...")
    trainer.standardize_data()
    
    # Initialize model (use integer 0 for Random Forest)
    print("   Initializing Random Forest model...")
    trainer.init_model(0, n_estimators=100, max_depth=10, random_state=42)
    
    # Cross-validation
    print("   Running cross-validation (5-fold)...")
    cv_results = trainer.cross_validate(model_idx=0, cv_folds=5)
    print(f"      CV R²: {cv_results.get('r2_mean', 'N/A'):.6f}")
    print(f"      CV MSE: {cv_results.get('mse_mean', 'N/A'):.6f}")
    
    # Train
    print("   Training model...")
    trainer.train(0)
    
    # Predict
    print("   Evaluating on test set...")
    trainer.predict_output(0)
    
    print("✅ Model trained successfully!")
    
    # Show results
    print("\n" + "=" * 70)
    print("📈 Model Performance")
    print("=" * 70)
    
    if hasattr(trainer, 'model_performance') and len(trainer.model_performance) > 0:
        perf = trainer.model_performance[0]
        print(f"\nModel: {perf.get('model_name', 'Random Forest')}")
        print(f"Training time: {perf.get('training_time', 'N/A'):.4f} seconds")
        
        if hasattr(trainer, 'MSE') and trainer.MSE is not None:
            print(f"\nTest Set Performance:")
            print(f"  MSE: {trainer.MSE:.6f}")
            print(f"  RMSE: {trainer.MSE**0.5:.6f}")
            
        if hasattr(trainer, 'R2') and trainer.R2 is not None:
            print(f"  R² Score: {trainer.R2:.6f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("💾 Saving Results")
    print("=" * 70)
    
    results_dir = Path("test_outputs")
    results_dir.mkdir(exist_ok=True)
    
    # Save results JSON
    if hasattr(trainer, 'save_results'):
        results_file = trainer.save_results(
            model_index=0,
            cv_results=cv_results,
            filepath=str(results_dir / 'hhfw_results.json'),
        )
        print(f"✅ Results saved to: {results_file}")
    
    # Save model
    import joblib
    if hasattr(trainer, 'models') and len(trainer.models) > 0:
        model_package = {
            'model': trainer.models[0].get_estimator(),
            'x_scaler': trainer.x_scaler,
            'y_scaler': trainer.y_scaler,
            'input_feature_names': input_cols,
            'output_feature_names': output_cols,
        }
        model_file = results_dir / 'hhfw_model.joblib'
        joblib.dump(model_package, model_file)
        print(f"✅ Model saved to: {model_file}")
    
    # Try visualization (may fail if matplotlib not configured)
    print("\n" + "=" * 70)
    print("📊 Generating Visualizations")
    print("=" * 70)
    
    try:
        if len(output_cols) > 0:
            output_idx = 0
            output_name = output_cols[output_idx]
            
            print(f"\nPlotting results for: {output_name}")
            
            # Plot regression results
            fig, axes = trainer.plot_regression_results(
                output_index=output_idx,
                output_name=output_name,
                dataset='test',
                bins=100,
                cmap='plasma_r',
                units='[a.u.]'
            )
            
            plot_file = results_dir / f'hhfw_regression_{output_idx}.png'
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✅ Regression plot saved to: {plot_file}")
            
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    print(f"\n📁 Output files saved in: {results_dir.absolute()}")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

