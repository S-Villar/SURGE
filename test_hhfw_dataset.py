#!/usr/bin/env python
"""
Test SURGE surrogate generator on HHFW-NSTX dataset.

This script demonstrates:
1. Loading the PwE_.pkl dataset (electron heating profiles)
2. Training a Random Forest Regressor
3. Cross-validation
4. Evaluation on test set
5. Visualization of results
6. Saving model and results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from surge import SurrogateEngine, SurrogateDataset

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

# Use SurrogateEngine with auto-detection (generalized pattern analysis)
try:
    engine = SurrogateEngine()
    input_cols, output_cols = engine.load_from_file(DATASET_PATH, auto_detect=True)
    df = engine.df
    
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

    # Use the unified SurrogateEngine
try:
    print("\n📋 Setting up SurrogateEngine...")
    trainer = engine  # Already has data loaded from load_from_file()
    
    # Split data
    print("   Splitting data (80% train, 20% test)...")
    trainer.train_test_split(test_split=0.2, random_state=42)
    
    # Standardize
    print("   Standardizing data...")
    trainer.standardize_data()
    
    # Initialize model
    print("   Initializing Random Forest model...")
    trainer.init_model('random_forest', n_estimators=100, max_depth=10, random_state=42)
    
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
        print(f"Training time: {perf.get('training_time', 'N/A')} seconds")
        
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
        # Use the cv_results from the cross_validate call above
        results_file = trainer.save_results(
            model_index=0,
            cv_results=cv_results,  # Use the cv_results from line 84
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
    
    # Create visualization
    print("\n" + "=" * 70)
    print("📊 Generating Visualizations")
    print("=" * 70)
    
    try:
        # Plot first output for test set
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
            
            # Plot all outputs comparison
            try:
                fig2, axes2 = trainer.plot_all_outputs(
                    output_indices=list(range(min(6, len(output_cols)))),
                    output_names=[output_cols[i] for i in range(min(6, len(output_cols)))],
                    dataset='test',
                    bins=100,
                    cmap='plasma_r',
                    units='[a.u.]'
                )
                plot_file2 = results_dir / 'hhfw_all_outputs.png'
                fig2.savefig(plot_file2, dpi=150, bbox_inches='tight')
                print(f"✅ All outputs plot saved to: {plot_file2}")
            except Exception as e:
                print(f"⚠️  Could not create all outputs plot: {e}")
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    print(f"\n📁 Output files saved in: {results_dir.absolute()}")
    print(f"   - Results: {results_dir / 'hhfw_results.json'}")
    print(f"   - Model: {results_dir / 'hhfw_model.joblib'}")
    print(f"   - Plots: {results_dir / '*.png'}")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

