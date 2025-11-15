#!/usr/bin/env python3
"""
Run M3DC1 Growth Rate Prediction Demo

This script runs the complete workflow for predicting growth rate from
shape and profile parameters, with visualizations.

Usage:
    python3 run_m3dc1_demo.py [--hdf5-file path/to/sdata.h5]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path (go up from scripts/m3dc1/ to SURGE root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from surge import (
    SurrogateDataset,
    SurrogateEngine,
    convert_to_dataframe,
    validate_dataset,
    print_validation_report,
)

def create_mock_data(n_samples=200):
    """Create realistic mock M3DC1 data for growth rate prediction."""
    np.random.seed(42)
    
    # Shape parameters (from equilibrium)
    delta = np.random.uniform(0.3, 0.6, n_samples)
    kappa = np.random.uniform(1.5, 2.0, n_samples)
    R0 = np.random.uniform(0.8, 1.2, n_samples)
    a = np.random.uniform(0.3, 0.5, n_samples)
    
    # Profile parameters
    q0 = np.random.uniform(0.8, 1.2, n_samples)
    q95 = np.random.uniform(2.5, 4.5, n_samples)
    p0 = np.random.uniform(0.5, 2.0, n_samples)
    
    # Generate growth rate (physics-informed)
    gamma = (
        0.1 * (1.0 / q0) +
        0.05 * (q95 - 3.0)**2 +
        0.15 * p0 * (1.0 / kappa) +
        -0.08 * delta +
        0.02 * (R0 / a - 2.0) +
        np.random.normal(0, 0.02, n_samples)
    )
    gamma = np.maximum(gamma, 0.01)
    
    df = pd.DataFrame({
        'delta': delta,
        'kappa': kappa,
        'R0': R0,
        'a': a,
        'q0': q0,
        'q95': q95,
        'p0': p0,
        'gamma': gamma,
    })
    
    return df

def main():
    parser = argparse.ArgumentParser(description='M3DC1 Growth Rate Prediction Demo')
    parser.add_argument('--hdf5-file', type=str, default='sdata.h5',
                       help='Path to M3DC1 HDF5 file (default: sdata.h5)')
    parser.add_argument('--output-dir', type=str, default='runs/m3dc1_demo',
                       help='Output directory for results (default: runs/m3dc1_demo)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 M3DC1 Growth Rate Prediction Demo")
    print("=" * 70)
    
    # Load or create data
    hdf5_file = Path(args.hdf5_file)
    if hdf5_file.exists():
        print(f"\n✅ Loading from HDF5: {hdf5_file}")
        df = convert_to_dataframe(
            hdf5_file,
            include_equilibrium_features=True,
            include_mesh_features=False,
        )
        use_real_data = True
    else:
        print(f"\n⚠️  HDF5 file not found: {hdf5_file}")
        print("   Creating mock dataset...")
        df = create_mock_data(n_samples=200)
        use_real_data = False
    
    # Prepare dataset
    if use_real_data:
        # Try to find shape and profile parameters
        shape_params = ['eq_delta', 'eq_kappa', 'eq_R0', 'eq_a']
        profile_params = ['q0', 'q95', 'p0']
        
        available_cols = df.columns.tolist()
        input_cols = [col for col in shape_params + profile_params if col in available_cols]
        output_cols = [col for col in available_cols if 'gamma' in col.lower()]
        
        if not output_cols:
            print("⚠️  Warning: 'gamma' not found in dataset")
            print(f"   Available columns: {available_cols[:10]}...")
            return
    else:
        input_cols = ['delta', 'kappa', 'R0', 'a', 'q0', 'q95', 'p0']
        output_cols = ['gamma']
    
    print(f"\n📥 Input features ({len(input_cols)}): {input_cols}")
    print(f"📤 Output target ({len(output_cols)}): {output_cols}")
    
    # Create clean dataset
    df_clean = df[input_cols + output_cols].dropna()
    print(f"\n✅ Clean dataset: {df_clean.shape[0]} samples")
    
    # Load with SURGE
    dataset = SurrogateDataset()
    dataset.load_from_dataframe(df_clean, input_cols=input_cols, output_cols=output_cols, auto_detect=False)
    
    # Validate
    validation_results = validate_dataset(dataset.df)
    print_validation_report(validation_results)
    
    # Train models
    engine = SurrogateEngine(dir_path=args.output_dir)
    engine.load_from_dataset(dataset)
    engine.train_test_split(test_split=0.2, random_state=42)
    engine.standardize_data()
    
    # Random Forest
    print(f"\n🎯 Training Random Forest...")
    engine.init_model('random_forest', n_estimators=200, max_depth=15, random_state=42)
    engine.train(0)
    engine.predict_output(0)
    
    # MLP
    print(f"\n🎯 Training MLP...")
    engine.init_model('sklearn.mlp', hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    engine.train(1)
    engine.predict_output(1)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Test MSE':<12}")
    print(f"{'-'*70}")
    print(f"{'Random Forest':<20} {engine.R2_train_val:.4f}      {engine.R2:.4f}      {engine.MSE:.6f}")
    print(f"{'MLP':<20} {engine.model_performance[1]['R2_train_val']:.4f}      {engine.model_performance[1]['R2']:.4f}      {engine.model_performance[1]['MSE']:.6f}")
    print(f"{'='*70}")
    
    # Feature importance
    rf_model = engine.models[0].get_estimator()
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n📊 Feature Importance (Random Forest):")
        for i, idx in enumerate(indices):
            print(f"   {i+1}. {dataset.input_columns[idx]:<15} {importances[idx]:.4f}")
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Creating visualizations...")
    
    # Prediction plots
    try:
        fig, axes, results = engine.plot_regression_results(
            model_index=0,
            dataset='both',
            bins=50,
            cmap='plasma_r',
            xlabel='Actual Growth Rate (gamma)',
            ylabel='Predicted Growth Rate (gamma)',
            units='1/s',
            save_path=str(output_dir / 'rf_predictions.png'),
        )
        print(f"   ✅ Saved: {output_dir / 'rf_predictions.png'}")
    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")
    
    # Model comparison
    try:
        fig, axes, results = engine.compare_models(
            model_indices=[0, 1],
            dataset='test',
            bins=50,
            cmap='plasma_r',
            save_path=str(output_dir / 'model_comparison.png'),
        )
        print(f"   ✅ Saved: {output_dir / 'model_comparison.png'}")
    except Exception as e:
        print(f"   ⚠️  Comparison plot error: {e}")
    
    # Feature importance plot
    if hasattr(rf_model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(dataset.input_columns)), importances[indices], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(dataset.input_columns)))
        ax.set_yticklabels([dataset.input_columns[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance for Growth Rate Prediction', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_dir / 'feature_importance.png'}")
    
    # Save results
    engine.save_results(model_index=0, filepath=str(output_dir / 'rf_results.json'))
    engine.save_results(model_index=1, filepath=str(output_dir / 'mlp_results.json'))
    
    print(f"\n✅ Demo complete! Results saved to: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()





