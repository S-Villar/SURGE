"""
Quick Start Demo for SURGE.

This script demonstrates the most common workflows for using SURGE,
showing both the detailed API and convenience helpers.
"""

import numpy as np
import pandas as pd

# Option 1: Using convenience helpers (simplest)
print("=" * 60)
print("OPTION 1: Quick Train (Simplest)")
print("=" * 60)

try:
    from surge.helpers import quick_train
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.random((100, 5))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
    
    # Train with one line!
    trainer = quick_train(X, y, model_type='random_forest', n_estimators=50)
    
    print(f"\n✅ Model trained!")
    print(f"   R² Score: {trainer.R2:.4f}")
    print(f"   MSE: {trainer.MSE:.6f}")
    
except ImportError:
    print("⚠️ Helper functions not available. Using detailed API...")


# Option 2: Using detailed MLTrainer API
print("\n" + "=" * 60)
print("OPTION 2: Detailed API (Full Control)")
print("=" * 60)

from surge import MLTrainer

# Generate sample data
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
y_multi = np.column_stack([y, y * 1.2])

# Create DataFrame
input_names = [f'x{i+1}' for i in range(n_features)]
output_names = [f'y{i+1}' for i in range(n_outputs)]
df = pd.DataFrame(X, columns=input_names)
for i, output_name in enumerate(output_names):
    df[output_name] = y_multi[:, i]

# Initialize trainer
trainer = MLTrainer(n_features=n_features, n_outputs=n_outputs)

# Load data
trainer.load_df_dataset(df, input_names, output_names)

# Split data
trainer.train_test_split(test_split=0.2, random_state=42)

# Standardize
trainer.standardize_data()

# Initialize and train model
trainer.init_model('random_forest', n_estimators=50, max_depth=8, random_state=42)
trainer.train(0)

# Evaluate
trainer.predict_output(0)

print(f"\n✅ Model trained!")
print(f"   R² Score (test): {trainer.R2:.4f}")
print(f"   MSE (test): {trainer.MSE:.6f}")


# Option 3: Compare multiple models
print("\n" + "=" * 60)
print("OPTION 3: Compare Multiple Models")
print("=" * 60)

try:
    from surge.helpers import train_and_compare
    
    # Generate data
    X = np.random.random((150, 4))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(150)
    
    # Train and compare multiple models
    trainer = train_and_compare(
        X, y,
        model_types=['random_forest', 'mlp'],  # or [0, 1] for legacy
        test_split=0.2,
        save_comparison_plots=False  # Set to True to save plots
    )
    
    print(f"\n✅ Trained {len(trainer.models)} models!")
    
    # Compare all models
    try:
        trainer.compare_all_models(output_index=0, dataset='test')
        print("   Comparison plots generated!")
    except Exception as e:
        print(f"   (Plots skipped: {e})")
    
except ImportError:
    print("⚠️ Helper functions not available for comparison")


# Option 4: Using model registry directly
print("\n" + "=" * 60)
print("OPTION 4: Using Model Registry")
print("=" * 60)

from surge.model import MODEL_REGISTRY

# List available models
print("\n📋 Available models:")
models = MODEL_REGISTRY.list_models()
for key, cls_name in list(models.items())[:5]:  # Show first 5
    print(f"   - {key}: {cls_name}")

print("\n✅ SURGE is ready to use!")
print("\n💡 Tip: Check tests/ directory for more examples")

