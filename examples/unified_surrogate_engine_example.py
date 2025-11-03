"""
Minimal Example: Unified SurrogateEngine Framework

This example demonstrates the new unified SurrogateEngine framework with
SurrogateDataset for generalized data loading and auto-detection.

The framework is generalized - it detects inputs/outputs by pattern analysis
(e.g., var_1, var_2, ...) rather than hardcoded prefixes.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from surge import SurrogateDataset, SurrogateEngine

print("=" * 70)
print("🚀 SURGE Unified Framework Example")
print("=" * 70)

# Example 1: Load dataset with auto-detection using SurrogateDataset
print("\n" + "=" * 70)
print("Example 1: SurrogateDataset - Auto-Detection")
print("=" * 70)

dataset = SurrogateDataset()
input_cols, output_cols = dataset.load_from_file(
    'data/datasets/HHFW-NSTX/PwE_.pkl',
    auto_detect=True  # Uses generalized pattern analysis (var_1, var_2, ...)
)

print(f"\n✅ Auto-detected:")
print(f"   Input columns: {len(input_cols)}")
print(f"   Output columns: {len(output_cols)}")
print(f"   Output groups: {len(dataset.output_groups)}")

# Example 2: Unified SurrogateEngine - Direct file loading
print("\n" + "=" * 70)
print("Example 2: SurrogateEngine - Direct File Loading")
print("=" * 70)

engine = SurrogateEngine()

# Option A: Load directly from file (convenient)
engine.load_from_file('data/datasets/HHFW-NSTX/PwE_.pkl', auto_detect=True)

# Option B: Use SurrogateDataset first (more control)
# dataset = SurrogateDataset()
# dataset.load_from_file('data.pkl', auto_detect=True)
# engine.load_from_dataset(dataset)

# Split and standardize
engine.train_test_split(test_split=0.2, random_state=42)
engine.standardize_data()

# Initialize model (using registry key - generalized)
engine.init_model('random_forest', n_estimators=100, max_depth=10, random_state=42)

# Cross-validation
cv_results = engine.cross_validate(model_idx=0, cv_folds=5)
print(f"\n📊 Cross-Validation:")
print(f"   CV R²: {cv_results.get('r2_mean', 'N/A'):.6f} ± {cv_results.get('r2_std', 0):.6f}")

# Train
engine.train(0)

# Predict and evaluate
engine.predict_output(0)
print(f"\n📈 Test Performance:")
print(f"   R²: {engine.R2:.6f}")
print(f"   MSE: {engine.MSE:.6f}")

# Save results
results_file = engine.save_results(
    model_index=0,
    cv_results=cv_results,
    filepath='results.json'
)
print(f"\n💾 Results saved to: {results_file}")

# Example 3: Manual specification (alternative approach)
print("\n" + "=" * 70)
print("Example 3: Manual Column Specification")
print("=" * 70)

print("\nFor datasets with non-standard naming, you can specify manually:")
print("  engine.load_from_file('data.pkl',")
print("                        input_cols=['x1', 'x2', 'x3'],")
print("                        output_cols=['y1', 'y2'],")
print("                        auto_detect=False)")

print("\n" + "=" * 70)
print("✅ Example Complete!")
print("=" * 70)

