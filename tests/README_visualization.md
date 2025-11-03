# Visualization in Tests

This document shows how visualization plots are created and saved in the test files.

## Overview

The test files (`test_complete_rfr_workflow.py` and `run_complete_rfr_workflow.py`) demonstrate how to:
1. Train a model using SURGE
2. Make predictions
3. Create publication-quality visualization plots
4. Save plots to the `tests/plots/` directory

## Plot Generation

After training and making predictions, the tests create several visualization plots:

### 1. Comparison Plot (Train vs Test)
```python
fig_comparison, axes_comparison, results_plot = trainer.plot_regression_results(
    model_index=0,
    output_index=0,  # First output
    dataset='both',  # Both train and test side-by-side
    bins=50,
    cmap='plasma_r',  # Inverse plasma colormap (default)
    save_path=str(plots_dir / 'regression_comparison_output1.png'),
    dpi=150,
)
```

**Output:** `tests/plots/regression_comparison_output1.png`
- Two subplots: (e) Training Set and (f) Test Set
- Density heatmap with 'plasma' colormap
- R² scores displayed
- Diagonal line for perfect prediction

### 2. Training Set Plot
```python
fig_train, ax_train, r2_train_plot = trainer.plot_regression_results(
    model_index=0,
    output_index=0,
    dataset='train',  # Training set only
    bins=50,
    cmap='plasma_r',  # Inverse plasma colormap (default)
    save_path=str(plots_dir / 'regression_train_output1.png'),
    dpi=150,
)
```

**Output:** `tests/plots/regression_train_output1.png`
- Single plot showing training set performance
- Density heatmap visualization

### 3. Test Set Plot
```python
fig_test, ax_test, r2_test_plot = trainer.plot_regression_results(
    model_index=0,
    output_index=0,
    dataset='test',  # Test set only
    bins=50,
    cmap='plasma_r',  # Inverse plasma colormap (default)
    save_path=str(plots_dir / 'regression_test_output1.png'),
    dpi=150,
)
```

**Output:** `tests/plots/regression_test_output1.png`
- Single plot showing test set performance
- Density heatmap visualization

### 4. Multi-Output Comparison (if multi-output)
```python
if trainer.T > 1:
    fig_all, axes_all, results_all = trainer.plot_all_outputs(
        model_index=0,
        max_outputs=min(3, trainer.T),  # Plot up to 3 outputs
        bins=50,
        cmap='plasma_r',  # Inverse plasma colormap (default)
        save_path=str(plots_dir / 'regression_all_outputs.png'),
        dpi=150,
    )
```

**Output:** `tests/plots/regression_all_outputs.png`
- Grid of plots for multiple outputs
- Each output gets train/test comparison

## Directory Structure

When tests run, plots are saved to:
```
tests/
├── plots/                          # Created automatically
│   ├── regression_comparison_output1.png    # Train vs Test comparison
│   ├── regression_train_output1.png         # Training set only
│   ├── regression_test_output1.png          # Test set only
│   └── regression_all_outputs.png           # All outputs (if multi-output)
├── test_complete_rfr_workflow.py
└── run_complete_rfr_workflow.py
```

## Key Features

### Plot Creation
- Plots are created using `trainer.plot_regression_results()` after `predict_output()`
- Plots use density heatmaps (2D histograms) with 'plasma' colormap
- R² scores are automatically calculated and displayed
- Diagonal reference line shows perfect prediction (y=x)

### Error Handling
- Tests gracefully handle missing matplotlib (ImportError)
- Plotting failures don't crash the test
- Warnings are printed if visualization is unavailable

### Path Management
```python
# Create plots directory in tests folder
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)
```
- Uses `Path(__file__).parent` to get tests directory
- Creates `plots/` subdirectory automatically
- All plots saved relative to tests folder

## Running the Tests

### Pytest Version
```bash
pytest tests/test_complete_rfr_workflow.py::test_complete_rfr_workflow -v -s
```

### Standalone Script
```bash
python tests/run_complete_rfr_workflow.py
```

Both will create plots in `tests/plots/` directory.

## Plot Specifications

- **Colormap:** 'plasma' (default, configurable)
- **Bins:** 50x50 2D histogram (configurable)
- **DPI:** 150 (publication quality)
- **Format:** PNG
- **Style:** Density scatter plots with logarithmic normalization

## Example Output

After running the tests, you'll see output like:
```
[Step 12] Creating visualization plots...
  Creating Ground Truth vs Prediction comparison plot...
✅ Comparison plot saved to: tests/plots/regression_comparison_output1.png
   Plot R² scores: Train=0.9876, Test=0.9234
  Creating training set plot...
✅ Training plot saved to: tests/plots/regression_train_output1.png
   Train R²: 0.9876
  Creating test set plot...
✅ Test plot saved to: tests/plots/regression_test_output1.png
   Test R²: 0.9234
✅ All visualization plots saved to: tests/plots
```

## Integration with Test Workflow

The visualization step is integrated into the complete workflow:
1. ✅ Dataset loading
2. ✅ Cross-validation
3. ✅ Training
4. ✅ Testing (`predict_output()`)
5. ✅ **Visualization** ← Creates and saves plots
6. ✅ Saving results to JSON
7. ✅ Saving model + scalers

All plots are automatically saved in the `tests/plots/` folder for easy inspection.

