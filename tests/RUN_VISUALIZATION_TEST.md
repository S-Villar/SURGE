# How to Run the Visualization Test

The visualization test creates Ground Truth vs Prediction plots and saves them to `tests/plots/` directory.

## Prerequisites

You need to have the SURGE environment set up with all dependencies.

### Option 1: Using Conda (Recommended)

```bash
# Create the minimal environment (includes matplotlib)
conda env create -f envs/environment_minimal.yml

# Activate the environment
conda activate surge-minimal

# Verify matplotlib is installed
python -c "import matplotlib; print('✅ Matplotlib available')"
```

### Option 2: Using Pip

```bash
# Install from requirements.txt (includes matplotlib)
pip install -r requirements.txt

# Or install SURGE with dependencies
pip install -e .
```

## Running the Visualization Test

### Quick Test Script

```bash
python tests/test_visualization_only.py
```

This will:
1. Create a synthetic dataset
2. Train a Random Forest model
3. Generate predictions
4. Create 3-4 visualization plots
5. Save them to `tests/plots/` directory

### Full Workflow Test

```bash
python tests/run_complete_rfr_workflow.py
```

This includes everything plus model saving, cross-validation, etc.

### Pytest Version

```bash
pytest tests/test_complete_rfr_workflow.py::test_complete_rfr_workflow -v -s
```

## Expected Output

After running successfully, you should see:

```
✅ VISUALIZATION TEST COMPLETE
============================================================

📁 All plots saved to: /path/to/SURGE/tests/plots

Generated files:
  1. regression_comparison_output1.png (XXX.X KB)
  2. regression_train_output1.png (XXX.X KB)
  3. regression_test_output1.png (XXX.X KB)
  4. regression_all_outputs.png (XXX.X KB)  # If multi-output

✅ Successfully created 4 visualization plots!
```

## Plot Files

All plots are saved in: `tests/plots/`

- `regression_comparison_output1.png` - Side-by-side train vs test comparison
- `regression_train_output1.png` - Training set only
- `regression_test_output1.png` - Test set only
- `regression_all_outputs.png` - All outputs grid (if multi-output)

## Troubleshooting

### "matplotlib not available"
```bash
pip install matplotlib
# or
conda install matplotlib
```

### "pandas not available"
```bash
pip install pandas
# or
conda install pandas
```

### "ModuleNotFoundError: No module named 'surge'"
Make sure SURGE is installed:
```bash
pip install -e .
```

### Plots directory doesn't exist
The script creates it automatically. If it fails, manually create:
```bash
mkdir -p tests/plots
```

