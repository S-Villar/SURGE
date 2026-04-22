# SURGE Environment Files

This directory contains conda environment files for different SURGE installation scenarios.

## Available Environments

### 1. `environment.yml` (Main/Recommended)
**Location:** `/SURGE/environment.yml`

Full-featured environment with all dependencies:
- ✅ Core ML libraries (scikit-learn, numpy, pandas, scipy)
- ✅ PyTorch (CPU version, can be modified for GPU)
- ✅ Visualization (matplotlib, seaborn)
- ✅ Hyperparameter optimization (Optuna, scikit-optimize)
- ✅ Advanced optimization (BoTorch, GPflow)
- ✅ Development tools (pytest, jupyter, code quality tools)

**To create:**
```bash
conda env create -f environment.yml
conda activate surge
```

### 2. `envs/environment_gpu.yml` (GPU-enabled)
Full-featured environment with GPU support:
- ✅ Everything from main environment
- ✅ PyTorch with CUDA support
- ✅ BoTorch for GPU-accelerated optimization
- ✅ TensorFlow GPU support for GPflow

**Requirements:**
- NVIDIA GPU with compatible drivers
- CUDA 11.8 or 12.1

**To create:**
```bash
conda env create -f envs/environment_gpu.yml
conda activate surge-gpu
```

### 3. `envs/environment_minimal.yml` (Minimal)
Lightweight environment for basic functionality:
- ✅ Core ML libraries
- ✅ Visualization (matplotlib) - **Required for regression plots**
- ✅ Basic optimization (Optuna, scikit-optimize)
- ❌ No PyTorch
- ❌ No GPflow
- ❌ No advanced optimization

**To create:**
```bash
conda env create -f envs/environment_minimal.yml
conda activate surge-minimal
```

## Required Packages for Visualization

All environments now include `matplotlib>=3.5.0` which is **required** for:
- `trainer.plot_regression_results()` - Ground Truth vs Prediction plots
- `trainer.plot_all_outputs()` - Multi-output comparison plots
- Density heatmap visualizations with 'plasma' colormap

## Requirements.txt

**Location:** `/SURGE/requirements.txt`

Pip-based installation file (use if not using conda):
- ✅ Updated to include matplotlib (required for visualization)
- ✅ Includes core dependencies
- ✅ Includes optional dependencies (commented)

**To install:**
```bash
pip install -r requirements.txt
# Or with editable install:
pip install -e ".[dev]"
```

## Package Summary

### Core (All Environments)
- `numpy>=1.20.0`
- `scipy>=1.5.0`
- `pandas>=1.5.0`
- `scikit-learn>=1.0.0`
- `joblib>=1.0.0`
- `matplotlib>=3.5.0` ← **Required for visualization**

### Optional Advanced Features
- `torch>=1.9.0` - PyTorch for enhanced MLP models
- `optuna>=3.0.0` - Hyperparameter optimization
- `scikit-optimize>=0.9.0` - Bayesian optimization
- `botorch>=0.8.0` - Advanced Bayesian optimization (GPU-enabled)
- `gpflow>=2.5.0` - Gaussian Process models (requires TensorFlow)

### Development Tools
- `pytest>=6.0` - Testing framework
- `jupyter` / `jupyterlab` - Notebook support
- `ruff`, `black`, `isort`, `mypy` - Code quality tools

## Verification

After installing an environment, verify visualization is available:

```python
import surge
import matplotlib.pyplot as plt
print(f"SURGE {surge.__version__}")
print(f"matplotlib {plt.__version__}")

from surge import SurrogateWorkflowSpec, run_surrogate_workflow  # workflow API
from surge.hpc import ResourceSpec                                # resource spec
```

## Custom Environment Path

On shared systems where the ``surge`` environment is installed at a
non-default location, activate it by path:

```bash
conda activate "$SURGE_CONDA_ENV"   # or the absolute path on your site
```

Then run the bundled quickstart to sanity-check the install:

```bash
conda run -p "$SURGE_CONDA_ENV" python -m examples.quickstart --dataset diabetes
```

Use ``-p`` (path) instead of ``-n`` (name) when the environment is not in
conda's default envs directory.

## Notes

- **Visualization is now a core requirement** - all environments include matplotlib
- The minimal environment is sufficient for running tests with plots
- For full functionality, use the main `environment.yml` or GPU version
- Python version: 3.8 to 3.11 (3.12 may have compatibility issues)

