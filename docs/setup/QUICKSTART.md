# SURGE Environment Quick Reference

## 🚀 Quick Setup Commands

```bash
# Full CPU environment (recommended)
conda env create -f environment.yml
conda activate surge

# Minimal environment
conda env create -f environment_minimal.yml
conda activate surge-minimal

# GPU environment
conda env create -f environment_gpu.yml
conda activate surge-gpu

# Or use the setup script
./setup_environment.sh
```

## 📦 Core Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0 | Numerical computing |
| pandas | ≥1.5.0 | Data manipulation |
| scikit-learn | ≥1.0.0 | Classical ML models |
| torch | ≥1.9.0 | Neural networks |
| optuna | ≥3.0.0 | Hyperparameter optimization |
| botorch | ≥0.8.0 | Bayesian optimization |
| matplotlib | ≥3.5.0 | Visualization |

## 🎯 Which Environment?

| Scenario | Use |
|----------|-----|
| Development/Research | `environment.yml` |
| Production Deployment | `environment_minimal.yml` |
| GPU Training | `environment_gpu.yml` |
| CI/CD Testing | `environment_minimal.yml` |

## 🔧 Common Commands

```bash
# Update environment
conda env update -f environment.yml --prune

# List environments
conda env list

# Remove environment
conda env remove -n surge

# Export current environment
conda env export > environment_current.yml

# Check SURGE installation
python -c "import surge; print(surge.__version__)"

# Check available backends
python -c "from surge import PYTORCH_AVAILABLE; print(f'PyTorch: {PYTORCH_AVAILABLE}')"

# Verify GPU (if using GPU environment)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🧪 Test Your Setup

```bash
# Quick test
python -c "from surge.trainer import MLTrainer; print('✅ SURGE working!')"

# Run test suite
pytest tests/ -v

# Run specific test
pytest tests/test_models.py::test_random_forest -v

# Test with coverage
pytest tests/ --cov=surge --cov-report=html
```

## 🎓 Getting Started

```python
from surge.trainer import MLTrainer
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Initialize trainer
trainer = MLTrainer(n_features=3, n_outputs=1)
trainer.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
trainer.train_test_split(test_split=0.2)
trainer.standardize_data()

# Train model
trainer.init_model(0)  # 0=RandomForest, 1=MLP, 2=PyTorchMLP
trainer.train(0)
trainer.predict_output(0)

# Hyperparameter tuning with CV
results = trainer.tune(
    model_index=0,
    method='optuna_botorch',  # or 'random_mem_eff', 'bayesian_skopt', 'optuna_tpe'
    n_trials=50,
    cv=5  # 5-fold cross-validation
)

print(f"Best R²: {results['best_r2']:.4f}")
print(f"Best params: {results['best_params']}")
```

## 🐛 Troubleshooting Quick Fixes

```bash
# PyTorch not detecting GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# BoTorch import error
pip install botorch gpytorch

# Out of memory
# Reduce batch_size or use minimal environment

# Slow conda
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

## 📊 Optimization Method Comparison

| Method | Speed | Sample Efficiency | Use Case |
|--------|-------|-------------------|----------|
| `random_mem_eff` | ⚡⚡⚡ | ⭐ | Quick baseline |
| `bayesian_skopt` | ⚡⚡ | ⭐⭐⭐ | Small search space |
| `optuna_tpe` | ⚡⚡⚡ | ⭐⭐ | Mixed parameters |
| `optuna_botorch` | ⚡ | ⭐⭐⭐⭐ | Continuous parameters |

## 🔗 Resources

- **Documentation**: `docs/`
- **Examples**: `examples/`
- **Environment Setup**: `ENVIRONMENT_SETUP.md`
- **Tests**: `tests/`

## ⚙️ Environment Variables (Optional)

```bash
# Optimize CPU performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Suppress warnings
export PYTHONWARNINGS="ignore"
```

---
**Tip**: For detailed setup instructions, see `ENVIRONMENT_SETUP.md`
