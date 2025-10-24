# SURGE Environment Setup Guide

This guide provides instructions for setting up the optimal environment to run SURGE's machine learning models and hyperparameter optimization.

## 📋 Available Environment Files

We provide three environment configurations:

1. **`environment.yml`** - Full-featured CPU environment (recommended for most users)
2. **`environment_minimal.yml`** - Minimal dependencies for basic functionality
3. **`environment_gpu.yml`** - GPU-accelerated environment with CUDA support

## 🚀 Quick Start

### Option 1: Full CPU Environment (Recommended)

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate surge

# Verify installation
python -c "import surge; surge._print_availability()"
```

### Option 2: Minimal Environment (Lightweight)

```bash
# For basic functionality without PyTorch or advanced features
conda env create -f environment_minimal.yml
conda activate surge-minimal
```

### Option 3: GPU Environment (For NVIDIA GPUs)

```bash
# For GPU-accelerated training and optimization
conda env create -f environment_gpu.yml
conda activate surge-gpu

# Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## 📦 What's Included

### Core Dependencies (All Environments)

- **NumPy** (≥1.20.0) - Numerical computing
- **Pandas** (≥1.5.0) - Data manipulation
- **Scikit-learn** (≥1.0.0) - ML algorithms (Random Forest, MLP, GP)
- **Matplotlib** (≥3.5.0) - Visualization
- **Joblib** (≥1.0.0) - Model serialization

### Full Environment Additional Features

- **PyTorch** (≥1.9.0) - Enhanced neural network models
- **Optuna** (≥3.0.0) - Tree-structured Parzen Estimator optimization
- **BoTorch** (≥0.8.0) - Bayesian optimization
- **Scikit-Optimize** (≥0.9.0) - Gaussian Process-based optimization
- **Development Tools** - pytest, ruff, black, jupyter

### GPU Environment Extra Features

- **CUDA Support** - GPU-accelerated PyTorch training
- **TensorFlow + GPflow** - Advanced Gaussian Process models
- **GPU-Accelerated BoTorch** - Faster Bayesian optimization

## 🔧 System Requirements

### Minimum Requirements (CPU)
- **OS**: Linux, macOS, or Windows
- **RAM**: 8 GB (16 GB recommended)
- **Python**: 3.8-3.11
- **Disk Space**: ~5 GB for environment

### GPU Requirements (Optional)
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥3.5
- **CUDA**: 11.8 or 12.1
- **VRAM**: 4 GB minimum (8+ GB recommended)
- **Drivers**: Latest NVIDIA drivers

### Apple Silicon (M1/M2/M3)
- PyTorch will automatically use **MPS** (Metal Performance Shaders)
- Remove `cpuonly` from `environment.yml` before creating
- GPU acceleration available without CUDA dependencies

## 🎯 Choosing the Right Environment

| Use Case | Environment | Key Features |
|----------|------------|--------------|
| Development & Research | `environment.yml` | Full features, CPU-optimized |
| Production/Deployment | `environment_minimal.yml` | Smallest footprint |
| GPU Training | `environment_gpu.yml` | CUDA acceleration |
| Testing & CI/CD | `environment_minimal.yml` | Fast setup |
| Hyperparameter Tuning | `environment.yml` or `_gpu.yml` | All optimization backends |

## 📝 Installation Steps (Detailed)

### Step 1: Install Conda

If you don't have conda installed:

```bash
# Download Miniconda (lightweight) or Anaconda (full-featured)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or for macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### Step 2: Clone SURGE Repository

```bash
cd /path/to/your/workspace
git clone <your-surge-repo-url>
cd SURGE
```

### Step 3: Create Environment

```bash
# Choose one:
conda env create -f environment.yml          # Full CPU
conda env create -f environment_minimal.yml  # Minimal
conda env create -f environment_gpu.yml      # GPU
```

### Step 4: Activate and Verify

```bash
conda activate surge  # or surge-minimal, surge-gpu

# Check SURGE installation
python -c "import surge; print(surge.__version__)"

# Check available backends
python -c "from surge import PYTORCH_AVAILABLE, GPFLOW_AVAILABLE; print(f'PyTorch: {PYTORCH_AVAILABLE}, GPflow: {GPFLOW_AVAILABLE}')"
```

## 🔄 Updating the Environment

If dependencies change:

```bash
# Update existing environment
conda activate surge
conda env update -f environment.yml --prune

# Or recreate from scratch
conda env remove -n surge
conda env create -f environment.yml
```

## 🐛 Troubleshooting

### Issue: PyTorch not detecting GPU

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue: BoTorch import fails

```bash
# Install BoTorch dependencies explicitly
pip install botorch gpytorch
```

### Issue: Out of memory during training

- Reduce batch size in model configuration
- Use `environment_minimal.yml` for lower memory footprint
- Enable gradient checkpointing for large models

### Issue: Conda environment creation is slow

```bash
# Use mamba (faster conda alternative)
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

## 🧪 Testing Your Setup

Run the test suite to verify everything works:

```bash
conda activate surge
cd SURGE

# Run basic tests
pytest tests/test_models.py -v

# Run hyperparameter optimization tests
pytest tests/test_cv_tuning.py -v

# Run all tests
pytest tests/ -v
```

## 📚 Key Dependencies Explained

### Core ML Frameworks

- **Scikit-learn**: Traditional ML models (Random Forest, MLP, Gaussian Processes)
- **PyTorch**: Deep learning and enhanced neural networks
- **GPflow**: Advanced Gaussian Process models (optional)

### Hyperparameter Optimization

- **Optuna**: Modern optimization framework with TPE and BoTorch samplers
- **Scikit-Optimize**: Gaussian Process-based Bayesian optimization
- **BoTorch**: State-of-the-art Bayesian optimization

### Why Multiple Optimization Backends?

SURGE supports 4 optimization methods:

1. **Random Search** (`random_mem_eff`) - Fast, memory-efficient baseline
2. **Bayesian Optimization** (`bayesian_skopt`) - Sample-efficient, good for expensive objectives
3. **Tree-structured Parzen Estimator** (`optuna_tpe`) - Fast, works well for categorical parameters
4. **BoTorch** (`optuna_botorch`) - State-of-the-art, best for continuous parameters

## 🎓 Next Steps

After setup:

1. Check out the [examples/](../examples/) directory for demos
2. Read the [documentation](../docs/) for API reference
3. Try the hyperparameter optimization examples:
   ```bash
   python examples/simple_optuna_demo.py
   python examples/hyperparameter_optimization_demo.py
   ```

## 💡 Tips for Performance

### CPU Optimization
- Set `OMP_NUM_THREADS` for optimal parallel performance:
  ```bash
  export OMP_NUM_THREADS=8  # Adjust based on your CPU
  ```
- Use Random Forest with `n_jobs=-1` for parallel training

### GPU Optimization
- Use PyTorch models and move tensors to GPU
- Increase batch size to fully utilize GPU memory
- Enable mixed precision training for faster training

### Memory Management
- Use `del` and `gc.collect()` after large operations
- Enable SURGE's resource monitoring: `plot_resources=True`
- Use memory-efficient random search for initial exploration

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the troubleshooting section above

## 🔗 Additional Resources

- [SURGE Documentation](../docs/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [BoTorch Documentation](https://botorch.org/)

---

**Last Updated**: October 2025
**SURGE Version**: 0.1.0
