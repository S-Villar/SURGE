# SURGE Environment Setup - Complete Summary

## 📋 What I've Created

I've analyzed your SURGE codebase and created comprehensive environment setup files tailored to your machine learning and optimization needs. Here's what's included:

### Files Created

1. **`environment.yml`** - Full-featured CPU environment (2.2 KB)
2. **`environment_minimal.yml`** - Lightweight minimal setup (627 B)  
3. **`environment_gpu.yml`** - GPU-accelerated environment (2.0 KB)
4. **`ENVIRONMENT_SETUP.md`** - Detailed setup guide (7.9 KB)
5. **`QUICKSTART.md`** - Quick reference card (3.9 KB)
6. **`setup_environment.sh`** - Interactive setup script (2.3 KB, executable)

## 🎯 Recommended Setup (Based on Your Code)

Based on analyzing your SURGE codebase, here's my recommendation:

### For You: **Full CPU Environment** (`environment.yml`)

**Why?** Your code includes:
- ✅ PyTorch neural networks (`surge/pytorch_models.py`)
- ✅ Multiple optimization backends (Optuna, BoTorch, scikit-optimize)
- ✅ Comprehensive hyperparameter tuning (`test_cv_tuning.py`)
- ✅ Random Forest, MLP, and Gaussian Process models
- ✅ Cross-validation with proper scaling per fold
- ✅ Resource monitoring and performance tracking

**Setup Command:**
```bash
cd /path/to/SURGE
conda env create -f environment.yml
conda activate surge
python -c "import surge; surge._print_availability()"
```

## 📦 What's Included in Each Environment

### 1. Full CPU Environment (`environment.yml`) - **RECOMMENDED**

**Size**: ~3-4 GB installed  
**Setup Time**: 5-10 minutes

**Includes:**
- ✅ All scikit-learn models (RandomForest, MLP, GPR)
- ✅ PyTorch (CPU-optimized)
- ✅ All 4 optimization backends:
  - Random search (memory-efficient)
  - Bayesian optimization (scikit-optimize)
  - Tree-structured Parzen Estimator (Optuna TPE)
  - BoTorch (state-of-the-art Bayesian optimization)
- ✅ Full development tools (pytest, ruff, jupyter)
- ✅ Visualization (matplotlib, seaborn)
- ✅ Documentation tools (sphinx)

**Best For:**
- Development and research
- Hyperparameter tuning with CV
- Running all SURGE examples
- Your typical workflow

### 2. Minimal Environment (`environment_minimal.yml`)

**Size**: ~1-2 GB installed  
**Setup Time**: 2-3 minutes

**Includes:**
- ✅ Core dependencies only
- ✅ Scikit-learn models
- ✅ Basic optimization (Optuna, scikit-optimize)
- ❌ No PyTorch
- ❌ No BoTorch
- ❌ No development tools

**Best For:**
- Production deployment
- CI/CD pipelines
- Lightweight testing
- Memory-constrained systems

### 3. GPU Environment (`environment_gpu.yml`)

**Size**: ~5-6 GB installed  
**Setup Time**: 10-15 minutes

**Includes:**
- ✅ Everything from Full CPU
- ✅ PyTorch with CUDA 11.8 support
- ✅ GPU-accelerated BoTorch
- ✅ TensorFlow + GPflow for advanced GP models
- ✅ CUDA toolkit

**Best For:**
- Training large neural networks
- GPU-accelerated optimization
- Large-scale hyperparameter tuning
- When you have NVIDIA GPU available

## 🚀 Quick Start (3 Steps)

### Step 1: Navigate to SURGE directory
```bash
cd /path/to/SURGE
```

### Step 2: Create environment (choose one)

**Option A: Automated (Recommended)**
```bash
./setup_environment.sh
# Then select option 1 (Full CPU)
```

**Option B: Manual**
```bash
conda env create -f environment.yml
```

### Step 3: Activate and verify
```bash
conda activate surge
python -c "import surge; surge._print_availability()"
```

## 🧪 Test Your Setup

After setup, verify everything works:

```bash
# Basic import test
python -c "from surge.trainer import MLTrainer; print('✅ SURGE working!')"

# Check available backends
python -c "from surge import PYTORCH_AVAILABLE; print(f'PyTorch: {PYTORCH_AVAILABLE}')"

# Run a simple example
python examples/simple_optuna_demo.py

# Run the test suite
pytest tests/ -v
```

## 📊 Dependency Breakdown

### Core Dependencies (All Models)
```yaml
numpy>=1.20.0          # Numerical computing
pandas>=1.5.0          # Data handling
scikit-learn>=1.0.0    # ML models
matplotlib>=3.5.0      # Plotting
joblib>=1.0.0          # Model saving
```

### Neural Network Models
```yaml
torch>=1.9.0           # PyTorch MLP models
```

### Hyperparameter Optimization
```yaml
optuna>=3.0.0                      # TPE optimization
scikit-optimize>=0.9.0             # Bayesian optimization
botorch>=0.8.0                     # BoTorch (state-of-the-art)
optuna-integration[botorch]>=3.5.0 # Optuna + BoTorch
```

### Development Tools
```yaml
pytest>=6.0            # Testing
ruff>=0.1.0            # Fast linting
jupyter                # Notebooks
psutil                 # Resource monitoring
```

## 🎓 What You Can Do With This Setup

### 1. Train Multiple Model Types
```python
from surge.trainer import MLTrainer

trainer = MLTrainer(n_features=3, n_outputs=1)
trainer.init_model(0)  # RandomForest
trainer.init_model(1)  # Scikit-learn MLP  
trainer.init_model(2)  # PyTorch MLP
```

### 2. Hyperparameter Tuning with 4 Methods
```python
# Method 1: Random search (fast baseline)
results = trainer.tune(0, method='random_mem_eff', n_trials=100)

# Method 2: Bayesian optimization (sample-efficient)
results = trainer.tune(0, method='bayesian_skopt', n_trials=50)

# Method 3: TPE (Optuna's default, fast)
results = trainer.tune(0, method='optuna_tpe', n_trials=100)

# Method 4: BoTorch (state-of-the-art)
results = trainer.tune(0, method='optuna_botorch', n_trials=50)
```

### 3. Proper Cross-Validation
```python
# CV-enhanced tuning (prevents data leakage)
results = trainer.tune(
    model_index=0,
    method='optuna_botorch',
    n_trials=50,
    cv=5  # 5-fold cross-validation with proper scaling
)
```

### 4. Resource Monitoring
```python
# Track compute resources during training
trainer.tune(0, method='optuna_botorch', plot_resources=True)
```

## 🔧 Customization

### Modify for Your System

**Apple Silicon (M1/M2/M3)?**
```yaml
# In environment.yml, remove this line:
# - cpuonly
# PyTorch will automatically use MPS acceleration
```

**Have NVIDIA GPU?**
```bash
# Use GPU environment
conda env create -f environment_gpu.yml
```

**Limited Disk Space?**
```bash
# Use minimal environment
conda env create -f environment_minimal.yml
```

**Different CUDA Version?**
```yaml
# In environment_gpu.yml, change:
- pytorch-cuda=11.8  # to your version (e.g., 12.1)
- cudatoolkit=11.8   # match the above
```

## 📈 Performance Tips

### CPU Optimization
```bash
# Set thread count for optimal performance
export OMP_NUM_THREADS=8  # adjust to your CPU core count
export MKL_NUM_THREADS=8
```

### Memory Management
```python
# Use memory-efficient random search first
trainer.tune(0, method='random_mem_eff', n_trials=100)

# Then use expensive methods on smaller space
trainer.tune(0, method='optuna_botorch', n_trials=20)
```

### GPU Usage (if GPU environment)
```python
# Verify GPU is being used
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")

# Models automatically use GPU when available
trainer.init_model(2)  # PyTorch MLP uses GPU
```

## 🔍 Troubleshooting

### Problem: "Conda not found"
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Problem: "Environment creation is slow"
```bash
# Use mamba (faster conda)
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

### Problem: "PyTorch not using GPU"
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with correct CUDA:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Problem: "BoTorch import error"
```bash
# Install explicitly
pip install botorch gpytorch linear-operator
```

## 📚 Documentation Structure

```
SURGE/
├── environment.yml              # Main environment file
├── environment_minimal.yml      # Lightweight version
├── environment_gpu.yml          # GPU-accelerated version
├── ENVIRONMENT_SETUP.md         # Detailed setup guide (you are here)
├── QUICKSTART.md                # Quick reference card
├── setup_environment.sh         # Interactive setup script
├── requirements.txt             # Pip requirements (for reference)
└── README.md                    # Project README
```

## 🎉 Next Steps

1. **Create environment**: `conda env create -f environment.yml`
2. **Activate**: `conda activate surge`
3. **Verify**: `python -c "import surge; surge._print_availability()"`
4. **Test**: `pytest tests/ -v`
5. **Learn**: Try examples in `examples/` directory
6. **Build**: Start training your models!

## 🔗 Quick Links

- **Detailed Setup**: See `ENVIRONMENT_SETUP.md`
- **Quick Reference**: See `QUICKSTART.md`
- **Examples**: Check `examples/` directory
- **Tests**: See `tests/` directory
- **Code**: Explore `surge/` package

## 📞 Support

If you encounter issues:
1. Check `ENVIRONMENT_SETUP.md` troubleshooting section
2. Verify your conda version: `conda --version`
3. Check Python version: `python --version` (should be 3.8-3.11)
4. Review terminal output for specific error messages

---

**Created**: October 24, 2025  
**SURGE Version**: 0.1.0  
**Conda Environments**: 3 variants provided  
**Total Size**: ~3-6 GB depending on variant

**Ready to optimize? Let's go! 🚀**
