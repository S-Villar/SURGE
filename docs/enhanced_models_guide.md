# SURGE Enhanced Models Guide

## Overview

SURGE now supports enhanced model implementations beyond the basic scikit-learn models:

- **PyTorch MLP**: Advanced neural network implementation with flexible architecture
- **GPflow GP**: Sophisticated Gaussian Process models with uncertainty quantification

## Available Model Types

### Basic Models (Always Available)
- `rfr`: Random Forest Regressor (scikit-learn)
- `mlp`: Multi-Layer Perceptron (scikit-learn) 
- `gpr`: Gaussian Process Regressor (scikit-learn)

### Enhanced Models (Require Optional Dependencies)
- `pytorch_mlp`: Enhanced MLP with PyTorch backend
- `gpflow_gpr`: Advanced GP with GPflow backend
- `gpflow_multi`: Multi-kernel GP with automatic kernel selection

## Installation

### Basic Installation
```bash
pip install -e .
```

### With PyTorch Support
```bash
pip install -e ".[pytorch]"
```

### With GPflow Support
```bash
pip install -e ".[gpflow]"
```

### With All Enhanced Features
```bash
pip install -e ".[all]"
```

## Usage Examples

### Basic Usage
```python
from surge import SurrogateTrainer
import numpy as np

# Generate sample data
X = np.random.random((100, 4))
y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

# Basic scikit-learn models
trainer_rf = SurrogateTrainer(model_type="rfr", n_estimators=100)
trainer_mlp = SurrogateTrainer(model_type="mlp", hidden_layer_sizes=(64, 32))
trainer_gp = SurrogateTrainer(model_type="gpr")
```

### PyTorch MLP (Enhanced)
```python
# Enhanced PyTorch MLP with advanced features
trainer_pytorch = SurrogateTrainer(
    model_type="pytorch_mlp",
    hidden_layers=[64, 32, 16],        # Custom architecture
    n_epochs=200,                      # Training epochs
    learning_rate=1e-3,                # Learning rate
    batch_size=32,                     # Batch size
    dropout_rate=0.1,                  # Dropout for regularization
    activation_fn="relu"               # Activation function
)

trainer_pytorch.fit(X, y)
results = trainer_pytorch.cross_validate(n_splits=5)
```

### GPflow Gaussian Process (Enhanced)
```python
# Single kernel GP
trainer_gp = SurrogateTrainer(
    model_type="gpflow_gpr",
    kernel_type="matern32",            # Kernel type
    lengthscales=None,                 # Auto-determined
    variance=1.0,                      # Signal variance
    noise_variance=0.1,                # Noise variance
    optimize=True,                     # Optimize hyperparameters
    maxiter=1000                       # Optimization iterations
)

trainer_gp.fit(X, y)
predictions, uncertainty = trainer_gp.predict_with_uncertainty(X_test)
```

### Multi-Kernel GP (Advanced)
```python
# Automatic kernel selection
trainer_multi = SurrogateTrainer(
    model_type="gpflow_multi",
    kernel_types=["matern32", "matern52", "rbf", "exponential"],
    lengthscales_range=(0.25, 2.0),
    variance_range=(0.1, 2.0),
    optimize=True,
    maxiter=500
)

trainer_multi.fit(X, y)
best_kernel = trainer_multi.get_best_kernel_info()
print(f"Best kernel: {best_kernel['name']}")
```

## PyTorch MLP Options

### Architecture Parameters
- `hidden_layers`: List of hidden layer sizes, e.g., `[64, 32, 16]`
- `dropout_rate`: Dropout probability for regularization (0.0 to 0.5)
- `activation_fn`: Activation function (`"relu"`, `"tanh"`, `"leaky_relu"`, `"sigmoid"`)

### Training Parameters
- `n_epochs`: Number of training epochs (default: 300)
- `learning_rate`: Learning rate for Adam optimizer (default: 1e-3)
- `batch_size`: Training batch size (default: 32)

### Example Configurations
```python
# Small, fast model
SurrogateTrainer(
    model_type="pytorch_mlp",
    hidden_layers=[32, 16],
    n_epochs=100,
    learning_rate=1e-2
)

# Large, high-capacity model
SurrogateTrainer(
    model_type="pytorch_mlp", 
    hidden_layers=[128, 64, 32, 16],
    n_epochs=500,
    learning_rate=5e-4,
    dropout_rate=0.2
)
```

## GPflow GP Options

### Kernel Types
- `"matern32"`: Matérn 3/2 kernel (smooth, twice differentiable)
- `"matern52"`: Matérn 5/2 kernel (very smooth)
- `"rbf"` or `"squared_exponential"`: RBF/Squared Exponential kernel
- `"exponential"`: Exponential kernel (rough)
- `"rational_quadratic"`: Rational Quadratic kernel

### Hyperparameters
- `lengthscales`: Array of length scales per input dimension
- `variance`: Signal variance (noise-free output variance)
- `noise_variance`: Observation noise variance

### Optimization
- `optimize`: Whether to optimize hyperparameters (recommended: True)
- `maxiter`: Maximum optimization iterations

### Example Configurations
```python
# Quick GP with default settings
SurrogateTrainer(
    model_type="gpflow_gpr",
    kernel_type="matern32"
)

# Highly optimized GP
SurrogateTrainer(
    model_type="gpflow_gpr",
    kernel_type="rbf",
    variance=2.0,
    noise_variance=0.01,
    optimize=True,
    maxiter=2000
)

# Multi-kernel comparison
SurrogateTrainer(
    model_type="gpflow_multi",
    kernel_types=["matern32", "matern52", "rbf"],
    optimize=True
)
```

## Uncertainty Quantification

GPflow models support uncertainty quantification:

```python
# Train GP model
trainer = SurrogateTrainer(model_type="gpflow_gpr")
trainer.fit(X, y)
trainer.model.fit(trainer.X_tv, trainer.y_tv)

# Get predictions with uncertainty
mean_pred, std_pred = trainer.predict_with_uncertainty(X_test)

# Sample from posterior distribution
samples = trainer.model.sample_posterior(X_test, num_samples=100)
```

## Model Comparison

```python
# Compare multiple model types
models = {
    "Random Forest": {"type": "rfr", "params": {"n_estimators": 100}},
    "PyTorch MLP": {"type": "pytorch_mlp", "params": {"hidden_layers": [64, 32]}},
    "GPflow GP": {"type": "gpflow_gpr", "params": {"kernel_type": "matern32"}}
}

results = {}
for name, config in models.items():
    trainer = SurrogateTrainer(model_type=config["type"], **config["params"])
    trainer.fit(X, y)
    cv_results = trainer.cross_validate(n_splits=5)
    results[name] = cv_results
    
# Compare performance
for name, result in results.items():
    print(f"{name}: R² = {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
```

## Integration with ml_utils.py

The enhanced models integrate seamlessly with your existing `ml_utils.py` workflow:

```python
# Import both SURGE and ml_utils
from surge import SurrogateTrainer
from scripts.ml_utils import MLTrainer

# Use SURGE for quick prototyping
surge_trainer = SurrogateTrainer(model_type="pytorch_mlp")
surge_trainer.fit(X, y)

# Use ml_utils for advanced workflows
ml_trainer = MLTrainer(X, y)
# ... custom ml_utils workflow
```

## Performance Tips

### PyTorch MLP
- Start with smaller networks and increase complexity if needed
- Use dropout for regularization with complex data
- Adjust learning rate based on convergence behavior
- Use larger batch sizes for stable training

### GPflow GP
- Enable optimization for better performance
- Try different kernels for different data characteristics
- Use multi-kernel mode for automatic kernel selection
- Consider computational cost for large datasets (>1000 points)

## Troubleshooting

### Import Errors
If you get import errors for enhanced models:

```python
from surge import PYTORCH_AVAILABLE, GPFLOW_AVAILABLE
print(f"PyTorch available: {PYTORCH_AVAILABLE}")
print(f"GPflow available: {GPFLOW_AVAILABLE}")
```

Install missing dependencies:
```bash
pip install torch  # For PyTorch models
pip install gpflow tensorflow  # For GPflow models
```

### Memory Issues
For large datasets with PyTorch:
- Reduce batch size
- Reduce network complexity
- Use data streaming if possible

For GPflow:
- Consider sparse GP approximations for >1000 points
- Reduce optimization iterations
- Use simpler kernels
