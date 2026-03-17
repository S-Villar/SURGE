# SURGE vs. Other Tools: What Makes SURGE Unique?

## Executive Summary

**SURGE** is a **domain-specific, end-to-end workflow framework** for building **scientific surrogate models**. Unlike general-purpose ML tools, SURGE provides a complete, opinionated pipeline specifically designed for replacing expensive physics simulations with fast, uncertainty-aware surrogates.

---

## Quick Comparison Table

| Feature | SURGE | skorch | Ergodic adept | phi-flow |
|---------|-------|--------|---------------|----------|
| **Primary Purpose** | Scientific surrogate modeling | PyTorch → scikit-learn bridge | Differentiable solvers (JAX) | Differentiable physics (JAX) |
| **Workflow Automation** | ✅ Complete end-to-end | ❌ Model training only | ❓ Solver-focused | ❌ Simulation-focused |
| **Dataset Handling** | ✅ Auto-detection, multi-format | ❌ User-provided arrays | ❓ Custom | ❌ Custom |
| **HPO Integration** | ✅ Optuna built-in | ⚠️ Via scikit-learn | ❓ Custom | ❌ No |
| **Artifact Management** | ✅ Standardized layout | ❌ Manual | ❓ Custom | ❌ Custom |
| **Uncertainty Quantification** | ✅ Built-in (RFR, GP, MC-Dropout) | ❌ Manual | ❓ Custom | ❌ No |
| **HPC Awareness** | ✅ Auto-detection | ❌ No | ❓ Custom | ❌ No |
| **Scientific Domain Focus** | ✅ Physics simulations | ❌ General ML | ✅ Solvers | ✅ Physics |
| **YAML Config Workflows** | ✅ First-class | ❌ No | ❓ Unknown | ❌ No |
| **Reproducibility** | ✅ Full provenance | ⚠️ Manual | ❓ Custom | ⚠️ Manual |

---

## Detailed Comparison

### 1. SURGE vs. skorch

#### What skorch provides:
- **Scikit-learn interface for PyTorch**: Wraps PyTorch models to use `fit()`, `predict()`, `score()` methods
- **Grid search & cross-validation**: Works with scikit-learn's `GridSearchCV`, `Pipeline`, etc.
- **Model training**: Simplifies PyTorch training loops

#### What SURGE provides that skorch doesn't:

1. **Complete Workflow Automation**
   ```python
   # SURGE: One call does everything
   summary = run_surrogate_workflow(spec)  # Dataset → Split → Train → HPO → Artifacts
   
   # skorch: You still need to write the workflow
   from skorch import NeuralNetRegressor
   net = NeuralNetRegressor(MyModule, ...)
   # But you still need to:
   # - Load/preprocess data
   # - Split train/val/test
   # - Standardize
   # - Run HPO
   # - Save artifacts
   # - Track metrics
   ```

2. **Scientific Dataset Handling**
   - Auto-detects input/output columns from CSV/HDF5/NetCDF/Pickle
   - Handles metadata YAML overrides
   - Profile groups (spatial/temporal outputs)
   - Scientific data format support (NetCDF, HDF5)

3. **Built-in HPO with Optuna**
   - YAML-configurable search spaces
   - TPE and BoTorch samplers
   - Automatic trial tracking and export
   - Per-model HPO configuration

4. **Standardized Artifact Management**
   ```
   runs/<tag>/
   ├── models/          # Trained models
   ├── scalers/         # Input/output scalers
   ├── predictions/     # Per-split predictions
   ├── metrics.json     # All metrics
   ├── hpo/             # HPO trials
   ├── spec.yaml        # Workflow spec
   ├── environment.txt  # Environment snapshot
   └── git.txt          # Git revision
   ```

5. **Uncertainty Quantification Built-in**
   - Random Forest: Tree variance
   - GPflow: Mean/variance
   - PyTorch MLP: MC-Dropout
   - Unified `predict_with_uncertainty()` interface

6. **HPC Awareness**
   - Auto-detects compute resources (CPU, GPU, scheduler)
   - Records hardware context in artifacts
   - SLURM-aware

7. **Scientific Visualization**
   - Density scatter plots (for large datasets)
   - Profile comparisons
   - SNR/violin plots
   - Correlation heatmaps

**When to use skorch**: When you want scikit-learn compatibility for a PyTorch model in a general ML pipeline.

**When to use SURGE**: When building surrogates for scientific simulations with full workflow automation and reproducibility.

---

### 2. SURGE vs. Ergodic adept

#### What Ergodic adept likely provides (based on context):
- **Differentiable solvers in JAX**: Tools for solving differential equations with gradient-based optimization
- **JAX-based**: High-performance automatic differentiation
- **Solver-focused**: Likely focused on the solver implementation itself

#### What SURGE provides that adept doesn't:

1. **Surrogate Modeling Focus**
   - SURGE is specifically designed for **replacing expensive simulations** with fast ML models
   - adept appears focused on **solving differential equations** (different problem)

2. **End-to-End Workflow**
   - SURGE: Dataset → Model → Deployment pipeline
   - adept: Likely solver implementation and optimization

3. **Model Registry & Multi-Backend**
   - SURGE supports multiple backends (sklearn, PyTorch, GPflow) through unified interface
   - adept: Likely JAX-only

4. **Workflow Automation**
   - SURGE: YAML-driven, reproducible workflows
   - adept: Likely requires custom workflow code

5. **Scientific Data Integration**
   - SURGE: Built-in support for scientific data formats (NetCDF, HDF5)
   - adept: Likely expects custom data handling

**Key Difference**: SURGE builds **surrogates** (fast approximations) of simulations, while adept likely provides **differentiable solvers** (the simulations themselves).

**When to use adept**: When you need differentiable solvers for optimization or inverse problems.

**When to use SURGE**: When you need to replace expensive simulations with fast ML surrogates.

---

### 3. SURGE vs. phi-flow

#### What phi-flow provides:
- **Differentiable physics simulations**: Fluid dynamics and physics-based deep learning
- **JAX-based**: High-performance differentiable programming
- **Simulation framework**: For running physics simulations with gradients

#### What SURGE provides that phi-flow doesn't:

1. **Surrogate Modeling (Not Simulation)**
   - SURGE: Builds **ML models** that approximate simulation outputs
   - phi-flow: Provides **differentiable simulations** themselves

2. **Workflow Automation**
   - SURGE: Complete pipeline from data to deployed surrogate
   - phi-flow: Simulation framework (you build the workflow)

3. **Model Training & HPO**
   - SURGE: Built-in training, validation, HPO
   - phi-flow: Simulation framework (training is separate)

4. **Artifact Management**
   - SURGE: Standardized artifact layout for reproducibility
   - phi-flow: Simulation outputs (custom handling)

5. **Multi-Backend Support**
   - SURGE: sklearn, PyTorch, GPflow
   - phi-flow: JAX-only

**Key Difference**: SURGE creates **surrogates** (fast ML approximations), while phi-flow provides **differentiable simulations** (the physics itself).

**When to use phi-flow**: When you need differentiable physics simulations for optimization or learning.

**When to use SURGE**: When you need to replace expensive simulations (from any source) with fast ML surrogates.

---

## SURGE's Unique Value Proposition

### 1. **Domain-Specific: Scientific Surrogate Modeling**

SURGE is **not** a general-purpose ML framework. It's specifically designed for:

- **Replacing expensive physics simulations** with fast ML models
- **Scientific data formats** (NetCDF, HDF5, scientific pickles)
- **Uncertainty-aware predictions** for scientific applications
- **Reproducible workflows** for scientific computing

### 2. **Complete Workflow Automation**

Unlike tools that focus on one aspect (model training, solvers, simulations), SURGE provides:

```
Dataset Loading → Preprocessing → Model Training → HPO → Validation → Artifact Export
```

All in one framework, with YAML configuration.

### 3. **Reproducibility First**

Every run captures:
- Environment snapshot (`environment.txt`)
- Git revision (`git.txt`)
- Complete workflow spec (`spec.yaml`)
- All artifacts in standardized layout
- Compute resource context

### 4. **Scientific Domain Features**

- **Profile outputs**: Handles spatial/temporal profiles (common in physics)
- **Multi-output**: Built-in support for multiple outputs per model
- **Uncertainty quantification**: Essential for scientific applications
- **HPC integration**: Works seamlessly on clusters

### 5. **Extensibility**

- **Model registry**: Easy to add new model types
- **Adapter pattern**: Clean interface for new backends
- **Workflow hooks**: Customizable at every step

---

## Use Case Scenarios

### Use SURGE when:
- ✅ You have expensive physics simulations to replace
- ✅ You need uncertainty quantification
- ✅ You want reproducible, automated workflows
- ✅ You work with scientific data formats (NetCDF, HDF5)
- ✅ You need HPC-friendly workflows
- ✅ You want YAML-driven configuration
- ✅ You need standardized artifact management

### Use skorch when:
- ✅ You want scikit-learn interface for PyTorch
- ✅ You're doing general ML (not scientific surrogates)
- ✅ You want to use scikit-learn's GridSearchCV, Pipeline, etc.
- ✅ You're building custom workflows from scratch

### Use Ergodic adept when:
- ✅ You need differentiable solvers
- ✅ You're solving differential equations with gradients
- ✅ You're doing optimization/inverse problems
- ✅ You need JAX-based high performance

### Use phi-flow when:
- ✅ You need differentiable physics simulations
- ✅ You're doing fluid dynamics or physics-based deep learning
- ✅ You need gradients through simulations
- ✅ You're building physics-informed neural networks

---

## Code Comparison

### SURGE: Complete Workflow
```python
# One YAML file + one function call
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

spec = SurrogateWorkflowSpec.from_dict(yaml.safe_load("config.yaml"))
summary = run_surrogate_workflow(spec)
# Done! Models trained, HPO run, artifacts saved, metrics computed
```

### skorch: Model Training Only
```python
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

# You still need to:
# 1. Load/preprocess data
# 2. Split train/val/test
# 3. Standardize
# 4. Define model architecture
# 5. Run grid search
# 6. Save artifacts
# 7. Compute metrics
net = NeuralNetRegressor(MyModule, ...)
gs = GridSearchCV(net, param_grid, ...)
gs.fit(X_train, y_train)
# Still need to handle everything else manually
```

### phi-flow: Simulation Framework
```python
# phi-flow provides simulations, not surrogates
from phi.flow import *

# You build simulations, then need to:
# 1. Run simulations to generate data
# 2. Build ML models separately (not in phi-flow)
# 3. Handle training/HPO separately
# 4. Manage artifacts separately
```

---

## Summary: SURGE's Unique Strengths

1. **End-to-End Automation**: Complete workflow from data to deployed surrogate
2. **Scientific Domain Focus**: Built for physics simulation surrogates
3. **Reproducibility**: Full provenance tracking and artifact management
4. **Uncertainty Quantification**: Built-in UQ for scientific applications
5. **HPC Integration**: Works seamlessly on clusters
6. **YAML-Driven**: Configuration-first approach
7. **Multi-Backend**: Unified interface across sklearn, PyTorch, GPflow
8. **Scientific Data**: Native support for NetCDF, HDF5, scientific formats

**Bottom Line**: SURGE is the **only tool** that provides a complete, domain-specific framework for building scientific surrogate models with full workflow automation, reproducibility, and uncertainty quantification.

---

*For questions or clarifications, see the SURGE documentation or GitHub repository.*



