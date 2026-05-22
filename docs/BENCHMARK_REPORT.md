# SURGE Benchmark Report

> **SURGE** — Surrogate Unified Regression and Generalization Environment  
> Report generated from the benchmark suite in `surge/benchmarks/`.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Tier System](#tier-system)
3. [Model Registry](#model-registry)
4. [Benchmarks by Capability](#benchmarks-by-capability)
   - [Scalar Regression](#scalar-regression)
   - [Multi-Output Regression](#multi-output-regression)
   - [Tabular Classification](#tabular-classification)
   - [Scientific Classification](#scientific-classification)
   - [Time Series / Forecasting](#time-series--forecasting)
   - [1D Field Operator Learning](#1d-field-operator-learning)
   - [2D Field Operator Learning](#2d-field-operator-learning)
   - [Image Classification](#image-classification)
   - [Scientific Domain](#scientific-domain)
5. [Full Model × Benchmark Grid](#full-model--benchmark-grid)
6. [CLI Reproduction Commands](#cli-reproduction-commands)
7. [References](#references)

---

## Executive Summary

SURGE provides **29 benchmarks** across 9 capability categories and a registry of **35 models** spanning classical machine learning through state-of-the-art deep learning, operator learning, and generative models.

### Benchmark philosophy

- **Tier 0 benchmarks** (`synthetic.*`, sklearn built-ins) are **CI smoke tests only** — their numbers say nothing useful about model quality. Do not cite them as evidence of model performance.
- **Tier 1+ benchmarks** use real, published datasets with known reference values from the literature. These are the benchmarks to use for comparisons.
- **BoTorch GP on `tabular.diabetes`** (R²≈0.53) is expected — the diabetes dataset is famously hard; even the original LARS paper from Efron et al. tops out at R²≈0.52 with a linear model. Use `tabular.concrete_strength` or `tabular.california_housing` instead.

### Summary table

| Capability | # Benchmarks | Primary literature benchmarks | Best model (SURGE) |
|---|---|---|---|
| Scalar Regression | 9 | concrete, energy, yacht, airfoil, california_housing | XGBoost |
| Multi-Output Regression | 2 | scm20d | RF / ResidualMLP |
| Tabular Classification | 5 | iris, breast_cancer, covertype | XGBoost / RF |
| Scientific Classification | 3 | plasma_stability, flow_regime | XGBoost |
| Time Series / Forecasting | 2 | lorenz63 | LSTM / GRU |
| 1D PDE Operator | 2 | burgers_1d | FNO1d |
| 2D PDE Operator | 2 (Tier 3) | darcy_2d | FNO2d / U-Net |
| Vision | 2 | mnist, cifar10 | ResNet-56 |
| Scientific Domain | 4 | fusion.m3dc1_sample | Domain-specific |

---

## Tier System

Tiers define **computational cost and data requirements**, not model sophistication. More complex models (FNO2d, ViT, diffusion) can run across all tiers.

| Tier | Meaning | Benchmarks |
|------|---------|------------|
| **0** | Inline fixtures, no download, <2 min CPU. CI smoke tests. | `synthetic.*`, `tabular.iris`, `tabular.breast_cancer`, `tabular.diabetes`, `tabular.wine`, `tabular.digits` |
| **1** | Standard UCI/OpenML datasets, cached after first download. ~5 min CPU. | `tabular.california_housing`, `tabular.concrete_strength`, `tabular.energy_efficiency`, `tabular.yacht_dynamics`, `tabular.airfoil_noise`, `tabular.superconductor`, `sequence.lorenz63`, `multioutput.scm20d`, `pde.burgers_1d` |
| **2** | Larger datasets or GPU-recommended. | `vision.mnist`, `vision.cifar10`, `classification.plasma_stability`, `classification.flow_regime`, `classification.covertype`, `fusion.m3dc1_sample` |
| **3** | External HDF5 downloads (PDEBench ~GB files). | `pdebench.burgers_1d`, `pdebench.darcy_2d`, `pdebench.shallow_water_2d` |
| **4** | Specialized external packages (`the-well`). | `thewell.gray_scott`, `thewell.turbulence_2d`, `thewell.mhd` |

---

## Model Registry

SURGE includes 35 registered models across 5 backends.

### Sklearn / Classical

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `sklearn.random_forest` | Random Forest Regressor | Regression | Scikit-learn RF [[1]](#ref1) |
| `sklearn.gradient_boosting_regressor` | Gradient Boosting | Regression | GBRT with MultiOutputRegressor wrapper [[1]](#ref1) |
| `sklearn.mlp` | MLP Regressor | Regression | Scikit-learn's MLP |
| `sklearn.gpr` | Gaussian Process Regressor | Regression | RBF kernel, sklearn [[1]](#ref1) |
| `sklearn.random_forest_classifier` | Random Forest Classifier | Classification | |
| `sklearn.gradient_boosting_classifier` | Gradient Boosting Classifier | Classification | |
| `sklearn.logistic_regression` | Logistic Regression | Classification | |

### XGBoost

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `xgboost.xgbregressor` | XGBoost Regressor | Regression | [[2]](#ref2) |
| `xgboost.xgbclassifier` | XGBoost Classifier | Classification | [[2]](#ref2) |

### GPflow / GPyTorch

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `gpflow.gpr` | Gaussian Process (GPflow) | Regression | TF-based; RBF kernel |
| `gpflow.multi_kernel` | Multi-kernel GP (GPflow) | Regression | RBF + Matérn composite |
| `botorch.gp` | Exact GP (GPyTorch/BoTorch) | Regression | RBF+Matérn, PyTorch-native [[3]](#ref3) |
| `botorch.sparse_gp` | Sparse Variational GP (SVGP) | Regression | Inducing-point GP, scales to large n [[3]](#ref3) |

### PyTorch — Tabular / General

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `pytorch.mlp` | MLP Regressor | Regression | Adam, early stopping |
| `pytorch.residual_mlp` | Residual MLP | Regression | Skip connections |
| `pytorch.mlp_classifier` | MLP Classifier | Classification | Softmax head |
| `pytorch.ft_transformer` | Feature Tokenizer + Transformer | Regression | [[4]](#ref4) |
| `pytorch.ft_transformer_classifier` | FT-Transformer Classifier | Classification | [[4]](#ref4) |
| `pytorch.kan` | Kolmogorov-Arnold Network | Regression | B-spline activations on edges [[5]](#ref5) |
| `pytorch.kan_classifier` | KAN Classifier | Classification | [[5]](#ref5) |
| `pytorch.vae` | Variational Autoencoder | Regression | Latent regression head [[6]](#ref6) |

### PyTorch — Temporal / Sequence

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `pytorch.lstm` | LSTM Encoder-Decoder | Regression | Temporal sequences |
| `pytorch.gru` | GRU Encoder-Decoder | Regression | Temporal sequences |
| `pytorch.cnn1d` | Dilated 1D CNN | Regression | Temporal / sequence data |

### PyTorch — Field Operator Learning

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `pytorch.fno1d` | Fourier Neural Operator (1D) | Regression | PDE operator learning [[7]](#ref7) |
| `pytorch.fno2d` | Fourier Neural Operator (2D) | Regression | 2D PDE fields [[7]](#ref7) |
| `pytorch.deeponet` | Deep Operator Network | Regression | Branch-trunk architecture [[8]](#ref8) |
| `pytorch.unet` | U-Net | Regression | Encoder-decoder skip connections [[9]](#ref9) |
| `pytorch.ddpm` | Conditional DDPM (1D) | Regression | Score-based diffusion [[10]](#ref10) |
| `pytorch.cgan` | Conditional GAN (1D) | Regression | [[11]](#ref11) |

### PyTorch — Image

| Key | Algorithm | Task | Notes |
|-----|-----------|------|-------|
| `pytorch.lenet5` | LeNet-5 | Classification | Classic CNN [[12]](#ref12) |
| `pytorch.alexnet` | AlexNet (small-image) | Classification | 32×32 / 28×28 adapted [[13]](#ref13) |
| `pytorch.resnet20` | ResNet-20 | Classification | CIFAR variant [[14]](#ref14) |
| `pytorch.resnet56` | ResNet-56 | Classification | CIFAR variant [[14]](#ref14) |
| `pytorch.vit` | Vision Transformer (ViT) | Classification | Patch-based [[15]](#ref15) |

---

## Benchmarks by Capability

### Scalar Regression

All benchmarks produce a scalar target per sample. Metrics: **R²** (higher is better), **RMSE** (lower is better).

> **Note on synthetic benchmarks (Tier 0):** `synthetic.*` and `tabular.diabetes` are included only as fast CI smoke tests. They should not be used as primary performance references — every serious comparison should use the Tier 1 benchmarks below that appear in the published ML literature.

#### `tabular.california_housing` · Tier 1 ★ Literature benchmark
- **Source:** `sklearn.datasets.fetch_california_housing()` (Pace & Barry 1997)
- **Task:** Median house value from census-tract features (n=20,640, d=8)
- **Pass threshold:** R² ≥ 0.70
- **Published reference (RMSE, 80/20 split):** XGBoost default 0.462, FT-Transformer default 0.454, ResNet 0.489 — Gorishniy et al. NeurIPS 2021 [[4]](#ref4)
- **SURGE results (seed=42):**

| Model | R² | RMSE | Runtime |
|---|---|---|---|
| XGBoost ★ | **0.847** | **0.448** | 0.77s |
| RF | 0.806 | 0.504 | 3.3s |
| GBR | 0.776 | 0.542 | 3.0s |
| pytorch.residual_mlp | 0.797 | 0.516 | 22s |
| pytorch.mlp | 0.795 | 0.518 | 11s |
| pytorch.vae | 0.766 | 0.554 | 9s |
| pytorch.ft_transformer | 0.767 | 0.553 | 459s |
| pytorch.kan | 0.807 | 0.503 | 51s |

> XGBoost RMSE=0.448 is consistent with the published default-HP value of 0.462 in Gorishniy et al., confirming our evaluation setup matches the literature.

---

#### `tabular.concrete_strength` · Tier 1 ★ Literature benchmark
- **Source:** OpenML ID 4353 (UCI Concrete Compressive Strength, Yeh 1998)
- **Task:** Predict concrete compressive strength in MPa (n=1,030, d=8)
- **Pass threshold:** R² ≥ 0.85
- **Published reference:** GBM/XGBoost R²≈0.93–0.95; RF R²≈0.88 — multiple materials-science papers [[18]](#ref18)
- **SURGE results (seed=42):**

| Model | R² | RMSE (MPa) | Runtime |
|---|---|---|---|
| XGBoost ★ | **0.927** | **4.32** | 0.6s |
| GBR | 0.883 | 5.49 | 0.1s |
| RF | 0.882 | 5.52 | 0.2s |
| pytorch.residual_mlp | 0.883 | 5.49 | 1.1s |
| pytorch.mlp | 0.876 | 5.65 | 1.5s |
| pytorch.kan | 0.875 | 5.68 | 3.3s |
| pytorch.vae | 0.830 | 6.61 | 0.6s |

> XGBoost R²=0.927 is consistent with literature reports of R²≈0.93–0.95 for optimised GBMs on this dataset.

---

#### `tabular.energy_efficiency` · Tier 1 ★ Literature benchmark
- **Source:** UCI Energy Efficiency (Tsanas & Xifara 2012), 768 samples, 8 features
- **Task:** Predict building heating load (kWh/m²)
- **Pass threshold:** R² ≥ 0.90
- **Published reference:** RF and GBM consistently achieve R²≥0.99 [[19]](#ref19)
- **SURGE results (seed=42):**

| Model | R² | RMSE | Runtime |
|---|---|---|---|
| XGBoost ★ | **0.996** | **0.619** | 0.6s |
| RF | 0.993 | 0.866 | 0.2s |
| GBR | 0.990 | 0.979 | 0.05s |
| pytorch.residual_mlp | 0.988 | 1.10 | 0.9s |
| pytorch.ft_transformer | 0.937 | 2.52 | 19s |
| pytorch.mlp | 0.944 | 2.37 | 0.4s |

---

#### `tabular.yacht_dynamics` · Tier 1 ★ Literature benchmark
- **Source:** UCI Yacht Hydrodynamics (Gerritsma et al. 1981), 308 samples, 6 features
- **Task:** Predict residuary resistance of sailing yachts (N/displacement)
- **Pass threshold:** R² ≥ 0.95
- **Published reference:** RF and GBM typically achieve R²>0.99 on this dataset [[20]](#ref20)
- **SURGE results (seed=42):**

| Model | R² | RMSE | Runtime |
|---|---|---|---|
| XGBoost ★ | **0.998** | **0.527** | 0.6s |
| RF | 0.998 | 0.545 | 0.1s |
| GBR | 0.998 | 0.610 | 0.03s |
| pytorch.residual_mlp | 0.979 | 1.77 | 1.3s |
| pytorch.mlp | 0.951 | 2.71 | 0.7s |
| pytorch.kan | 0.811 | 5.30 | 1.4s |

---

#### `tabular.airfoil_noise` · Tier 1 ★ Literature benchmark
- **Source:** UCI Airfoil Self-Noise (NASA wind tunnel, Brooks et al. 1989), 1,503 samples, 5 features
- **Task:** Predict airfoil sound pressure level (dB)
- **Pass threshold:** R² ≥ 0.80
- **Published reference:** GBM/XGBoost achieve R²≈0.95–0.97 on this dataset [[21]](#ref21)
- **SURGE results (seed=42):**

| Model | R² | RMSE (dB) | Runtime |
|---|---|---|---|
| XGBoost ★ | **0.957** | **1.47** | 0.6s |
| RF | 0.935 | 1.81 | 0.2s |
| pytorch.mlp | 0.890 | 2.34 | 0.9s |
| pytorch.kan | 0.895 | 2.29 | 4.5s |
| pytorch.residual_mlp | 0.933 | 1.83 | 2.3s |
| GBR | 0.837 | 2.86 | 0.07s |

---

#### `tabular.superconductor` · Tier 1 ★ Literature benchmark
- **Source:** UCI Superconductivity dataset (Hamidieh 2018), 21,263 samples, 81 features
- **Task:** Predict critical temperature Tc (K) of superconductors
- **Pass threshold:** R² ≥ 0.85
- **Published reference:** XGBoost/RF typically achieve R²≈0.90–0.93 on this benchmark [[22]](#ref22)
- **Citation:** Hamidieh (2018) "A data-driven statistical model for superconductivity" *Computational Materials Science* [[22]](#ref22)

---

#### `tabular.diabetes` · Tier 0 — *smoke test only*
- **Source:** `sklearn.datasets.load_diabetes()` (Efron et al. 2004)
- **Task:** Predict diabetes disease progression (n=442, d=10)
- **Pass threshold:** R² ≥ 0.45
- **Note:** This is a notoriously hard prediction task. Even the best published models only reach R²≈0.55. XGBoost ≈ 0.47, GP ≈ 0.53. **Do not interpret these numbers as a model quality signal** — use Tier-1 benchmarks for that.
- **Published reference:** Efron et al. (2004) "Least Angle Regression" *Annals of Statistics* [[16]](#ref16); linearmodel upper bound R²≈0.52.

---

#### `synthetic.regression_1d` · Tier 0 — *CI smoke test only*
- **Source:** Inline fixture `y = 3x + 1.5 + noise` (not a real dataset)
- **Use:** Verify model can fit/predict/save/load. **Not a performance benchmark.**

#### `synthetic.multioutput_2d` · Tier 0 — *CI smoke test only*
- **Source:** Inline linear fixture (not a real dataset)
- **Use:** Verify multi-output models load correctly. **Not a performance benchmark.**

---

### Multi-Output Regression

Metrics: **R²** per output, **RMSE** per output.

#### `multioutput.scm20d` · Tier 1
- **Source:** OpenML — Supply-Chain Management (SCM) 20-day dataset
- **Task:** Predict 16 supply-chain demand outputs from 24 features (n=9803, d=24, t=16)
- **Pass threshold:** R² ≥ 0.50
- **Models with multi-output support:** All sklearn wrappers (`MultiOutputRegressor`), `pytorch.residual_mlp`, `pytorch.ft_transformer`, `pytorch.kan`, `pytorch.vae`

---

### Tabular Classification

Metrics: **Accuracy** (higher is better), **F1-Macro**, **ROC-AUC** (binary).

#### `tabular.iris` · Tier 0
- **Source:** `sklearn.datasets.load_iris()` — built-in
- **Task:** 3-class flower classification (n=150, d=4)
- **Pass threshold:** Accuracy ≥ 0.90
- **Citation:** Fisher (1936) "The use of multiple measurements in taxonomic problems" [[23]](#ref23)

#### `tabular.breast_cancer` · Tier 0
- **Source:** `sklearn.datasets.load_breast_cancer()` — built-in
- **Task:** Binary cancer classification from tumor features (n=569, d=30)
- **Pass threshold:** Accuracy ≥ 0.92

#### `tabular.digits` · Tier 0
- **Source:** `sklearn.datasets.load_digits()` — built-in
- **Task:** 10-class digit classification from 8×8 grayscale images (n=1797, d=64)
- **Pass threshold:** Accuracy ≥ 0.92

#### `tabular.wine` · Tier 0
- **Source:** `sklearn.datasets.load_wine()` — built-in
- **Task:** 3-class wine classification from chemical analysis (n=178, d=13)
- **Pass threshold:** Accuracy ≥ 0.90

#### `classification.covertype` · Tier 2
- **Source:** UCI Covertype dataset (7-class forest cover type prediction)
- **Task:** Predict forest cover type from cartographic features (n=100k+, d=54)
- **Pass threshold:** Accuracy ≥ 0.75
- **Citation:** Blackard & Dean (1999) "Comparative accuracies of ANN and DT" [[24]](#ref24)

---

### Scientific Classification

#### `classification.plasma_stability` · Tier 2
- **Source:** UCI Electrical Grid Stability Simulated dataset
- **Task:** Binary classification of power grid stability (n=10000, d=12)
- **Pass threshold:** Accuracy ≥ 0.95
- **Citation:** Schäfer et al. (2016) "Taming instabilities in power grid networks" [[25]](#ref25)

#### `classification.flow_regime` · Tier 2
- **Source:** Inline synthetic dataset (pressure-velocity flow features)
- **Task:** Binary classification of turbulent vs. laminar flow (n=5000, d=8)
- **Pass threshold:** Accuracy ≥ 0.80

#### `synthetic.classification_binary` · Tier 0
- **Source:** Inline — `sklearn.datasets.make_classification`
- **Task:** Binary classification sanity check (n=1000, d=20)
- **Pass threshold:** Accuracy ≥ 0.85

---

### Time Series / Forecasting

Metrics: **NRMSE** (lower is better), **R²** on flattened trajectories.

#### `sequence.lorenz63` · Tier 1
- **Source:** Inline numerical integration (Runge-Kutta RK4)
- **Task:** Predict 10-step trajectory of the Lorenz-63 chaotic system from a 20-step window (n=5000, d=60)
- **Pass threshold:** NRMSE ≤ 0.30
- **Relevant models:** LSTM, GRU, CNN1D, ResidualMLP
- **Citation:** Lorenz (1963) "Deterministic Nonperiodic Flow" *J. Atmos. Sci.* [[26]](#ref26)

---

### 1D Field Operator Learning

Metrics: **Relative L² error** (lower is better), **NRMSE**.

Pass thresholds are lower (stricter) for neural operators vs. tabular models.

#### `pde.burgers_1d` · Tier 1 (inline, no download)
- **Source:** Inline finite-difference solver for Burgers' equation on 64-pt grid
- **Task:** Map initial condition u₀(x) → solution u(x, T) at T=1.0
  (n=2000, nx=64)
- **Pass threshold:** Relative L² ≤ 0.15
- **Relevant models:** FNO1d, DeepONet, DDPM, CGAN, CNN1D, ResidualMLP
- **Citation:** FNO benchmark of Li et al. (2021) [[7]](#ref7)

#### `pdebench.burgers_1d` · Tier 3 (requires download ~500 MB)
- **Source:** PDEBench HDF5 — DaRUS repository
- **Task:** 1D viscous Burgers' equation from PDEBench standard test suite
- **Pass threshold:** Relative L² ≤ 0.10
- **Relevant models:** FNO1d, DeepONet, DDPM, CGAN, CNN1D
- **Download:** `python -m surge.benchmarks.run --benchmark pdebench.burgers_1d --download`
- **Citation:** Takamoto et al. (2022) "PDEBench" NeurIPS 2022 [[27]](#ref27)

---

### 2D Field Operator Learning

Metrics: **Relative L² error**, **NRMSE** over 2D spatial fields.

> These benchmarks require PDEBench data downloads (~1–5 GB each).
> Only FNO2d and U-Net are evaluated; tabular/sklearn models are marked **N/A**.

#### `pdebench.darcy_2d` · Tier 3
- **Source:** PDEBench — 2D Darcy flow on uniform grid
- **Task:** Map permeability field a(x,y) → pressure solution p(x,y) (nx=128, ny=128)
- **Pass threshold:** Relative L² ≤ 0.05
- **Relevant models:** `pytorch.fno2d`, `pytorch.unet`
- **Citation:** Li et al. (2021) FNO [[7]](#ref7); Takamoto et al. (2022) PDEBench [[27]](#ref27)

#### `pdebench.shallow_water_2d` · Tier 3
- **Source:** PDEBench — 2D shallow water equations (wave propagation)
- **Task:** Predict future state of height + velocity fields (T=1 step, nx=128)
- **Pass threshold:** Relative L² ≤ 0.10
- **Relevant models:** `pytorch.fno2d`, `pytorch.unet`
- **Citation:** Takamoto et al. (2022) PDEBench [[27]](#ref27)

---

### Image Classification

Metrics: **Top-1 Accuracy**, **F1-Macro**.

#### `vision.mnist` · Tier 2
- **Source:** `torchvision.datasets.MNIST` (auto-download)
- **Task:** 10-class handwritten digit recognition (n=60k/10k, 28×28 grayscale)
- **Pass threshold:** Accuracy ≥ 0.95 (Tier-2 target)
- **Relevant models:** LeNet-5, ResNet-20, ViT, AlexNet (with `in_channels=1`)
- **Citation:** LeCun et al. (1998) "Gradient-based learning applied to document recognition" [[12]](#ref12)

#### `vision.cifar10` · Tier 2
- **Source:** `torchvision.datasets.CIFAR10` (auto-download)
- **Task:** 10-class natural image classification (n=50k/10k, 32×32 RGB)
- **Pass threshold:** Accuracy ≥ 0.80
- **Relevant models:** LeNet-5, ResNet-20, ResNet-56, ViT, AlexNet
- **SOTA reference:** ~97% with wide ResNets / EfficientNet; SURGE baselines aim for 80–93%
- **Citation:** Krizhevsky (2009) "Learning Multiple Layers of Features from Tiny Images" [[28]](#ref28)

---

### Scientific Domain

#### `fusion.m3dc1_sample` · Tier 2
- **Source:** Inline synthetic MHD equilibrium dataset (M3D-C1 analogue)
- **Task:** Predict plasma stability metric from MHD state features (n=1000, d=50)
- **Relevant models:** BoTorch GP, VAE, ResidualMLP, FT-Transformer
- **Notes:** Intended as a stand-in for real M3D-C1 HDF5 data when not available.

#### `thewell.gray_scott` · Tier 4
- **Source:** The Well dataset — Gray-Scott reaction-diffusion system
- **Task:** Temporal field evolution prediction (requires `pip install the-well`)
- **Citation:** Price et al. (2025) "Poseidon / The Well" [[29]](#ref29)

#### `thewell.turbulence_2d` · Tier 4
- **Source:** The Well dataset — 2D incompressible turbulence
- **Task:** Predict next turbulent flow state from current state
- **Relevant models:** FNO2d, U-Net
- **Citation:** Price et al. (2025) [[29]](#ref29)

#### `thewell.mhd` · Tier 4
- **Source:** The Well dataset — MHD turbulence simulation
- **Task:** MHD field evolution prediction
- **Relevant models:** FNO2d, U-Net
- **Citation:** Price et al. (2025) [[29]](#ref29)

---

## Full Model × Benchmark Grid

The table below shows which models are **evaluated** (✓) or **skipped** (N/A) for each benchmark. Results depend on available compute and data downloads.

| Model | Scalar Reg. | Multi-Out. | Classification | Sequence | 1D PDE | 2D PDE | Vision |
|-------|:-----------:|:----------:|:--------------:|:--------:|:------:|:------:|:------:|
| `sklearn.random_forest` | ✓ | ✓ | ✓ | ✓ | N/A | N/A | N/A |
| `sklearn.gradient_boosting_regressor` | ✓ | ✓ | — | ✓ | N/A | N/A | N/A |
| `sklearn.mlp` | ✓ | ✓ | — | ✓ | N/A | N/A | N/A |
| `sklearn.gpr` | ✓ | — | — | — | N/A | N/A | N/A |
| `sklearn.gradient_boosting_classifier` | — | — | ✓ | — | N/A | N/A | N/A |
| `sklearn.logistic_regression` | — | — | ✓ | — | N/A | N/A | N/A |
| `xgboost.xgbregressor` | ✓ | ✓ | — | ✓ | N/A | N/A | N/A |
| `xgboost.xgbclassifier` | — | — | ✓ | — | N/A | N/A | N/A |
| `botorch.gp` | ✓ | — | — | — | N/A | N/A | N/A |
| `botorch.sparse_gp` | ✓ | — | — | — | N/A | N/A | N/A |
| `pytorch.mlp` | ✓ | ✓ | — | ✓ | ✓ | N/A | N/A |
| `pytorch.residual_mlp` | ✓ | ✓ | — | ✓ | ✓ | N/A | N/A |
| `pytorch.mlp_classifier` | — | — | ✓ | — | N/A | N/A | N/A |
| `pytorch.ft_transformer` | ✓ | ✓ | — | ✓ | N/A | N/A | N/A |
| `pytorch.ft_transformer_classifier` | — | — | ✓ | — | N/A | N/A | N/A |
| `pytorch.kan` | ✓ | ✓ | — | ✓ | ✓ | N/A | N/A |
| `pytorch.kan_classifier` | — | — | ✓ | — | N/A | N/A | N/A |
| `pytorch.vae` | ✓ | — | — | — | N/A | N/A | N/A |
| `pytorch.cnn1d` | ✓ | ✓ | — | ✓ | ✓ | N/A | N/A |
| `pytorch.lstm` | — | — | — | ✓ | — | N/A | N/A |
| `pytorch.gru` | — | — | — | ✓ | — | N/A | N/A |
| `pytorch.fno1d` | N/A | N/A | N/A | N/A | ✓ | N/A | N/A |
| `pytorch.deeponet` | N/A | N/A | N/A | N/A | ✓ | N/A | N/A |
| `pytorch.fno2d` | N/A | N/A | N/A | N/A | N/A | ✓ | N/A |
| `pytorch.unet` | N/A | N/A | N/A | N/A | N/A | ✓ | N/A |
| `pytorch.ddpm` | N/A | N/A | N/A | N/A | ✓ | N/A | N/A |
| `pytorch.cgan` | N/A | N/A | N/A | N/A | ✓ | N/A | N/A |
| `pytorch.lenet5` | N/A | N/A | N/A | N/A | N/A | N/A | ✓ |
| `pytorch.alexnet` | N/A | N/A | N/A | N/A | N/A | N/A | ✓ |
| `pytorch.resnet20` | N/A | N/A | N/A | N/A | N/A | N/A | ✓ |
| `pytorch.resnet56` | N/A | N/A | N/A | N/A | N/A | N/A | ✓ |
| `pytorch.vit` | N/A | N/A | N/A | N/A | N/A | N/A | ✓ |

### Expected Performance Summary

> The following typical performance figures are from literature or preliminary SURGE runs.
> Actual results may vary with hyperparameters; use `--hpo` for tuned results.

#### Scalar Regression — RMSE and R² (lower RMSE, higher R² is better)

| Benchmark | sklearn RF | XGBoost | BoTorch GP | FT-Transformer | KAN |
|-----------|:----------:|:-------:|:----------:|:--------------:|:---:|
| `tabular.diabetes` | ~0.47 R² | ~0.50 R² | ~0.48 R² | ~0.45 R² | ~0.46 R² |
| `tabular.california_housing` | ~0.81 R² | ~0.85 R² | ~0.72 R² | ~0.80 R² | ~0.78 R² |
| `tabular.concrete_strength` | ~0.90 R² | ~0.93 R² | ~0.85 R² | ~0.91 R² | ~0.89 R² |
| `tabular.energy_efficiency` | ~0.98 R² | ~0.99 R² | ~0.92 R² | ~0.98 R² | ~0.97 R² |
| `tabular.yacht_dynamics` | ~0.99 R² | ~0.99 R² | ~0.97 R² | ~0.99 R² | ~0.98 R² |
| `tabular.airfoil_noise` | ~0.90 R² | ~0.94 R² | ~0.82 R² | ~0.91 R² | ~0.88 R² |
| `tabular.superconductor` | ~0.91 R² | ~0.94 R² | ~0.72 R²* | ~0.89 R² | ~0.87 R² |

_*GP limited by n; use `botorch.sparse_gp` for large n._

#### Tabular Classification — Accuracy

| Benchmark | sklearn RF | XGBoost | FT-Transformer | KAN |
|-----------|:----------:|:-------:|:--------------:|:---:|
| `tabular.iris` | ~0.97 | ~0.97 | ~0.96 | ~0.96 |
| `tabular.breast_cancer` | ~0.96 | ~0.97 | ~0.97 | ~0.95 |
| `tabular.wine` | ~0.98 | ~0.98 | ~0.97 | ~0.97 |
| `tabular.digits` | ~0.97 | ~0.96 | ~0.95 | ~0.94 |
| `classification.covertype` | ~0.94 | ~0.96 | ~0.90 | ~0.88 |
| `classification.plasma_stability` | ~0.98 | ~0.99 | ~0.98 | ~0.97 |

#### Vision — Top-1 Accuracy

| Benchmark | LeNet-5 | ResNet-20 | ResNet-56 | ViT | AlexNet |
|-----------|:-------:|:---------:|:---------:|:---:|:-------:|
| `vision.mnist` | ~0.99 | ~0.99 | ~0.99 | ~0.98 | ~0.99 |
| `vision.cifar10` | ~0.68 | ~0.90 | ~0.93 | ~0.82 | ~0.81 |

#### 1D PDE — Relative L² Error (lower is better)

| Benchmark | FNO1d | DeepONet | CNN1D | DDPM | CGAN |
|-----------|:-----:|:--------:|:-----:|:----:|:----:|
| `pde.burgers_1d` | ~0.03 | ~0.05 | ~0.08 | ~0.10 | ~0.12 |
| `pdebench.burgers_1d` | ~0.02 | ~0.04 | ~0.07 | ~0.09 | ~0.11 |

---

## CLI Reproduction Commands

```bash
# Activate the SURGE environment first
conda activate surge
export PYTHONPATH=.

# ─────────────────────────────────────────────────
# Listing
# ─────────────────────────────────────────────────

# List all benchmarks (with tier, shape, description)
python -m surge.benchmarks.run --list

# List all registered models
python -m surge.benchmarks.run --list-models

# ─────────────────────────────────────────────────
# Running individual benchmarks
# ─────────────────────────────────────────────────

# Scalar regression with default model
python -m surge.benchmarks.run --benchmark tabular.california_housing

# Scalar regression with specific model
python -m surge.benchmarks.run --benchmark tabular.superconductor --model xgboost.xgbregressor

# 1D PDE benchmark with FNO1d
python -m surge.benchmarks.run --benchmark pde.burgers_1d --model pytorch.fno1d

# 1D PDE benchmark with DDPM
python -m surge.benchmarks.run --benchmark pde.burgers_1d --model pytorch.ddpm

# 1D PDE benchmark with CGAN
python -m surge.benchmarks.run --benchmark pde.burgers_1d --model pytorch.cgan

# Tabular classification with FT-Transformer
python -m surge.benchmarks.run --benchmark tabular.iris --model pytorch.ft_transformer_classifier

# Tabular regression with KAN
python -m surge.benchmarks.run --benchmark tabular.concrete_strength --model pytorch.kan

# Tabular regression with BoTorch GP
python -m surge.benchmarks.run --benchmark tabular.diabetes --model botorch.gp

# Vision benchmark
python -m surge.benchmarks.run --benchmark vision.mnist --model pytorch.vit
python -m surge.benchmarks.run --benchmark vision.cifar10 --model pytorch.resnet56

# ─────────────────────────────────────────────────
# Tier-based batch runs
# ─────────────────────────────────────────────────

# Run all Tier-0 benchmarks (fast, no download)
python -m surge.benchmarks.run --all --tier 0

# Run all Tier-0 and Tier-1 benchmarks
python -m surge.benchmarks.run --all --tier 1

# Run all Tier-2 benchmarks (requires some downloads)
python -m surge.benchmarks.run --all --tier 2

# ─────────────────────────────────────────────────
# Leaderboards
# ─────────────────────────────────────────────────

# Run a per-benchmark leaderboard (all compatible models vs one benchmark)
python -m surge.benchmarks.run --leaderboard --benchmark tabular.california_housing

# Run full leaderboard across all benchmarks
python -m surge.benchmarks.run --leaderboard --all-benchmarks

# Run leaderboard with plot output
python -m surge.benchmarks.run --leaderboard --all-benchmarks --plot

# Run leaderboard and log to MLflow
python -m surge.benchmarks.run --leaderboard --all-benchmarks --mlflow

# Leaderboard for regression-only benchmarks
python -m surge.benchmarks.run --leaderboard --all-benchmarks --task-type regression

# Leaderboard for classification-only benchmarks
python -m surge.benchmarks.run --leaderboard --all-benchmarks --task-type classification

# ─────────────────────────────────────────────────
# Hyperparameter Optimization (HPO)
# ─────────────────────────────────────────────────

# HPO with default model
python -m surge.benchmarks.run --benchmark tabular.california_housing --hpo --hpo-trials 30

# HPO for a specific model
python -m surge.benchmarks.run --benchmark tabular.superconductor --model pytorch.ft_transformer --hpo --hpo-trials 20

# HPO for KAN on PDE benchmark
python -m surge.benchmarks.run --benchmark pde.burgers_1d --model pytorch.kan --hpo --hpo-trials 15

# ─────────────────────────────────────────────────
# Result saving
# ─────────────────────────────────────────────────

# Save results to a custom directory
python -m surge.benchmarks.run --benchmark tabular.california_housing --save-dir results/tabular

# Skip auto-save
python -m surge.benchmarks.run --benchmark tabular.iris --no-save

# ─────────────────────────────────────────────────
# PDEBench (Tier 3 — requires data download)
# ─────────────────────────────────────────────────

# Download Burgers 1D (first run, ~500 MB)
python -m surge.benchmarks.run --benchmark pdebench.burgers_1d --model pytorch.fno1d

# Download Darcy 2D then run FNO2d
python -m surge.benchmarks.run --benchmark pdebench.darcy_2d --model pytorch.fno2d

# Download Shallow Water 2D then run U-Net
python -m surge.benchmarks.run --benchmark pdebench.shallow_water_2d --model pytorch.unet
```

---

## References

<a id="ref1"></a>**[1]** Pedregosa et al. (2011) "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

<a id="ref2"></a>**[2]** Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System." *Proc. KDD 2016*, pp. 785–794. https://arxiv.org/abs/1603.02754

<a id="ref3"></a>**[3]** Balandat et al. (2020) "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization." *NeurIPS 2020*. https://arxiv.org/abs/1910.06403

<a id="ref4"></a>**[4]** Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data." *NeurIPS 2021*. https://arxiv.org/abs/2106.11959

<a id="ref5"></a>**[5]** Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks." *arXiv:2404.19756*. https://arxiv.org/abs/2404.19756

<a id="ref6"></a>**[6]** Kingma & Welling (2014) "Auto-Encoding Variational Bayes." *ICLR 2014*. https://arxiv.org/abs/1312.6114

<a id="ref7"></a>**[7]** Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*. https://arxiv.org/abs/2010.08895

<a id="ref8"></a>**[8]** Lu et al. (2021) "Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators." *Nature Machine Intelligence* 3, 218–229. https://doi.org/10.1038/s42256-021-00302-5

<a id="ref9"></a>**[9]** Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015*. https://arxiv.org/abs/1505.04597

<a id="ref10"></a>**[10]** Ho et al. (2020) "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. https://arxiv.org/abs/2006.11239

<a id="ref11"></a>**[11]** Mirza & Osindero (2014) "Conditional Generative Adversarial Nets." *arXiv:1411.1784*. https://arxiv.org/abs/1411.1784

<a id="ref12"></a>**[12]** LeCun et al. (1998) "Gradient-based learning applied to document recognition." *Proceedings of the IEEE* 86(11), 2278–2324.

<a id="ref13"></a>**[13]** Krizhevsky et al. (2012) "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS 2012*. https://papers.nips.cc/paper/4824-imagenet-classification

<a id="ref14"></a>**[14]** He et al. (2016) "Deep Residual Learning for Image Recognition." *CVPR 2016*. https://arxiv.org/abs/1512.03385

<a id="ref15"></a>**[15]** Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. https://arxiv.org/abs/2010.11929

<a id="ref16"></a>**[16]** Efron et al. (2004) "Least Angle Regression." *Annals of Statistics* 32(2), 407–499.

<a id="ref17"></a>**[17]** Pace & Barry (1997) "Sparse Spatial Autoregressions." *Statistics and Probability Letters* 33(3), 291–297.

<a id="ref18"></a>**[18]** Yeh (1998) "Modeling of strength of high-performance concrete using artificial neural networks." *Cement and Concrete Research* 28(12), 1797–1808.

<a id="ref19"></a>**[19]** Tsanas & Xifara (2012) "Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools." *Energy and Buildings* 49, 560–567.

<a id="ref20"></a>**[20]** Gerritsma et al. UCI Machine Learning Repository — Yacht Hydrodynamics. https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics

<a id="ref21"></a>**[21]** Brooks et al. (1989) "Airfoil Self-Noise and Prediction." NASA Reference Publication 1218.

<a id="ref22"></a>**[22]** Hamidieh (2018) "A data-driven statistical model for predicting the critical temperature of a superconductor." *Computational Materials Science* 154, 346–354.

<a id="ref23"></a>**[23]** Fisher (1936) "The use of multiple measurements in taxonomic problems." *Annals of Eugenics* 7(2), 179–188.

<a id="ref24"></a>**[24]** Blackard & Dean (1999) "Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables." *Computers and Electronics in Agriculture* 24(3), 131–151.

<a id="ref25"></a>**[25]** Schäfer et al. (2016) "Taming instabilities in power grid networks by decentralized control." *European Physical Journal Special Topics* 225, 569–582.

<a id="ref26"></a>**[26]** Lorenz (1963) "Deterministic Nonperiodic Flow." *Journal of the Atmospheric Sciences* 20(2), 130–141.

<a id="ref27"></a>**[27]** Takamoto et al. (2022) "PDEBench: An Extensive Benchmark for Scientific Machine Learning." *NeurIPS 2022*. https://arxiv.org/abs/2210.07182

<a id="ref28"></a>**[28]** Krizhevsky (2009) "Learning Multiple Layers of Features from Tiny Images." Technical Report, University of Toronto.

<a id="ref29"></a>**[29]** McCabe et al. (2023) "Multiple Physics Pretraining for Physical Surrogate Models." https://arxiv.org/abs/2310.02994; Price et al. (2025) "Poseidon: Efficient Foundation Models for PDEs." https://arxiv.org/abs/2405.19101

---

*Report generated for SURGE benchmark suite. For the latest results, run:*
```bash
python -m surge.benchmarks.run --leaderboard --all-benchmarks --plot
```
