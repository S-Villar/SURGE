# XGC A_parallel Surrogate – Data Format and Variable Mapping

## Overview

This document describes the OLCF AI Hackathon 2025 XGC dataset used for A_parallel (parallel vector potential) surrogate modeling, based on prior work in `Ah_prediction_NN1_ASV_Geometrical.ipynb`.

## Data Layout

| File | Shape | Description |
|------|-------|-------------|
| `data_nprev5_set1_data.npy` | (28994400, 201) | 201 input features |
| `data_nprev5_set1_target.npy` | (28994400, 2) | 2 output targets |
| `data_nprev5_set1_tags.npy` | (28994400, 3) | Tags (e.g., mesh indices, timestep) |
| `data_nprev5_set1_var_all.npy` | (14,) | Variable names |
| `data_nprev5_set1_add_n0.npy` | — | Additional n0 data |

## Variable Names (var_all)

14 variables in order:

1. `aparh` – parallel vector potential (hamiltonian)
2. `apars` – parallel vector potential (symplectic)
3. `dBphi` – magnetic perturbation (phi)
4. `dBpsi` – magnetic perturbation (psi)
5. `dBtheta` – magnetic perturbation (theta)
6. `ejpar` – electron parallel current
7. `ijpar` – ion parallel current
8. `dpot` – potential
9. `epara` – parallel electric field
10. `epara2` – parallel electric field (2)
11. `epsi` – electric field (psi)
12. `etheta` – electric field (theta)
13. `eden` – electron density
14. `iden` – ion density

## 201 Input → 14 Variables Mapping

**Decomposition:** 201 = 14 × 14 + 5

- **14 variables × 14 spatial points = 196 columns**  
  Layout is row-major by spatial point: for each of 14 points, all 14 variables are given in `var_all` order.

  - Columns 0–13: point 0 (aparh, apars, dBphi, …, iden)
  - Columns 14–27: point 1
  - …
  - Columns 182–195: point 13

- **5 extra columns (196–200):** Likely related to `nprev=5` (5 previous timesteps). Exact semantics not documented in the notebook.

**Column naming convention:** `{var}_{point}` for 0–195, e.g. `aparh_0`, `apars_0`, …, `iden_13`; `extra_0` … `extra_4` for 196–200.

## Output (Target)

- **Column 0:** First target (exact meaning not specified in notebook).
- **Column 1:** A_parallel (primary prediction target in prior work).

The notebook uses `target[:, 1]` as the regression target.

## Model Architecture (Prior Work)

From `Ah_prediction_NN1_ASV_Geometrical.ipynb`:

- **GeometricalMeanNN:** Hidden layer sizes use geometric mean of input and output dims.
- **Input:** 201
- **Output:** 1 (A_parallel) or 2
- **Training:** StandardScaler on X and y, train/test split 0.2, MSE loss, Adam optimizer.
- **Checkpoint:** `model_epoch_500.pth`, ONNX export `model.onnx`.

## Data Loading (Prior Work)

```python
path_data = '/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025/data_nprev5_set1_'
data = np.load(path_data + 'data.npy')      # (N, 201)
target = np.load(path_data + 'target.npy')  # (N, 2)
var_all = np.load(path_data + 'var_all.npy')  # 14 names
```
