# XGC A_parallel Surrogate – Full Capabilities Summary

Comprehensive recap of XGC surrogate capabilities developed in SURGE, including training, HPO, SHAP, drift detection, fine-tuning, inference, and integration into XGC source code.

---

## 0. Input Variable Structure (201 columns)

The OLCF hackathon data uses **201 input features** (`input_0` … `input_200`) and **2 outputs** (`output_0`, `output_1`; `output_1` = A_parallel). Layout from `Ah_data_preparation6_low_beta.ipynb`:

| Block | Columns | Description |
|-------|---------|-------------|
| **Current step (61)** | 0–60 | Single timestep features |
| **Previous 5 timesteps (140)** | 61–200 | 5 × 28 = 140 columns |

### Current step (61 columns)

| Sub-block | Count | Description |
|-----------|-------|-------------|
| single | 14 | Raw values at 14 flux-surface points |
| log10(norm) | 14 | log10 of normalized values |
| single_n0 | 14 | n=0 mode values |
| log10(norm_n0) | 14 | log10 of n=0 normalized values |
| const | 5 | Extra constant/global terms |

### Previous 5 timesteps (140 columns)

For each of the 5 previous timesteps: **28 columns** (14 full/norm1 + 14 n0/norm1_n0). No log10 norms for previous steps; only normalized values.

**Total:** 61 + 140 = **201**

---

## 1. Training & Main Regression Results

Train RF and MLP surrogates on OLCF hackathon data (201 inputs → 2 outputs; output_1 = A_parallel).

```bash
surge run configs/xgc_aparallel_set1.yaml --run-tag xgc_aparallel_set1_v3
```

**Data:** `/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025/` (set1, set2_beta0p5)

### Results (set1, 50k samples)

| Model | Train R² | Val R² | Test R² | Test RMSE |
|-------|----------|--------|---------|-----------|
| xgc_mlp_aparallel | 0.987 | 0.980 | 0.980 | 0.068 |
| xgc_rf_aparallel  | 0.987 | 0.960 | 0.958 | 0.100 |

---

## 2. Hyperparameter Optimization (Bayesian)

Optuna-based HPO with TPE (Tree-structured Parzen Estimator) sampler:

- **MLP:** 50 trials, search over hidden_layers, dropout, learning_rate, batch_size
- **RF:** 5 trials, search over n_estimators, max_depth, min_samples_leaf

```bash
surge run configs/xgc_aparallel_set1.yaml
```

![HPO convergence](../../runs/xgc_aparallel_set1_v3/plots/hpo_convergence.png)

---

## 3. Inference Comparison Plots

GT vs prediction density for train/val/test splits.

```bash
surge viz runs/xgc_aparallel_set1_v3
```

![Inference comparison grid](../../runs/xgc_aparallel_set1_v3/plots/inference_comparison_grid.png)

---

## 4. SHAP Feature Importance

Explain output_1 (A_parallel) with SHAP. Grouped bar: 14 stages + 5 extra columns.

```bash
surge viz runs/xgc_aparallel_set1_v3 --shap
```

![SHAP grouped](../../runs/xgc_aparallel_set1_v3/plots/shap_xgc_rf_aparallel/shap_summary_tree_grouped.png)

---

## 5. Evaluation on New Datasets (Datastreamsets)

Evaluate held-out contiguous regions; detect OOD and accuracy drop.

```bash
surge viz runs/xgc_aparallel_set1_v3 --datastreamset-eval
```

![Datastreamset R²/RMSE](../../runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_r2_rmse.png)

![Datastreamset degradation](../../runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_degradation.png)

**Cross-set evaluation** (set1 model on set2):

```bash
surge viz runs/xgc_aparallel_set1_v3 --datastreamset-eval --datastreamset-eval-set set2_beta0p5
```

Or with the generalization script:

```bash
python scripts/xgc_datastreamset_generalization.py \
    --run-dir runs/xgc_aparallel_set1_v3 --datastreamset-eval --eval-set set2_beta0p5
```

---

## 6. Drift Detection & Fine-Tuning Algorithm

### Drift Detection Policy

Three triggers per datastreamset:

| Trigger | Condition |
|---------|-----------|
| **Input OOD** | Any input min/max outside training range |
| **Output OOD** | Any output min/max outside training range |
| **Accuracy drop** | R² drop > 0.1 or RMSE ratio > 1.5× |

When drift is detected, SURGE recommends:

> **DRIFT DETECTED**: Evaluation data is outside training regime or prediction accuracy has dropped. Recommend ingesting new data from these regions and retraining the surrogate.

### Fine-Tuning Algorithm

Restart training whenever drift is detected:

1. **Train on set1** (lower beta): `surge run configs/xgc_aparallel_set1.yaml`
2. **Fine-tune on set2** (beta 0.5%): `surge run configs/xgc_aparallel_set2_finetune.yaml`

The workflow loads pretrained scalers and model weights, then continues training with scaled LR (`finetune_lr_scale: 0.1`).

```yaml
# configs/xgc_aparallel_set2_finetune.yaml
pretrained_run_dir: runs/xgc_aparallel_set1_v3
finetune_lr_scale: 0.1
metadata_overrides:
  set_name: set2_beta0p5
```

**Load compat:** `load_model_compat(..., for_finetune=True)` returns full PyTorch adapter with `fit()` for fine-tuning.

---

## 7. Training & Fine-Tuning on Lower Beta (set2)

| Config | Purpose |
|--------|---------|
| `xgc_aparallel_set1.yaml` | Train from scratch on set1 (lower beta) |
| `xgc_aparallel_set2_beta0p5.yaml` | Train from scratch on set2 (beta 0.5%) |
| `xgc_aparallel_set2_finetune.yaml` | Fine-tune set1 model on set2 |

```bash
# From scratch on set2
surge run configs/xgc_aparallel_set2_beta0p5.yaml

# Fine-tune set1 → set2
surge run configs/xgc_aparallel_set2_finetune.yaml
```

---

## 8. Inference Script

Standalone inference with SURGE models on hackathon data:

```bash
# Same-set (set1)
python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025

# Cross-set (set2_beta0p5)
python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \
    --eval-set set2_beta0p5 --sample 10000

# Save predictions
python scripts/xgc_inference.py ... --out predictions_xgc_set1.csv
```

---

## 9. Model Cards & MLflow

**Model cards:** Each trained model produces `model_card_<name>.json` with:

- `n_params`, `file_size_bytes`, `backend`
- `training_config` (run_tag, dataset, set_name, splits)
- `created_at` (ISO timestamp)

**MLflow:** Optional tracking:

```bash
pip install surge[mlflow]
surge run configs/xgc_aparallel_set1.yaml --mlflow
surge viz runs/xgc_aparallel_set1_v3 --mlflow
```

---

## 10. Coupling Model into XGC Source Code

To integrate the surrogate into XGC source code via compilation, the model can be loaded in memory and exposed as a callable inference routine.

### Option A: ONNX Export + Runtime

1. **Export ONNX** (PyTorch model):

```python
import torch
from surge.io.load_compat import load_model_compat

adapter = load_model_compat(Path("runs/xgc_aparallel_set1_v3/models/xgc_mlp_aparallel.joblib"))
model = adapter._model  # or adapter.get_estimator()
dummy = torch.randn(1, 201)
torch.onnx.export(
    model, (dummy,), "xgc_aparallel.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

2. **Python inference** (model in memory):

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("xgc_aparallel.onnx", providers=["CPUExecutionProvider"])
# Load scalers once
input_scaler = joblib.load("runs/.../scalers/input_scaler.joblib")
output_scaler = joblib.load("runs/.../scalers/output_scaler.joblib")

def infer(X: np.ndarray) -> np.ndarray:
    X_scaled = input_scaler.transform(X)
    out = session.run(None, {"input": X_scaled.astype(np.float32)})[0]
    return output_scaler.inverse_transform(out)
```

3. **C/C++ inference** via ONNX Runtime C API:

- Link `onnxruntime` C library
- `OrtSession*` created once at init
- `OrtSession::Run()` for each inference call
- Pass inputs as contiguous float arrays; receive outputs

### Option B: C/Fortran-Callable Wrapper

**Pattern:** Load model once at startup; expose a thin inference function.

```c
// xgc_surrogate_infer.h
void xgc_surrogate_init(const char* model_path, const char* scaler_path);
void xgc_surrogate_infer(const float* inputs, int n_inputs, float* outputs, int n_outputs);
void xgc_surrogate_finalize(void);
```

**Implementation options:**

- **ONNX Runtime C API:** Load model, create session; `infer` calls `OrtSession::Run`
- **Manual weights:** Export weights to JSON/binary; implement forward pass in C (for simple MLPs)
- **libtorch (C++):** Load torchscript model; `torch::jit::load` + `forward`

### Option C: Scaler Integration

The model expects **scaled inputs** and returns **scaled outputs**. Store scaler parameters (mean, scale) in `train_data_ranges.json` or alongside the model. In XGC:

1. Apply input scaling: `x_scaled = (x - mean) / scale`
2. Call model inference
3. Apply output inverse: `y = y_scaled * scale + mean`

### Summary: Callable Inference in XGC

| Step | Action |
|------|--------|
| 1 | Export model to ONNX (or TorchScript) |
| 2 | Save scalers (mean, scale) to JSON/binary |
| 3 | At XGC init: load ONNX session, load scalers |
| 4 | At inference: scale inputs → `session.Run()` → inverse scale outputs |
| 5 | Link ONNX Runtime (or libtorch) into XGC build |

---

## Artifact Layout

```
runs/xgc_aparallel_set1_v3/
├── models/           # RF, MLP (load_model_compat)
├── scalers/          # input/output StandardScaler
├── train_data_ranges.json
├── model_card_*.json
├── predictions/
└── plots/
    ├── inference_comparison_grid.png
    ├── hpo_convergence.png
    ├── datastreamset_evaluation_r2_rmse.png
    ├── datastreamset_evaluation_degradation.png
    ├── xgc_predictions_animation_set1.gif
    ├── xgc_spatial_aparallel_animation_*.gif
    └── shap_xgc_rf_aparallel/
```

---

## XGC Predictions Animation

Animated GIF showing how model predictions track ground truth across contiguous datastreamsets. Each frame is one datastreamset with a hexbin scatter of GT vs pred (A_parallel) and R² in the title.

### Animation locations

| Run | Path | Description |
|-----|------|-------------|
| **Model1** | `runs/xgc_model1_61cols/plots/xgc_predictions_animation_set1.gif` | 61-col model on set1 datastreamsets |
| **Model12** | `runs/xgc_model12_finetune/plots/xgc_predictions_animation_set2_beta0p5.gif` | Finetuned model on set2 datastreamsets |
| **set1_v3** | `runs/xgc_aparallel_set1_v3/plots/xgc_predictions_animation_set1.gif` | 201-col model on set1 |

**Absolute paths (NERSC):**
- Model1: `${SURGE_HOME}/runs/xgc_model1_61cols/plots/`
- Model12: `${SURGE_HOME}/runs/xgc_model12_finetune/plots/`

### Script usage

```bash
# Model1 (61 cols) on set1
python scripts/xgc_predictions_animation.py --run-dir runs/xgc_model1_61cols \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1

# Model12 (61 cols) on set2
python scripts/xgc_predictions_animation.py --run-dir runs/xgc_model12_finetune \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set2_beta0p5

# set1_v3 (201 cols)
python scripts/xgc_predictions_animation.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run-dir` | required | SURGE run directory |
| `--data-dir` | required | OLCF hackathon data dir |
| `--eval-set` | set1 | set1 or set2_beta0p5 |
| `--datastreamset-size` | 5000 | Rows per frame |
| `--max-frames` | 10 | Number of datastreamsets (frames) |
| `--model` | (RF) | Model name (e.g. xgc_rf_aparallel, xgc_mlp_aparallel) |
| `--fps` | 2.0 | Frames per second for GIF |
| `--output-name` | xgc_predictions_animation_{eval_set} | Output filename stem |

### Supported models

- **Model1 / Model12:** 61 columns (uses `spec.n_inputs`)
- **ModelS:** 20 columns (uses `spec.input_columns`)
- **set1_v3:** 201 columns (full dataset)

Uses `surge/viz` datastreamset logic: spec-based input selection so scalers and models receive the correct feature count.

### Environment

Run with the SURGE conda env (e.g. `/global/cfs/projectdirs/m3716/software/asvillar/envs/surge`):

```bash
/global/cfs/projectdirs/m3716/software/asvillar/envs/surge/bin/python scripts/xgc_predictions_animation.py ...
```

---

## XGC Spatial A_parallel Animation

2D spatial animations comparing ground truth vs surrogate inference for A_parallel on the circular (R–Z) geometry. Each frame shows two panels: **left** = GT contour, **right** = model prediction. One GIF per model over multiple timesteps.

### Animation locations

| Model | Path | Description |
|-------|------|-------------|
| **Model1** | `runs/xgc_model1_61cols/plots/xgc_spatial_aparallel_animation_model1.gif` | 61-col model on set1 timesteps |
| **Model12** | `runs/xgc_model12_finetune/plots/xgc_spatial_aparallel_animation_model12.gif` | Finetuned model on set2 timesteps |
| **ModelS** | `runs/xgc_modelS_shap20/plots/xgc_spatial_aparallel_animation_modelS.gif` | 20-col SHAP model on set1 timesteps |
| **set1_v3** | `runs/xgc_aparallel_set1_v3/plots/xgc_spatial_aparallel_animation_set1_v3.gif` | 201-col model on set1 timesteps |

### Script usage

```bash
# Model1 (61 cols) on set1
python scripts/xgc_spatial_aparallel_animation.py --run-dir runs/xgc_model1_61cols \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1

# Model12 (61 cols) on set2
python scripts/xgc_spatial_aparallel_animation.py --run-dir runs/xgc_model12_finetune \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set2_beta0p5

# ModelS (20 cols) on set1
python scripts/xgc_spatial_aparallel_animation.py --run-dir runs/xgc_modelS_shap20 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1

# set1_v3 (201 cols)
python scripts/xgc_spatial_aparallel_animation.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1

# Batch: all four models
for r in runs/xgc_model1_61cols runs/xgc_model12_finetune runs/xgc_modelS_shap20 runs/xgc_aparallel_set1_v3; do
  python scripts/xgc_spatial_aparallel_animation.py --run-dir $r \
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025
done
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run-dir` | required | SURGE run directory |
| `--data-dir` | required | OLCF hackathon data dir |
| `--eval-set` | set1 | set1 or set2_beta0p5 |
| `--max-frames` | 10 | Number of timesteps (frames) |
| `--model` | (MLP) | Model name (e.g. xgc_mlp_aparallel) |
| `--output-name` | xgc_spatial_aparallel_animation_{run_tag} | Output filename stem |
| `--fps` | 2.0 | Frames per second for GIF |

### Data and geometry

- **Source:** `Ah_data_preparation6_visualization_test.ipynb` – extracts timestep slices and plots on unstructured R–Z grid.
- **Files:** `data_nprev5_{set}_data.npy`, `_target.npy`, `_tags.npy` from `/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025/`.
- **R–Z coords:** `data[mask, 56:58]` for iphi=0 points.
- **Display value:** `z = target[:,1] * (10 ** norm)` where `norm = data[:,14]`.
- **Contour:** `matplotlib.tri.Triangulation` + `tricontourf` for unstructured R–Z.

---

## Implementation Plan & Status

### Model1 / Model12 workflow (61-column baseline)

| Step | Description | Status |
|------|-------------|--------|
| 1 | Train Model1 on first 61 columns only (set1) | **Done** – `surge run configs/xgc_aparallel_set1_61cols.yaml` |
| 2 | Eval Model1 on set1 datastreamsets | **Done** – `surge viz ... --datastreamset-eval` |
| 3 | Eval Model1 on set2 datastreamsets | **Done** – `--datastreamset-eval-set set2_beta0p5` |
| 4 | Fine-tune Model1 with set2 → Model12 | **Done** – `surge run configs/xgc_model12_finetune.yaml` |
| 5 | Eval Model12 on set2 datastreamsets | **Done** – run viz on finetune run |
| 6 | Eval Model12 on set1 datastreamsets | **Done** – cross-set eval on finetune run |

### Visualizations

| Viz | Description | Status |
|-----|-------------|--------|
| All workflow viz | Inference, SHAP, datastreamset for Model1/Model12 | **Done** |
| **Loss-curve continuation** | Save loss after refinement, continue after fine-tuning | **Done** – `surge viz RUN_DIR --loss-curve-continuation` |
| **XGC predictions animation** | GIF of GT vs pred across datastreamsets (Model1, Model12, set1_v3) | **Done** – see [Animation section](#xgc-predictions-animation) below |
| **XGC spatial A_parallel animation** | 2D R–Z contour GIF: GT vs inference per timestep (Model1, Model12, ModelS, set1_v3) | **Done** – see [Spatial Animation section](#xgc-spatial-a_parallel-animation) below |

### ModelS (SHAP-based, top 20 features)

| Step | Description | Status |
|------|-------------|--------|
| 1 | Train ModelS on top 20 SHAP features | **Done** – `surge run configs/xgc_aparallel_set1_shap20.yaml` |
| 2 | Eval ModelS on set1 datastreamsets | **Done** – `surge viz runs/xgc_modelS_shap20 --datastreamset-eval` |
| 3 | Fine-tune ModelS on set2 | **Done** – add config with pretrained_run_dir |
| 4 | **XGC predictions animation** | **Done** – `python scripts/xgc_predictions_animation.py --run-dir runs/...` |

**Top 20 SHAP features** (from `shap_summary_tree_dot.png`):  
`input_0`, `input_61`, `input_117`, `input_89`, `input_14`, `input_145`, `input_19`, `input_20`, `input_173`, `input_17`, `input_42`, `input_25`, `input_21`, `input_27`, `input_45`, `input_9`, `input_16`, `input_26`, `input_8`, `input_44`

### C++ integration (callable in XGC)

| Step | Description | Status |
|------|-------------|--------|
| 1 | Callable C++ function for Model1 and Model12 | **Done** – `xgc_cpp/` |
| 2 | Alternative to `set_nn_correction_decomp()` in `get_dAdt_guess.cpp` | **Done** – standalone API |
| 3 | Copy/generate alternative (do not modify XGC-Devel) | **Done** – `xgc_cpp/` is separate |
| 4 | ONNX Runtime for inference | **Done** – `scripts/xgc_export_onnx.py` + `xgc_cpp/` |

**Reference:** `/global/cfs/projectdirs/m499/sku/XGC-Devel/XGC_core/cpp/solvers/get_dAdt_guess.cpp` → `set_nn_correction_decomp()`

**C++ implementation (61-variable surrogates):** `xgc_cpp/` in the SURGE repo:

- `xgc_surrogate_infer.h` – API: `xgc_surrogate_init()`, `xgc_surrogate_infer_single()`, `xgc_surrogate_switch_model()`, `xgc_surrogate_finalize()`
- `xgc_surrogate_infer.cpp` – ONNX Runtime inference with input/output scaling from `scaler_params.json`
- `main_test.cpp` – minimal test; run from SURGE root with `runs/xgc_model1_61cols/onnx/` paths

**Usage example:**

```c
#include "xgc_surrogate_infer.h"

xgc_surrogate_init("runs/xgc_model1_61cols/onnx/xgc_mlp_aparallel.onnx",
                   "runs/xgc_model1_61cols/onnx/scaler_params.json");

float inputs[61];   // your 61 variables
float outputs[2];   // outputs[1] = A_parallel
xgc_surrogate_infer_single(inputs, 61, outputs, 2);

// Switch to Model12
xgc_surrogate_switch_model("runs/xgc_model12_finetune/onnx/xgc_mlp_aparallel.onnx",
                           "runs/xgc_model12_finetune/onnx/scaler_params.json");

xgc_surrogate_finalize();
```

**Build:**

```bash
cd xgc_cpp && mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
make
```

### Quick commands

```bash
# Model1 (61 cols)
surge run configs/xgc_aparallel_set1_61cols.yaml --run-tag xgc_model1_61cols

# Model12 (finetune)
surge run configs/xgc_model12_finetune.yaml

# ModelS (SHAP top 20)
surge run configs/xgc_aparallel_set1_shap20.yaml --run-tag xgc_modelS_shap20

# Loss-curve continuation
surge viz runs/xgc_model12_finetune --loss-curve-continuation

# Animation (Model1, Model12, or set1_v3)
python scripts/xgc_predictions_animation.py --run-dir runs/xgc_model1_61cols --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1
python scripts/xgc_predictions_animation.py --run-dir runs/xgc_model12_finetune --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set2_beta0p5
python scripts/xgc_predictions_animation.py --run-dir runs/xgc_aparallel_set1_v3 --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025

# Export ONNX for C++
python scripts/xgc_export_onnx.py --run-dir runs/xgc_aparallel_set1_v3

# Full workflow (all steps)
bash scripts/xgc_full_workflow.sh

# Orchestrator: runs steps sequentially, polls for completion before next
python scripts/xgc_workflow_orchestrator.py --quick
# If Model1 already running elsewhere:
python scripts/xgc_workflow_orchestrator.py --model1-already-running --poll-interval 60
```

### Execution results (set1 201-col model)

| Step | Status | Output |
|------|--------|--------|
| Datastreamset eval (set1) | Done | `runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_*` |
| Cross-set eval (set1→set2) | Done | Drift detected on set2 datastreamsets |
| XGC predictions animation (set1_v3) | Done | `runs/xgc_aparallel_set1_v3/plots/xgc_predictions_animation_set1.gif` |
| XGC predictions animation (Model1) | **Done** | `runs/xgc_model1_61cols/plots/xgc_predictions_animation_set1.gif` |
| XGC predictions animation (Model12) | **Done** | `runs/xgc_model12_finetune/plots/xgc_predictions_animation_set2_beta0p5.gif` |
| XGC spatial A_parallel animation (Model1) | **Done** | `runs/xgc_model1_61cols/plots/xgc_spatial_aparallel_animation_model1.gif` |
| XGC spatial A_parallel animation (Model12) | **Done** | `runs/xgc_model12_finetune/plots/xgc_spatial_aparallel_animation_model12.gif` |
| XGC spatial A_parallel animation (ModelS) | **Done** | `runs/xgc_modelS_shap20/plots/xgc_spatial_aparallel_animation_modelS.gif` |
| XGC spatial A_parallel animation (set1_v3) | **Done** | `runs/xgc_aparallel_set1_v3/plots/xgc_spatial_aparallel_animation_set1_v3.gif` |
| Model1 (61 cols) training | **Done** | `runs/xgc_model1_61cols/` |
| Model12 finetune | **Done** | `runs/xgc_model12_finetune/` |
| ModelS (SHAP 20) training | **Done** | `runs/xgc_modelS_shap20/` |
| ModelS finetune | **Done** | `runs/xgc_modelS_finetune/` |
| ONNX export | **Done** | `runs/xgc_model1_61cols/onnx/`, `runs/xgc_model12_finetune/onnx/` |

---

## 12. Progress Report (Workflow Execution)

### Phase 1: RF Fix and Model1

- **RF predict_with_uncertainty:** Disabled (`request_uncertainty: false`) for RF in all XGC configs. sklearn `RandomForestRegressor` does not support `return_std`; only GPR does. This unblocked training.
- **Model1 (61 cols):** Trained on set1 with quick config (10k samples). MLP val R² 0.95, RF val R² 0.95. Outputs: `runs/xgc_model1_61cols/`.

### Phase 2: Model12 Finetune

- **Load compat:** Fixed `PyTorchMLPModel.load` to use `weights_only=False` (PyTorch 2.6+ default change). Fixed engine to accept `backend="pytorch"` for finetune (was checking only `"torch"`).
- **Model12:** Fine-tuned Model1 on set2. MLP test R² 0.96 (output_1). Outputs: `runs/xgc_model12_finetune/`.
- **Loss-curve continuation:** `runs/xgc_model12_finetune/plots/loss_curve_continuation.png`.

### Phase 3: ModelS and ModelS Finetune

- **ModelS:** Trained on top 20 SHAP features (set1, 10k samples). MLP test R² 0.96. Outputs: `runs/xgc_modelS_shap20/`.
- **ModelS finetune:** Fine-tuned ModelS on set2. Config: `configs/xgc_modelS_finetune.yaml`. Outputs: `runs/xgc_modelS_finetune/`.

### Phase 4: Viz and ONNX

- **Inference plots:** Generated for Model1, Model12, ModelS.
- **Datastreamset eval:** Fixed for 61/20-col models. `viz_datastreamset_evaluation()` and `_load_or_build_train_ranges()` now use `spec.input_columns` or `spec.n_inputs` to select the correct input subset before building X, so scalers receive 61 or 20 features instead of 201.
- **ONNX export:** Model1 and Model12 exported to ONNX. Script uses `dynamo=False` (legacy exporter) and `pt_model.network.cpu().eval()` for compatibility. Outputs: `runs/xgc_model1_61cols/onnx/`, `runs/xgc_model12_finetune/onnx/`.

### Phase 5: XGC Predictions Animation

- **Script:** `scripts/xgc_predictions_animation.py` – creates GIF of GT vs pred across datastreamsets.
- **Model1 animation:** `runs/xgc_model1_61cols/plots/xgc_predictions_animation_set1.gif` (5 frames, RF).
- **Model12 animation:** `runs/xgc_model12_finetune/plots/xgc_predictions_animation_set2_beta0p5.gif` (5 frames, RF).
- **Features:** Standalone XGC loader (no SurrogateDataset), spec-based input selection, RF default (joblib loads reliably), matplotlib `buffer_rgba()` for frame capture.

### Phase 6: XGC Spatial A_parallel Animation

- **Script:** `scripts/xgc_spatial_aparallel_animation.py` – 2D spatial animation of A_parallel on R–Z geometry.
- **Layout:** Left panel = ground truth contour, right panel = surrogate inference; shared color scale.
- **Outputs:** One GIF per model (Model1, Model12, ModelS, set1_v3) with 10 frames per timestep.
- **Paths:** `runs/xgc_model1_61cols/plots/xgc_spatial_aparallel_animation_model1.gif`, `runs/xgc_model12_finetune/plots/xgc_spatial_aparallel_animation_model12.gif`, `runs/xgc_modelS_shap20/plots/xgc_spatial_aparallel_animation_modelS.gif`, `runs/xgc_aparallel_set1_v3/plots/xgc_spatial_aparallel_animation_set1_v3.gif`.
- **Features:** Tags-based timestep extraction, iphi=0 mask for 2D plot, `matplotlib.tri.Triangulation` + `tricontourf`, fallback for runs without `workflow_summary.json`, `load_model_compat` fallback for PyTorchMLPModel with mismatched `hidden_layers` in config.

### Input Structure (from user workflow)

- **Model1:** First 61 columns (current timestep only).
- **Model12:** Fine-tune Model1 with set2.
- **ModelS:** Top 20 SHAP features: `input_0`, `input_61`, `input_117`, `input_89`, `input_14`, `input_145`, `input_19`, `input_20`, `input_173`, `input_17`, `input_42`, `input_25`, `input_21`, `input_27`, `input_45`, `input_9`, `input_16`, `input_26`, `input_8`, `input_44`.

### Suggested commits for push

1. **feat(xgc): add xgc_predictions_animation.py** – GIF of GT vs pred across datastreamsets for Model1, Model12, set1_v3.
2. **fix(viz): datastreamset eval for 61/20-col models** – Use spec.input_columns or spec.n_inputs for effective input selection.
3. **fix(models): optional TorchMLPAdapter/GPflowGPRAdapter in adapters** – Avoid import failure when torch_adapter missing.
4. **fix(xgc): disable RF uncertainty to avoid predict_with_uncertainty failure** – Set `request_uncertainty: false` for RF in all XGC configs (sklearn RF does not support `return_std`).
5. **fix(io): torch-first load for PyTorch models** – `load_model_compat` tries `torch.load` first when backend is pytorch to avoid joblib "persistent IDs" error.
6. **fix(pytorch): weights_only=False in PyTorchMLPModel.load** – Required for PyTorch 2.6+ when checkpoint contains StandardScaler.
7. **fix(engine): accept backend pytorch for finetune** – Engine checked only `"torch"`; PyTorchMLPAdapter uses `"pytorch"`.
8. **feat(xgc): add ModelS finetune config** – `configs/xgc_modelS_finetune.yaml` with `pretrained_run_dir`, `input_columns`.
9. **feat(xgc): add ModelS finetune and eval to orchestrator** – Steps 7b–7d in `xgc_workflow_orchestrator.py`.
10. **feat(xgc): quick configs for ModelS** – `configs/xgc_aparallel_set1_shap20_quick.yaml`; orchestrator uses it with `--quick`.
11. **fix(onnx): CPU eval and legacy exporter for ONNX export** – `pt_model.network.cpu().eval()`, `dynamo=False` for compatibility.
12. **docs(xgc): progress report and execution results** – XGC_SUMMARY.md Section 12, updated execution table, animation section.
13. **feat(xgc): add xgc_spatial_aparallel_animation.py** – 2D R–Z contour GIF of GT vs inference per timestep for Model1, Model12, ModelS, set1_v3.

---

## 11. Datastreamset Comparison & Catastrophic Forgetting

**Goal:** Evaluate Set1-trained vs finetuned models across Set1 and Set2 datastreamsets to quantify cross-set generalization, fine-tuning gains, and catastrophic forgetting.

### Four Evaluation Scenarios

| Scenario | Model | Eval Data | Purpose |
|----------|-------|-----------|---------|
| Set1 on Set1 | xgc_aparallel_set1_v3 | Set1 datastreamsets | Baseline: in-distribution |
| Set1 on Set2 | xgc_aparallel_set1_v3 | Set2 datastreamsets | Cross-set generalization |
| Finetuned on Set2 | xgc_aparallel_set2_finetune | Set2 datastreamsets | Fine-tuned on target regime |
| Finetuned on Set1 | xgc_aparallel_set2_finetune | Set1 datastreamsets | **Catastrophic forgetting** test |

### Regenerate

```bash
python scripts/xgc_datastreamset_comparison.py
```

### Plot Locations

- **Combined:** `runs/xgc_datastreamset_comparison/datastreamset_comparison_all.png`
- **Set1 on Set1:** `runs/xgc_aparallel_set1_v3/plots/eval_set1/`
- **Set1 on Set2:** `runs/xgc_aparallel_set1_v3/plots/eval_set2/`
- **Finetuned on Set2:** `runs/xgc_aparallel_set2_finetune/plots/eval_set2/`
- **Finetuned on Set1 (forgetting):** `runs/xgc_aparallel_set2_finetune/plots/eval_set1_forgetting/`

### Results (MLP A_parallel, 10 datastreamsets)

![Datastreamset comparison all](../../runs/xgc_datastreamset_comparison/datastreamset_comparison_all.png)

| Scenario | R² (approx) | RMSE (approx) | Interpretation |
|----------|-------------|---------------|-----------------|
| Set1 model on Set1 | ~0.98 | ~0.10 | Strong in-distribution performance |
| Set1 model on Set2 | ~0.05–0.18 | ~0.60 | Poor cross-set generalization |
| Finetuned on Set2 | ~0.98 | ~0.10 | Fine-tuning restores Set2 performance |
| Finetuned on Set1 | ~0.63–0.80 | ~0.30–0.38 | **Catastrophic forgetting** – loss on Set1 after fine-tuning on Set2 |

### What We Are Trying to Do

1. **Cross-set generalization:** Measure how well a Set1-trained model transfers to Set2 (different beta regime). Result: poor without fine-tuning.
2. **Fine-tuning effectiveness:** Confirm that fine-tuning on Set2 recovers high R² on Set2 datastreamsets.
3. **Catastrophic forgetting:** Quantify how much the finetuned model forgets Set1 when evaluated back on Set1. Result: R² drops from ~0.98 to ~0.7–0.8; RMSE rises from ~0.1 to ~0.35.
4. **Continual learning trigger:** When drift is detected on new datastreamsets, recommend fine-tuning; but monitor forgetting on original regime and consider replay or regularization in future work.

---

## Related Docs

- [XGC Training Guide](XGC_TRAINING.md) – cross-set eval, fine-tuning, model cards, MLflow
- [XGC Features](XGC_FEATURES_2025-03.md) – sequential feature documentation
- [Drift Detection Policy](DRIFT_DETECTION_POLICY.md) – OOD and accuracy triggers
- [AI_OCLF Architecture](AI_OCLF_ARCHITECTURE.md) – prior notebook model format

---

## Slide Deck: SURGE–XGC Effort (Minimal Set)

Minimal slide structure for presenting the SURGE–XGC surrogate effort: goals, capabilities, current status, and future steps. Image paths are relative to the SURGE repo root.

---

### Slide 1: Title & Objective

**Title:** SURGE–XGC: Surrogate Models for A_parallel with Continual Learning

**Objective:** Replace or augment expensive XGC physics with fast ML surrogates that (1) predict A_parallel from flux-surface inputs, (2) adapt to new regimes via fine-tuning when drift is detected, and (3) export to C++ for embedding in the XGC simulation loop.

**Image:** None (title slide).

---

### Slide 2: What We Are Trying to Accomplish

**Bullets:**
- **Surrogate for A_parallel:** 201 (or 61) inputs → 2 outputs; output_1 = A_parallel.
- **Continual learning loop:** Train on regime A → evaluate on regime B → detect drift → refine weights on B → save and re-evaluate.
- **Deployment in XGC:** Export model (ONNX) and scalers; wrap in C++ inference; embed in `get_dAdt_guess` or equivalent.

**Image:** None, or a simple schematic (train → save → evaluate → refine → export → embed).

---

### Slide 3: Input Structure & Data Regimes

**Bullets:**
- **201 inputs:** 61 (current step) + 140 (5 previous timesteps). Layout from `Ah_data_preparation6_low_beta.ipynb`.
- **61-input baseline (Model1):** Current step only; suitable for C++ integration.
- **Regimes:** set1 (lower beta), set2_beta0p5 (beta 0.5%); cross-set generalization is poor without fine-tuning.

**Image:** None, or a table of input blocks (Section 0 of this doc).

---

### Slide 4: SURGE Capabilities Implemented for XGC

**Bullets:**
- **Training:** MLP and RF surrogates; HPO (Optuna/TPE).
- **Partial model saving:** Weights, scalers, metadata; load for fine-tuning.
- **Weight refining:** Fine-tune on new dataset with reduced LR (`finetune_lr_scale`).
- **Drift detection:** Input OOD, output OOD, accuracy drop (R², RMSE).
- **Datastreamset evaluation:** Held-out contiguous regions; cross-set eval.
- **SHAP:** Feature importance for A_parallel.
- **ONNX export:** Model + scalers for C++ inference.
- **C++ API:** `xgc_surrogate_init`, `xgc_surrogate_infer_single`, `xgc_surrogate_switch_model`, `xgc_surrogate_finalize`.

**Image:** None.

---

### Slide 5: Training & Inference Results

**Bullets:**
- **set1 (50k samples):** MLP test R² 0.98, RMSE 0.068; RF test R² 0.96, RMSE 0.10.
- **Model1 (61 cols, 10k):** MLP val R² 0.95, RF val R² 0.95.
- **Model12 (finetuned on set2):** MLP test R² 0.96 on set2.

**Image:** `runs/xgc_aparallel_set1_v3/plots/inference_comparison_grid.png`  
*Caption: Ground truth vs prediction (train/val/test) for MLP and RF.*

---

### Slide 6: HPO & SHAP

**HPO (Hyperparameter Optimization):**
- **Type:** Bayesian optimization via Optuna with TPE (Tree-structured Parzen Estimator) sampler.
- **Method:** TPE models the objective (e.g. val RMSE) as a distribution over the search space; samples hyperparameters that balance exploration and exploitation.
- **Search space:** MLP – hidden_layers, dropout, learning_rate, batch_size (50 trials); RF – n_estimators, max_depth, min_samples_leaf (5 trials).

**SHAP (SHapley Additive exPlanations):**
- **What it is:** Model-agnostic interpretability method that attributes each input feature a contribution (Shapley value) to the model’s prediction, based on cooperative game theory.
- **Use here:** Identify which of the 201 (or 61) inputs most influence A_parallel; supports tree explainer for RF and kernel explainer for MLP.
- **Output:** Top features for A_parallel; used to build ModelS (20-col) from SHAP-selected inputs.

**Images:**
- `runs/xgc_aparallel_set1_v3/plots/hpo_convergence.png`
- `runs/xgc_aparallel_set1_v3/plots/shap_xgc_rf_aparallel/shap_summary_tree_grouped.png`

---

### Slide 7: Datastreamset Evaluation & Drift Detection

**Bullets:**
- Evaluate on contiguous datastreamsets; compute R², RMSE per segment.
- Drift triggers: input OOD, output OOD, accuracy drop (R² drop > 0.1 or RMSE ratio > 1.5×).
- When drift detected → recommend fine-tuning.

**Images:**
- `runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_r2_rmse.png`
- `runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_degradation.png`

---

### Slide 8: Cross-Set Generalization & Catastrophic Forgetting

**Bullets:**
- **Set1 model on Set2:** R² ~0.05–0.18 (poor cross-set).
- **Finetuned on Set2:** R² ~0.98 on Set2 (fine-tuning effective).
- **Finetuned on Set1:** R² ~0.63–0.80 (catastrophic forgetting).

**Image:** `runs/xgc_datastreamset_comparison/datastreamset_comparison_all.png`  
*Caption: R² and RMSE across four scenarios (Set1/Set2 model × Set1/Set2 data).*

---

### Slide 9: Fine-Tuning & Loss Continuation

**Bullets:**
- Load pretrained model and scalers; continue training on new dataset with `finetune_lr_scale: 0.1`.
- Loss-curve continuation plot shows training before and after fine-tuning.

**Image:** `runs/xgc_model12_finetune/plots/loss_curve_continuation.png`

---

### Slide 10: Predictions Animation

**Bullets:**
- GIF of GT vs pred across datastreamsets; one frame per segment.
- Available for Model1 (set1), Model12 (set2), set1_v3.

**Images:**
- `runs/xgc_model1_61cols/plots/xgc_predictions_animation_set1.gif`
- `runs/xgc_model12_finetune/plots/xgc_predictions_animation_set2_beta0p5.gif`

### Slide 10b: Spatial A_parallel Animation

**Bullets:**
- 2D R–Z contour GIF: left = ground truth A_parallel, right = surrogate inference.
- One frame per timestep; shared color scale.
- Available for Model1, Model12, ModelS, set1_v3.

**Images:**
- `runs/xgc_model1_61cols/plots/xgc_spatial_aparallel_animation_model1.gif`
- `runs/xgc_aparallel_set1_v3/plots/xgc_spatial_aparallel_animation_set1_v3.gif`

---

### Slide 11: C++ Integration & ONNX Export

**Bullets:**
- **Export:** `scripts/xgc_export_onnx.py` → ONNX + `scaler_params.json`.
- **C++ API:** `xgc_cpp/` – init, infer, switch_model, finalize.
- **Embedding:** Replace or augment `set_nn_correction_decomp()` in `get_dAdt_guess.cpp`.

**Image:** None, or snippet of `xgc_surrogate_infer.h` API.

---

### Slide 12: Current Status Summary

**Table:**

| Capability              | Status   | Output / Path                                      |
|-------------------------|----------|----------------------------------------------------|
| Model1 (61 cols)        | Done     | `runs/xgc_model1_61cols/`                          |
| Model12 (finetune)      | Done     | `runs/xgc_model12_finetune/`                       |
| ModelS (20 SHAP cols)   | Done     | `runs/xgc_modelS_shap20/`                          |
| Datastreamset eval      | Done     | `runs/*/plots/datastreamset_evaluation_*`          |
| SHAP                    | Done     | `runs/*/plots/shap_*/`                             |
| Predictions animation   | Done     | `runs/*/plots/xgc_predictions_animation_*.gif`     |
| Spatial A_parallel animation | Done | `runs/*/plots/xgc_spatial_aparallel_animation_*.gif` |
| ONNX export             | Done     | `runs/xgc_model1_61cols/onnx/`, `.../model12/onnx/`|
| C++ API                 | Done     | `xgc_cpp/`                                         |

**Image:** `runs/xgc_model1_61cols/plots/inference_comparison_output_1.png` (A_parallel) or `inference_comparison_grid.png`.

---

### Slide 13: Future Steps

**Bullets:**
- **Mitigate catastrophic forgetting:** Replay, regularization, or elastic weight consolidation when fine-tuning on new regimes.
- **Full XGC integration:** Link `xgc_cpp` into XGC-Devel build; validate against production runs.
- **Extended input variants:** Evaluate 201-col model in C++ if memory/latency allow; compare with 61-col.
- **Automated continual learning:** Orchestrator that trains → evaluates → triggers fine-tuning when drift detected.
- **Uncertainty quantification:** GPR or ensemble-based UQ for surrogate predictions in XGC.

**Image:** None.

---

### Image Path Reference (for slide assembly)

| Slide | Image path(s) |
|-------|---------------|
| 5 | `runs/xgc_aparallel_set1_v3/plots/inference_comparison_grid.png` |
| 6 | `runs/xgc_aparallel_set1_v3/plots/hpo_convergence.png`, `runs/xgc_aparallel_set1_v3/plots/shap_xgc_rf_aparallel/shap_summary_tree_grouped.png` |
| 7 | `runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_r2_rmse.png`, `runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_degradation.png` |
| 8 | `runs/xgc_datastreamset_comparison/datastreamset_comparison_all.png` |
| 9 | `runs/xgc_model12_finetune/plots/loss_curve_continuation.png` |
| 10 | `runs/xgc_model1_61cols/plots/xgc_predictions_animation_set1.gif`, `runs/xgc_model12_finetune/plots/xgc_predictions_animation_set2_beta0p5.gif` |
| 10b | `runs/xgc_model1_61cols/plots/xgc_spatial_aparallel_animation_model1.gif`, `runs/xgc_aparallel_set1_v3/plots/xgc_spatial_aparallel_animation_set1_v3.gif` |
| 12 | `runs/xgc_model1_61cols/plots/inference_comparison_output_1.png` or `inference_comparison_grid.png` |
