# SURGE Overview

This page summarizes the current SURGE philosophy, primary modules, and how to
compose them into end-to-end surrogate workflows. It is intentionally concise
and Markdown-only so it can be viewed easily (e.g., via `less` on remote HPC
systems).

---

## 1. Philosophy

- **Single workflow surface** – the entire surrogate pipeline (dataset ingest →
  splits → model registry → Optuna HPO → artifacts) lives in the refactored
  `surge/` package. No more `src/` vs. `surge/`.
- **Config-first, but hackable** – YAML specs (`configs/*.yaml`) drive the CLI
  + notebook experiences, yet the same classes are usable directly from Python.
- **Reproducible artifacts** – every run records models, scalers, metrics,
  predictions, Optuna trials, environment, and git revision under `runs/<tag>/`.
- **HPC-aware** – basic resource detection (host, CPUs, GPUs, scheduler) is
  embedded so artifacts document the compute context.

---

## 2. Key Modules & Responsibilities

| Module / Class                              | Responsibility                                                                 |
|---------------------------------------------|---------------------------------------------------------------------------------|
| `surge.dataset.SurrogateDataset`            | Load CSV/Parquet/Excel/HDF5/NetCDF/Pickle, auto-detect I/O columns, apply metadata |
| `surge.engine.SurrogateEngine`              | Split + standardize data, train models from the registry, capture metrics/preds/timings |
| `surge.registry.MODEL_REGISTRY`             | Register adapters (`sklearn.random_forest`, `torch.mlp`, `gpflow.gpr`, …)        |
| `surge.workflow.spec.SurrogateWorkflowSpec` | YAML/Python spec describing dataset, models, HPO, artifact layout                |
| `surge.workflow.run.run_surrogate_workflow` | Orchestrate dataset load → engine prep → training → Optuna HPO → artifact save   |
| `surge.io.artifacts.*`                      | Persist models, scalers, predictions, metrics, spec, env, git, HPO trials        |
| `surge.viz`                                 | Density scatter, violin/SNR, correlation heatmap, profile bands                  |
| `notebooks/M3DC1_demo.ipynb`                | End-to-end demo with plots, dataset analysis, and artifact inspection            |

---

## 3. Step-by-Step: Minimal Workflow

### 3.1 Load or define a dataset

```python
from pathlib import Path
from surge.dataset import SurrogateDataset

dataset = SurrogateDataset.from_path(
    Path("data/datasets/SPARC/sparc-m3dc1-D1.pkl"),
    metadata_path="data/datasets/SPARC/m3dc1_metadata.yaml",
)
print(dataset.summary())
```

### 3.2 Configure & run the engine manually

```python
from surge.engine import EngineRunConfig, ModelSpec, SurrogateEngine

engine = SurrogateEngine(run_config=EngineRunConfig(test_fraction=0.2, val_fraction=0.2))
engine.configure_from_dataset(dataset)
results = engine.run([
    ModelSpec(key="sklearn.random_forest", name="rf_unit", params={"n_estimators": 200}),
])
print(results[0].val_metrics)
```

### 3.3 YAML-driven workflow (CLI)

```bash
# Baseline 1k-row smoke test
conda run -n surge python -m examples.m3dc1_workflow \
    --spec configs/m3dc1_demo.yaml \
    --run-tag m3dc1_demo_cli

# Augmented: 9,981 rows + 50 Optuna trials/model
conda run -n surge python -m examples.m3dc1_workflow \
    --spec configs/m3dc1_demo_augmented.yaml \
    --run-tag m3dc1_demo_full
```

### 3.4 Programmatic workflow execution

```python
import yaml
from surge.workflow.spec import SurrogateWorkflowSpec
from surge.workflow.run import run_surrogate_workflow

spec = SurrogateWorkflowSpec.from_dict(
    yaml.safe_load(Path("configs/m3dc1_demo.yaml").read_text())
)
spec.run_tag = "prog_example"
spec.models[0].hpo.enabled = False  # adjust per run if desired
summary = run_surrogate_workflow(spec)
print(summary["models"][0]["timings"])
```

### 3.5 What You Get: Results Summary

Running a SURGE workflow produces:

- **Models**: `runs/<tag>/models/*.joblib` (trained models)
- **Metrics**: R², RMSE, MAE, MAPE per split (train/val/test)
  ```
  Model: random_forest_profiles
    Test: R²=0.854, RMSE=0.008, MAE=0.004
  ```
- **Predictions**: `runs/<tag>/predictions/<model>_{train,val,test}.csv` (+ `*_val_uq.json` if UQ enabled)
- **HPO**: `runs/<tag>/hpo/<model>_hpo.json` (best params, trials, top-5 configs)
- **Artifacts**: `metrics.json`, `workflow_summary.json`, `spec.yaml`, `scalers/`
- **Timings**: Training and inference speeds per model

All saved to `runs/<tag>/`; accessed via returned `summary` dictionary.

### 3.6 Typical Summary Structure

After execution, `run_surrogate_workflow()` returns a dictionary with the following structure:

```python
{
    "run_tag": "m3dc1_demo",
    "dataset": {
        "file_path": "data/datasets/SPARC/sparc-m3dc1-D1.pkl",
        "n_rows": 1000,
        "n_inputs": 13,
        "n_outputs": 1,
        "input_columns": [...],
        "output_columns": [...],
        "output_groups": {...}
    },
    "resources": {
        "hostname": "...",
        "cpu_cores_physical": 64,
        "cpu_cores_logical": 128,
        "total_ram_gb": 512.0,
        "gpu_available": false,
        "scheduler": null
    },
    "splits": {
        "train": 640,
        "val": 160,
        "test": 200
    },
    "scalers": {
        "input_scaler": "runs/m3dc1_demo/scalers/inputs.joblib",
        "output_scaler": "runs/m3dc1_demo/scalers/outputs.joblib"
    },
    "models": [
        {
            "name": "random_forest_profiles",
            "backend": "sklearn",
            "params": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_leaf": 2
            },
            "metrics": {
                "train": {"r2": 0.924, "rmse": 0.006, "mae": 0.002, "mape": 0.854},
                "val": {"r2": 0.737, "rmse": 0.014, "mae": 0.006, "mape": 0.879},
                "test": {"r2": 0.854, "rmse": 0.008, "mae": 0.004, "mape": 1.386}
            },
            "timings": {
                "train_seconds": 0.199,
                "train_inference_seconds": 0.045,
                "val_inference_seconds": 0.050,
                "test_inference_seconds": 0.050,
                "train_inference_per_sample": 7.5e-05,
                "val_inference_per_sample": 0.00025,
                "test_inference_per_sample": 0.00025
            },
            "artifacts": {
                "model": "runs/m3dc1_demo/models/random_forest_profiles.joblib",
                "predictions": {
                    "train": "runs/m3dc1_demo/predictions/random_forest_profiles_train.csv",
                    "val": "runs/m3dc1_demo/predictions/random_forest_profiles_val.csv",
                    "test": "runs/m3dc1_demo/predictions/random_forest_profiles_test.csv",
                    "val_uq": "runs/m3dc1_demo/predictions/random_forest_profiles_val_uq.json"
                },
                "hpo": "runs/m3dc1_demo/hpo/random_forest_profiles_hpo.json"  # if HPO enabled
            }
        },
        # ... additional models (torch.mlp, gpflow.gpr, etc.)
    ],
    "registry": "Registered Models:\n- gpflow.gpr: GPflow GPR [...]\n- sklearn.random_forest: [...]\n- torch.mlp: [...]",
    "artifacts": {
        "root": "runs/m3dc1_demo",
        "metrics": "runs/m3dc1_demo/metrics.json",
        "summary": "runs/m3dc1_demo/workflow_summary.json",
        "spec": "runs/m3dc1_demo/spec.yaml"
    }
}
```

**Accessing summary data:**
```python
# Get all model names
model_names = [m["name"] for m in summary["models"]]

# Get best test R² score
best_model = max(summary["models"], key=lambda m: m["metrics"]["test"]["r2"])
print(f"Best model: {best_model['name']} (R²={best_model['metrics']['test']['r2']:.3f})")

# Get dataset info
print(f"Dataset: {summary['dataset']['n_rows']} samples, "
      f"{summary['dataset']['n_inputs']} inputs, "
      f"{summary['dataset']['n_outputs']} outputs")

# Get split sizes
print(f"Splits: train={summary['splits']['train']}, "
      f"val={summary['splits']['val']}, test={summary['splits']['test']}")

# Get artifact paths
print(f"Models saved to: {summary['artifacts']['root']}/models/")
print(f"Metrics: {summary['artifacts']['metrics']}")
```

---

## 4. Capabilities Matrix

| Capability              | Where it lives / how to use                                                   |
|------------------------|--------------------------------------------------------------------------------|
| Dataset auto-detection | `SurrogateDataset.from_path(...)` (metadata YAML overrides available)          |
| Splitting/standardize  | `SurrogateEngine.prepare()` (train/val/test fractions, StandardScaler toggles) |
| Model registry         | `surge.registry.MODEL_REGISTRY` adapters (`sklearn`, `torch`, `gpflow`)        |
| Optuna HPO             | `HPOConfig` per model (`sampler=tpe` or `botorch`, search space dict)          |
| Uncertainty estimates  | RFR tree variance, Torch MLP MC-Dropout, GPflow mean/variance                  |
| Artifact management    | `surge.io.artifacts.*`, run summaries under `runs/<tag>/`                      |
| Visualization          | `surge.viz` helpers + `notebooks/M3DC1_demo.ipynb`                             |
| HPC awareness          | `surge.hpc.resources.detect_compute_resources()` (auto-called by workflow)     |

---

## 5. Example Config (`configs/m3dc1_demo_augmented.yaml`)

Highlights:

- Uses the real SPARC D1 pickle (`sample_rows: 9981`)
- Defines three models with 50 Optuna trials each:
  - `sklearn.random_forest` (n_estimators, depth, min_samples_leaf)
  - `torch.mlp` (hidden layer templates, dropout, LR, batch size, epochs)
  - `gpflow.gpr` (noise variance, maxiter)
- Turns on CSV prediction exports for train/val/test splits

This fuels both the CLI and the notebook. To switch specs in the notebook,
change `spec_path = Path("configs/m3dc1_demo.yaml")` to the augmented file and
rerun the workflow cell.

---

## 6. Artifact Layout

```
runs/<run_tag>/
├── models/<model>.joblib
├── scalers/inputs.joblib, outputs.joblib
├── predictions/<model>_{train,val,test}.csv
├── predictions/<model>_val_uq.json
├── metrics.json
├── workflow_summary.json
├── spec.yaml
├── environment.txt     # pip freeze
├── git.txt             # `git describe --always --dirty`
└── hpo/<model>_hpo.json
```

Use these artifacts to power publication-quality plots, regression dashboards,
or downstream analysis scripts.

---

## 7. Visualization & Notebook Handoff

- Run `conda run -n surge jupyter lab notebooks/M3DC1_demo.ipynb`.
- The notebook mirrors the config-driven workflow and layers on:
  - Dataset memory/missingness stats
  - Timing/metric tables (train/val/test, per-model)
  - GT vs. prediction density heatmaps (`surge.viz.plot_density_scatter`)
  - Per-mode scatter (color-coded by `input_ntor`)
  - Violin/SNR/correlation panels (via `surge.viz`)
  - Table of model timings for quick comparisons

---

## 8. Deployment Checklist

1. **Select spec** – baseline (`configs/m3dc1_demo.yaml`) or augmented
   (`configs/m3dc1_demo_augmented.yaml`).
2. **Run workflow** – CLI or Python API.
3. **Verify artifacts** – `runs/<tag>/workflow_summary.json` contains dataset,
   split sizes, metrics, timings, HPO winners, and registry summary.
4. **Visualize** – notebook or custom scripts reading the CSV prediction files.
5. **Package docs** – `make -C docs html` to regenerate local ReadTheDocs-style
   documentation (see `docs/_build/html/index.html`).

That’s it—SURGE is ready for iterative surrogate development on remote clusters
and desktop environments alike.


