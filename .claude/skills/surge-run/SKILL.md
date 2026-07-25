---
name: surge-run
description: Run a SURGE surrogate workflow (train models on a dataset) and inspect its run artifacts. Use when asked to train a surrogate, run a workflow spec, reproduce a run, or explain what is inside runs/<tag>/.
---

# Running SURGE workflows

The canonical execution path is `run_surrogate_workflow(spec)` on a
`SurrogateWorkflowSpec` — everything else (benchmarks, examples) should be a
caller of this. There is no `surge run` CLI yet; use the Python API or the
quickstart.

## Fast end-to-end check (~5 s)

```bash
python -m examples.quickstart --dataset diabetes            # RF baseline
python -m examples.quickstart --dataset california --model mlp
python -m examples.quickstart --dataset diabetes --model mlp --n-trials 5  # +HPO
```

## Python API / YAML spec

```python
from surge import SurrogateWorkflowSpec, run_surrogate_workflow
spec = SurrogateWorkflowSpec.from_yaml("examples/configs/qlknn_multi_hpo.yaml")
result = run_surrogate_workflow(spec)
```

Key spec fields: `dataset_path`, `input_columns`/`output_columns` (or
metadata YAML), `models: [{name, params, hpo}]`, `test_fraction`,
`val_fraction`, `standardize_inputs`, `run_tag`, `output_dir`, `resources`
(device/num_workers/strict), `mlflow_tracking`.

Gotchas:
- `cv_folds` is accepted but currently IGNORED (single random split runs
  instead) — do not promise CV results.
- Splits are random only: no stratified/group/temporal options yet; flag
  leakage risk when samples are grouped (e.g., per-shot data).
- Optional-dependency models silently disappear from the registry if their
  import fails; verify with `python -c "from surge.model import list_models;
  print(sorted(list_models()))"` before debugging a KeyError.

## Run artifact layout (`runs/<tag>/`)

`spec.yaml` (re-runnable config) · `metrics.json` (per-model
train/val/test R²/RMSE/MAE/MAPE + timings) · `workflow_summary.json`
(dataset, resources, registry, profile) · `models/*.joblib|pt` ·
`scalers/inputs.joblib` · `predictions/<model>_{train,val,test}.parquet`
(y_true/y_pred columns) · `training_log_<model>.jsonl` (per-epoch) ·
`hpo/*_hpo.json` (trials, best_trial, direction) · `git_rev.txt`, `env.txt`,
`model_card_*.json`.

To evaluate a finished run, read `metrics.json` and the parquet predictions —
never re-train just to get numbers.
