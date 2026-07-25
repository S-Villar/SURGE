---
name: surge-build-surrogate
description: End-to-end playbook to build a surrogate model for a NEW dataset with SURGE - characterize the data, pick candidate models by task shape, run HPO, evaluate, and report. Use when asked to "train a surrogate for this data", "model this dataset", or "which model should I use".
---

# Building a surrogate for a new dataset

Follow the pipeline order; do not skip characterization.

## 1. Ingest + characterize (before any training)

Accepted formats: CSV, Parquet, Pickle, HDF5, NetCDF, JSON, Excel
(`SurrogateDataset.from_path`, optional metadata YAML naming
inputs/outputs). Check: n samples, input/output dimensionality and shape
(scalar vs multi-output vs field), NaNs, target distribution (log-scale if
heavy-tailed), input–target correlations, class balance. Reference figure:
`characterization` in `examples/viz_theme_gallery.py`. Note current core
limits: data must fit a DataFrame; NaNs are dropped silently; splits are
random-only (no group/temporal) — warn the user when samples are grouped
(shots, trajectories) because random splits leak.

## 2. Shortlist models by task shape

| task shape | first tries | stronger / slower |
|---|---|---|
| tabular → scalar | sklearn.random_forest, lgbm/xgboost.regressor, sklearn.gradient_boosting | pytorch.residual_mlp, pytorch.ft_transformer, pytorch.kan |
| tabular → multi-output | sklearn.random_forest, pytorch.mlp | pytorch.residual_mlp (joint), one-model-per-output loop |
| tabular → class | sklearn.logistic_regression, random_forest_classifier | pytorch.mlp_classifier, ft_transformer_classifier |
| 1D field → 1D field | pytorch.fno1d | pytorch.deeponet, pytorch.cnn1d |
| 2D field → 2D field | pytorch.fno2d, pytorch.unet | (no 3D operator model exists yet) |
| sequence window → window | pytorch.lstm, pytorch.gru | pytorch.cnn1d |
| images → class | pytorch.lenet5 / resnet20 | pytorch.vit, resnet56 |
| needs uncertainty | sklearn.gpr (exact, <5k rows), botorch.gp / botorch.sparse_gp | mlp_ensemble (no save/load); UQ API returns (mean, std) tuple for GPR |

Always include one cheap baseline (RF or ridge) — it calibrates whether
deep models are earning their runtime. Verify keys exist:
`python -c "from surge.model import list_models; print(sorted(list_models()))"`
(optional-dep models vanish silently if their import fails).

## 3. Train + HPO

Quick loop: `MODEL_REGISTRY.create(key, **params)` → `fit/predict`, or a
workflow spec with `hpo:` per model (Optuna TPE; `sampler: botorch` for
expensive objectives). Benchmark-side HPO has ready search spaces for
sklearn/GBM/mlp/residual_mlp/ft_transformer/kan/cnn1d/lstm/fno1d/fno2d/
deeponet/unet in `surge/benchmarks/hpo.py`. Budget guidance: 20–40 trials
for MLP-family, cap epochs during search (`--hpo-epochs-cap`), retrain the
best config at full epochs. Record: per-trial JSON lands in
`runs/<tag>/hpo/`; plot with the starred-best HPO recipe (surge-viz skill).

## 4. Evaluate + report

Read `runs/<tag>/metrics.json` (never re-train to get numbers). Regression:
R², RMSE + parity/residual figure; fields: rel-L2 + field triptych;
classification: Acc/F1/AUROC + ROC/PR/confusion/ECE; UQ: coverage of 95%
band. Generate figures per the surge-viz skill; for multi-model comparisons
use the leaderboard report.

## Known capability edges (set expectations honestly)

- 2D fields: FNO2d/UNet are implemented with HPO spaces but verified only
  on smoke-scale data locally; PDEBench 2D needs a multi-GB fetch first.
- 3D fields (e.g. TheWell MHD 64³): no registered model — requires a new
  adapter (see surge-add-model skill).
- Single-GPU only (`max_gpus=1` enforced); large 2D training is slow.
- No conformal/ensemble UQ wrapper yet — GP-family only for uncertainty.
- `cv_folds` ignored; no stratified/group splits in the engine.
