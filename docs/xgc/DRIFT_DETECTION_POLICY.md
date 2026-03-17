# SURGE Drift Detection Policy

## Purpose

When evaluating a **static model** (fixed weights) on **held-out datastreamsets** of data (regions not used for training), we need to detect when the evaluation regime differs from the training regime. This enables:

1. **OOD (out-of-distribution) characterization** – flag datastreamsets where inputs/outputs fall outside training bounds
2. **Accuracy degradation detection** – flag datastreamsets where prediction quality drops significantly
3. **Continual learning triggers** – drive policies for ingesting new data and retraining (future)

## Evaluation Setup

- **Model**: Static, trained on a random sample (e.g. 50k rows)
- **Segments**: Contiguous regions of the full dataset (e.g. rows 0–50k, 50k–100k, …)
- **Segments are not trained on** – they represent new regimes the model has never seen

## Drift Triggers

### 1. Input OOD (out-of-distribution)

**Trigger:** Any input feature in the datastreamset has min or max outside the training data min/max.

- **Stored**: `train_data_ranges.json` (inputs min/max from training)
- **Check**: For each datastreamset, compare datastreamset min/max per feature to training ranges
- **Flag**: `inputs_in_range: false` → **OOD input trigger**

**Interpretation:** The model is being asked to extrapolate on inputs it has never seen. Prediction accuracy may drop.

### 2. Output OOD (out-of-distribution)

**Trigger:** Any output (ground truth) in the datastreamset has min or max outside the training output ranges.

- **Check**: Segment output min/max vs training output min/max
- **Flag**: `outputs_in_range: false` → **OOD output trigger**

**Interpretation:** The regime produces outputs outside the training distribution. The model may struggle to predict these values.

### 3. Accuracy Drop

**Trigger:** Segment R² falls below a threshold relative to training, or RMSE rises above a threshold.

- **R² drop**: `datastreamset_r2 < train_r2 - r2_drop_threshold` (default: 0.1)
- **RMSE rise**: `datastreamset_rmse > train_rmse * rmse_ratio_threshold` (default: 1.5)
- **Flag**: Either condition → **Accuracy drop trigger**

**Interpretation:** Even if inputs appear in-range, the model performs poorly. Possible causes: different physics regime, covariate shift, or model capacity limits.

## Per-Segment Drift Summary

For each datastreamset we compute:

| Field | Description |
|-------|-------------|
| `drift_detected` | `true` if any trigger fired |
| `ood_triggers` | List of fired triggers: `["input_ood", "output_ood", "accuracy_drop"]` |
| `in_range` | Input/output range check (existing) |
| `models` | R², RMSE, MAE per model (when available) |

## Overall Drift Warning

- **`drift_warning`**: `true` if any datastreamset has `drift_detected`
- **`datastreamsets_with_drift`**: List of datastreamset keys that triggered drift
- **`continual_learning_recommendation`**: Human-readable message when drift is detected

## Continual Learning Policy (Future)

When drift is detected, SURGE flags:

> **DRIFT DETECTED**: Evaluation data is outside training regime or prediction accuracy has dropped. Recommend ingesting new data from these regions and retraining the surrogate.

Future extensions (not yet implemented):

- Automatic retraining triggers
- Data ingestion pipelines for flagged regimes
- Incremental / continual learning workflows

## Usage

```bash
surge viz runs/xgc_aparallel_set1_v3 --datastreamset-eval
```

Output:

- `datastreamset_evaluation_stats.json` – includes `drift_detection` section with per-datastreamset and overall flags
- Plots – R²/RMSE vs datastreamset with out-of-range shading

## Thresholds (Configurable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r2_drop_threshold` | 0.1 | Max allowed drop from train R² before accuracy trigger |
| `rmse_ratio_threshold` | 1.5 | Max allowed RMSE ratio (datastreamset/train) before accuracy trigger |

These can be overridden in `viz_datastreamset_evaluation()` or via CLI in the future.
