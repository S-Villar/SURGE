# XGC Features – Sequential Documentation

**Date:** 2025-03

This document describes XGC-related features added in 2025-03, in the order they were implemented.

---

## 1. row_range support for datastreamset loading

**Purpose:** Load contiguous row regions from XGC data for held-out evaluation.

**Changes:**
- `SurrogateDataset.from_path()` accepts `analyzer_kwargs` with `hints` (e.g. `row_range`)
- XGC loader `_load_olcf_npy()` accepts `row_range=(start, end)` to load rows `[start:end]` instead of a random sample

**Usage:** Pass `hints={"row_range": (50000, 100000)}` when loading for datastreamset 1 (rows 50k–100k).

**Files:** `surge/dataset.py`, `surge/datasets/xgc.py`

---

## 2. train_data_ranges.json for in-distribution checks

**Purpose:** Store training input/output min/max for OOD detection on held-out datastreamsets.

**Changes:**
- `save_train_data_ranges()` writes `train_data_ranges.json` after training
- Workflow calls it when scalers exist
- JSON schema: `inputs`/`outputs` with `columns`, `min`, `max` arrays

**Files:** `surge/io/artifacts.py`, `surge/workflow/run.py`

---

## 3. Model loading compatibility

**Purpose:** Load models trained in different environments (sklearn version mismatch, PyTorch pickle protocol).

**Changes:**
- `load_model_compat()` patches sklearn `DecisionTreeRegressor`/`DecisionTreeClassifier` for missing `monotonic_cst` (1.3.2 → 1.8.0)
- PyTorch fallback when `joblib.load` fails with persistent ID error: `torch.load` + reconstruction
- New saves use `joblib.dump(..., protocol=4)` for pickle compatibility

**Files:** `surge/io/load_compat.py`, `surge/viz/run_viz.py`, `surge/io/artifacts.py`

---

## 4. Datastreamset evaluation and drift detection

**Purpose:** Evaluate trained models on held-out datastreamsets; detect OOD and accuracy drop.

**Changes:**
- CLI: `--datastreamset-eval`, `--datastreamset-size`, `--datastreamset-max`, `--datastreamset-offset`
- `viz_datastreamset_evaluation()` loads contiguous datastreamsets, runs inference, computes R²/RMSE/MAE
- In-range check: compare datastreamset min/max to `train_data_ranges.json`
- Drift triggers: `input_ood`, `output_ood`, `accuracy_drop`
- Outputs: `datastreamset_evaluation_stats.json`, `datastreamset_evaluation_r2_rmse.png`

**Policy:** `docs/xgc/DRIFT_DETECTION_POLICY.md`

**Files:** `surge/viz/run_viz.py`, `surge/cli.py`, `docs/xgc/DRIFT_DETECTION_POLICY.md`

---

## 5. Input range evolution and R² zoom

**Purpose:** Visualize how OOD input ranges change across datastreamsets; zoom R² plot around train/val.

**Changes:**
- `_check_datastreamset_in_range()` returns `input_min_max_ood` (min/max per OOD input vs training range)
- New subplot: "Input range evolution" – top 6 OOD inputs, normalized min/max vs datastreamset index (0–1 = train range)
- R² subplot y-axis zooms to `[min(r2, refs) - 0.02, max(r2, refs) + 0.01]` for better visibility

**Files:** `surge/viz/run_viz.py`

---

## 6. SHAP on A_parallel, grouped SHAP, multi-output fix

**Purpose:** Explain output_1 (A_parallel) by default; group 201 inputs into 14 stages + 5 extra; fix 3D indexing.

**Changes:**
- Default `output_index=1` for XGC with 2 outputs
- CLI: `--shap-output-index`
- Multi-output RF: use `shap_vals[:, :, output_index]`
- Grouped bar: 14 stages (14 vars × 14 slots), 5 extra columns
- Output: `shap_summary_tree_grouped.png`

**Files:** `surge/viz/importance.py`, `surge/viz/run_viz.py`, `surge/cli.py`, `tests/test_viz_importance.py`

---

## 7. XGC datastreamset generalization script

**Purpose:** Train on a single datastreamset or evaluate existing models on held-out datastreamsets.

**Modes:**
- `--run-dir ... --datastreamset-eval`: evaluate saved models
- `--data-dir ... --train-on-datastreamset --hpo-trials N`: train RF on datastreamset 0, evaluate on 1, 2, …

**File:** `scripts/xgc_datastreamset_generalization.py`

---

## 8. Nomenclature: datastreamset

**Purpose:** Consistent naming for contiguous evaluation regions.

**Changes:** CLI and outputs use `datastreamset` (replacing "chunk", "segment"). Default `max_datastreamsets=10`.

---

## Quick reference

```bash
# Full viz with SHAP and datastreamset eval
surge viz runs/xgc_aparallel_set1_v3 --shap --datastreamset-eval

# More datastreamsets
surge viz runs/xgc_aparallel_set1_v3 --datastreamset-eval --datastreamset-max 20

# Train-on-datastreamset
python scripts/xgc_datastreamset_generalization.py \
  --data-dir /path/to/olcf_ai_hackathon_2025 \
  --train-on-datastreamset --hpo-trials 10
```
