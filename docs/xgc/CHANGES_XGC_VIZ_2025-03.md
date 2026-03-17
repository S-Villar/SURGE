# XGC Viz & Generalization Analysis – Changes Summary

**Date:** 2025-03-17

> **Full sequential documentation:** `docs/xgc/XGC_FEATURES_2025-03.md`

## Summary

SHAP analysis now targets **A_parallel (output_1)** instead of output_0. Grouped SHAP plots show importance of the 14 stages of 14 vars and the 5 extra columns. Datastreamset evaluation plots R² and RMSE vs datastreamset index, with drift detection, input range evolution, and R² zoom. Model loading compatibility fixes sklearn and PyTorch version mismatches. A new script trains simple models on datastreamsets with limited HPO.

---

## 1. SHAP on A_parallel (output_1)

**Issue:** SHAP was explaining output_0, not A_parallel (output_1).

**Change:** For XGC runs with 2 outputs, SHAP now uses `output_index=1` by default.

- **Files:** `surge/viz/run_viz.py`, `surge/cli.py`
- **CLI:** `--shap-output-index 1` to override
- **Inference:** From `workflow_summary.json` when `dataset_format=xgc` and `n_outputs>=2`

---

## 2. Multi-output SHAP indexing fix

**Issue:** For multi-output RandomForest, SHAP returns a 3D array `(n_samples, n_features, n_outputs)`. The code indexed the wrong axis, producing wrong shapes and empty plots.

**Change:** Use `shap_vals[:, :, output_index]` instead of `shap_vals[output_index]`.

- **File:** `surge/viz/importance.py` in `compute_shap_tree()`

---

## 3. Grouped SHAP (14 stages + 5 extra)

**New:** Bar plot of mean |SHAP| per group for XGC 201 inputs:

- **14 stages:** `stage_0` (inputs 0–13), `stage_1` (14–27), …, `stage_13` (182–195)
- **5 extra:** `extra_5` (inputs 196–200)

**Output:** `plots/shap_{model}/shap_summary_tree_grouped.png`

- **File:** `surge/viz/importance.py` – `xgc_201_group_specs()`, `plot_shap_grouped_bar()`, `shap_grouped_importance()`

---

## 4. Datastreamset evaluation and drift detection

**New:** R² and RMSE vs datastreamset index for held-out datastreamsets. Drift detection flags OOD inputs/outputs and accuracy drop.

- **Output:** `plots/datastreamset_evaluation_r2_rmse.png`, `datastreamset_evaluation_stats.json`
- **CLI:** `surge viz --datastreamset-eval --datastreamset-size 50000 --datastreamset-max 10`
- **Drift policy:** `docs/xgc/DRIFT_DETECTION_POLICY.md`
- **File:** `surge/viz/run_viz.py` – `viz_datastreamset_evaluation()`, `_apply_drift_detection()`

## 5. Model loading compatibility

**New:** `load_model_compat()` fixes sklearn `monotonic_cst` (1.3.2→1.8.0) and PyTorch pickle protocol errors. Viz uses it for SHAP and datastreamset eval.

- **File:** `surge/io/load_compat.py`

## 6. Input range evolution and R² zoom

**New:** Fourth subplot shows top 6 OOD inputs' normalized min/max vs datastreamset index. R² subplot y-axis zooms around train/val range.

- **File:** `surge/viz/run_viz.py` – `input_min_max_ood`, `_plot_datastreamset_evaluation()`

---

## 7. XGC datastreamset generalization script

**New:** `scripts/xgc_datastreamset_generalization.py`

**Modes:**

1. **Existing models:** `--run-dir runs/xgc_aparallel_set1_v3 --datastreamset-eval`
2. **Train on datastreamset 0:** `--data-dir /path/to/olcf_ai_hackathon_2025 --train-on-datastreamset --hpo-trials 10`

The train-on-datastreamset mode trains an RF on datastreamset 0 with 10 HPO trials and evaluates on datastreamsets 1, 2, … to assess generalization.

---

## 8. Data structure docs (56 vars, 4 timesteps)

**Update:** `docs/xgc/XGC_APARALLEL_DATA.md`

- 14 base vars × 4 variants (axisym, non-axisym, normalized, flux-averaged) = 56
- 4 timesteps
- 196 = 14 × 14 (slots); 5 extra tied to nprev=5

---

## 9. MLP SHAP pickle error

**Status:** MLP models trained with older pickle protocols may fail with `persistent IDs in protocol 0 must be ASCII strings`. `load_model_compat` provides a fallback; new saves use `protocol=4`. RF SHAP is well-tested.

---

## Suggested commits

See `docs/dev/XGC_COMMIT_PLAN.md` for a one-commit-per-feature sequence (8 commits).

---

## Usage

```bash
# Full viz with SHAP and datastreamset eval
surge viz runs/xgc_aparallel_set1_v3 --shap --datastreamset-eval

# Train-on-datastreamset analysis (10 HPO trials)
python scripts/xgc_datastreamset_generalization.py \
  --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \
  --train-on-datastreamset --hpo-trials 10 --datastreamset-size 50000 --max-datastreamsets 10
```
