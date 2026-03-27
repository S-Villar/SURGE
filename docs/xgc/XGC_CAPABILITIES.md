# XGC A_parallel Capabilities – SURGE

Concise recap of XGC surrogate capabilities developed in SURGE, with examples from `runs/xgc_aparallel_set1_v3`.

---

## 1. Training

Train RF and MLP surrogates on OLCF hackathon data (201 inputs → 2 outputs, output_1 = A_parallel).

```bash
surge run configs/xgc_aparallel_set1.yaml --run-tag xgc_aparallel_set1_v3
```

**Data:** `/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025/` (set1, set2_beta0p5)

---

## 2. Inference Comparison Plots

GT vs prediction density for train/val/test splits.

```bash
surge viz runs/xgc_aparallel_set1_v3
```

![Inference comparison grid](../../runs/xgc_aparallel_set1_v3/plots/inference_comparison_grid.png)

---

## 3. HPO Convergence

Optuna HPO convergence for RF and MLP.

![HPO convergence](../../runs/xgc_aparallel_set1_v3/plots/hpo_convergence.png)

---

## 4. SHAP Importance (A_parallel)

Explain output_1 (A_parallel) with SHAP. Grouped bar: 14 stages + 5 extra columns.

```bash
surge viz runs/xgc_aparallel_set1_v3 --shap
```

![SHAP grouped](../../runs/xgc_aparallel_set1_v3/plots/shap_xgc_rf_aparallel/shap_summary_tree_grouped.png)

---

## 5. Datastreamset Evaluation & Drift Detection

Evaluate on held-out contiguous regions; detect OOD and accuracy drop.

```bash
surge viz runs/xgc_aparallel_set1_v3 --datastreamset-eval
```

![Datastreamset R²/RMSE](../../runs/xgc_aparallel_set1_v3/plots/datastreamset_evaluation_r2_rmse.png)

**Drift triggers:** input OOD, output OOD, accuracy drop. Policy: `docs/xgc/DRIFT_DETECTION_POLICY.md`

---

## 6. Standalone Inference Script

Run inference with SURGE models on hackathon data (same data as `Ah_prediction_NN1_ASV_Geometrical_inference.ipynb`).

```bash
# Same-set (set1)
python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025

# Cross-set (set2_beta0p5)
python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \
    --eval-set set2_beta0p5 --sample 10000

# Save predictions
python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \
    --out predictions_xgc_set1.csv
```

---

## 7. Datastreamset Generalization Script

Train on one datastreamset, evaluate on others; or evaluate existing models on held-out datastreamsets.

```bash
# Evaluate existing models
python scripts/xgc_datastreamset_generalization.py --run-dir runs/xgc_aparallel_set1_v3 \
    --datastreamset-eval --max-datastreamsets 10

# Train on datastreamset 0, evaluate on 1+
python scripts/xgc_datastreamset_generalization.py \
    --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \
    --train-on-datastreamset --hpo-trials 10 --max-datastreamsets 5
```

---

## 8. Combined Viz

```bash
surge viz runs/xgc_aparallel_set1_v3 --shap --datastreamset-eval
```

---

## Artifact Layout

```
runs/xgc_aparallel_set1_v3/
├── models/           # RF, MLP (load_model_compat)
├── scalers/          # input/output StandardScaler
├── train_data_ranges.json
├── predictions/      # train/val/test CSVs
└── plots/
    ├── inference_comparison_grid.png
    ├── inference_comparison_output_0.png
    ├── inference_comparison_output_1.png
    ├── hpo_convergence.png
    ├── datastreamset_evaluation_r2_rmse.png
    ├── datastreamset_evaluation_degradation.png
    ├── chunk_evaluation_r2_rmse.png
    ├── segment_evaluation_r2_rmse.png
    └── shap_xgc_rf_aparallel/
        ├── shap_summary_tree_grouped.png
        ├── shap_summary_tree_bar.png
        └── shap_dependence_tree_*.png
```

---

## Related

- **Ah_prediction notebook:** `/global/u2/a/asvillar/AI_OCLF/Ah_prediction_NN1_ASV_Geometrical_inference.ipynb` – raw .npy loading, custom NN
- **SURGE inference:** `scripts/xgc_inference.py` – SURGE models + scalers, same data format
- **Detailed features:** `docs/xgc/XGC_FEATURES_2025-03.md`
