# XGC Commit Plan – One Commit Per Feature

Run from project root. Each commit corresponds to one feature in `docs/xgc/XGC_FEATURES_2025-03.md`.

---

## Commit 1: row_range support for datastreamset loading

```bash
git add surge/dataset.py surge/datasets/xgc.py
git commit -m "feat(xgc): add row_range support for datastreamset loading

- SurrogateDataset accepts analyzer_kwargs with hints
- XGC loader loads contiguous rows [start:end] via row_range
- Enables held-out datastreamset evaluation"
```

---

## Commit 2: train_data_ranges.json for in-distribution checks

```bash
git add surge/io/artifacts.py surge/workflow/run.py
git commit -m "feat(io): save train_data_ranges.json for in-distribution checks

- save_train_data_ranges() writes inputs/outputs min/max
- Workflow calls it after training when scalers exist
- Used by datastreamset eval for OOD detection"
```

---

## Commit 3: model loading compatibility

```bash
git add surge/io/load_compat.py surge/io/artifacts.py surge/viz/run_viz.py
git commit -m "feat(io): model loading compatibility for sklearn and PyTorch

- load_model_compat(): sklearn monotonic_cst patch, PyTorch pickle fallback
- joblib.dump uses protocol=4 for pickle compatibility
- Viz uses load_model_compat for SHAP and datastreamset eval"
```

---

## Commit 4: datastreamset evaluation and drift detection

```bash
git add surge/viz/run_viz.py surge/cli.py docs/xgc/DRIFT_DETECTION_POLICY.md
git commit -m "feat(viz): datastreamset evaluation and drift detection

- CLI: --datastreamset-eval, --datastreamset-size, --datastreamset-max
- R²/RMSE/MAE per model per datastreamset
- Drift triggers: input_ood, output_ood, accuracy_drop
- Policy: docs/xgc/DRIFT_DETECTION_POLICY.md"
```

---

## Commit 5: input range evolution and R² zoom

```bash
git add surge/viz/run_viz.py
git commit -m "feat(viz): input range evolution plot and R² zoom

- input_min_max_ood per OOD input for range evolution subplot
- Top 6 OOD inputs: normalized min/max vs datastreamset index
- R² subplot y-axis zooms around train/val range"
```

---

## Commit 6: SHAP on A_parallel, grouped SHAP, multi-output fix

```bash
git add surge/viz/importance.py surge/viz/run_viz.py surge/cli.py tests/test_viz_importance.py
git commit -m "feat(viz): SHAP on A_parallel, grouped SHAP, multi-output fix

- Default output_index=1 for XGC; --shap-output-index override
- Fix 3D indexing for multi-output RF SHAP
- Grouped bar: 14 stages + 5 extra columns"
```

---

## Commit 7: XGC datastreamset generalization script

```bash
git add scripts/xgc_datastreamset_generalization.py
git commit -m "feat(scripts): XGC datastreamset generalization script

- Evaluate existing models on held-out datastreamsets
- Train-on-datastreamset mode with limited HPO"
```

---

## Commit 8: XGC documentation

```bash
git add docs/xgc/XGC_FEATURES_2025-03.md docs/xgc/CHANGES_XGC_VIZ_2025-03.md docs/xgc/XGC_INPUT_STRUCTURE.md docs/xgc/XGC_APARALLEL_DATA.md docs/dev/XGC_COMMIT_PLAN.md
git commit -m "docs(xgc): sequential documentation of XGC features

- XGC_FEATURES_2025-03.md: features in implementation order
- CHANGES_XGC_VIZ_2025-03.md: viz changes summary
- XGC_INPUT_STRUCTURE.md, XGC_APARALLEL_DATA.md: data layout
- XGC_COMMIT_PLAN.md: one-commit-per-feature sequence"
```

---

## Quick apply (all 8 commits)

```bash
cd ${SURGE_HOME}

git add surge/dataset.py surge/datasets/xgc.py
git commit -m "feat(xgc): add row_range support for datastreamset loading"

git add surge/io/artifacts.py surge/workflow/run.py
git commit -m "feat(io): save train_data_ranges.json for in-distribution checks"

git add surge/io/load_compat.py surge/io/artifacts.py surge/viz/run_viz.py
git commit -m "feat(io): model loading compatibility for sklearn and PyTorch"

git add surge/viz/run_viz.py surge/cli.py docs/xgc/DRIFT_DETECTION_POLICY.md
git commit -m "feat(viz): datastreamset evaluation and drift detection"

git add surge/viz/run_viz.py
git commit -m "feat(viz): input range evolution plot and R² zoom"

git add surge/viz/importance.py surge/viz/run_viz.py surge/cli.py tests/test_viz_importance.py
git commit -m "feat(viz): SHAP on A_parallel, grouped SHAP, multi-output fix"

git add scripts/xgc_datastreamset_generalization.py
git commit -m "feat(scripts): XGC datastreamset generalization script"

git add docs/xgc/XGC_FEATURES_2025-03.md docs/xgc/CHANGES_XGC_VIZ_2025-03.md docs/xgc/XGC_INPUT_STRUCTURE.md docs/xgc/XGC_APARALLEL_DATA.md docs/dev/XGC_COMMIT_PLAN.md
git commit -m "docs(xgc): sequential documentation of XGC features"
```

---

## Note on overlapping files

`surge/viz/run_viz.py` and `surge/io/artifacts.py` appear in multiple commits.

- **Commit 3:** For `run_viz.py`, stage only the `load_model_compat` import and the two adapter-loading replacements (use `git add -p` if needed). For `artifacts.py`, stage only the `protocol=4` change in `joblib.dump`.
- **Commits 4–6:** Stage the full `run_viz.py`; each commit builds on the previous.
