# XGC Workflow – Commits to Push

Run these in order from the SURGE project root. Adjust paths if needed.

```bash
# 1. fix(xgc): disable RF uncertainty to avoid predict_with_uncertainty failure
git add configs/xgc_aparallel_set1.yaml configs/xgc_aparallel_set1_61cols.yaml \
  configs/xgc_aparallel_set1_61cols_quick.yaml configs/xgc_aparallel_set1_shap20.yaml \
  configs/xgc_aparallel_set2_beta0p5.yaml configs/xgc_aparallel_set2_finetune.yaml \
  configs/xgc_finetune_smoke.yaml configs/xgc_model12_finetune.yaml configs/xgc_set2_smoke.yaml
git commit -m "fix(xgc): disable RF uncertainty to avoid predict_with_uncertainty failure

sklearn RandomForestRegressor does not support return_std; only GPR does.
Set request_uncertainty: false for RF in all XGC configs."

# 2. fix(io): torch-first load for PyTorch models
git add surge/io/load_compat.py
git commit -m "fix(io): torch-first load for PyTorch models

Try torch.load first when backend is pytorch to avoid joblib
'persistent IDs in protocol 0 must be ASCII strings' error."

# 3. fix(pytorch): weights_only=False in PyTorchMLPModel.load
git add surge/model/pytorch_impl.py
git commit -m "fix(pytorch): weights_only=False in PyTorchMLPModel.load

Required for PyTorch 2.6+ when checkpoint contains StandardScaler."

# 4. fix(engine): accept backend pytorch for finetune
git add surge/engine.py
git commit -m "fix(engine): accept backend pytorch for finetune

Engine checked only 'torch'; PyTorchMLPAdapter uses 'pytorch'."

# 5. feat(xgc): add ModelS finetune config
git add configs/xgc_modelS_finetune.yaml
git commit -m "feat(xgc): add ModelS finetune config

pretrained_run_dir: runs/xgc_modelS_shap20, input_columns (top 20 SHAP)."

# 6. feat(xgc): add ModelS finetune and eval to orchestrator
git add scripts/xgc_workflow_orchestrator.py
git commit -m "feat(xgc): add ModelS finetune and eval to orchestrator

Steps 7b-7d: finetune ModelS on set2, eval on set2 and set1."

# 7. feat(xgc): quick configs for ModelS
git add configs/xgc_aparallel_set1_shap20_quick.yaml
git commit -m "feat(xgc): quick config for ModelS (no HPO, 10k samples)

Orchestrator uses it with --quick."

# 8. fix(onnx): CPU eval and legacy exporter for ONNX export
git add scripts/xgc_export_onnx.py
git commit -m "fix(onnx): CPU eval and legacy exporter for ONNX export

pt_model.network.cpu().eval() and dynamo=False for compatibility."

# 9. docs(xgc): progress report and execution results
git add docs/xgc/XGC_SUMMARY.md
git commit -m "docs(xgc): progress report and execution results

Section 12: workflow execution phases, input structure, updated status table."

# Optional: batch remaining XGC configs and scripts
git add configs/xgc_aparallel_set1_61cols.yaml configs/xgc_aparallel_set1_shap20.yaml
git add scripts/xgc_export_onnx.py scripts/xgc_full_workflow.sh scripts/xgc_workflow_orchestrator.py
git add scripts/xgc_predictions_animation.py scripts/xgc_datastreamset_comparison.py scripts/xgc_inference.py
git add xgc_cpp/
# (Some of these may already be committed above; adjust as needed)

# Push
git push origin ai4fusion-dev
```

## Note

Some configs (xgc_model12_finetune, xgc_modelS_finetune) were added in commits 1 and 5. The RF fix touches 9 configs. Run `git status` between commits to verify what is staged.
