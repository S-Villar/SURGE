# M3DC1 Session Context: Lessons, Workflows, and Agent Skills

## Purpose

This document captures what was learned during the M3DC1 SURGE interaction, what workflows are now operational, and how to convert those workflows into reusable AI Agent skills.

Primary goals addressed in-session:

- monitor and diagnose long-running Slurm jobs
- extract best HPO trial settings and run single-model training without HPO
- evaluate intermediate checkpoints on test-only unique cases in `(m, psi_N)` maps
- debug scaling/metrics/path issues in post-processing
- produce physics-facing RZ reconstruction comparisons using `m3dc1_python_code`
- document reproducible commands and pitfalls for future runs

## What We Implemented

### 1) Training workflow (single model, no HPO)

- Created single-model config from best trial architecture/hyperparameters.
- Enabled validation logging while preserving full-epoch training (`patience > 0`, very large).
- Trained with mini-batches (`batch_size=128`) and full-pass epochs over train split.
- Saved checkpoints every 100 epochs for later evaluation.

Operational behavior observed:

- `train_rows=1183080/1183080` means each epoch traverses full train split.
- mini-batch learning is still used within each epoch.
- training/validation curves are noisy but trend downward overall.

### 2) Checkpoint-capable 2D map evaluation

`scripts/m3dc1/eval_test_delta_p_2d_maps.py` was expanded to:

- load model from `.pt` checkpoint or `.joblib`
- reconstruct adapter and set model fitted state for inference
- recover dataset path from config/spec and fallback paths robustly
- enforce consistent scaler usage for checkpoint and non-checkpoint paths
- select unique test cases from test split only
- evaluate over explicit `m` range (e.g. `-100..100`)
- write:
  - `per_case_2d_metrics.csv`
  - `summary_2d_maps.json`
  - `predictions_2d_rows.csv`
- save best-case artifacts for downstream plotting:
  - `best_case_data.pkl`
  - `best_case_y_true.npy`, `best_case_y_pred.npy`, `best_case_m.npy`, `best_case_psi.npy`
- support log-scale plotting controls and NaN handling options for R2.

### 3) RZ comparison workflow from saved best-case prediction

Created `scripts/m3dc1/plot_best_case_rz_compare.py` to ingest best-case prediction outputs and compare:

- ground-truth RZ field
- predicted-spectrum reconstruction
- error panel
- true-spectrum reconstruction

Capabilities added:

- configurable `--m3dc1-python-code` path injection
- fallback to `.npy` inputs when `.pkl` protocol is incompatible
- compatibility fixes for older Python type hints
- selectable colormap (`jet`, `viridis`, etc.)
- additional `*_shared_recon_cbar.png` output with shared reconstruction colorbar

## Key Lessons Learned

### Environment and execution

- Wrong interpreter (`python` -> Py2) produced syntax errors on modern Python code.
- Correct bootstrap sequence for this project is mandatory before training/eval:
  - `module load conda`
  - `source scripts/m3dc1/surge_slurm_env.sh`
  - `surge_slurm_setup_python`

### Validation behavior and patience coupling

- In current implementation, validation loop is active only when `patience > 0` and val data exists.
- `patience: 0` effectively disables val logging in this code path.
- To keep full training horizon while logging val, use large patience (e.g., `1000` for `500` epochs).

### Metric interpretation and noisy curves

- Per-epoch train/val metrics can show large stochastic jumps in MLP training.
- Single-epoch spikes were observed but did not imply immediate divergence.
- Prefer smoothed trend analysis and checkpoint-based model selection over raw per-epoch minima.

### Post-processing and physics data integration

- Relative paths in `m3dc1_python_code` may assume case working directory (e.g., `equilibrium.h5`).
- Running plotting scripts from wrong cwd causes file-open errors even with valid absolute args.
- HDF5 library path may need explicit `LD_LIBRARY_PATH` on Cray environments.

### Robust data artifacts for analysis loops

- Saving best-case arrays and map predictions was critical for iterative plotting/debug cycles.
- Storing downstream-ready artifacts (`.npy`, `.pkl`, CSV metrics) decouples expensive inference from plotting.

## Frequent Failure Modes and Fixes

- **`SyntaxError` around f-strings** -> wrong Python interpreter (Py2).  
  Fix: activate SURGE Python 3 environment first.

- **`ModelRegistry` API mismatch (`get_entry` vs `get`)** -> registry compatibility drift.  
  Fix: use compatibility helper in workflow code.

- **Checkpoint evaluation says model not fitted** -> adapter state not marked fitted.  
  Fix: set `is_fitted=True` after loading checkpoint weights.

- **Ground-truth map appears near zero/white** -> physically small magnitudes and selection bias.  
  Fix: best-case filters (`--best-case-metric`, `--min-truth-max-for-best`) and log-scale visualization.

- **NaN R2 values** on flat maps -> undefined metric when truth variation is tiny.  
  Fix: fill/override using configurable fallback (`--nan-r2-value`).

- **`equilibrium.h5` open errors** in RZ plotting -> wrong working directory assumption.  
  Fix: run plotting command from simulation case directory.

## Reusable Command Patterns

### Single-model training (no HPO)

```bash
python -m surge.cli run configs/m3dc1_delta_p_per_mode_cfs_mlp_trial12_single.yaml \
  --run-tag m3dc1_trial12_e500_bs128_withval_$(date +%m%d_%H%M%S) \
  2>&1 | tee -a surge_output_m3dc1_eigenmode.log
```

### Checkpoint evaluation on test maps

```bash
/global/cfs/projectdirs/m3716/software/asvillar/envs/surge/bin/python \
  scripts/m3dc1/eval_test_delta_p_2d_maps.py \
  runs/<tag> \
  --model torch_mlp_profiles_deltap \
  --checkpoint runs/<tag>/checkpoints/torch_mlp_profiles_deltap/epoch_0300.pt \
  --m-source range --m-min -100 --m-max 100 \
  --max-cases 1000 \
  --best-case-metric nrmse \
  --min-truth-max-for-best 1e-3 \
  --nan-r2-value 0.0 \
  --save-best-case-data
```

### RZ compare from saved best-case prediction

```bash
export LD_LIBRARY_PATH="/opt/cray/pe/hdf5/1.14.3.7/gnu/12.3/lib:${LD_LIBRARY_PATH}"
python3 scripts/m3dc1/plot_best_case_rz_compare.py \
  --best-case-pkl /pscratch/.../best_case_data.pkl \
  --sdata /global/cfs/projectdirs/amsc007/data/m3dc1/run11/sparc_1319/sdata_complex_v2.h5 \
  --c1 /global/cfs/projectdirs/amsc007/data/m3dc1/run11/sparc_1319/C1.h5 \
  --run run_0001 \
  --field p \
  --time-index 1 \
  --cmap viridis \
  --out /pscratch/.../best_case_rz_compare_run11_sparc_1319_viridis.png
```

## AI Agent Skills to Create from This Session

The following are high-value candidate skills for project automation.

### Skill 1: `m3dc1-slurm-triage`

- **Use when:** user asks about queue status, log location, stalled runs, or tailing progress.
- **Core actions:** find job state, locate stdout/stderr, detect active terminal context, provide non-disruptive monitoring commands.
- **Output:** short status snapshot + exact follow commands.

### Skill 2: `surge-single-model-from-hpo`

- **Use when:** user wants to convert best trial from HPO into one deterministic training run.
- **Core actions:** parse best trial, create config, enforce batch/epoch/checkpoint/val settings, generate launch command with `tee`.
- **Output:** validated config + run command + expected artifacts checklist.

### Skill 3: `surge-checkpoint-map-eval`

- **Use when:** user asks to evaluate intermediate/final model on full test set or sampled cases.
- **Core actions:** load checkpoint/joblib safely, reconstruct split-aware test cases, evaluate over configured `m` range, export robust metrics/artifacts.
- **Output:** csv/json metrics summary + best-case selection + plotting paths.

### Skill 4: `m3dc1-rz-reconstruction-compare`

- **Use when:** user requests physics-space reconstruction comparison from predicted spectra.
- **Core actions:** handle environment setup (`LD_LIBRARY_PATH`, cwd), call RZ compare script, produce standard + shared-colorbar figures.
- **Output:** generated figure set + quantitative errors (`relL2`).

### Skill 5: `surge-metric-debugger`

- **Use when:** user reports suspicious maps/NaNs/flat truth or inconsistent metrics.
- **Core actions:** diagnose scaler path, split leakage, near-zero truth regimes, R2 undefined cases, and plotting scale artifacts.
- **Output:** root-cause matrix + corrective flag recipe.

### Skill 6: `surge-loss-curve-analyst`

- **Use when:** user asks if training is unstable/noisy or overfitting.
- **Core actions:** parse JSONL training history, report trend + spike stats, compute smoothed series, recommend noise-reduction hyperparameter changes.
- **Output:** concise stability diagnosis + next-run tuning deltas.

## Suggested Skill Authoring Plan

1. Start with `surge-checkpoint-map-eval` and `surge-loss-curve-analyst` (highest immediate reuse).
2. Add `m3dc1-rz-reconstruction-compare` for domain plotting consistency.
3. Add `m3dc1-slurm-triage` for operations support on Perlmutter.
4. Keep each `SKILL.md` short; move long examples to reference docs.
5. Include trigger terms in each skill description so auto-discovery works.

## Traceability Note

This context captures one end-to-end interactive session where training, evaluation, and physics reconstruction workflows were refined under real HPC constraints. Keep this file updated when scripts or expected outputs change.
