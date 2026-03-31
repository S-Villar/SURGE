# M3DC1 surrogate results (summary)

This page pulls together **where artifacts live**, **what each training line is doing**, and **embedded figures** copied under `m3dc1/figures/` so they can live in git (the default `.gitignore` ignores `runs/` and most `*.png`).

**Related docs**

- Data layout and postprocessing status: [`docs/m3dc1/M3DC1_DATA_MAPPING.md`](../docs/m3dc1/M3DC1_DATA_MAPPING.md), [`docs/m3dc1/BATCH_DIRS_AND_DATA.md`](../docs/m3dc1/BATCH_DIRS_AND_DATA.md)
- Delta *p* spectra training: [`docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md`](../docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md)
- Per-mode δ*p*(ψ) guide: [`docs/m3dc1/DELTA_P_PER_MODE_GUIDE.md`](../docs/m3dc1/DELTA_P_PER_MODE_GUIDE.md)
- Development log for delta *p* tooling: [`docs/m3dc1/M3DC1_COMMITS_AND_SUMMARY.md`](../docs/m3dc1/M3DC1_COMMITS_AND_SUMMARY.md)

---

## What `M3DC1_DATA_MAPPING.md` is for

That file is **not** training output—it is an **inventory** of M3DC1 batch data on Perlmutter scratch (`/pscratch/sd/a/asvillar/mp288/jobs`):

- It records which run directories have **`sdata_pertfields_grid_complex_v2.h5`** (postprocessed δ*p* spectra) vs only raw **`C1.h5`**.
- **CFS bulk δ*p*:** [`configs/m3dc1_delta_p_per_mode_cfs.yaml`](../configs/m3dc1_delta_p_per_mode_cfs.yaml) trains on **`sdata_complex_v2.h5`** under `/global/cfs/projectdirs/amsc007/data/m3dc1` (~9859 cases), materialized as **Parquet** (~**1.97M** rows, one per (*case*, *m*)). That path is independent of scratch postprocessing above.

So **M3DC1 “training”** in SURGE is: load inputs/metadata from those HDF5-derived pickles or directly from batch dirs (`dataset_source: m3dc1_batch` / `m3dc1_batch_per_mode`), then run the standard workflow (`surge.cli` / `run_surrogate_workflow`).

---

## Where the latest numerical results are (local machine)

Run outputs are written under **`runs/<run_tag>/`** (gitignored). On the workspace used to build this page, these directories existed:

| Run tag | Target | Config (repo) | Notes |
|---------|--------|---------------|--------|
| **`m3dc1_aug_r75`** | **`output_gamma`** (growth rate), SPARC M3DC1 D1 | [`configs/m3dc1_aug_r75.yaml`](../configs/m3dc1_aug_r75.yaml) | Full-size study (9981 rows after `sample_rows`, 75 HPO trials / model). **Primary “production” γ surrogate.** |
| **`m3dc1_demo`** | **`output_gamma`**, smaller demo | [`configs/m3dc1_demo.yaml`](../configs/m3dc1_demo.yaml) | Includes RF / MLP / GPR; quick plots. |
| **`m3dc1_delta_p_batch16`** | Full δ*p* spectrum (many outputs), ~30 cases | [`configs/m3dc1_delta_p_batch16.yaml`](../configs/m3dc1_delta_p_batch16.yaml) | High dimensional; metrics poor (expected at *n*≈30). |
| **`m3dc1_delta_p_profile_mode0`** | Single-mode profile | [`configs/m3dc1_delta_p_profile_mode0.yaml`](../configs/m3dc1_delta_p_profile_mode0.yaml) | Still *n*≈30; severe overfit / bad test. |
| **`m3dc1_delta_p_per_mode`** | Rows = (case, *n*, *m*) → profile | [`configs/m3dc1_delta_p_per_mode.yaml`](../configs/m3dc1_delta_p_per_mode.yaml) | 6000 rows in summary. |
| **`m3dc1_delta_p_per_mode_hpo`** | Same family with HPO | [`configs/m3dc1_delta_p_per_mode_hpo.yaml`](../configs/m3dc1_delta_p_per_mode_hpo.yaml) | Produces `plots/eigenmode_best_worst.png`. |
| **`m3dc1_per_mode_from_batch`** | Load per-mode directly from batch dir | [`configs/m3dc1_delta_p_per_mode_from_batch.yaml`](../configs/m3dc1_delta_p_per_mode_from_batch.yaml) | Similar metrics to pickle-based per-mode. |
| **`m3dc1_delta_p_per_mode_cfs`** | Per-mode δ*p* from **CFS** `sdata_complex_v2.h5` (~9.8k cases) | [`configs/m3dc1_delta_p_per_mode_cfs.yaml`](../configs/m3dc1_delta_p_per_mode_cfs.yaml) | Dataset: `data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet` (~1.97M rows). Batch: [`scripts/m3dc1/train_delta_p_per_mode_cfs.slurm`](../scripts/m3dc1/train_delta_p_per_mode_cfs.slurm). |
| **`m3dc1_delta_p_per_mode_cfs_hpo`** | Same data + HPO | [`configs/m3dc1_delta_p_per_mode_cfs_hpo.yaml`](../configs/m3dc1_delta_p_per_mode_cfs_hpo.yaml) | Long run; use compute node. |
| **CFS trial suite** | Batch launch (CPU + GPU) | [`scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh`](../scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh) | Submits 5 Slurm jobs (baseline, HPO, MLP BoTorch ×2, GPR). Job IDs logged under `logs/cfs_trial_suite_jobids_*.txt`. |

**Parquet / pickle vs `m3dc1_batch_per_mode`:** `dataset_source: m3dc1_batch_per_mode` reads HDF5 during the workflow and **does not** write a dataset file. For the full CFS table (~2M rows), use **Parquet** (chunked write, lower RAM than one giant pickle):

`python scripts/m3dc1/build_delta_p_per_mode.py /global/cfs/projectdirs/amsc007/data/m3dc1 --out data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet --filename sdata_complex_v2.h5`

(Requires **`pyarrow`**; see [`requirements.txt`](../requirements.txt). `.pkl` output still works for smaller sets but may OOM on login nodes for the full bulk.)

For any run: see **`metrics.json`**, **`workflow_summary.json`**, **`predictions/*.csv`**, **`models/*.joblib`**, and optional **`plots/`**.

---

## Growth-rate (γ) surrogates — metrics snapshot

Values from `runs/m3dc1_aug_r75/metrics.json` (scalar **`output_gamma`**; test fraction 0.2):

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| `random_forest_profiles` | 0.919 | 0.00613 | 0.00207 |
| `torch_mlp_mc_dropout` | 0.884 | 0.00733 | 0.00322 |
| `gpflow_gpr_profiles` | 0.885 | 0.00729 | 0.00288 |

**Note:** `runs/m3dc1_aug_r75/performance_summary.md` documents HPO winners and states GPflow did not complete in some rebuilds—check `models/` and `hpo/` for your local copy.

Smaller reference run **`m3dc1_demo`** (`metrics.json`): RF test R² ≈ **0.854**, MLP test R² ≈ **0.853** (subset / different spec).

### γ — figures

**Full study — prediction grid and HPO**

![Growth rate: inference comparison grid (m3dc1_aug_r75)](figures/gamma_aug_r75_inference_comparison_grid.png)

![Growth rate: HPO convergence (m3dc1_aug_r75)](figures/gamma_aug_r75_hpo_convergence.png)

**Demo run — model comparison, RF parity plot, feature importance**

![Growth rate: demo model comparison](figures/gamma_demo_model_comparison.png)

![Growth rate: demo RF predictions](figures/gamma_demo_rf_predictions.png)

![Growth rate: demo feature importance](figures/gamma_demo_feature_importance.png)

---

## Delta *p* surrogates — metrics snapshot

### Full spectrum (`m3dc1_delta_p_batch16`)

From `runs/m3dc1_delta_p_batch16/metrics.json` (~2500 outputs, ~30 samples):

| Model | Test R² |
|-------|---------|
| `rf_delta_p` | −76.8 |
| `mlp_delta_p` | −17.3 |

Interpretation: **severely underdetermined**; documented expectation in [`docs/m3dc1/M3DC1_COMMITS_AND_SUMMARY.md`](../docs/m3dc1/M3DC1_COMMITS_AND_SUMMARY.md).

### Single-mode profile — mode 0 (`m3dc1_delta_p_profile_mode0`)

| Model | Test R² |
|-------|---------|
| `rf_profile_mode0` | −104.4 |
| `mlp_profile_mode0` | −23.8 |

### Per-mode dataset (`m3dc1_delta_p_per_mode`)

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| `rf_per_mode` | 0.505 | 1901 |
| `mlp_per_mode` | 0.530 | 2822 |

(Scale of RMSE reflects δ*p* amplitude units in the pickle.)

### Per-mode with HPO (`m3dc1_delta_p_per_mode_hpo`)

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| `rf_per_mode` | 0.348 | 1647 |
| `mlp_per_mode` | 0.492 | 2730 |

### Per-mode — CFS bulk `sdata_complex_v2.h5` (`m3dc1_delta_p_per_mode_cfs`)

Full-tree δ*p* (magnitude of complex `spectrum/p`): **1,971,800** rows (**9859** cases × **200** m-modes), same 12 inputs and 200 profile outputs as the older per-mode line. Stored as **`data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet`** (build command above).

#### Batch trial suite (CPU + GPU)

All workflows below use the same Parquet unless noted. **Launch** only from the **repository root** (so `SLURM_SUBMIT_DIR` points at SURGE and Parquet paths resolve):

`./scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh`

*(2026-03-28 launch **failed** when `SURGE_ROOT` resolved to spool; scripts use **`$SLURM_SUBMIT_DIR`** and **`cd` before Python**.)*

Override Slurm account: `SURGE_SLURM_ACCOUNT=YOUR_REPO sbatch …` or edit `#SBATCH -A` in each script. Job IDs are written to **`logs/cfs_trial_suite_jobids_*.txt`** (symlink **`logs/cfs_trial_suite_jobids_latest.txt`**).

**Run 2026-03-31 (relaunch):** Perlmutter jobs **50753544** (T1), **50753545** (T2), **50753546** (T3), **50753547** (T4 GPU), **50753548** (T5). Track with `squeue -u $USER` and `sacct -j …`; logs: `surge_*.log` in the submit directory.

| Trial | Resource | Slurm script | Config / `run_tag` |
|-------|----------|--------------|---------------------|
| **T1** | CPU (`-C cpu`) | [`train_delta_p_per_mode_cfs.slurm`](../scripts/m3dc1/train_delta_p_per_mode_cfs.slurm) | [`m3dc1_delta_p_per_mode_cfs.yaml`](../configs/m3dc1_delta_p_per_mode_cfs.yaml) → `m3dc1_delta_p_per_mode_cfs` |
| **T2** | CPU | [`train_delta_p_per_mode_cfs_hpo.slurm`](../scripts/m3dc1/train_delta_p_per_mode_cfs_hpo.slurm) | [`m3dc1_delta_p_per_mode_cfs_hpo.yaml`](../configs/m3dc1_delta_p_per_mode_cfs_hpo.yaml) → `m3dc1_delta_p_per_mode_cfs_hpo` |
| **T3** | CPU | [`train_delta_p_per_mode_cfs_mlp_hpo.slurm`](../scripts/m3dc1/train_delta_p_per_mode_cfs_mlp_hpo.slurm) | [`m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible.yaml`](../configs/m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible.yaml) → `m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible` |
| **T4** | GPU (`-C gpu`, 1 GPU) | [`train_delta_p_per_mode_cfs_mlp_hpo_gpu.slurm`](../scripts/m3dc1/train_delta_p_per_mode_cfs_mlp_hpo_gpu.slurm) | Same YAML, `--run-tag` `m3dc1_delta_p_per_mode_cfs_mlp_hpo_gpu` |
| **T5** | CPU | [`train_delta_p_per_mode_cfs_gpr_hpo.slurm`](../scripts/m3dc1/train_delta_p_per_mode_cfs_gpr_hpo.slurm) | [`m3dc1_delta_p_per_mode_cfs_gpr_linear_matern52_botorch.yaml`](../configs/m3dc1_delta_p_per_mode_cfs_gpr_linear_matern52_botorch.yaml) → `m3dc1_delta_p_per_mode_cfs_gpr_lin_matern52_botorch` |

**Refresh metrics table** after jobs finish (reads `runs/<run_tag>/metrics.json`; repo **`runs/`** may be a symlink to scratch):

`python scripts/m3dc1/harvest_cfs_trial_metrics.py`

| Trial | Resource | Run tag | Models (test R² / RMSE) | Status |
|-------|----------|---------|--------------------------|--------|
| **T1** | CPU | `m3dc1_delta_p_per_mode_cfs` | — | pending until `metrics.json` exists |
| **T2** | CPU | `m3dc1_delta_p_per_mode_cfs_hpo` | — | pending |
| **T3** | CPU | `m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible` | — | pending |
| **T4** | GPU | `m3dc1_delta_p_per_mode_cfs_mlp_hpo_gpu` | — | pending |
| **T5** | CPU | `m3dc1_delta_p_per_mode_cfs_gpr_lin_matern52_botorch` | — | pending *(subset: `sample_rows`, single `output_p_0`)* |

*Re-run [`harvest_cfs_trial_metrics.py`](../scripts/m3dc1/harvest_cfs_trial_metrics.py) and paste the printed table over the rows above to update R² / RMSE.*

### δ*p* — figures

**Best/worst eigenmode-style comparison (HPO run)**

![Delta p per-mode HPO: best vs worst cases](figures/delta_p_per_mode_hpo_eigenmode_best_worst.png)

**Example δ*p*(ψ) profile for (*n*, *m*) = (9, −7)** — from `scripts/m3dc1/plot_profile.py` workflow (`runs/delta_p_n9_m-7.png` copy):

![Delta p profile n=9, m=-7](figures/delta_p_profile_n9_m-7.png)

---

## Regenerating figures

- γ visualization helper: `python examples/viz_m3dc1_predictions.py --run-dir runs/m3dc1_aug_r75` (see [`docs/dev/UNSTAGED_CHANGES_SUMMARY.md`](../docs/dev/UNSTAGED_CHANGES_SUMMARY.md) if the script path differs on your branch).
- δ*p* profile plot (scratch pickle): `python scripts/m3dc1/plot_profile.py data/datasets/SPARC/delta_p_per_mode.pkl --n 9 --m -7`
- CFS Parquet: `python scripts/m3dc1/plot_profile.py data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet --n 9 --m -7`
- Best/worst plot script: `scripts/m3dc1/plot_best_worst_eigenmode.py` (used for the HPO run plots directory).

---

*Generated as a consolidation of repo docs and local `runs/` metrics; re-copy `m3dc1/figures/` after new training if you want this page to stay visually in sync.*
