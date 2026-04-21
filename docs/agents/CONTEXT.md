# SURGE + M3DC1: repository context for agents

Consolidated **domain context** for this repo. Operational procedures live in [`skills/`](skills/) as executable specs.

---

## What SURGE does here

- **YAML-driven** surrogate workflows: `python -m surge.cli run <config>`.
- **Outputs:** `runs/<run_tag>/` → `metrics.json`, `workflow_summary.json`, `models/`, `checkpoints/`, optional `plots/`, `predictions/`.
- **M3DC1** spans:
  - **Growth rate γ** (`output_gamma`): flagship run **`m3dc1_aug_r75`**.
  - **δ*p* per-mode:** HDF5 batch paths and/or **large CFS Parquet** (~2M rows), HPO trials, PyTorch profile MLPs, **2D evaluation** scripts.

**Disambiguation:** γ and δ*p* use **different** configs, metrics files, and figures—do not merge them when answering “latest results.”

---

## Slurm (summary)

- Batch scripts run from a **copy**; resolve repo with **`SLURM_SUBMIT_DIR`** and **`cd "$SURGE_ROOT"`** before Python.
- Submit from **repository root** (see skill **`surge.slurm.bootstrap`**).

---

## Conda / Parquet (summary)

- Batch nodes may lack conda on PATH until **`module load conda`** (Perlmutter-style).
- Never **`conda activate … || true`** for production jobs; verify **`import pyarrow`** in the **same** interpreter as `surge.cli`.
- Shared helper: [`scripts/m3dc1/surge_slurm_env.sh`](../../scripts/m3dc1/surge_slurm_env.sh).

---

## Data & git

| Asset | Policy |
|-------|--------|
| Large CFS Parquet | Ignored in git; build locally |
| `runs/` | Usually gitignored or symlinked |
| `m3dc1/figures/` | Curated PNGs referenced from `m3dc1/results.md` |
| `logs/` | Launch / job-id logs ignored |

---

## Results quick reference

| Target | Metrics | Figures / docs |
|--------|---------|----------------|
| **γ** | `runs/m3dc1_aug_r75/metrics.json` | `m3dc1/figures/gamma_*`, table in `m3dc1/results.md` |
| **δ*p*** | `runs/<run_tag>/metrics.json` | Run `plots/`, `m3dc1/results.md`, eval scripts |

---

## CFS trial suite (batch)

- Launcher: [`scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh`](../../scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh)
- Harvest: [`scripts/m3dc1/harvest_cfs_trial_metrics.py`](../../scripts/m3dc1/harvest_cfs_trial_metrics.py)

---

## Changelog (high level)

- Slurm `SURGE_ROOT` fix; `surge_slurm_env.sh` with strict conda + pyarrow; default CFS conda prefix for this project.
- Agent docs refactored into **`docs/agents/skills/`** + manifest (blueprint-style).

---

*For Academy alignment, see [`blueprint/README.md`](blueprint/README.md).*
