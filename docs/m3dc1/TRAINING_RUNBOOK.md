# M3DC1 Surrogate Training Runbook

A reproducible, copy-pasteable record of how the M3DC1 surrogates in this branch
(`feat/m3dc1-on-model-bench`) were trained, so others can learn and repeat it.

Every command was run from the SURGE repo root:

```bash
cd /global/homes/a/asvillar/src/SURGE
```

Two prediction tasks are covered:

| task | target | rows | dataset |
|---|---|---|---|
| **γ** (growth rate) | one scalar per case | ~10k | pre-built Parquet of case scalars |
| **Re(δp) per mode** | 201-pt profile per (case, n, m) | ~2M | loaded straight from the M3DC1 batch dir |

Every model in a run shares **one** train/val/test split, grouped by physical
case (`group_columns: [run_id, eq_id]`) so per-mode rows never leak across
splits. The split is written to `splits.json` in each run directory.

---

## 0. Environment

Interactive (login/compute node):

```bash
module load conda                                   # NERSC Perlmutter
export SURGE_CONDA_ENV=/global/cfs/projectdirs/m3716/software/asvillar/envs/surge
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python                            # activates env, checks pyarrow/torch
```

SLURM batch scripts do this automatically (they source the same helper).

---

## 1. Build the γ (growth-rate) dataset

One row per `(run_id, eq_id)`; scalar features + `gamma`. Fast (scalars only,
no 2D spectra). Emits a Parquet table + a metadata YAML (declares inputs,
output, and `group_columns`) that SURGE reads as a generic dataframe.

```bash
python scripts/m3dc1/internal/build_case_scalar_dataset.py \
    /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
    --filename csdata_deltap_b_ver.h5 \
    --out data/datasets/SPARC/case_scalars_ver.parquet \
    --metadata-out data/datasets/SPARC/case_scalars_ver_metadata.yaml \
    --target gamma
```

The Re(δp) per-mode task needs **no** pre-build step — the loader reads the
batch dir directly (see §3).

---

## 2. Train all models — γ leaderboard

The workflow runner drives training. On the `model-bench` line `surge.cli`
only exposes the *benchmark runner*, so workflows (which keep our case-grouped
`splits.json`) are launched through a thin wrapper:

```bash
python scripts/m3dc1/internal/run_workflow.py \
    configs/internal/m3dc1_gamma_ver_allmodels.yaml
```

- Trains the framework's regression roster (RF, GBR, ridge, sklearn MLP,
  pytorch MLP/residual/geom-residual/ensemble, FT-Transformer, VAE, …) on one
  shared split.
- Missing-backend / incompatible models are **skipped, not fatal**.
- Artifacts land in `runs/m3dc1_gamma_ver_allmodels/`
  (`metrics.json`, `splits.json`, `predictions/`, `models/`, per-model cards).

Result (test R²): mlp_ensemble 0.857, random_forest 0.852, torch_mlp 0.852.

---

## 3. Train all models — Re(δp) per mode

### 3a. Smoke test first (small subset, fast, login node)

Always validate the per-mode pipeline on a few cases before the big job:

```bash
python scripts/m3dc1/internal/run_workflow.py \
    configs/internal/m3dc1_deltap_real_per_mode_smoke.yaml
```

The δp loader options live in the config under `dataset_kwargs`:

```yaml
dataset_source: m3dc1_batch_per_mode
batch_dir_filename: csdata_deltap_b_ver.h5
dataset_kwargs:
  component: real        # real | imag | magnitude  (target = Re(δp))
  profile_inputs: true   # add q(ψ_N) and p(ψ_N) samples as inputs
  profile_points: 16     # samples per profile
  max_cases: 40          # cap for the smoke test (drop for the full run)
```

### 3b. Full flagship run (~2M rows) — SLURM

```bash
sbatch scripts/m3dc1/internal/train_deltap_real_per_mode.slurm
```

Runs `configs/internal/m3dc1_deltap_real_per_mode_ver_allmodels.yaml` (all 9976
cases). Inputs = shaping + q/p profile scalars + n,m + drive + q(ψ)/p(ψ)
profile samples (44 features) → 201-point Re(δp) profile. Output in
`runs/m3dc1_deltap_real_per_mode_ver_allmodels/`.

---

## 4. Hyperparameter optimization (HPO)

HPO uses built-in per-model **recipes** (Optuna search spaces in
`surge/benchmarks/hpo.py`). In a workflow, set `hpo.enabled: true` with **no**
`search_space` and the recipe is used automatically — every trial evaluated on
the shared case-grouped split:

```yaml
models:
  - {key: pytorch.mlp, name: torch_mlp,
     hpo: {enabled: true, n_trials: 20, metric: val_rmse, direction: minimize}}
```

γ HPO (validated: torch_mlp 0.852 → 0.866):

```bash
python scripts/m3dc1/internal/run_workflow.py \
    configs/internal/m3dc1_gamma_ver_hpo.yaml
```

Re(δp) HPO on a capped subset (best params transfer to the full run) — SLURM:

```bash
sbatch scripts/m3dc1/internal/train_deltap_real_per_mode_hpo.slurm
```

HPO artifacts per run: `hpo/<model>_hpo.json` (best trial + top-k),
`hpo_trials_manifest.jsonl`, and training-progress JSONL streams.

List models that have an HPO recipe:

```bash
surge run --list-hpo-models
```

---

## 5. Alternative: the benchmark runner (leaderboards + MLflow)

Good for standard datasets. NOTE: it makes its **own row-random split**, so it
is *not* used for the per-mode δp task (which needs case grouping).

```bash
surge models                          # list registered models
surge run --list                      # list benchmarks (by category)
surge run -b <benchmark> -m all       # leaderboard over all compatible models
surge run -b <benchmark> -m pytorch.ft_transformer --hpo --hpo-trials 40
surge run -b <benchmark> -m all --mlflow   # log to MLflow
```

---

## 6. Inspect results

```bash
# leaderboard from a run's metrics
python - <<'PY'
import json; from pathlib import Path
m=json.loads(Path("runs/m3dc1_gamma_ver_allmodels/metrics.json").read_text())
rows=[(n, d.get("test",{}).get("r2")) for n,d in m.items() if "error" not in d]
for n,r2 in sorted(rows, key=lambda x:-(x[1] or -9)):
    print(f"{n:<20}{r2:.4f}")
PY

# split integrity (zero leakage across train/val/test groups)
python -c "import json;s=json.load(open('runs/m3dc1_gamma_ver_allmodels/splits.json'));print(s['group_columns'], s['leakage_check'])"
```

---

## 7. Files registered this session

Configs (`configs/internal/`):
- `m3dc1_gamma_ver_allmodels.yaml` — γ, all models, shared split
- `m3dc1_gamma_ver_hpo.yaml` — γ, recipe HPO
- `m3dc1_deltap_real_per_mode_smoke.yaml` — Re(δp) smoke test
- `m3dc1_deltap_real_per_mode_ver_allmodels.yaml` — Re(δp) flagship (all cases)
- `m3dc1_deltap_real_per_mode_hpo.yaml` — Re(δp) recipe HPO (capped subset)

Scripts (`scripts/m3dc1/internal/`):
- `build_case_scalar_dataset.py` — build the γ Parquet dataset
- `run_workflow.py` — launch a workflow spec (keeps `splits.json`)
- `train_deltap_real_per_mode.slurm` — flagship Re(δp) SLURM launcher
- `train_deltap_real_per_mode_hpo.slurm` — Re(δp) HPO SLURM launcher

Core features added: case-grouped splitting (`group_columns` + `splits.json`),
`dataset_kwargs` passthrough, real/imag/magnitude + profile inputs in the
per-mode loader, resilient multi-model loop, and the recipe→workflow HPO bridge.
