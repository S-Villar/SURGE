# M3DC1 Delta p Per-Mode Surrogate Training

Train surrogates that predict **δp_n,m(ψ_N)** — the amplitude profile vs normalized flux for each (n, m) mode. One row per (case, m); 12 inputs → 200 outputs.

## Quick Start

```bash
# 1. Build dataset from batch dir
python scripts/m3dc1/build_delta_p_per_mode.py ${SURGE_SCRATCH}/mp288/jobs/batch_16 \
    --out data/datasets/SPARC/delta_p_per_mode.pkl

# 2. Inspect dataset
python -m surge.cli analyze configs/m3dc1_delta_p_per_mode.yaml

# 3. Train (from file)
python -m surge.cli run configs/m3dc1_delta_p_per_mode.yaml

# Or train directly from batch dir (no pre-build)
python -m surge.cli run configs/m3dc1_delta_p_per_mode_from_batch.yaml
```

## Inputs (12)

| Input | Description |
|-------|-------------|
| eq_R0, eq_a, eq_kappa, eq_delta | Equilibrium geometry |
| q0, q95, qmin, p0 | Safety factor and pressure |
| n, m | Toroidal and poloidal mode numbers |
| input_pscale, input_batemanscale | Run parameters |

## Outputs (200)

`output_p_0` … `output_p_199` — δp amplitude at 200 ψ_N points (0.0001 → 1.0).

## Configs

| Config | Description |
|--------|-------------|
| `m3dc1_delta_p_per_mode.yaml` | From pre-built .pkl, no HPO |
| `m3dc1_delta_p_per_mode_from_batch.yaml` | Load from batch dir |
| `m3dc1_delta_p_per_mode_hpo.yaml` | With HPO (15 RF + 30 MLP trials), patience=20 |
| `m3dc1_delta_p_per_mode_hpo_extended.yaml` | Extended HPO (15 RF + 250 MLP trials) |

## SLURM (Perlmutter)

```bash
# Debug queue (30 min)
sbatch scripts/m3dc1/train_delta_p_per_mode.slurm

# Regular queue, HPO (2 hr)
sbatch scripts/m3dc1/train_delta_p_per_mode_hpo.slurm

# Extended HPO (4 hr, 250 MLP trials)
sbatch scripts/m3dc1/train_delta_p_per_mode_hpo_extended.slurm
```

## Early Stopping (Patience)

The PyTorch MLP uses `patience: 20` — training stops if validation loss does not improve for 20 epochs. Set in config:

```yaml
params:
  max_epochs: 200
  patience: 20
```

## Workflow Modes

- **From file:** `dataset_path: data/.../delta_p_per_mode.pkl`
- **From batch:** `dataset_source: m3dc1_batch_per_mode`, `dataset_path: /path/to/batch_16`

See `tests/test_m3dc1_workflow_modes.py` for both modes.
