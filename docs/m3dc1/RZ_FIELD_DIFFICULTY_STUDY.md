# RZ field difficulty study — experiment plan

Branch: `feat/rz-difficulty-study`  
Baseline: `runs/rz_field_gaugefix_complex_g201` (FNO2d modes=64, ep 688 best, test align med ≈ 0.72)

## Step 1 — Diagnosis (zero-GPU, gate)

```bash
python scripts/m3dc1/internal/analyze_rz_field_difficulty.py \
  --run runs/rz_field_gaugefix_complex_g201
```

Outputs: `difficulty_analysis/difficulty_scatter.png`, `difficulty_summary.json`

## Step 2 — Unstable-only (single lever)

```bash
OUT=runs/rz_field_gaugefix_unstable_only \
bash scripts/m3dc1/internal/run_RZ_gaugefix_unstable.sh
```

Adds `--stability-filter unstable` (default OFF in baseline script).

## Step 3 — Architecture sweep (one change each)

| Run | Lever | Command |
|-----|-------|---------|
| 3a U-Net | `--models unet` | `run_RZ_gaugefix_unet.sh` |
| 3b Hybrid | `--models fno_unet` | `run_RZ_gaugefix_fno_unet.sh` |
| 3c FNO96 | `--fno-modes 96` | `run_RZ_gaugefix_fno96.sh` |

## Step 4 — |δp|² target (magnitude fidelity)

```bash
OUT=runs/rz_field_gaugefix_mag2 \
bash scripts/m3dc1/internal/run_RZ_gaugefix_mag2.sh
```

`--target-mode mag2` (default `legacy` = unchanged behavior).

## Constraints

- All new flags default OFF — baseline `run_RZ_gaugefix_complex.sh` unchanged semantics
- 4h `salloc`, `--time-budget-min 235`, `--patience 0`, `--epochs 1000`
- Resume via `ckpt_fno2d_last.pt` with persisted `no_improve` / `best_epoch`
