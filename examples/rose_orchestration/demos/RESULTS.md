# Recorded M3DC1 demo results

These results were recorded from the `radical-integration` branch in:

`/global/cfs/cdirs/amsc007/asvillar/radical-integration-smoke/SURGE`

## Environment

- dataset: `m3dc1`
- split policy inside SURGE workflows: `train=0.6`, `val=0.2`, `test=0.2`
- source PKL: `/global/homes/a/asvillar/src/SURGEROSE/data/datasets/M3DC1/sparc-m3dc1-D1.pkl`

## Demo 1: Better surrogate selection

Command:

```bash
python example_04_parallel_model_race.py \
  --dataset m3dc1 \
  --growing-pool \
  --max-iter 1 \
  --candidates rf,mlp,gpr,gpflow_gpr \
  --quiet \
  --no-live-progress
```

Observed summary:

```text
Example 4 summary
Wall time: 32.5s
rank  learner      workflow            val_r2    val_rmse  run_tag
   1  rf           m3dc1_rf           0.83884    0.008262  rose_parallel_0_rf_m3dc1_rf_iter_0
   2  mlp          m3dc1_mlp          0.80211    0.009156  rose_parallel_1_mlp_m3dc1_mlp_iter_0
   3  gpflow_gpr   m3dc1_gpflow_gpr   0.79932    0.009220  rose_parallel_3_gpflow_gpr_m3dc1_gpflow_gpr_iter_0
   4  gpr          m3dc1_gpr          0.78847    0.009466  rose_parallel_2_gpr_m3dc1_gpr_iter_0
```

Notes:

- All four models trained on the same 600-row shuffled M3DC1 slice.
- The effective SURGE split at this scale was `360 / 120 / 120`.
- On this run, `rf` ranked first on validation `R2`.

## Demo 2: Smarter campaign control

Command:

```bash
python example_05_uq_guided_mlp_ensemble.py \
  --dataset m3dc1 \
  --growing-pool \
  --max-iter 3 \
  --ensemble-n 3 \
  --sklearn-mlp-max-iter 250 \
  --uncertainty-threshold 0.015 \
  --quiet \
  --no-live-progress
```

Observed summary:

```text
Example 5 summary
Wall time: 33.6s
  #   samples     val_r2    mean_std     max_std
  0       600    0.82459    0.099952    0.304668
  1      1200    0.61474    0.080929    0.375287
  2      1800    0.85108    0.090653    0.365113
```

Notes:

- ROSE monitored SURGE's validation uncertainty artifact at each iteration.
- The mean validation standard deviation stayed above the configured stop threshold
  of `0.015`, so the campaign did not stop early.
- The uncertainty signal changed across iterations, but not monotonically. That is
  consistent with the current state of the example: it is a monitoring/control demo,
  not a full acquisition-policy implementation.

## Interpretation

These demos already show two useful integration points:

1. **Parallel model race**: ROSE and Rhapsody can evaluate several SURGE surrogate
   families concurrently and return a ranked result.
2. **UQ-driven monitoring**: SURGE can emit uncertainty artifacts that ROSE can use
   to track campaign quality and make control decisions.

What they do **not** show yet:

- adaptive point acquisition from an unlabeled candidate pool
- distributed placement through RADICAL-Pilot
- a persistent campaign database or dashboard
