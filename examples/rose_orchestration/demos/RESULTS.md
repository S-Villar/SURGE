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
  --max-iter 1 \
  --candidates rf,mlp \
  --quiet \
  --no-live-progress
```

Observed summary:

```text
Example 4 summary
Wall time: 53.0s
rank  learner   workflow         val_r2    val_rmse  run_tag
   1  mlp       m3dc1_mlp       0.87105    0.008094  rose_parallel_1_mlp_m3dc1_mlp_iter_0
   2  rf        m3dc1_rf        0.85180    0.008677  rose_parallel_0_rf_m3dc1_rf_iter_0
```

Notes:

- This demo uses the full complete-case M3DC1 table: `9891` rows.
- The effective SURGE split in the recorded run was `5934 / 1978 / 1979`.
- On this run, `mlp` ranked ahead of `rf` on validation `R2`.

## Demo 2: Smarter campaign control

Command:

```bash
python example_05_uq_guided_mlp_ensemble.py \
  --dataset m3dc1 \
  --max-iter 1 \
  --ensemble-n 3 \
  --sklearn-mlp-max-iter 250 \
  --uncertainty-threshold 0.015 \
  --quiet \
  --no-live-progress
```

Observed summary:

```text
Example 5 summary
Wall time: 51.1s
  #     total     train       val      test     val_r2    mean_std     max_std
  0      9891      5934      1978      1979    0.86638    0.080005    0.458289
```

Notes:

- This demo uses the full complete-case M3DC1 table in one ROSE-controlled run.
- The effective SURGE split in the recorded run was `5934 / 1978 / 1979`.
- ROSE monitors SURGE's validation uncertainty artifact and compares it to the
  configured threshold of `0.015`.

## Interpretation

These demos already show two useful integration points:

1. **Parallel model race**: ROSE and Rhapsody can evaluate several SURGE surrogate
   families concurrently and return a ranked result on the full dataset.
2. **UQ-driven monitoring**: SURGE can emit uncertainty artifacts that ROSE can use
   to track surrogate trustworthiness and make control decisions on the full dataset.

What they do **not** show yet:

- adaptive point acquisition from an unlabeled candidate pool
- distributed placement through RADICAL-Pilot
- a persistent campaign database or dashboard

## Demo 3: Resource-aware surrogate search

Command used for verification:

```bash
python demos/demo_03_resource_aware_search.py \
  --max-trials 2 \
  --cpus-per-trial 4 \
  --sklearn-mlp-max-iter 300
```

Observed summary:

```text
Demo 3 summary
Wall time: 160.7s
rank  trial  family   train     val    test     val_r2    val_rmse  detail
   1      1  mlp      5934    1978    1979    0.88257    0.007724  mlp[64, 96, 128, 96]
   2      0  rf       5934    1978    1979    0.85180    0.008677  baseline_rf
```

Notes:

- The script detected the available CPU budget and converted it into a parallel
  trial count.
- For this bounded verification run, the search width was capped at `2` trials.
- Both trials used the full M3DC1 table with SURGE's internal `60 / 20 / 20`
  split.
- The randomized MLP trial beat the RF baseline on validation `R2`.

## Demo 4: GPU-aware surrogate search

Command used for fallback verification in a non-GPU shell:

```bash
python demos/demo_04_gpu_aware_search.py \
  --allow-cpu-fallback \
  --max-trials 1 \
  --n-epochs 10 \
  --batch-size 512 \
  --num-workers 0
```

Observed summary:

```text
Demo 4 summary
Wall time: 51.1s
rank  trial  device     train     val    test     val_r2    val_rmse  arch
   1      0  cpu         5934    1978    1979    0.77286    0.010743  [96, 128, 160, 64]
```

Notes:

- The important verification point was `effective_device=cpu` in the trial output.
- That confirms the SURGE resource policy is now reaching the `pytorch.mlp`
  backend rather than only being logged.
- On a GPU allocation, the same demo should report `effective_device=cuda:0`
  (or one slot-local CUDA device per trial).
