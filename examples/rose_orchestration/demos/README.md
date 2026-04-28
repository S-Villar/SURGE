# M3DC1 demos

This directory contains two runnable demos built on top of the
`examples/rose_orchestration` integration layer.

They are intentionally narrow:

- **Demo 1** shows **better surrogate selection** by racing multiple SURGE model
  families in parallel on the same M3DC1 subset.
- **Demo 2** shows **smarter campaign control** by letting ROSE monitor SURGE UQ
  outputs from an MLP ensemble across a staged M3DC1 campaign.

## Why these two demos

They exercise different parts of the stack:

- `SURGE` does the actual surrogate training, splitting, metrics, and artifacts.
- `ROSE` owns the outer learner loop and the decision logic.
- `Rhapsody` plus `radical.asyncflow` own the execution backend.

Demo 1 emphasizes **parallel candidate evaluation**.
Demo 2 emphasizes **campaign-state monitoring from uncertainty artifacts**.

## Prerequisites

The scripts default to:

- SURGE checkout:
  `/global/cfs/cdirs/amsc007/asvillar/radical-integration-smoke/SURGE`
- venv:
  `/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv`
- M3DC1 source PKL:
  `/global/homes/a/asvillar/src/SURGEROSE/data/datasets/M3DC1/sparc-m3dc1-D1.pkl`

Override them with environment variables if needed:

```bash
export SURGE_ROSE_VENV=/path/to/venv
export M3DC1_SOURCE=/path/to/sparc-m3dc1-D1.pkl
```

## Run

From `examples/rose_orchestration`:

```bash
bash demos/demo_01_better_surrogate_selection.sh
bash demos/demo_02_smarter_campaign_control.sh
```

Both scripts:

- activate the default venv when present
- ensure `PYTHONPATH` points at this SURGE checkout
- symlink the M3DC1 PKL into `data/datasets/M3DC1/`
- write detailed execution logs under `demos/output/`

## Demo definitions

### Demo 1: Better surrogate selection

Command:

```bash
python example_04_parallel_model_race.py \
  --dataset m3dc1 \
  --growing-pool \
  --max-iter 1 \
  --candidates rf,mlp,gpr,gpflow_gpr
```

Reasoning:

- `--growing-pool --max-iter 1` keeps all four model families on the same 600-row
  M3DC1 slice.
- That makes the model race fair enough to compare `rf`, `mlp`, `sklearn.gpr`,
  and `gpflow.gpr`.
- ROSE `ParallelActiveLearner` and the Rhapsody backend execute the race
  concurrently, while SURGE handles training for each candidate.

### Demo 2: Smarter campaign control

Command:

```bash
python example_05_uq_guided_mlp_ensemble.py \
  --dataset m3dc1 \
  --growing-pool \
  --max-iter 3 \
  --ensemble-n 3 \
  --sklearn-mlp-max-iter 250 \
  --uncertainty-threshold 0.015
```

Reasoning:

- The M3DC1 pool grows `600 -> 1200 -> 1800` rows.
- SURGE trains an sklearn MLP ensemble on each iteration.
- SURGE writes validation uncertainty artifacts.
- ROSE `SeqUQLearner` reads those artifacts and monitors the uncertainty threshold.

This is not acquisition yet. It is a control-plane demo that shows the loop can
make decisions from UQ signals emitted by SURGE.

## Recorded results

See [RESULTS.md](RESULTS.md) for the latest recorded M3DC1 run.
