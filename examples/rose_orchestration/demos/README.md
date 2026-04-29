# M3DC1 demos

This directory contains two runnable demos built on top of the
`examples/rose_orchestration` integration layer.

They are intentionally narrow:

- **Demo 1** shows **better surrogate selection** by racing multiple SURGE model
  families in parallel on the full M3DC1 table.
- **Demo 2** shows **smarter campaign control** by letting ROSE monitor SURGE UQ
  outputs from an MLP ensemble trained on the full M3DC1 table.
- **Demo 3** shows **resource-aware surrogate search** by sizing the number of
  concurrent trials from the CPUs you allocate and racing those trials together.
- **Demo 4** shows **GPU-aware surrogate search** by sizing PyTorch MLP trials
  from the visible GPUs and binding one trial per GPU.

## Why these two demos

They exercise different parts of the stack:

- `SURGE` does the actual surrogate training, splitting, metrics, and artifacts.
- `ROSE` owns the outer learner loop and the decision logic.
- `Rhapsody` plus `radical.asyncflow` own the execution backend.

Demo 1 emphasizes **parallel candidate evaluation**.
Demo 2 emphasizes **uncertainty-aware model assessment and control signals**.

## Prerequisites

The scripts default to:

- SURGE checkout:
  `/global/cfs/cdirs/amsc007/asvillar/radical-integration-smoke/SURGE`
- venv:
  `/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv`
- M3DC1 source PKL:
  `/global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl`

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
bash demos/demo_03_resource_aware_search.sh
bash demos/demo_04_gpu_aware_search.sh --allow-cpu-fallback --max-trials 1
```

Both scripts:

- activate the default venv when present
- ensure `PYTHONPATH` points at this SURGE checkout
- symlink the M3DC1 PKL into `data/datasets/M3DC1/`
- write detailed execution logs under `demos/output/`

To run Demo 1 and Demo 2 through an interactive Slurm allocation, use the
dedicated launchers:

```bash
cd examples/rose_orchestration
TIME=02:00:00 CPUS_PER_TASK=8 bash demos/launch_demo_01_interactive.sh
TIME=02:00:00 CPUS_PER_TASK=8 bash demos/launch_demo_02_interactive.sh
```

Those wrappers do the `salloc` step for you and then run the demo on the
allocated node.

To run the CPU demo suite in one interactive allocation:

```bash
cd examples/rose_orchestration
TIME=04:00:00 CPUS_PER_TASK=16 bash demos/launch_all_demos_cpu_interactive.sh
```

To run Demo 3 with an interactive allocation:

```bash
cd examples/rose_orchestration
TIME=02:00:00 CPUS_PER_TASK=16 bash demos/run_demo_interactive.sh demos/demo_03_resource_aware_search.sh
```

To run Demo 3 as a Slurm job submission:

```bash
cd examples/rose_orchestration
TIME=02:00:00 CPUS_PER_TASK=16 bash demos/run_demo_03_slurm.sh --cpus-per-trial 4
```

That submits an `sbatch` job and lets the demo size its parallel trial budget
from the CPUs granted to the job.

To record scaling curves for Demo 3 across several CPU allocations:

```bash
cd examples/rose_orchestration
CPU_SWEEP="8 16 32" CPUS_PER_TRIAL=4 TIME=02:00:00 bash demos/run_demo_03_scaling_sweep.sh
```

Then aggregate the completed runs into JSON, CSV, and Markdown:

```bash
python demos/demo_03_scaling_report.py
```

Then turn the scaling CSV into PNG plots:

```bash
python demos/plot_demo_03_scaling.py
```

To run the GPU-aware demo on an interactive GPU node:

```bash
cd examples/rose_orchestration
TIME=02:00:00 CPUS_PER_TASK=8 GPUS_PER_NODE=1 bash demos/launch_demo_04_gpu_interactive.sh
```

To run the GPU demo suite launcher directly:

```bash
cd examples/rose_orchestration
TIME=04:00:00 CPUS_PER_TASK=8 GPUS_PER_NODE=1 bash demos/launch_all_demos_gpu_interactive.sh
```

## Demo definitions

### Demo 1: Better surrogate selection

Command:

```bash
python example_04_parallel_model_race.py \
  --dataset m3dc1 \
  --max-iter 1 \
  --candidates rf,mlp
```

Reasoning:

- Both candidates see the full complete-case M3DC1 table.
- SURGE then applies its internal `60 / 20 / 20` split.
- The default full-dataset race uses `rf` and `mlp` because the current Gaussian
  process workflows are intentionally capped on M3DC1 and would not be a fair
  "entire sample set" comparison.
- ROSE `ParallelActiveLearner` and the Rhapsody backend execute the race
  concurrently, while SURGE handles training for each candidate.

### Demo 2: Smarter campaign control

Command:

```bash
python example_05_uq_guided_mlp_ensemble.py \
  --dataset m3dc1 \
  --max-iter 1 \
  --ensemble-n 3 \
  --sklearn-mlp-max-iter 250 \
  --uncertainty-threshold 0.015
```

Reasoning:

- The M3DC1 complete-case table is used as the full outer dataset.
- SURGE applies the internal `60 / 20 / 20` split once for the run.
- SURGE trains an sklearn MLP ensemble on that split.
- SURGE writes validation uncertainty artifacts.
- ROSE `SeqUQLearner` reads those artifacts and monitors the uncertainty threshold.

This is not acquisition yet. It is a control-plane demo that shows the loop can
make decisions from UQ signals emitted by SURGE, even when the model is trained
on the full available dataset.

### Demo 3: Resource-aware surrogate search

Command:

```bash
python demos/demo_03_resource_aware_search.py --cpus-per-trial 4
```

Reasoning:

- the script reads `SLURM_CPUS_PER_TASK` from your interactive allocation
- it converts that into a parallel trial budget: `parallel_trials = allocated_cpus / cpus_per_trial`
- it launches one or more RF baseline trials plus randomized MLP trials on the
  full M3DC1 table
- all trials race concurrently, and the script ranks them by validation `R2`

This is the first demo here that actually changes campaign width based on the
resources you allocate.

Each Demo 3 run now records:

- allocated CPUs
- CPUs per trial
- parallel trial width
- completed trial count
- wall time
- trials/hour
- best validation `R2`
- best validation `RMSE`

### Demo 4: GPU-aware surrogate search

Command:

```bash
python demos/demo_04_gpu_aware_search.py --allow-cpu-fallback --max-trials 1
```

Reasoning:

- the script detects visible GPUs from `CUDA_VISIBLE_DEVICES` or `torch.cuda`
- it launches one PyTorch MLP trial per visible GPU
- each subprocess is bound to one GPU slot and requests `resources.device=cuda:0`
  inside that slot-local environment
- if no GPU is visible, `--allow-cpu-fallback` verifies the same code path on CPU

GPU-capable models in SURGE today:

- `pytorch.mlp`
- `gpflow.gpr`
- `gpflow.multi_kernel`

CPU-only models in SURGE today:

- `sklearn.random_forest`
- `sklearn.mlp`
- `sklearn.gpr`

This demo uses `pytorch.mlp` because its device placement is explicit and
verifiable through SURGE's `resources_used` summary.

## Artifact visualization

After examples and demos finish, generate PNG summaries from saved artifacts:

```bash
cd examples/rose_orchestration
python viz_examples_and_demos.py
```

Plots are written under `examples/rose_orchestration/viz/`.

## What the UQ means

For Demo 2, SURGE writes ensemble uncertainty on the validation split:

- `mean_std`: average predictive standard deviation across validation points
- `max_std`: largest predictive standard deviation across validation points

In plain terms:

- low `mean_std` means the ensemble members mostly agree
- high `mean_std` means the ensemble members disagree more
- large `max_std` highlights at least one validation region where the model is not stable

## What ROSE can do with it

In the current implementation, ROSE uses that UQ signal for **monitoring and
decision logic**, not yet for full acquisition. Concretely, ROSE can:

- stop a campaign when uncertainty drops below a target
- flag that the current surrogate is still not trustworthy enough
- compare campaign states or retraining attempts using the same UQ metric

The current Demo 2 therefore demonstrates **uncertainty-aware control**, not
"using less data". The earlier staged version used less data to make the control
signal evolve across iterations; this full-dataset version keeps the data fixed
and shows the same ROSE-to-SURGE contract on the complete M3DC1 table.

## Recorded results

See [RESULTS.md](RESULTS.md) for the latest recorded M3DC1 run.
