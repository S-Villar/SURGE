# SURGE + ROSE + Rhapsody reference demos

This directory is the `radical-integration` reference integration for:

- `SURGE` as the surrogate training engine
- `ROSE` as the outer loop controller
- `Rhapsody` + `radical.asyncflow` as the execution backend

The goal is to show a clean composition, not just a pile of wrappers.

## Setup, Examples, and Demos

### Setup

- `SURGE` provides the surrogate workflows, dataset splitting, preprocessing, model training, metrics, uncertainty outputs, and run artifacts.
- `ROSE` provides learner orchestration, stopping criteria, monitoring, and outer-loop control.
- `Rhapsody` plus `radical.asyncflow` provide the execution backend used to run the ROSE tasks.

### Examples

- **Example 1**: `ROSE` uses a sequential learner and `Rhapsody/radical.asyncflow` in-process tasks to repeatedly call a `SURGE` workflow, where `SURGE` performs dataset splitting, preprocessing, model training, metrics, and artifact writing.
- **Example 2**: `ROSE` uses sequential subprocess-based orchestration through `Rhapsody/radical.asyncflow` to launch distinct `SURGE` workflows (`rf`, `mlp`, `sklearn.gpr`, `gpflow.gpr`), with `SURGE` handling the actual model fit, validation metrics, and run artifacts in each external process.
- **Example 3**: `ROSE` uses stop criteria and learner-state tracking with in-process execution to explore multiple `SURGE` sklearn MLP architectures, while `SURGE` trains each candidate and returns the validation scores that drive the HPO loop.
- **Example 4**: `ROSE ParallelActiveLearner` and `Rhapsody/radical.asyncflow` run several `SURGE` model families concurrently, with `SURGE` training each surrogate independently and producing the metrics used to rank them.
- **Example 5**: `ROSE` uses a UQ-oriented learner flow to monitor uncertainty signals, while `SURGE` trains an sklearn MLP ensemble, computes predictive uncertainty, and writes the UQ artifacts that `ROSE` reads.

### Demos

- **Demo 1**: `ROSE` and `Rhapsody/radical.asyncflow` orchestrate a parallel full-dataset surrogate race, while `SURGE` trains and evaluates the candidate models and provides the metrics used for model selection.
- **Demo 2**: `ROSE` monitors campaign quality through uncertainty-aware control logic, while `SURGE` trains the full-dataset ensemble surrogate and emits the validation UQ artifacts that drive that decision signal.
- **Demo 3**: A resource-aware orchestration layer scales the number of concurrent `SURGE` trials with allocated CPUs, and `SURGE` trains each trial model and returns the quality metrics used to compare the search outcomes.
- **Demo 4**: A device-aware orchestration layer maps trials onto visible accelerators, while `SURGE` runs the GPU-capable PyTorch MLP backend and reports the effective device and training artifacts.

### RADICAL-Cybertools note

These examples use `radical.asyncflow` via `Rhapsody` as the execution backend.

## Demo wrappers

For a shorter, presentation-oriented entry point on M3DC1, use the scripts under
[demos/](demos/README.md):

- `demos/demo_01_better_surrogate_selection.sh`
- `demos/demo_02_smarter_campaign_control.sh`

Recorded outputs for those two runs are in [demos/RESULTS.md](demos/RESULTS.md).

## What is implemented

Examples 1-4 use the same four ROSE stages:

1. `simulation`
2. `training`
3. `active_learn`
4. `criterion`

`training` is always real SURGE training through `run_surrogate_workflow(...)` in
[surge_train.py](surge_train.py). The examples differ in execution style and policy:

| Example | Purpose | Execution | Policy |
|---|---|---|---|
| [example_01_rose_inprocess_verbose.py](example_01_rose_inprocess_verbose.py) | Canonical reference workflow | in-process Python tasks | campaign loop with fixed SURGE workflow family |
| [example_02_rose_subprocess_shell.py](example_02_rose_subprocess_shell.py) | Same logic, external process boundaries | subprocess stages | same campaign policy as Example 1 |
| [example_03_rose_mlp_random_r2_stop.py](example_03_rose_mlp_random_r2_stop.py) | Explicit toy HPO loop | in-process Python tasks | random search over MLP architectures with ROSE stop criterion on `val_r2` |
| [example_04_parallel_model_race.py](example_04_parallel_model_race.py) | Parallel model evaluation | `ParallelActiveLearner` + process-isolated SURGE training | train several SURGE model families concurrently and rank by validation score |
| [example_05_uq_guided_mlp_ensemble.py](example_05_uq_guided_mlp_ensemble.py) | UQ-guided decision monitoring | `SeqUQLearner` + thread backend | train SURGE MLP ensemble, read validation uncertainty, and let ROSE monitor a UQ threshold |

## What is not implemented

- Example 1 and Example 2 do **not** implement a real acquisition function yet.
- Example 3 is **random search**, not Bayesian optimization or a full HPO service.
- Example 4 compares model families concurrently; it is not yet an adaptive scheduler.
- Example 5 monitors SURGE ensemble uncertainty; it does not yet select new simulation points from an unlabeled candidate pool.
- The branch is intended for integration and workflow development, not benchmark-grade HPO.

## State layout

Each iteration writes explicit state under:

```text
examples/rose_orchestration/workspace/example_01/iter_000/
examples/rose_orchestration/workspace/example_04/0_rf/iter_000/
...
```

Per-iteration files include:

- `simulation.json`
- `surge_metrics.json`
- `active.json`
- `criterion.json`
- `candidate.json` for Example 3
- `prediction.json` and `uncertainty.json` for Example 5

The old `workspace/last_*.json` files are still written as convenience aliases, but
the canonical state is now the per-iteration directory.

## Dataset semantics

The examples hand SURGE a tabular dataset for each outer iteration. SURGE then performs
its own train/validation/test split internally.

The workflow YAMLs in this directory now use:

- `train = 0.6`
- `val = 0.2`
- `test = 0.2`

That is implemented through:

- `test_fraction: 0.2`
- `val_fraction: 0.2`

inside the local `workflow_*.yaml` files.

### Example 1 and 2

These are reference orchestration demos.

- `m3dc1` is the default dataset.
- `m3dc1` uses the full available complete-case pool by default.
- `synthetic` remains available for fast smoke tests.
- `--growing-pool` restores the staged campaign behavior (`600`, `1200`, `1800`, ... for M3DC1).

Default workflow family is `rf`. You can also use `mlp`, `gpr`, and `gpflow_gpr`.
For `m3dc1`, the Gaussian-process workflows default to a capped 600-row subset unless
you explicitly opt into `--growing-pool`; full 9891-row GP training is not a practical default.

### Example 3

This is a fixed-dataset HPO-style demo.

- for `m3dc1`, each candidate sees the same shuffled complete-case table
- SURGE then applies its internal 60/20/20 split
- the outer loop varies only the MLP architecture and stop condition

### Example 4

This demonstrates Rhapsody/ROSE parallel orchestration.

- ROSE `ParallelActiveLearner` launches multiple sequential learners.
- Each learner evaluates a different SURGE workflow family, for example `rf` and `mlp`.
- Each SURGE workflow is launched in its own Python process because the current SURGE workflow logger redirects process-global stdout while writing run logs.
- States stream back as each learner completes, and the script ranks candidates by validation `R2`.

### Example 5

This demonstrates the UQ path.

- SURGE trains an sklearn MLP ensemble with `ensemble_n > 1`.
- SURGE writes `*_val_uq.json` containing ensemble mean/std on the validation split.
- ROSE `SeqUQLearner` reads the mean validation standard deviation and monitors a UQ threshold.

## Installation and setup

### Recommended path

Use the dedicated setup script:

```bash
cd examples/rose_orchestration
./setup_surge_rose_demo_env.sh
```

That script:

1. creates or reuses a venv on CFS
2. installs this SURGE checkout editable
3. installs `requirements-rose-demo.txt`
4. installs your `ROSE` checkout editable
5. verifies imports for `surge`, `rose`, `radical.asyncflow`, `rhapsody`, `pandas`, and `pyarrow`

Default assumptions:

- base Python: `/global/common/software/m3716/asvillar/envs/surge/bin/python`
- venv: `/global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos`
- `ROSE_ROOT`: `/global/homes/a/asvillar/src/ROSE`

Override them with environment variables before running the script:

```bash
export ROSE_ROOT=/path/to/ROSE
export SURGE_ROSE_DEMO_VENV=/path/to/venv
export SURGE_PY=/path/to/python
./setup_surge_rose_demo_env.sh
```

### Manual activation after setup

```bash
source /global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos/bin/activate
export SURGE_ROOT=/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/SURGE
export PYTHONPATH="$SURGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SURGE_ROOT/examples/rose_orchestration"
```

You can also use:

```bash
source ./activate_surge_rose_cfs_venv.sh
```

## M3DC1 dataset

Expected location inside the SURGE checkout:

```bash
$SURGE_ROOT/data/datasets/M3DC1/sparc-m3dc1-D1.pkl
```

If your source dataset lives elsewhere, symlink it:

```bash
ln -sf /global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl \
  "$SURGE_ROOT/data/datasets/M3DC1/sparc-m3dc1-D1.pkl"
```

## Run commands

Examples 1, 3, 4, and 5 default to a refreshing `tqdm` progress display. Detailed
ROSE/SURGE output is written under `workspace/example_XX/execution.log`. Use
`--no-live-progress` when you want the older line-oriented trace in the terminal.

### Example 1

```bash
python example_01_rose_inprocess_verbose.py --max-iter 2
python example_01_rose_inprocess_verbose.py --workflow-family mlp --max-iter 2
python example_01_rose_inprocess_verbose.py --workflow-family gpr --dataset synthetic --max-iter 1
python example_01_rose_inprocess_verbose.py --workflow-family gpflow_gpr --dataset synthetic --max-iter 1
python example_01_rose_inprocess_verbose.py --dataset synthetic --max-iter 2
python example_01_rose_inprocess_verbose.py --growing-pool --max-iter 3
```

### Example 2

```bash
python example_02_rose_subprocess_shell.py --max-iter 2
python example_02_rose_subprocess_shell.py --dataset synthetic --max-iter 2
python example_02_rose_subprocess_shell.py --dataset synthetic --workflow-sequence rf,mlp,gpr,gpflow_gpr
```

The `--workflow-sequence` form is the explicit subprocess sweep over:

- `sklearn.random_forest`
- `sklearn.mlp`
- `sklearn.gpr`
- `gpflow.gpr`

### Example 3

```bash
python example_03_rose_mlp_random_r2_stop.py --dataset synthetic --max-iter 8
python example_03_rose_mlp_random_r2_stop.py --max-iter 12 --r2-threshold 0.9 --r2-operator ">="
```

### Example 4

```bash
python example_04_parallel_model_race.py --max-iter 1 --candidates rf,mlp
python example_04_parallel_model_race.py --dataset synthetic --max-iter 1 --candidates gpr,gpflow_gpr
python example_04_parallel_model_race.py --dataset synthetic --max-iter 1 --candidates rf,mlp
```

Default live output is a refreshing progress bar plus a final ranking:

```text
SURGE candidates: 100%|██████████| 2/2 [..., learner=mlp, r2=0.9008, rmse=0.1957]

Example 4 summary
rank  learner   workflow         val_r2    val_rmse  run_tag
   1  mlp       mlp             0.90078    0.195668  rose_parallel_1_mlp_mlp_iter_0
   2  rf        rf              0.87164    0.222551  rose_parallel_0_rf_rf_iter_0
```

### Example 5

```bash
python example_05_uq_guided_mlp_ensemble.py --max-iter 1 --ensemble-n 3
python example_05_uq_guided_mlp_ensemble.py --dataset synthetic --max-iter 1 --ensemble-n 2 --sklearn-mlp-max-iter 80
```

Default live output shows the uncertainty monitor in the progress postfix:

```text
SURGE ensemble UQ: 100%|██████████| 1/1 [..., r2=0.4370, mean_std=0.22848, target=0.01500]
```

## How the layers fit together

### ROSE

Owns the outer loop and the stop criterion.

### Rhapsody + radical.asyncflow

Own the execution backend used by ROSE:

- `ThreadPoolExecutor` in Examples 1, 3, and 5
- `ProcessPoolExecutor` in Example 2
- `ParallelActiveLearner` in Example 4, with SURGE training subprocesses for safe concurrent workflow execution

### SURGE

Owns data splitting, scaling, fitting, metrics, and run artifacts. The training wrapper
in [surge_train.py](surge_train.py) loads a workflow spec, resolves paths, calls
`run_surrogate_workflow(...)`, and writes the iteration metrics that ROSE reads.

## Practical interpretation

- Example 1 is the canonical integration example.
- Example 2 is the same pattern with explicit process boundaries.
- Example 3 is a toy HPO loop with explicit candidate state and a ROSE stop criterion.
- Example 4 shows multiple SURGE surrogate candidates being orchestrated concurrently.
- Example 5 connects SURGE ensemble UQ outputs to ROSE UQ decision monitoring.

If you want one place to start refining the branch, start with Example 1.

## Slurm launch helpers

The scripts mirror the allocation style used under `/global/cfs/projectdirs/amsc007/demo`.

Interactive allocation:

```bash
./run_rose_example_interactive.sh example_01_rose_inprocess_verbose.py --max-iter 2
```

Batch job:

```bash
sbatch --export=ALL,EXAMPLE=example_01_rose_inprocess_verbose.py,ARGS="--max-iter 2" \
  run_rose_example_batch.sh
```

Useful overrides:

```bash
TIME=04:00:00 CPUS_PER_TASK=16 \
  ./run_rose_example_interactive.sh example_03_rose_mlp_random_r2_stop.py --max-iter 12

sbatch --time=04:00:00 \
  --export=ALL,EXAMPLE=example_05_uq_guided_mlp_ensemble.py,ARGS="--max-iter 1 --ensemble-n 3" \
  run_rose_example_batch.sh
```

Run the whole example suite on one interactive CPU node:

```bash
TIME=04:00:00 CPUS_PER_TASK=16 ./launch_all_examples_interactive.sh
```

Generate PNG summaries from saved artifacts:

```bash
python viz_examples_and_demos.py
```
