# Quickstart

This is the shortest reproducible path for the `radical-integration` demos.

## 1. Build the demo environment

```bash
cd /global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/SURGE/examples/rose_orchestration
./setup_surge_rose_demo_env.sh
```

If your ROSE checkout is not at `/global/homes/a/asvillar/src/ROSE`:

```bash
export ROSE_ROOT=/path/to/ROSE
./setup_surge_rose_demo_env.sh
```

## 2. Activate it

```bash
source /global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos/bin/activate
export SURGE_ROOT=/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/SURGE
export PYTHONPATH="$SURGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SURGE_ROOT/examples/rose_orchestration"
```

Or:

```bash
source ./activate_surge_rose_cfs_venv.sh
```

## 3. Set the M3DC1 dataset path if needed

```bash
export M3DC1_SOURCE=/global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl
```

## 4. Verify imports and wiring

```bash
python verify_demo_env.py
```

## 5. Run the examples

### What each example and demo shows

- **Example 1**: `ROSE` uses a sequential learner and `Rhapsody/radical.asyncflow` in-process tasks to repeatedly call a `SURGE` workflow, where `SURGE` performs dataset splitting, preprocessing, model training, metrics, and artifact writing.
- **Example 2**: `ROSE` uses sequential subprocess-based orchestration through `Rhapsody/radical.asyncflow` to launch distinct `SURGE` workflows (`rf`, `mlp`, `sklearn.gpr`, `gpflow.gpr`), with `SURGE` handling the actual model fit, validation metrics, and run artifacts in each external process.
- **Example 3**: `ROSE` uses stop criteria and learner-state tracking with in-process execution to explore multiple `SURGE` sklearn MLP architectures, while `SURGE` trains each candidate and returns the validation scores that drive the HPO loop.
- **Example 4**: `ROSE ParallelActiveLearner` and `Rhapsody/radical.asyncflow` run several `SURGE` model families concurrently, with `SURGE` training each surrogate independently and producing the metrics used to rank them.
- **Example 5**: `ROSE` uses a UQ-oriented learner flow to monitor uncertainty signals, while `SURGE` trains an sklearn MLP ensemble, computes predictive uncertainty, and writes the UQ artifacts that `ROSE` reads.
- **Demo 1**: `ROSE` and `Rhapsody/radical.asyncflow` orchestrate a parallel full-dataset surrogate race, while `SURGE` trains and evaluates the candidate models and provides the metrics used for model selection.
- **Demo 2**: `ROSE` monitors campaign quality through uncertainty-aware control logic, while `SURGE` trains the full-dataset ensemble surrogate and emits the validation UQ artifacts that drive that decision signal.
- **Demo 3**: A resource-aware orchestration layer scales the number of concurrent `SURGE` trials with allocated CPUs, and `SURGE` trains each trial model and returns the quality metrics used to compare the search outcomes.
- **Demo 4**: A device-aware orchestration layer maps trials onto visible accelerators, while `SURGE` runs the GPU-capable PyTorch MLP backend and reports the effective device and training artifacts.
- **RADICAL-Cybertools note**: These examples use `radical.asyncflow` via `Rhapsody` as the execution backend.

Examples 1, 3, 4, and 5 show refreshing progress bars by default and write detailed
logs to `workspace/example_XX/execution.log`. Add `--no-live-progress` for the
line-oriented terminal trace.

Example 1:

```bash
python example_01_rose_inprocess_verbose.py --max-iter 2
python example_01_rose_inprocess_verbose.py --workflow-family gpr --dataset synthetic --max-iter 1
python example_01_rose_inprocess_verbose.py --workflow-family gpflow_gpr --dataset synthetic --max-iter 1
```

Example 2:

```bash
python example_02_rose_subprocess_shell.py --max-iter 2
python example_02_rose_subprocess_shell.py --dataset synthetic --workflow-sequence rf,mlp,gpr,gpflow_gpr
```

Example 3:

```bash
python example_03_rose_mlp_random_r2_stop.py --max-iter 12 --r2-threshold 0.9 --r2-operator ">="
```

Example 4:

```bash
python example_04_parallel_model_race.py --max-iter 1 --candidates rf,mlp
python example_04_parallel_model_race.py --dataset synthetic --max-iter 1 --candidates gpr,gpflow_gpr
```

Example 5:

```bash
python example_05_uq_guided_mlp_ensemble.py --max-iter 1 --ensemble-n 3
```

## Notes

- Example 1 is the reference workflow.
- Example 2 is the same logic with subprocess boundaries.
- Example 3 is a toy HPO loop over MLP architectures with a ROSE stop criterion.
- Example 4 uses ROSE `ParallelActiveLearner` to evaluate multiple SURGE model candidates concurrently.
- Example 4 runs each SURGE training workflow in a separate Python process; that is intentional because the current SURGE workflow logger is not thread-local.
- Example 5 uses ROSE's UQ learner flow to monitor SURGE MLP ensemble uncertainty.
- `m3dc1` is the default dataset. Use `--dataset synthetic` for faster smoke tests.
- Examples 1 and 2 use the full M3DC1 pool by default. Use `--growing-pool` for staged campaign growth.
- For `gpr` and `gpflow_gpr` on `m3dc1`, the examples cap the default dataset slice at 600 rows; full-row Gaussian-process training is not the default.
- All local workflow YAMLs use a 60/20/20 train/val/test split through SURGE.

## Slurm Launch

Interactive allocation:

```bash
./run_rose_example_interactive.sh example_01_rose_inprocess_verbose.py --max-iter 2
```

Batch job:

```bash
sbatch --export=ALL,EXAMPLE=example_01_rose_inprocess_verbose.py,ARGS="--max-iter 2" \
  run_rose_example_batch.sh
```

Override allocation details with environment variables, for example:

```bash
TIME=04:00:00 CPUS_PER_TASK=16 \
  ./run_rose_example_interactive.sh example_03_rose_mlp_random_r2_stop.py --max-iter 12
```

Run the entire example suite on one interactive CPU node:

```bash
TIME=04:00:00 CPUS_PER_TASK=16 ./launch_all_examples_interactive.sh
```

Generate visualization PNGs from saved example/demo artifacts:

```bash
python viz_examples_and_demos.py
```
