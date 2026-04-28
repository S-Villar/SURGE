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

## 3. Link the M3DC1 dataset if needed

```bash
ln -sf /global/homes/a/asvillar/src/SURGEROSE/data/datasets/M3DC1/sparc-m3dc1-D1.pkl \
  "$SURGE_ROOT/data/datasets/M3DC1/sparc-m3dc1-D1.pkl"
```

## 4. Verify imports and wiring

```bash
python verify_demo_env.py
```

## 5. Run the examples

Examples 1, 3, 4, and 5 show refreshing progress bars by default and write detailed
logs to `workspace/example_XX/execution.log`. Add `--no-live-progress` for the
line-oriented terminal trace.

Example 1:

```bash
python example_01_rose_inprocess_verbose.py --max-iter 2
```

Example 2:

```bash
python example_02_rose_subprocess_shell.py --max-iter 2
```

Example 3:

```bash
python example_03_rose_mlp_random_r2_stop.py --max-iter 12 --r2-threshold 0.9 --r2-operator ">="
```

Example 4:

```bash
python example_04_parallel_model_race.py --max-iter 1 --candidates rf,mlp
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
