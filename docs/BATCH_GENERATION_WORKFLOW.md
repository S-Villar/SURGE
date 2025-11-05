# M3DC1 Batch Generation and Launching Workflow

This document explains how to generate parameterized batches of M3DC1 simulations and launch them on SLURM clusters.

## Overview

The workflow consists of three main steps:

1. **Configure**: Create a YAML configuration file defining parameters, ranges, and sampling strategy
2. **Generate**: Use `surge_batch_setup.py` to create batch directories with parameterized simulation cases
3. **Launch**: Submit SLURM job arrays to run all simulations in parallel

## Quick Start Example

```bash
# 1. Create config file (examples/batch_setup_m3dc1_scratch_logspace.yml)
# 2. Generate batch
python surge_batch_setup.py --config examples/batch_setup_m3dc1_scratch_logspace.yml

# 3. Navigate to batch and submit job
cd ${SURGE_SCRATCH}/m3dc1_batch/batch_N
sbatch batchjob.perlmutter
```

## Prerequisites

- SURGE environment activated (`conda activate surge`)
- Access to source equilibria (`batch_0` with equilibrium cases)
- SLURM cluster access (Perlmutter/NERSC)
- Sufficient disk quota in scratch filesystem

## Step-by-Step Workflow

### Step 1: Create Configuration File

Create a YAML configuration file specifying:

- **Output location**: Where to create batches (scratch or project directory)
- **Parameters to vary**: Which simulation parameters to modify
- **Parameter ranges**: Min/max values for each parameter
- **Sampling method**: LHS (Latin Hypercube) or random
- **Log-space sampling**: Optional concentration of samples at low/high values
- **Number of runs**: How many parameter sets to generate

**Example Configuration (`batch_setup_m3dc1_scratch_logspace.yml`):**

```yaml
# Output location (scratch filesystem)
out_root: ${SURGE_SCRATCH}/m3dc1_batch

# Template input file
inpfile: /global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC/batch_0/run1/C1input

# Parameters to vary
params: [ntor, pscale, batemanscale]

# Ranges for each parameter (min, max)
ranges:
  - [1, 20]         # ntor: toroidal mode number (integer)
  - [0.5, 2.0]      # pscale: pressure scaling (float)
  - [0.1, 1.0]      # batemanscale: Bateman scaling (float)

# Which parameters are integers
integer_mask: [true, false, false]

# Log-space sampling: concentrate samples at low values for ntor
log_space: [true, false, false]

# Number of parameter sets (runs) to generate
nsamples: 10

# Sampling method
spl: lhs  # Latin Hypercube Sampling

# Random seed for reproducibility
seed: 42

# Equilibria mode: fixed means all equilibria in a run share same parameters
equilibria: fixed

# Source equilibria directory
eqsetpath: /global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC/batch_0

# Options
use_python_replacement: true
confirm_dirs: false
save_plots: true
```

### Step 2: Generate Batch

Run the batch generator:

```bash
python surge_batch_setup.py --config examples/batch_setup_m3dc1_scratch_logspace.yml
```

**What happens:**
- Creates a new `batch_N` directory in the specified `out_root`
- Generates parameter samples using the specified method (LHS or random)
- Creates `run1/`, `run2/`, ..., `runN/` directories
- Copies all equilibrium cases (`sparc_*`) from source to each run
- Modifies `C1input` files in each case with sampled parameter values
- Saves `samples.npz` with all parameter values
- Creates `meta.json` and `README.md` with batch metadata

**Output:**
```
${SURGE_SCRATCH}/m3dc1_batch/batch_3/
├── run1/
│   ├── sparc_1300/
│   │   ├── C1input          # Modified with run1 parameters
│   │   ├── geqdsk
│   │   ├── equilibrium.h5
│   │   └── ...
│   ├── sparc_1400/
│   │   └── ...
│   └── ... (101 equilibria total)
├── run2/
│   └── ... (same structure, different parameters)
├── ...
├── run10/
│   └── ...
├── samples.npz              # Parameter samples array
├── meta.json                # Batch metadata
├── README.md                # Batch documentation
└── sampling_plot.png        # Distribution visualization (if save_plots=true)
```

### Step 3: Prepare Batch Job Script

The batch generator doesn't automatically create `batchjob.perlmutter`. Create it manually with:

**Key features:**
- SLURM job array: `--array=1-N` (one task per run)
- Shared mesh symlink: Task 1 creates symlink, others wait
- Processes all `sparc_*` cases in each run directory
- Removes existing outputs to force fresh runs

**Example `batchjob.perlmutter`:**

```bash
#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH -L SCRATCH
#SBATCH -C cpu
#SBATCH -A mp288
#SBATCH -o M3DC1log.o%j.%a
#SBATCH -e M3DC1log.e%j.%a
#SBATCH -J m3dc1_batch
#SBATCH --array=1-10  # Adjust to number of runs

BATCH_DIR="${SLURM_SUBMIT_DIR}"
RUN_NUM=$SLURM_ARRAY_TASK_ID
RUN_DIR="${BATCH_DIR}/run${RUN_NUM}"

# Setup shared mesh (symlink, not copy - saves disk space)
if [ "$RUN_NUM" -eq 1 ]; then
    MESH_SRC="/global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC/batch_0/run1/mesh"
    MESH_DST="${BATCH_DIR}/mesh"
    ln -sfn "$MESH_SRC" "$MESH_DST"
    touch "${BATCH_DIR}/.mesh_ready"
fi

# Wait for mesh (other tasks)
if [ "$RUN_NUM" -ne 1 ]; then
    while [ ! -f "${BATCH_DIR}/.mesh_ready" ]; do sleep 1; done
fi

# Process all cases in this run
cd "$RUN_DIR"
for sparc_dir in sparc_*; do
    cd "$sparc_dir"
    rm -f finished C1.h5 started  # Force fresh run
    $M3DC1_MPIRUN -n $SLURM_NTASKS m3dc1_2d_complex -pc_factor_mat_solver_type mumps
    touch finished
    cd ..
done
```

### Step 4: Launch Job

```bash
cd ${SURGE_SCRATCH}/m3dc1_batch/batch_3
sbatch batchjob.perlmutter
```

**Output:**
```
Submitted batch job 44853645
```

### Step 5: Monitor Progress

**Check job status:**
```bash
squeue -j 44853645
```

**Count completed cases:**
```bash
cd ${SURGE_SCRATCH}/m3dc1_batch/batch_3
find . -name "finished" | wc -l
```

**View logs:**
```bash
# Output logs
cat M3DC1log.o44853645.1
cat M3DC1log.o44853645.2

# Error logs
cat M3DC1log.e44853645.1
```

## Example Results

### Batch Configuration
- **Batch**: `batch_3`
- **Location**: `${SURGE_SCRATCH}/m3dc1_batch/batch_3`
- **Runs**: 10
- **Equilibria per run**: 101
- **Total simulations**: 1,010 cases
- **Sampling**: LHS with log-space for `ntor`

### Parameter Distribution (Log-Space Sampling)

**ntor Distribution:**
```
Total runs: 10
ntor range: 1 - 18
ntor mean: 6.50

Distribution by value:
  ntor= 1:  1 runs ( 10.0%)
  ntor= 2:  2 runs ( 20.0%)
  ntor= 3:  1 runs ( 10.0%)
  ntor= 4:  1 runs ( 10.0%)
  ntor= 6:  1 runs ( 10.0%)
  ntor= 7:  1 runs ( 10.0%)
  ntor= 9:  1 runs ( 10.0%)
  ntor=13:  1 runs ( 10.0%)
  ntor=18:  1 runs ( 10.0%)

Low values (1-5):   5 runs ( 50.0%)
High values (15-20): 1 runs ( 10.0%)
```

**Key Observation:** With log-space sampling enabled, 50% of runs have low `ntor` values (1-5), while only 10% have high values (15-20). This concentrates sampling where it's most needed.

### Sample Parameter Values

**First 5 runs:**
| Run | ntor | pscale | batemanscale |
|-----|------|--------|--------------|
| 1   | 1    | 1.78   | 0.85         |
| 2   | 2    | 0.53   | 0.33         |
| 3   | 2    | 1.43   | 0.74         |
| 4   | 3    | 0.81   | 0.22         |
| 5   | 4    | 0.65   | 0.59         |

### Job Execution

**Job Details:**
```
Job ID: 44853645
Array: 1-10
Status: Pending → Running → Completed
Resources: 8 nodes × 12 tasks/node = 96 tasks
Time limit: 8 hours
```

**Progress Tracking:**
```bash
# Check completed cases
$ find . -name "finished" | wc -l
750  # 750 out of 1,010 cases completed

# Check specific run progress
$ ls run1/sparc_*/finished | wc -l
85   # 85 out of 101 cases in run1 completed
```

## Key Features

### 1. Log-Space Sampling

For parameters spanning wide ranges (e.g., `ntor` from 1 to 20), log-space sampling concentrates more samples at lower values:

- **Linear sampling**: Uniform distribution across range
- **Log-space sampling**: More samples at lower values, fewer at higher values

**Use case:** When you need more data at low parameter values for better model training.

**Configuration:**
```yaml
log_space: [true, false, false]  # Log-space for first parameter only
```

### 2. Shared Mesh Symlink

Instead of copying mesh files (which can be large), use a symlink:

- **Saves disk space**: Mesh files stored once, referenced by all runs
- **Faster setup**: Symlink creation is instant vs. copying many files
- **Automatic**: Task 1 creates symlink, others wait for it

### 3. Fresh Runs

Each simulation case removes existing outputs before running:

```bash
rm -f finished C1.h5 started
```

Ensures clean execution and prevents reusing stale results.

### 4. Job Arrays

SLURM job arrays parallelize across runs:

- Each array task processes one run directory
- All tasks run in parallel (subject to resource availability)
- Separate log files for each task: `M3DC1log.o<JOBID>.<ARRAY_ID>`

## Troubleshooting

### Issue: "out of range" error

**Cause:** Batch directory not found or incorrect `--array` range.

**Solution:** Check that `BATCH_DIR` is set correctly in batchjob script:
```bash
BATCH_DIR="${SLURM_SUBMIT_DIR}"
```

### Issue: Mesh symlink not found

**Cause:** Task 1 hasn't created symlink yet, or source mesh doesn't exist.

**Solution:** Verify mesh source exists:
```bash
ls /global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC/batch_0/run1/mesh
```

### Issue: Parameter values not applied

**Cause:** Input file modification failed or wrong parameter names.

**Solution:** Check parameter names match exactly:
```bash
grep "ntor" run1/sparc_1300/C1input
```

### Issue: Job stuck in pending

**Cause:** Insufficient resources or queue limits.

**Solution:** 
- Check queue limits: `squeue -u $USER`
- Reduce node count for debug queue
- Check priority: `squeue -j <JOBID> -o "%.P %.Q"`

## Best Practices

1. **Start small**: Test with 3-5 runs before launching large batches
2. **Use scratch**: Generate batches in scratch filesystem to avoid quota issues
3. **Verify distribution**: Check `samples.npz` to confirm parameter distribution
4. **Monitor resources**: Track disk usage and job runtime
5. **Save configs**: Keep YAML config files for reproducibility
6. **Use log-space**: For wide parameter ranges, log-space sampling often improves results

## File Structure Reference

```
batch_N/
├── run1/
│   ├── sparc_1300/
│   │   ├── C1input          # Modified with run1 parameters
│   │   ├── geqdsk
│   │   ├── equilibrium.h5
│   │   ├── finished         # Created when case completes
│   │   └── C1.h5            # Output file
│   └── ... (other equilibria)
├── run2/
│   └── ...
├── mesh -> /path/to/source/mesh  # Symlink to shared mesh
├── samples.npz              # Parameter samples (numpy array)
├── meta.json                # Batch metadata
├── README.md                # Batch documentation
├── batchjob.perlmutter      # SLURM job script
├── M3DC1log.o<jobid>.<task> # Output logs
└── M3DC1log.e<jobid>.<task> # Error logs
```

## Summary

The workflow enables efficient generation and execution of large-scale parameter studies:

1. **Configure** parameters and sampling strategy in YAML
2. **Generate** batch directories with parameterized cases
3. **Launch** parallel SLURM job arrays
4. **Monitor** progress and collect results

With log-space sampling, shared mesh symlinks, and job arrays, you can efficiently explore large parameter spaces while optimizing resource usage.
