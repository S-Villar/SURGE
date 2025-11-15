# SURGE Batch Dataset Generation and Launching User Manual

This manual covers how to generate batches of M3DC1 simulation cases and launch them on SLURM clusters.

## Table of Contents

1. [Batch Generation](#batch-generation)
2. [Batch Launching](#batch-launching)
3. [Verification](#verification)
4. [Examples](#examples)

---

## Batch Generation

### Overview

The batch generator (`scripts/datagen/surge_batch_setup.py`) creates parameterized simulation batches by:
- Sampling parameter values using Latin Hypercube Sampling (LHS) or random sampling
- Creating directories for each run/case
- Copying template input files and modifying them with sampled parameter values
- Supporting shared equilibrium cases across multiple parameter sets

### Quick Start

```bash
python scripts/datagen/surge_batch_setup.py --config examples/batch_setup_m3dc1.yml
```

### Configuration File

Create a YAML configuration file to define your batch. Here's the structure:

```yaml
# Where to create the next batch_N directory
out_root: /path/to/output/directory
scratch: false  # Use $SCRATCH/mp288/jobs if true (default: true)

# Full path to the reference input file (template)
inpfile: /path/to/template/C1input

# Parameters to vary in simulations
params: [ntor, pscale, batemanscale]

# Ranges for each parameter (min, max)
ranges:
  - [1, 20]         # ntor (integer)
  - [0.5, 2.0]      # pscale (float)
  - [0.1, 1.0]      # batemanscale (float)

# Whether each parameter is integer
integer_mask: [true, false, false]

# Whether each parameter should be sampled in log space
# Useful for concentrating samples at lower values
log_space: [true, false, false]

# Number of runs to create
nsamples: 100

# Sampling method: lhs | random
spl: lhs

# Random seed for reproducibility
seed: 42

# Equilibria mode: fixed | per_case
# fixed: all equilibria in a run share the same parameters
# per_case: parameters vary per equilibrium case
equilibria: fixed

# Path to source equilibria (batch_0 or a run folder)
eqsetpath: /path/to/batch_0

# Use in-Python replacement (recommended)
use_python_replacement: true

# Non-interactive mode
confirm_dirs: false

# Save visualization plots of the sampling
save_plots: true
```

### Configuration Options

#### Required Fields

- **`inpfile`**: Full path to the template input file (e.g., `C1input`) that will be copied and modified
- **`params`**: List of parameter names to vary
- **`ranges`**: List of `[min, max]` pairs for each parameter
- **`integer_mask`**: Boolean list indicating which parameters are integers
- **`nsamples`**: Number of runs/cases to generate

#### Optional Fields

- **`out_root`**: Output directory (default: `$SCRATCH/mp288/jobs` if `scratch=true`, else `SURGE/examples/datagen`)
- **`scratch`**: Boolean (default: `true`). If `true` and `out_root` not set, uses `$SCRATCH/mp288/jobs`
- **`spl`**: Sampling method - `lhs` (Latin Hypercube) or `random` (default: `lhs`)
- **`seed`**: Random seed for reproducibility
- **`log_space`**: Boolean list for log-space sampling (concentrates samples at lower values)
- **`equilibria`**: Mode - `fixed` (same params per run) or `per_case` (varies per case)
- **`eqsetpath`**: Path to source equilibrium cases (required if `equilibria` is set)
- **`save_plots`**: Save sampling distribution plots (default: `true`)

### Command-Line Interface

You can override any config file option via CLI:

```bash
python scripts/datagen/surge_batch_setup.py \
    --config examples/batch_setup_m3dc1.yml \
    --nsamples 50 \
    --seed 123 \
    --scratch true \
    --out_root /custom/path
```

#### CLI Options

- `--config`: Path to YAML config file
- `--out_root`: Override output directory
- `--inpfile`: Override template input file
- `--nsamples`: Override number of samples
- `--spl`: Override sampling method (`lhs` or `random`)
- `--seed`: Override random seed
- `--equilibria`: Override equilibria mode (`fixed` or `per_case`)
- `--eqsetpath`: Override equilibria source path
- `--scratch`: Use scratch directory (takes precedence over config)
- `--no-scratch`: Disable scratch mode (takes precedence over `--scratch`)
- `--dry-run`: Show what would be done without writing files

### Log-Space Sampling

For parameters that span a wide range (e.g., `ntor` from 1 to 20), you may want more samples at lower values. Use `log_space: [true, false, false]` to enable logarithmic sampling:

```yaml
params: [ntor, pscale, batemanscale]
ranges:
  - [1, 20]         # ntor - will have more samples near 1
  - [0.5, 2.0]      # pscale - uniform distribution
  - [0.1, 1.0]      # batemanscale - uniform distribution
log_space: [true, false, false]
```

**Example distribution with log-space for `ntor`:**
- `ntor=1`: 14% of samples
- `ntor=5`: 6% of samples
- `ntor=10`: 4% of samples
- `ntor=20`: 2% of samples

### Output Structure

The generator creates a `batch_N` directory with:

```
batch_N/
├── run1/
│   ├── sparc_1300/
│   │   ├── C1input          # Modified with sampled parameters
│   │   ├── geqdsk
│   │   ├── equilibrium.h5
│   │   └── ...
│   ├── sparc_1400/
│   │   └── ...
│   └── ...
├── run2/
│   └── ...
├── ...
├── samples.npz              # Parameter samples array
├── meta.json                # Batch metadata
├── sampling_plot.png        # Distribution visualization (if save_plots=true)
└── C1input                  # Reference template copy
```

### Equilibria Modes

#### Fixed Mode (`equilibria: fixed`)

All equilibrium cases in a run share the same parameter values:

```yaml
equilibria: fixed
nsamples: 100
# Creates 100 runs, each with all equilibria from batch_0
# Each run has the same parameters for all equilibria
```

#### Per-Case Mode (`equilibria: per_case`)

Each equilibrium case gets different parameter values:

```yaml
equilibria: per_case
nsamples: 100
# Creates 100 runs, each with all equilibria from batch_0
# Each equilibrium case within a run has different parameters
```

---

## Batch Launching

### Overview

After generating a batch, you need to submit it to SLURM. Each batch includes a `batchjob.perlmutter` template that processes all runs using SLURM job arrays.

### Quick Start

1. **Navigate to the batch directory:**
   ```bash
   cd /path/to/batch_N
   ```

2. **Submit the job:**
   ```bash
   sbatch batchjob.perlmutter
   ```

### Batch Job Template

The `batchjob.perlmutter` file is a SLURM script that:
- Uses job arrays to process multiple runs in parallel
- Sets up shared mesh directory (task 1 only)
- Processes all `sparc_*` cases in each run
- Forces fresh runs by removing existing outputs

#### Key SLURM Directives

```bash
#!/bin/bash
#SBATCH -N 8                          # Number of nodes
#SBATCH --ntasks-per-node=12          # Tasks per node
#SBATCH --cpus-per-task=1             # CPUs per task
#SBATCH --time=08:00:00               # Time limit
#SBATCH --qos=regular                 # Queue (regular | debug)
#SBATCH --array=1-100                 # Job array: tasks 1-100
#SBATCH -L SCRATCH                    # Use scratch filesystem
#SBATCH -C cpu                        # CPU constraint
#SBATCH -A mp288                      # Account
#SBATCH -o M3DC1log.o%j.%a            # Output log (%j=jobid, %a=array_id)
#SBATCH -e M3DC1log.e%j.%a            # Error log
```

#### Resource Configuration

**For Debug Queue (limited nodes):**
```bash
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
```

**For Regular Queue (production runs):**
```bash
#SBATCH --qos=regular
#SBATCH -N 8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
```

### Job Array Behavior

- **Task 1**: Copies shared mesh to `batch_N/mesh/` and updates all `C1input` files
- **Tasks 2-N**: Process their assigned run directories in parallel
- Each task processes all `sparc_*` cases in its run directory

### Monitoring Jobs

**Check job status:**
```bash
squeue -u $USER
squeue -j <job_id>
```

**View output logs:**
```bash
# For task 1:
cat M3DC1log.o<job_id>.1

# For task 5:
cat M3DC1log.o<job_id>.5
```

**Check which runs completed:**
```bash
cd batch_N
find . -name "finished" | sort
```

**Count completed cases:**
```bash
find . -name "finished" | wc -l
```

### Shared Mesh Setup

The batch job automatically sets up a shared mesh directory:

1. **Task 1** copies pre-partitioned mesh files from `batch_0/run1/mesh/` to `batch_N/mesh/`
2. All `C1input` files are updated to reference `../../mesh/part.smb`
3. Other tasks wait for mesh setup to complete

**Mesh files copied:**
- `part*.smb` (mesh partitions)
- `sparc_fw1.txt` (mesh metadata)

**Note:** The source mesh must already be partitioned. If you need to partition a mesh, do that separately before generating batches.

### Time Estimation

Estimate time based on:
- Number of equilibria per run
- Simulation complexity (ntor, mesh size)
- Typical runtime per case: 5-30 minutes

**Example:**
- 100 runs × 101 equilibria = 10,100 cases
- ~10 minutes per case = ~168 hours total
- With 100 parallel tasks: ~1.7 hours wall time

Adjust `--time` accordingly:
```bash
#SBATCH --time=04:00:00  # 4 hours for large batches
```

---

## Verification

### Verify Batch Generation

After generating a batch, verify it's correct:

```bash
python surge/verify_batch.py /path/to/batch_N
```

**Checks performed:**
- All expected input files present
- Parameter values correctly set
- Parameter values within expected ranges
- Consistent parameters across equilibria (in `fixed` mode)
- Properly copied equilibrium directories

**Verbose output:**
```bash
python surge/verify_batch.py /path/to/batch_N --verbose
```

### Manual Verification

**Check parameter distribution:**
```bash
cd batch_N
python3 -c "
import numpy as np
data = np.load('samples.npz')
samples = data['X']
print(f'Total samples: {len(samples)}')
print(f'Parameter ranges:')
for i, param in enumerate(['ntor', 'pscale', 'batemanscale']):
    print(f'  {param}: {samples[:, i].min():.2f} - {samples[:, i].max():.2f}')
"
```

**Check specific parameter values:**
```bash
grep "ntor" run1/sparc_1300/C1input
grep "pscale" run1/sparc_1300/C1input
```

**Verify mesh paths:**
```bash
grep "mesh_filename" run1/sparc_1300/C1input
# Should show: mesh_filename = '../../mesh/part.smb'
```

---

## Examples

### Example 1: Small Test Batch

**Config file (`test_batch.yml`):**
```yaml
out_root: /path/to/output
inpfile: /path/to/batch_0/run1/C1input
params: [ntor, pscale]
ranges:
  - [1, 5]
  - [0.5, 2.0]
integer_mask: [true, false]
nsamples: 5
spl: lhs
seed: 42
equilibria: fixed
eqsetpath: /path/to/batch_0
use_python_replacement: true
confirm_dirs: false
save_plots: true
```

**Generate:**
```bash
python scripts/datagen/surge_batch_setup.py --config test_batch.yml
```

**Launch:**
```bash
cd /path/to/output/batch_1
sbatch batchjob.perlmutter
```

### Example 2: Large Production Batch with Log-Space Sampling

**Config file (`production_batch.yml`):**
```yaml
out_root: /global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC
scratch: false
inpfile: /path/to/batch_0/run1/C1input
params: [ntor, pscale, batemanscale]
ranges:
  - [1, 20]         # ntor - log-space for more low values
  - [0.5, 2.0]
  - [0.1, 1.0]
integer_mask: [true, false, false]
log_space: [true, false, false]  # Log-space for ntor
nsamples: 100
spl: lhs
seed: 42
equilibria: fixed
eqsetpath: /path/to/batch_0
use_python_replacement: true
confirm_dirs: false
save_plots: true
```

**Generate:**
```bash
python scripts/datagen/surge_batch_setup.py --config production_batch.yml
```

**Verify:**
```bash
python surge/verify_batch.py /path/to/batch_7
```

**Launch:**
```bash
cd /path/to/batch_7
# Edit batchjob.perlmutter to set appropriate resources
sbatch batchjob.perlmutter
```

### Example 3: Using Scratch Directory

**Generate batch in scratch:**
```bash
python scripts/datagen/surge_batch_setup.py \
    --config examples/batch_setup_m3dc1.yml \
    --scratch true
# Creates batch in $SCRATCH/mp288/jobs/batch_N
```

**Or use config file:**
```yaml
scratch: true  # Default
# out_root not needed - will use $SCRATCH/mp288/jobs
```

### Example 4: CLI Overrides

**Override specific options:**
```bash
python scripts/datagen/surge_batch_setup.py \
    --config examples/batch_setup_m3dc1.yml \
    --nsamples 50 \
    --seed 999 \
    --no-scratch \
    --out_root /custom/path
```

---

## Troubleshooting

### Common Issues

**1. "Reference input file not found"**
- Ensure `inpfile` path is correct and absolute
- Check file permissions

**2. "Could not determine equilibria source_run_dir"**
- Verify `eqsetpath` points to a directory containing `sparc_*` folders
- Or ensure `inpfile` is inside a run directory with `sparc_*` cases

**3. "Length mismatch among params, ranges, and integer_mask"**
- Ensure all three lists have the same length
- Check for typos or extra commas

**4. SLURM job fails with "out of range"**
- Verify `batchjob.perlmutter` has correct `--array` range
- Check that batch directory exists and contains `run*/` directories

**5. "Shared mesh directory not found"**
- Ensure task 1 completes mesh setup before other tasks
- Check source mesh exists at `batch_0/run1/mesh/`

**6. Slow batch generation**
- Large equilibrium directories (with big mesh files) take time to copy
- Consider using scratch filesystem for faster I/O

### Debugging

**Dry run to see what would be generated:**
```bash
python scripts/datagen/surge_batch_setup.py --config my_config.yml --dry-run
```

**Check batch structure:**
```bash
cd batch_N
ls -la run1/
ls -la run1/sparc_1300/
cat run1/sparc_1300/C1input | grep -E "(ntor|pscale|mesh_filename)"
```

**Check SLURM job logs:**
```bash
cat M3DC1log.o<job_id>.* | grep -i error
cat M3DC1log.e<job_id>.*
```

---

## Best Practices

1. **Always verify batches** before launching large jobs
2. **Use log-space sampling** for parameters with wide ranges where you want more low values
3. **Start with small batches** (5-10 runs) to test configuration
4. **Set appropriate time limits** in `batchjob.perlmutter` based on expected runtime
5. **Use scratch filesystem** for temporary batches to avoid quota issues
6. **Save config files** for reproducibility
7. **Check parameter distributions** using `save_plots: true` and examine `sampling_plot.png`
8. **Monitor job progress** regularly, especially for large batches

---

## Additional Resources

- Example config files: `examples/batch_setup_m3dc1.yml`
- Verification script: `surge/verify_batch.py`
- Source code: `surge/datagen/generator.py`, `scripts/datagen/surge_batch_setup.py`


