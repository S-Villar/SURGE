# Minimal Batch Generation Recipe

This document provides the minimal workflow for generating batches using SURGE's DataGenerator.

## Complete Workflow

### STEP 1: Activate environment and generate batch

```bash
cd /global/homes/a/asvillar/src/SURGE
conda activate surge
python scripts/datagen/surge_batch_setup.py --config examples/batch_setup_m3dc1_5runs.yml
```

**What this does:**
- Creates a new batch directory (e.g., `$SCRATCH/mp288/jobs/batch_5`)
- Generates parameter samples using LHS or random sampling
- Copies equilibria directories and modifies input files
- Creates `run1..runN` folders with `sparc_*` subdirectories
- Saves `samples.npz` and `metadata.json`

### STEP 2: Verify the batch

```bash
python -m surge.verify_batch $SCRATCH/mp288/jobs/batch_5
```

**What this checks:**
- Parameter assignments are correct
- File structure matches expected format
- Input files contain correct parameter values

### STEP 3: Copy batchjob template

```bash
cp /pscratch/sd/a/asvillar/mp288/jobs/batch_3/batchjob.perlmutter \
   $SCRATCH/mp288/jobs/batch_5/batchjob.perlmutter
```

**About the batchjob script:**
- Uses self-submitting SLURM array jobs
- Bootstrap job `[1]` counts runs/equilibria and resubmits with full array
- Each job handles one case (run, equilibrium combination)
- Handles mesh partitioning and M3DC1 execution

### STEP 4: Submit the job

```bash
cd $SCRATCH/mp288/jobs/batch_5
sbatch batchjob.perlmutter
```

**What happens:**
- Initially you'll see job `[1]` (bootstrap)
- Bootstrap job counts number of runs and equilibria
- Calculates `TOTAL_CASES = N_RUNS × N_EQUILIBRIA`
- Resubmits with full array range (e.g., `--array=1-505`)

### STEP 5: Monitor progress (optional)

```bash
squeue -u $USER                                    # Check job status
find run*/sparc_*/finished -type f 2>/dev/null | wc -l  # Count completed cases
```

## Important Notes

- Replace `batch_5` with your actual batch name from step 1
- Ensure config file points to correct paths
- The 20-minute time limit is per individual job, not entire batch
- Each job in the array handles one case independently

## Equilibria Modes

### Mode 1: `fixed` - Single equilibrium for all runs

All runs (`run1..runN`) use the **SAME** equilibrium (single geqdsk file).  
Different runs have different parameter combinations.

**Structure:**
```
batch_N/
  run1/sparc_1300/
  run2/sparc_1300/  (same equilibrium)
  run3/sparc_1300/  (same equilibrium)
```

**Config example:**
```yaml
equilibria: fixed
eqsetpath: /path/to/source/run1  # Contains sparc_* directories
nsamples: 5  # Creates run1..run5
```

### Mode 2: `set` - Multiple equilibria per run

Each run tests **ALL** equilibria from the set.  
All equilibria in a run share the same parameter values.

**Structure:**
```
batch_N/
  run1/
    sparc_1300/
    sparc_1301/
    sparc_234321/
  run2/
    sparc_1300/  (same equilibria set)
    sparc_1301/
    sparc_234321/
```

**Config example:**
```yaml
equilibria: set
eqsetpath: /path/to/source/run1  # Contains multiple sparc_* directories
nsamples: 5  # Creates run1..run5, each with all equilibria
```

### Mode 3: `per_case` - Independent parameters per equilibrium (advanced)

Each equilibrium gets its own independent set of runs with different parameter samples.

**Structure:**
```
batch_N/
  sparc_1300/
    run1/
    run2/
  sparc_1301/
    run1/
    run2/
```

**Config example:**
```yaml
equilibria: per_case
eqsetpath: /path/to/source/run1
nsamples: 5  # Creates 5 runs per equilibrium
```

## Required Configuration Parameters

### Required:
- `inpfile`: Full path to reference input file (template)
- `params`: List of parameter names to vary
- `ranges`: List of `[min, max]` pairs for each parameter
- `integer_mask`: List of booleans indicating integer parameters
- `nsamples`: Number of runs/samples to generate

### Optional but recommended:
- `equilibria`: `'fixed'`, `'set'`, or `'per_case'` (if using equilibria)
- `eqsetpath`: Path to source equilibria directory
- `spl`: Sampling method (`'lhs'` or `'random'`)
- `seed`: Random seed for reproducibility
- `log_space`: List of booleans for log-space sampling
- `scratch`: `true`/`false` (default: `true`, uses `$SCRATCH`)
- `use_python_replacement`: `true` (recommended)
- `save_plots`: `true` (for visualization)
