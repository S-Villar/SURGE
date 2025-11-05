#!/usr/bin/env python3
"""
Minimal Recipe Test for Batch Generation with DataGenerator.

This test documents the minimal workflow for generating batches using SURGE's
DataGenerator, including verification and SLURM job submission steps.

The recipe covers:
1. Activating environment and generating batch
2. Verifying the batch
3. Copying batchjob template
4. Submitting SLURM job
5. Monitoring progress

This serves as both documentation and a verifiable example.
"""

import os
import pytest
from pathlib import Path


def test_batch_generation_recipe_documentation():
    """
    Minimal Recipe for Batch Generation - Documentation/Example.
    
    This test documents the complete workflow without actually executing it
    (which requires actual data and HPC resources).
    
    Recipe Steps:
    =============
    
    STEP 1: Activate environment and generate batch
    ------------------------------------------------
    cd ${SURGE_HOME}
    conda activate surge
    python surge_batch_setup.py --config examples/batch_setup_m3dc1_5runs.yml
    
    This will:
    - Create a new batch directory (e.g., $SCRATCH/mp288/jobs/batch_5)
    - Generate parameter samples using LHS or random sampling
    - Copy equilibria directories and modify input files
    - Create run1..runN folders with sparc_* subdirectories
    - Save samples.npz and metadata.json
    
    STEP 2: Verify the batch
    -------------------------
    python -m surge.verify_batch $SCRATCH/mp288/jobs/batch_5
    
    This checks:
    - Parameter assignments are correct
    - File structure matches expected format
    - Input files contain correct parameter values
    
    STEP 3: Copy batchjob template
    -------------------------------
    cp ${SURGE_SCRATCH}/mp288/jobs/batch_3/batchjob.perlmutter \\
       $SCRATCH/mp288/jobs/batch_5/batchjob.perlmutter
    
    The batchjob.perlmutter script:
    - Uses self-submitting SLURM array jobs
    - Bootstrap job [1] counts runs/equilibria and resubmits with full array
    - Each job handles one case (run, equilibrium combination)
    - Handles mesh partitioning and M3DC1 execution
    
    STEP 4: Submit the job
    -----------------------
    cd $SCRATCH/mp288/jobs/batch_5
    sbatch batchjob.perlmutter
    
    Initially you'll see job [1] (bootstrap), which will:
    - Count number of runs and equilibria
    - Calculate TOTAL_CASES = N_RUNS × N_EQUILIBRIA
    - Resubmit with full array range (e.g., --array=1-505)
    
    STEP 5: Monitor progress (optional)
    ------------------------------------
    squeue -u $USER                                    # Check job status
    find run*/sparc_*/finished -type f 2>/dev/null | wc -l  # Count completed cases
    
    Notes:
    ------
    - Replace 'batch_5' with actual batch name from step 1
    - Ensure config file points to correct paths
    - The 20-minute time limit is per individual job, not entire batch
    """
    # This is a documentation test - it always passes
    # but documents the complete recipe
    assert True, "This test documents the batch generation recipe"


def test_batch_generation_modes():
    """
    Document the three equilibria modes available.
    
    Mode 1: 'fixed' - Single equilibrium for all runs
    -------------------------------------------------
    - All runs (run1..runN) use the SAME equilibrium (single geqdsk file)
    - Different runs have different parameter combinations
    - Structure: batch_N/run1/sparc_1300/, run2/sparc_1300/, run3/sparc_1300/
    
    Config example:
        equilibria: fixed
        eqsetpath: /path/to/source/run1  # Contains sparc_* directories
        nsamples: 5  # Creates run1..run5
    
    Mode 2: 'set' - Multiple equilibria per run
    --------------------------------------------
    - Each run tests ALL equilibria from the set
    - All equilibria in a run share the same parameter values
    - Structure: batch_N/run1/sparc_1300/, run1/sparc_1301/, run1/sparc_234321/
                  run2/sparc_1300/, run2/sparc_1301/, run2/sparc_234321/
    
    Config example:
        equilibria: set
        eqsetpath: /path/to/source/run1  # Contains multiple sparc_* directories
        nsamples: 5  # Creates run1..run5, each with all equilibria
    
    Mode 3: 'per_case' - Independent parameters per equilibrium (advanced)
    -----------------------------------------------------------------------
    - Each equilibrium gets its own independent set of runs
    - Structure: batch_N/sparc_1300/run1/, sparc_1300/run2/
                  batch_N/sparc_1301/run1/, sparc_1301/run2/
    
    Config example:
        equilibria: per_case
        eqsetpath: /path/to/source/run1
        nsamples: 5  # Creates 5 runs per equilibrium
    """
    modes = ['fixed', 'set', 'per_case']
    assert all(mode in ['fixed', 'set', 'per_case'] for mode in modes)


def test_required_config_parameters():
    """
    Document required and optional parameters for batch generation.
    
    Required parameters:
    --------------------
    - inpfile: Full path to reference input file (template)
    - params: List of parameter names to vary
    - ranges: List of [min, max] pairs for each parameter
    - integer_mask: List of booleans indicating integer parameters
    - nsamples: Number of runs/samples to generate
    
    Optional but recommended:
    -------------------------
    - equilibria: 'fixed', 'set', or 'per_case' (if using equilibria)
    - eqsetpath: Path to source equilibria directory
    - spl: Sampling method ('lhs' or 'random')
    - seed: Random seed for reproducibility
    - log_space: List of booleans for log-space sampling
    - scratch: true/false (default: true, uses $SCRATCH)
    - use_python_replacement: true (recommended)
    - save_plots: true (for visualization)
    """
    required = ['inpfile', 'params', 'ranges', 'integer_mask', 'nsamples']
    optional = ['equilibria', 'eqsetpath', 'spl', 'seed', 'log_space', 
                'scratch', 'use_python_replacement', 'save_plots']
    
    assert len(required) == 5
    assert len(optional) >= 8


if __name__ == "__main__":
    # When run directly, print the recipe
    print("=" * 70)
    print("MINIMAL BATCH GENERATION RECIPE")
    print("=" * 70)
    print()
    
    recipe = """
# STEP 1: Generate batch
cd ${SURGE_HOME}
conda activate surge
python surge_batch_setup.py --config examples/batch_setup_m3dc1_5runs.yml

# STEP 2: Verify batch
python -m surge.verify_batch $SCRATCH/mp288/jobs/batch_5

# STEP 3: Copy batchjob template
cp ${SURGE_SCRATCH}/mp288/jobs/batch_3/batchjob.perlmutter \\
   $SCRATCH/mp288/jobs/batch_5/batchjob.perlmutter

# STEP 4: Submit job
cd $SCRATCH/mp288/jobs/batch_5
sbatch batchjob.perlmutter

# STEP 5: Monitor (optional)
squeue -u $USER
find run*/sparc_*/finished -type f 2>/dev/null | wc -l
"""
    print(recipe)
    print("=" * 70)
    print("Note: Replace 'batch_5' with your actual batch name")
    print("=" * 70)
