# Testing Eigenmode Extraction

This script tests extraction of perturbed fields (delta B and delta p) from M3DC1 simulations.

## Prerequisites

1. **Fusion-io environment** - Required for `m3dc1` and `fpy` modules
2. **Conda environment** - For matplotlib and other dependencies
3. **Test case** - A completed M3DC1 simulation with `C1.h5` file

## Running the Test

### Option 1: Using the wrapper script (recommended)

```bash
cd /global/homes/a/asvillar/src/SURGE
./scripts/m3dc1/interfaces/run_test_eigenmode.sh
```

The wrapper script will attempt to:
- Load fusion-io module (if available)
- Activate conda environment (surge or surge-devel)
- Run the test script

### Option 2: Manual setup

```bash
# Load fusion-io (adjust for your system)
module load fusion-io  # or your system's equivalent

# Activate conda environment
conda activate surge  # or surge-devel

# Run the script
cd /global/homes/a/asvillar/src/SURGE
python3 scripts/m3dc1/interfaces/test_eigenmode_extraction.py
```

## What the Script Does

1. **Reads perturbed fields** from `C1.h5`:
   - Perturbed magnetic field magnitude (|δB|)
   - Perturbed pressure (δp)

2. **Computes flux averages** using `m1.flux_average()`:
   - Flux-averages the fields over flux surfaces
   - Returns values as a function of normalized flux (psi_N)

3. **Interpolates to standard grid**:
   - Creates a uniform psi_N grid from 0 to 0.995
   - Interpolates the flux-averaged data to this grid

4. **Plots results**:
   - Saves plot to `test_eigenmode_plot.png`
   - Shows δB and δp vs psi_N

## Test Case

The script is configured to test with:
```
/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run12/sparc_1429
```

To test with a different case, edit the `test_case` variable in `main()` function.

## Expected Output

- Console output showing:
  - Field extraction progress
  - Data ranges and statistics
  - Success/failure messages

- Plot file: `test_eigenmode_plot.png`
  - Top panel: δB vs ψ_N
  - Bottom panel: δp vs ψ_N

## Troubleshooting

### "m3dc1 and fpy modules are required"

**Solution**: Load fusion-io environment
```bash
module load fusion-io  # or equivalent for your system
```

### "matplotlib not found"

**Solution**: Activate conda environment
```bash
conda activate surge
```

### "C1.h5 not found"

**Solution**: Check that the test case path is correct and the simulation completed successfully.

### Field extraction fails

The script tries multiple methods:
1. Direct flux_average with 'B'
2. Computing |B| from components (Br, Bz, Bphi)
3. Alternative field names

If all fail, check:
- Simulation completed successfully
- C1.h5 file is valid
- Fusion-io version compatibility

## Next Steps

Once the test works:
1. Verify the plots look reasonable
2. Check the data ranges make sense
3. Integrate this extraction into `collect_from_batch.py` for database creation

