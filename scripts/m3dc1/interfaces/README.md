# M3DC1 Interface/Visualization Scripts

This directory contains scripts for generating visualizations and interfaces with M3DC1 simulation data, particularly focused on eigenmode analysis and field visualization.

## Scripts

### Visualization Scripts

- **`plot_comprehensive_eigenmode.py`** - Comprehensive eigenmode visualization
  - Creates 8-panel figure (2 rows × 4 columns) showing:
    - Row 1 (δp): Dominant mode, Spectrum, Flux-averaged amplitude, 2D field
    - Row 2 (δB): Dominant mode, Spectrum, Flux-averaged amplitude, 2D field
  - Output: `comprehensive_eigenmode_plot.png` (or custom filename)
  - Usage: `python plot_comprehensive_eigenmode.py <sparc_dir> [--output <file>] [--max-modes <n>]`

- **`plot_eigenmode_complex.py`** - Complex eigenmode visualization
  - Plots complex eigenmode structures and phase information
  - Output: `eigenmode_complex_plot.png`
  
- **`plot_2d_field_from_eigenmodes.py`** - 2D field reconstruction from eigenmodes
  - Reconstructs 2D R-Z plane fields by summing over all poloidal modes
  - Generates separate plots for pressure and magnetic field
  - Output: `2d_field_pressure_summed_modes.png`, `2d_field_B_summed_modes.png`, `2d_field_pressure_and_B_summed_modes.png`

- **`reconstruct_field_from_eigenmodes.py`** - Field reconstruction validation
  - Reconstructs total field from eigenmode amplitudes
  - Compares with direct field access from HDF5
  - Output: `field_reconstruction_plot.png`

### Test/Validation Scripts

- **`test_eigenmode_extraction.py`** - Test eigenmode extraction
  - Extracts and validates eigenmode data from C1.h5
  - Tests flux-averaged profiles for δB and δp
  - Output: `eigenmode_test_plot.png` or `test_eigenmode_plot.png`
  - See `TEST_EIGENMODE_README.md` for detailed usage

- **`test_resolution_convergence.py`** - Resolution convergence testing
  - Tests eigenmode extraction at different resolutions
  - Validates convergence behavior
  - Output: `test_resolution_convergence_p.png`, `test_resolution_convergence_B.png`

### Helper Scripts

- **`run_test_eigenmode.sh`** - Wrapper script for test_eigenmode_extraction.py
  - Automatically loads fusion-io module and activates conda environment
  - Usage: `./run_test_eigenmode.sh`

## Output Files

All visualization scripts generate PNG image files. These files are **excluded from version control** (see `.gitignore`). Common output patterns:

- `comprehensive_eigenmode*.png` - Comprehensive eigenmode plots
- `test_*.png` - Test and validation plots
- `eigenmode_*.png` - Eigenmode-specific visualizations
- `field_reconstruction*.png` - Field reconstruction validation plots
- `2d_field_*.png` - 2D field visualizations

## Dependencies

All scripts require:
- `m3dc1` Python module (from M3DC1/unstructured/python)
- `fpy` module (fusion-io library)
- `numpy`, `matplotlib`, `scipy`
- Access to fusion-io environment (typically via module system)

## Usage Example

```bash
# Generate comprehensive eigenmode plot for a simulation
python plot_comprehensive_eigenmode.py /path/to/batch/run1/sparc_1300 \
    --output comprehensive_eigenmode_run1_sparc_1300.png \
    --max-modes 50

# Test eigenmode extraction
./run_test_eigenmode.sh
```

## Integration

These scripts are designed to work with M3DC1 batch processing workflows. They can be called from batch processing scripts or run interactively for visualization and validation.

For batch processing integration, see the main `scripts/m3dc1/` README.
