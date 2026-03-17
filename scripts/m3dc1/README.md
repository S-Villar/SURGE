# M3DC1 Scripts

This directory contains M3DC1-specific scripts and utilities for processing M3DC1 simulation data and generating surrogate models.

## Scripts

### Core Processing Scripts

- **`loader.py`** - Loads M3DC1 HDF5 data files and converts to pandas DataFrame
  - `load_m3dc1_hdf5()`, `convert_to_dataframe()`, `read_m3dc1_hdf5_structure()`
  - `convert_sdata_complex_v2_to_dataframe()` for single aggregated sdata_complex_v2.h5
  - Re-exports from `dataset_complex_v2.py` for per-run batch loading

- **`dataset_complex_v2.py`** - Dataset generator for **per-run** sdata_pertfields_grid_complex_v2.h5
  - Each `run*/sparc_*/` has its own complex_v2 file with spectrum, miller, parset, flux_average
  - `find_complex_v2_files(batch_dir)` - discover all files
  - `build_dataframe_from_batch(batch_dir)` - load all cases into DataFrame
  - `load_complex_v2_for_surge(batch_dir)` - returns (df, input_cols, output_cols) for SurrogateDataset

- **`write.py`** - Parallel processing script to convert M3DC1 simulation outputs to HDF5 format
  - Processes run directories and extracts equilibrium, profiles (q, p), growth rates, etc.
  - Supports multiprocessing for large batch processing
  - Output: Creates sdataXX.h5 files with structured HDF5 groups

- **`collect_from_batch.py`** - Collects M3DC1 data from batch folders
  - Extracts growth rates, profiles, equilibrium parameters from C1.h5 files
  - Creates pandas DataFrame with all simulation data

- **`collect_data.py`** - Alternative data collection utility
  - Similar functionality to `collect_from_batch.py` with different data sources

### Utility Scripts

- **`fix_inputs.py`** - Fixes C1input files to ensure integer values (e.g., ntor=2 instead of ntor=2.0)
  - Supports parallel processing for batch fixes
  - Ensures M3DC1 input files have correct format

### Visualization/Interface Scripts

See the **`interfaces/`** subdirectory for visualization and plotting scripts:
- Comprehensive eigenmode plotting
- 2D field visualization
- Eigenmode extraction testing
- Field reconstruction validation

For details, see `interfaces/README.md`.

## Integration with SURGE

These scripts are M3DC1-specific and kept separate from the core SURGE package to maintain framework simulator-agnosticism. The `loader.py` module is optionally importable by SURGE's `SurrogateDataset` class for M3DC1 data format support.

## Usage

### Standalone Usage

```python
import sys
from pathlib import Path

# Add scripts/m3dc1 to path
scripts_m3dc1 = Path(__file__).parent / "scripts" / "m3dc1"
sys.path.insert(0, str(scripts_m3dc1))

from loader import convert_to_dataframe, convert_sdata_complex_v2_to_dataframe

# Standard sdata.h5 (gamma, profiles)
df = convert_to_dataframe('sdata03.h5')

# Delta p spectra from sdata_complex_v2.h5 (auto-detected from filename)
df = convert_sdata_complex_v2_to_dataframe('sdata_complex_v2.h5')

# Per-run sdata_pertfields_grid_complex_v2.h5 (one file per run*/sparc_*/)
from loader import load_complex_v2_for_surge
df, input_cols, output_cols = load_complex_v2_for_surge('/path/to/batch_16')
dataset = SurrogateDataset.from_dataframe(df, input_columns=input_cols, output_columns=output_cols)
```

### Via SURGE

```python
from surge import SurrogateDataset

# SURGE will automatically try to use M3DC1 loader if available
dataset = SurrogateDataset()
dataset.load_from_file('sdata03.h5', auto_detect=True)
```

## Dependencies

- `h5py` - HDF5 file support
- `numpy`, `pandas` - Data manipulation
- `m3dc1`, `fpy` - M3DC1-specific libraries (optional, for advanced features)
- `scipy` - Interpolation utilities











