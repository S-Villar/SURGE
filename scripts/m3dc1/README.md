# M3DC1 Scripts

This directory contains M3DC1-specific scripts and utilities for processing M3DC1 simulation data and generating surrogate models.

## Scripts

### Core Processing Scripts

- **`loader.py`** - Loads M3DC1 HDF5 data files (sdata.h5 format) and converts to pandas DataFrame
  - Functions: `load_m3dc1_hdf5()`, `convert_to_dataframe()`, `read_m3dc1_hdf5_structure()`
  - Usage: Can be imported by SURGE's `SurrogateDataset` class or used standalone

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

from loader import convert_to_dataframe
df = convert_to_dataframe('sdata03.h5')
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







