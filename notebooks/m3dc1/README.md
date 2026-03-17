# M3DC1 Notebooks

This directory contains Jupyter notebooks for M3DC1-specific surrogate modeling workflows.

## Notebooks

### `data_analysis.ipynb`
Comprehensive data analysis and curation for M3DC1 datasets:
- Data loading from sdata.h5 files
- Data completeness analysis
- Profile visualization (q and p profiles)
- Data curation (removing incomplete/invalid cases)
- Distribution analysis and completeness matrices

### `surrogate_training.ipynb`
Complete surrogate model training workflow:
- Load curated M3DC1 dataset
- Data preprocessing (train/validation/test split, standardization)
- Model training (Random Forest and MLP)
- Performance evaluation and feature importance

### `delta_p_spectra_demo.ipynb`
Load, plot, and prepare delta p spectra for surrogate training:
- Load from `sdata_pertfields_grid.h5` or `sdata_complex_v2.h5`
- Plot 2D perturbed fields, complex field components, and spectra (m vs psi_N)
- Convert to DataFrame for SURGE surrogate training
- Uses `plotting_helpers.py` (adapted from `plot_sdata_pertfields.py`)
- Optional: `m3dc1_python_code/` for C1.h5-based visualization

### Supporting files
- **`plotting_helpers.py`** – `_get_mesh_coords`, `_select_time_positions`, `_time_label`, `plot_2d_fields`, `plot_pertfield_complex`, `plot_spectrum`
- **`m3dc1_python_code/`** – Copied from batch_16 for `plot_field`, `plot_flux_average`, `eigenfunction`, etc.

## Workflow

1. **Data Analysis** - Run `data_analysis.ipynb` to curate and validate your M3DC1 dataset
2. **Model Training** - Run `surrogate_training.ipynb` to train surrogate models on curated data

## Data Requirements

- Input HDF5 file: `sdata03.h5` (or similar) with M3DC1 simulation outputs
- Curated dataset: `sdata03_curated.pkl` (created by data_analysis notebook)
- Required fields: R0, a, kappa, delta, q0, q95, qmin, p0, gamma

## Integration

These notebooks demonstrate how to use SURGE framework with M3DC1 data. They can be adapted for other simulators by modifying the data loading sections.











