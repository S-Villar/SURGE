# SURGE Datasets

This directory contains datasets used for SURGE surrogate modeling demonstrations and testing.

## HHFW-NSTX RF Heating Dataset

### Files
- **`HHFW-NSTX/PwE_.pkl`**: HHFW electron heating power deposition profiles from TORIC simulations
- **`HHFW-NSTX/PwIF_.pkl`**: HHFW ion and fast particle heating power deposition profiles

### Dataset Information
- **Source**: TORIC RF heating simulations for NSTX High Harmonic Fast Wave scenarios
- **Format**: Pandas DataFrame stored as pickle files
- **Size**: ~41MB each (~82MB total)
- **Features**: Input parameters for HHFW heating scenarios + output power deposition profiles
- **Use Case**: Training surrogate models for RF heating prediction in spherical tokamaks

### Usage

```python
from surge import SurrogateTrainer

# Load the HHFW electron heating dataset
trainer = SurrogateTrainer()
input_vars, output_vars = trainer.load_dataset_pickle('data/datasets/HHFW-NSTX/PwE_.pkl')

# The trainer automatically identifies input/output variables:
# - Input variables: HHFW simulation parameters (frequency, power, plasma conditions, etc.)
# - Output variables: PwE_* columns (electron heating power deposition profiles)
```

### Dataset Structure

The datasets contain:
- **Input Features**: HHFW heating simulation parameters
  - **n_phi**: Toroidal mode number spectrum
  - **Te0, Te1**: Core and edge electron temperatures [keV]
  - **ne0, ne1**: Core and edge electron densities [m⁻³]  
  - **alpha_T, alpha_n**: Profile exponents for flux coordinate dependence
  - Plasma geometry and antenna configuration parameters
  
- **Output Targets**: Power deposition profiles
  - `PwE_*`: Electron heating power deposition [W/m³] (spatial distribution)
  - `PwIF_*`: Ion and fast particle heating power deposition [W/m³] (spatial distribution)

### Technical Notes

- Files are in pandas DataFrame format with proper column labeling
- All datasets use consistent naming conventions for automated variable detection
- Power deposition profiles are discretized across spatial grid points in NSTX geometry
- Suitable for multi-output regression surrogate modeling of RF heating physics

### Example Analysis

For a complete demonstration of using these datasets, see:
- `notebooks/RF_Heating_Surrogate_Demo.ipynb`: Full workflow with HHFW-NSTX data
- `HHFW-NSTX/README.md`: Detailed technical documentation

### Data Licensing

These datasets are provided for research and educational purposes. Please cite appropriately if used in publications.
