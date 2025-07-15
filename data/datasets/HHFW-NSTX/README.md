# HHFW-NSTX Dataset

## Overview

This dataset contains High Harmonic Fast Wave (HHFW) heating simulation results for NSTX (National Spherical Torus Experiment) plasma scenarios. The data represents power deposition profiles computed using the TORIC RF heating code for various plasma conditions and antenna configurations.

## Dataset Contents

### Files

- **`PwE_.pkl`**: Electron heating power deposition profiles
- **`PwIF_.pkl`**: Ion and fast particle heating power deposition profiles

### File Format

All files are pandas DataFrames saved in pickle format (.pkl) containing:

- **Input Features**: HHFW simulation parameters
  - Plasma parameters (density, temperature profiles)
  - Magnetic field configuration
  - Antenna parameters (frequency, power, phasing)
  - Geometry settings
  
- **Output Targets**: Spatial power deposition profiles
  - **PwE_**: Electron heating power density [W/m³] across spatial grid points
  - **PwIF_**: Ion and fast particle heating power density [W/m³] across spatial grid points

## Scientific Context

### HHFW Heating in NSTX

High Harmonic Fast Wave (HHFW) heating is a radio frequency heating method used in spherical tokamaks like NSTX. The waves are launched from antennas and deposit power into the plasma through various absorption mechanisms:

- **Electron Heating**: Direct electron Landau damping and transit-time magnetic pumping
- **Ion Heating**: Ion cyclotron damping and other ion absorption processes
- **Fast Particle Heating**: Heating of energetic particles (e.g., neutral beam ions)

### TORIC Simulation Code

The data was generated using TORIC, a full-wave electromagnetic code that solves the wave equation in toroidal geometry to compute RF power deposition profiles. TORIC accounts for:

- Full electromagnetic wave propagation
- Finite Larmor radius effects
- Multiple ion species
- Non-Maxwellian velocity distributions

## Dataset Structure

### Input Variables (Example subset)

- `freq`: RF frequency [Hz]
- `power`: Total antenna power [W]
- `n_phi`: Toroidal mode number spectrum
- `Te0`: Central electron temperature [keV]
- `ne0`: Central electron density [m⁻³]
- `Bt0`: Toroidal magnetic field [T]
- Plasma geometry parameters
- Antenna configuration parameters

### Output Variables

- **Electron Power (`PwE_*`)**: Power deposited to electrons at different spatial locations
- **Ion/Fast Particle Power (`PwIF_*`)**: Power deposited to ions and fast particles

Each spatial location corresponds to a specific radial position in the plasma, allowing reconstruction of complete power deposition profiles.

## Usage Examples

### Basic Data Loading

```python
import pandas as pd
from surge import SurrogateTrainer

# Load electron heating data
trainer = SurrogateTrainer()
inputs, outputs = trainer.load_dataset_pickle('data/datasets/HHFW-NSTX/PwE_.pkl')

print(f"Input features: {len(inputs)}")
print(f"Output profiles: {len(outputs)}")
```

### Surrogate Model Training

```python
from surge import MLTrainer

# Load and prepare data
trainer = SurrogateTrainer()
inputs, outputs = trainer.load_dataset_pickle('data/datasets/HHFW-NSTX/PwE_.pkl')

# Get the underlying MLTrainer
ml_trainer = trainer.get_trainer()

# Load data into ML framework
ml_trainer.load_df_dataset(trainer.df, inputs, outputs)
ml_trainer.train_test_split(test_split=0.2)
ml_trainer.standardize_data()

# Train surrogate model
ml_trainer.init_model(0)  # Random Forest
ml_trainer.train(0)
ml_trainer.predict_output(0)

print(f"Model R² score: {ml_trainer.R2:.4f}")
```

### Hyperparameter Optimization

```python
# Optimize model hyperparameters
results = ml_trainer.tune(
    method='optuna_botorch',
    n_trials=50
)
print(f"Best R² after tuning: {results['best_r2']:.4f}")
```

## Data Statistics

- **File sizes**: ~41MB each (~82MB total)
- **Number of scenarios**: Multiple HHFW parameter combinations
- **Spatial resolution**: High-resolution radial grid for power profiles
- **Physics coverage**: Wide range of NSTX-relevant plasma conditions

## Applications

This dataset is ideal for:

1. **Surrogate Model Development**: Training fast ML models to replace expensive TORIC simulations
2. **Real-time Control**: Developing predictive models for plasma control systems
3. **Optimization Studies**: Rapid exploration of HHFW parameter space
4. **Physics Studies**: Understanding HHFW heating efficiency trends
5. **Machine Learning Research**: Multi-output regression with physics constraints

## Citation

If you use this dataset in research, please cite:

- TORIC code development and physics
- NSTX experimental program
- SURGE surrogate modeling framework

## Technical Notes

- All power units are in watts per cubic meter [W/m³]
- Spatial coordinates follow NSTX magnetic coordinate system
- Data includes both fundamental and harmonic absorption physics
- Suitable for both single-output and multi-output regression tasks

## Related Files

- `../../../notebooks/RF_Heating_Surrogate_Demo.ipynb`: Complete demonstration workflow
- `../../../examples/`: Additional analysis scripts
- `../README.md`: General datasets overview
