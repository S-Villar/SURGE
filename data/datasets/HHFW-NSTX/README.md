# HHFW-NSTX Dataset

> **Public source releases** do not include `*.pkl` binaries. This README describes
> the layout for developers who hold the data under the appropriate collaboration
> or data-use terms. Place approved files locally if you use them.

## Overview

This dataset contains High Harmonic Fast Wave (HHFW) heating simulation results for NSTX (National Spherical Torus Experiment) plasma scenarios. The data represents power deposition profiles computed using the TORIC RF heating code for various plasma conditions and antenna configurations.

## Dataset Contents

### Files

- **`PwE_.pkl`**: Electron heating power deposition profiles
- **`PwIF_.pkl`**: Ion and fast particle heating power deposition profiles

### File Format

All files are pandas DataFrames saved in pickle format (.pkl) containing:

- **Input Features**: HHFW simulation parameters
  - **n_phi**: Toroidal mode number spectrum
  - **Te0**: Core electron temperature [keV] 
  - **Te1**: Edge electron temperature [keV]
  - **ne0**: Core electron density [m⁻³]
  - **ne1**: Edge electron density [m⁻³]
  - **alpha_T**: Temperature profile exponent (for flux coordinate dependence)
  - **alpha_n**: Density profile exponent (for flux coordinate dependence)
  - Additional plasma geometry and antenna parameters
  
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

### Input Variables (Key parameters)

- **`n_phi`**: Toroidal mode number spectrum for HHFW launch
- **`Te0`**: Core electron temperature [keV]
- **`Te1`**: Edge electron temperature [keV] 
- **`ne0`**: Core electron density [m⁻³]
- **`ne1`**: Edge electron density [m⁻³]
- **`alpha_T`**: Temperature profile exponent (assumes T ∝ (1-ψ)^α_T where ψ is normalized flux)
- **`alpha_n`**: Density profile exponent (assumes n ∝ (1-ψ)^α_n where ψ is normalized flux)
- Additional geometric and antenna configuration parameters

### Output Variables

- **Electron Power (`PwE_*`)**: Power deposited to electrons at different spatial locations
- **Ion/Fast Particle Power (`PwIF_*`)**: Power deposited to ions and fast particles

Each spatial location corresponds to a specific radial position in the plasma, allowing reconstruction of complete power deposition profiles.

## Usage Examples

### Basic Data Loading

```python
import pandas as pd

df = pd.read_pickle('data/datasets/HHFW-NSTX/PwE_.pkl')
print(f"shape: {df.shape}")
print(df.columns.tolist())
```

### Surrogate Model Training

Convert the pickle to CSV (or pass the DataFrame directly via the spec's
``dataset`` field) and drive the standard SURGE workflow:

```python
import pandas as pd
from surge import SurrogateWorkflowSpec, run_surrogate_workflow
from surge.hpc import ResourceSpec

df = pd.read_pickle('data/datasets/HHFW-NSTX/PwE_.pkl')
df.to_csv('hhfw_pwe.csv', index=False)

spec = SurrogateWorkflowSpec(
    dataset_path='hhfw_pwe.csv',
    models=[{'key': 'sklearn.random_forest',
             'params': {'n_estimators': 200}}],
    resources=ResourceSpec(device='cpu', num_workers=4),
    output_dir='.', run_tag='hhfw_pwe',
    overwrite_existing_run=True,
)
summary = run_surrogate_workflow(spec)
print(summary['models'][0]['metrics']['test'])
```

### Hyperparameter Optimization

Add an ``hpo`` block to any model entry; see
[`examples/quickstart.py`](../../../examples/quickstart.py) for a worked
Optuna-driven MLP example, or `docs/quickstart.rst` for the YAML form.

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

- `../../../examples/quickstart.py`: Public quickstart CLI (generic datasets)
- `../README.md`: General datasets overview
