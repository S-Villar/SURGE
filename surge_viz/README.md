# SURGE Visualization Platform

Interactive web-based visualization and inference platform for SURGE surrogate models.

## Features

- **Dataset Loading & Analysis**: Load datasets (PwE_.pkl, CSV, etc.) and explore statistics, correlations, and distributions
- **Model Training & Management**: Train SURGE models (Random Forest, MLP, PyTorch MLP, GPflow GPR) and load existing models
- **Interactive Inference**: Run predictions with parameter sliders and visualize results
- **Multiple Visualization Types**: 
  - Point predictions (scalar values)
  - Profile predictions (1D profiles with ground truth comparison)
  - Image predictions (2D heatmaps)
- **Comparison Tools**: Compare ground truth vs predictions with multiple metrics (RMSE, MAE, R²)

## Installation

### Option 1: Using Conda Environment

```bash
cd surge_viz
conda env create -f env.yml
conda activate surge-viz
```

### Option 2: Using pip

```bash
pip install panel>=1.5.0 holoviews>=1.19.0 hvplot bokeh datashader numpy pandas scipy scikit-learn xarray pyyaml
```

## Usage

### Start the Application

```bash
# From the SURGE root directory
panel serve surge_viz/app.py --dev --autoreload --port 5007

# Or using Python
python -m surge_viz.app
```

The application will be available at `http://localhost:5007`

### Quick Start with PwE_.pkl

1. **Load Dataset**: Click "Load Dataset" and select `data/datasets/HHFW-NSTX/PwE_.pkl`
2. **Select Parameters**: Choose input parameter columns from the dropdown
3. **Select Outputs**: Choose output target columns
4. **Train Model**: Go to "Model" tab, select model type, click "Train Model"
5. **Run Inference**: Go to "Inference & Compare" tab, adjust parameter sliders, click "Run Inference"
6. **View Results**: See predictions, residuals, and metrics

## Project Structure

```
surge_viz/
├── app.py              # Main Panel application
├── surge_api.py        # SURGE backend wrapper
├── data_ops.py         # Dataset operations (load, stats, correlations)
├── viz.py              # HoloViews plotting functions
├── components/
│   ├── controls.py     # Param-based control widgets
│   └── views.py         # Composable Panel views
├── env.yml             # Conda environment
└── README.md           # This file
```

## Development

The application uses:
- **Panel**: For web UI framework
- **HoloViews**: For interactive scientific visualizations
- **Bokeh**: Backend for interactive plots
- **SURGE**: Backend ML framework

## Future Enhancements

- HPC integration (Frontier/Perlmutter via SSH+SLURM)
- Async job submission and status polling
- Profile visualization with radial coordinate (ρ)
- Advanced image comparison tools
- Model comparison across multiple trained models

