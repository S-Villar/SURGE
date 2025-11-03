# Install Dependencies for SURGE Visualizer

## Current Issue: Missing Dependencies

The 500 Internal Server Error is likely due to missing Python packages.

## Quick Fix

The Panel server is running with Python 3.12 from Anaconda, but `holoviews` and other dependencies are not installed.

### Install Dependencies Now:

```bash
# Option 1: Install everything at once
pip install panel>=1.5.0 holoviews>=1.19.0 hvplot bokeh pandas numpy scipy scikit-learn

# Option 2: Use conda (if you prefer)
conda install -c conda-forge panel holoviews hvplot bokeh pandas numpy scipy scikit-learn

# Option 3: Create dedicated environment
conda env create -f surge_viz/env.yml
conda activate surge-viz
```

### Then Restart the Server:

1. **Stop the current server**: Press `Ctrl+C` in the terminal where Panel is running
2. **Restart it**:
   ```bash
   panel serve surge_viz/app.py --dev --autoreload --port 5007
   ```
3. **Refresh your browser**

## Verify Installation

Check if dependencies are installed:
```bash
python -c "import panel, holoviews, pandas, numpy; print('✅ All dependencies installed!')"
```

## What's Required

- **Panel** >= 1.5.0 (for web UI)
- **HoloViews** >= 1.19.0 (for interactive plots)
- **hvplot** (for pandas integration)
- **Bokeh** >= 3.0.0 (backend for plots)
- **Pandas** (for data handling)
- **NumPy** (for numerical operations)
- **SciPy** (for correlations, distance metrics)
- **scikit-learn** (for SURGE backend)

