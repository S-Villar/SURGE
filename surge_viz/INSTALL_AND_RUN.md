# Install Dependencies and Run SURGE Visualizer

## Problem
The app won't open because `panel` (and possibly other dependencies) are not installed.

## Quick Fix

### Option 1: Install dependencies directly (Quick)
```bash
cd /Users/asanche2/res/repo/SURGE
pip install panel holoviews hvplot bokeh pandas numpy scipy scikit-learn
```

### Option 2: Create conda environment (Recommended)
```bash
cd /Users/asanche2/res/repo/SURGE

# Create environment from the provided file
conda env create -f surge_viz/env.yml

# Activate it
conda activate surge-viz

# Then run the app
python surge_viz/run.py
```

### Option 3: Check what's missing
```bash
cd /Users/asanche2/res/repo/SURGE
python surge_viz/check_app.py
```

This will tell you exactly what's missing.

## After Installing

Once dependencies are installed, run:
```bash
cd /Users/asanche2/res/repo/SURGE
python surge_viz/run.py
```

Or:
```bash
panel serve surge_viz/app.py --dev --autoreload --port 5007
```

## Verify Installation

Check if Panel is installed:
```bash
python -c "import panel; print(f'Panel {panel.__version__} installed!')"
```

If you get an error, Panel is not installed.

