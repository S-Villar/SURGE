# How to Run the SURGE Visualizer App

## Directory Structure

```
SURGE/                          ← You should be HERE (project root)
├── surge/                      ← SURGE package
├── surge_viz/                  ← Visualizer app
│   ├── app.py                  ← Main app file
│   ├── run.py                  ← Alternative launcher
│   └── ...
├── data/
│   └── datasets/
│       └── HHFW-NSTX/
│           └── PwE_.pkl       ← Example dataset
└── ...
```

## Quick Start

### Option 1: From SURGE Root Directory (Recommended)

```bash
# You should be in: /Users/asanche2/res/repo/SURGE
cd /Users/asanche2/res/repo/SURGE

# Install dependencies (if not already installed)
conda env create -f surge_viz/env.yml
conda activate surge-viz

# OR install manually
pip install panel>=1.5.0 holoviews>=1.19.0 hvplot bokeh pandas numpy

# Run the app
panel serve surge_viz/app.py --dev --autoreload --port 5007
```

### Option 2: Using the Run Script

```bash
# From SURGE root directory
cd /Users/asanche2/res/repo/SURGE

# Activate environment
conda activate surge-viz

# Run with Python
python surge_viz/run.py
```

### Option 3: Direct Python Import

```bash
# From SURGE root directory
cd /Users/asanche2/res/repo/SURGE

# Activate environment
conda activate surge-viz

# Run
python -c "from surge_viz.app import serve; serve(port=5007, dev=True)"
```

## Important Notes

1. **Always run from SURGE root directory** (`/Users/asanche2/res/repo/SURGE`)
   - This ensures proper imports
   - Paths to datasets are relative to root
   - SURGE package is importable

2. **Current directory when running:**
   ```bash
   cd /Users/asanche2/res/repo/SURGE  # ← You should be here
   panel serve surge_viz/app.py       # ← Then run this
   ```

3. **Access the app:**
   - Open browser: `http://localhost:5007`
   - The app will auto-open if `--show` or `show=True` is set

## Troubleshooting

If you get import errors:
- Make sure you're in the SURGE root directory
- Ensure SURGE package is installed: `pip install -e .`
- Check Python path includes current directory

If Panel doesn't start:
- Check Panel is installed: `python -c "import panel; print(panel.__version__)"`
- Verify port 5007 is not in use
- Try different port: `panel serve surge_viz/app.py --port 5008`

