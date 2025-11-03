# Fix: App Not Opening

## The Problem
The app won't open because **Panel is not installed** in your Python environment.

## The Solution

### Step 1: Install Panel and Dependencies

**Option A: Using pip (Quick)**
```bash
cd /Users/asanche2/res/repo/SURGE
pip install panel holoviews hvplot bokeh pandas numpy scipy scikit-learn
```

**Option B: Using conda (Recommended)**
```bash
cd /Users/asanche2/res/repo/SURGE
conda install -c conda-forge panel holoviews hvplot bokeh pandas numpy scipy scikit-learn
```

### Step 2: Verify Installation

Check if Panel is installed:
```bash
python -c "import panel; print(f'✅ Panel {panel.__version__} is installed!')"
```

If you see an error, Panel is **not installed** yet.

### Step 3: Run the App

Once Panel is installed:
```bash
cd /Users/asanche2/res/repo/SURGE
python surge_viz/run.py
```

The browser should open automatically at `http://localhost:5007/app`

## Troubleshooting

### If pip install fails:
- Try: `pip install --upgrade pip` first
- Try: `conda install panel` instead

### If you have multiple Python environments:
1. Check which Python you're using: `which python`
2. Make sure you install to the same Python that will run the app
3. Or activate the correct conda environment first

### Check what's missing:
```bash
python surge_viz/check_app.py
```

This will tell you exactly which dependencies are missing.

## Common Issues

**"No module named 'panel'"**
- Solution: Install Panel (see Step 1)

**"Port 5007 already in use"**
- Solution: Change port or kill existing process:
  ```bash
  lsof -ti:5007 | xargs kill -9
  ```

**"Browser doesn't open automatically"**
- Solution: Manually open `http://localhost:5007/app`

