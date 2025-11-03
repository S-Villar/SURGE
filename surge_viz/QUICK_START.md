# Quick Start - SURGE Visualizer

## Simple Launch (Easiest)

### Option 1: Using the run script
```bash
cd /Users/asanche2/res/repo/SURGE
python surge_viz/run.py
```

This will:
- Start the Panel server on port 5007
- Open your browser automatically
- Enable auto-reload when you change code

### Option 2: Direct Panel command
```bash
cd /Users/asanche2/res/repo/SURGE
panel serve surge_viz/app.py --dev --autoreload --port 5007
```

Then open your browser to: **http://localhost:5007/app**

### Option 3: With show=False (manual browser open)
```bash
panel serve surge_viz/app.py --dev --autoreload --port 5007 --show
```

## Stop the Server

Press `Ctrl+C` in the terminal where Panel is running.

## Troubleshooting

If you get "port already in use":
1. Find the process: `lsof -ti:5007`
2. Kill it: `kill -9 $(lsof -ti:5007)`
3. Or use a different port: `--port 5008`

If the browser doesn't open automatically:
- Manually go to: `http://localhost:5007/app`

## What to Expect

1. Terminal shows: "Bokeh app running at: http://localhost:5007/app"
2. Browser opens automatically (or open manually)
3. You see the SURGE Visualizer interface with 3 tabs:
   - 📊 Data
   - 🤖 Model  
   - 🔮 Inference & Compare

