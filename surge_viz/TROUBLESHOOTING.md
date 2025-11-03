# Troubleshooting SURGE Visualizer

## Server Started But Browser Didn't Open

### Solution 1: Manually Open Browser

The server is running! Just open your browser and navigate to:

```
http://localhost:5007/app
```

**Note:** The URL includes `/app` at the end, not just `:5007`

### Solution 2: Install watchfiles for Better Auto-reload

The warning mentions `watchfiles` for autoreload. Install it:

```bash
pip install watchfiles
```

Or:
```bash
conda install -c conda-forge watchfiles
```

### Solution 3: Open Browser Automatically

Add `--show` flag to automatically open browser:

```bash
panel serve surge_viz/app.py --dev --autoreload --port 5007 --show
```

### Solution 4: Check for Errors

If the page doesn't load, check the terminal for error messages. Common issues:

1. **Import errors**: Make sure you're in SURGE root directory
2. **Missing dependencies**: Install all requirements
3. **Port in use**: Try different port: `--port 5008`

## Common Issues

### Issue: "ModuleNotFoundError"
**Solution:** Make sure you're running from SURGE root directory:
```bash
cd /Users/asanche2/res/repo/SURGE
```

### Issue: Panel version mismatch
**Solution:** Update Panel:
```bash
pip install --upgrade panel
```

### Issue: Port already in use
**Solution:** Use different port or kill existing process:
```bash
panel serve surge_viz/app.py --port 5008
```

### Issue: App loads but shows errors
**Solution:** Check browser console (F12) for JavaScript errors. Check terminal for Python errors.

