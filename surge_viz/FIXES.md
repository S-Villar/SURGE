# Fixed Import Error

## Problem
The app was using relative imports (`.components.controls`) which don't work when Panel serves a file directly (not as a package).

## Solution
Changed all relative imports to absolute imports:
- `from .components.controls` → `from surge_viz.components.controls`
- `from .components.views` → `from surge_viz.components.views`
- `from .data_ops` → `from surge_viz.data_ops`
- `from .surge_api` → `from surge_viz.surge_api`

Also added path manipulation to ensure the parent directory is in sys.path.

## Next Steps

1. **The server should auto-reload** - Check your browser, the error should be gone
2. **If you still see errors**, you may need to:
   - Restart the server: Stop (Ctrl+C) and run again:
     ```bash
     panel serve surge_viz/app.py --dev --autoreload --port 5007
     ```
   - Hard refresh browser: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)

3. **If you see new errors about missing modules** (like holoviews), install dependencies:
   ```bash
   pip install panel holoviews hvplot bokeh pandas numpy
   ```

## Status

✅ **Fixed**: Relative import error resolved
⚠️ **Note**: You may need to install dependencies if not already installed

