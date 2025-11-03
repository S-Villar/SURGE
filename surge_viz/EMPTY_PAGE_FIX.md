# Fix: Empty Page in Safari

## The Problem
The app opens in Safari but shows an empty/blank page.

## Common Causes and Fixes

### 1. Wrong URL

**Make sure you're using the correct URL:**

- ✅ **Correct:** `http://localhost:5007/app`
- ❌ **Wrong:** `http://localhost:5007` or `http://localhost:5007/`

Panel serves apps at `/app` path by default when using `panel serve app.py`.

### 2. Check Browser Console

1. Open Safari Developer Tools:
   - Safari → Settings → Advanced → Enable "Show Develop menu"
   - Develop → Show Web Inspector
   
2. Look for JavaScript errors in the Console tab

3. Check the Network tab to see if resources are loading

### 3. Try Different Browser

Sometimes Safari has caching issues. Try:
- Chrome: `http://localhost:5007/app`
- Firefox: `http://localhost:5007/app`

### 4. Clear Browser Cache

1. Safari → Develop → Empty Caches
2. Or hard refresh: Cmd+Shift+R
3. Or try incognito/private window

### 5. Check Terminal Output

Look at the terminal where Panel is running. You should see:
- `Starting Bokeh server...`
- `Bokeh app running at: http://localhost:5007/app`

If you see errors, that's the problem.

### 6. Check What's Being Served

The app should show a title "🚀 SURGE Visualization Platform" at the top. If you see nothing:
- The page might not be rendering
- There might be a JavaScript error
- The Panel app might not be initialized correctly

## Quick Debug Steps

1. **Verify URL is correct:**
   ```
   http://localhost:5007/app
   ```

2. **Check terminal for errors:**
   - Look for any Python tracebacks
   - Look for "Application did not publish any contents" error

3. **Check browser console:**
   - Open Developer Tools (F12 or Cmd+Option+I)
   - Look for red error messages

4. **Try a different port:**
   ```bash
   panel serve surge_viz/app.py --port 5008
   ```
   Then go to: `http://localhost:5008/app`

5. **Verify the app loads:**
   ```bash
   python surge_viz/check_app.py
   ```

## If Still Empty

Share:
1. What you see in the browser (completely blank? error message?)
2. Any errors from the terminal
3. Any errors from browser console

