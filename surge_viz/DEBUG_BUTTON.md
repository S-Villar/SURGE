# Debugging Load Button Issue

## What to Check

1. **Check Terminal Output**
   When you click the button, you should see:
   ```
   ======================================================================
   🔵 LOAD BUTTON CLICKED!
   ======================================================================
   ```
   
   If you DON'T see this, the button callback isn't wired correctly.

2. **Check Browser Console**
   - Safari → Develop → Show Web Inspector
   - Look for JavaScript errors in Console tab
   - Look for network errors in Network tab

3. **Check Button State**
   - Is the button disabled? (It should be enabled after selecting a file)
   - Does the button show the filename after selecting a file?

4. **What the Terminal Should Show:**
   
   **If button is wired correctly:**
   ```
   DEBUG: Load button wired directly
   ```
   
   **When you click the button:**
   ```
   ======================================================================
   🔵 LOAD BUTTON CLICKED!
   ======================================================================
   DEBUG: File input value type: <class 'bytes'>
   DEBUG: File bytes length: <number>
   ```
   
   **If it fails, you'll see:**
   ```
   ❌ DEBUG: No file selected!
   ```
   or
   ```
   ❌ ERROR in _on_dataset_load: <error>
   ```

## Common Issues

1. **Button callback not firing** → Check terminal for "LOAD BUTTON CLICKED!" message
2. **No file selected** → Button is disabled or file wasn't actually selected
3. **JavaScript error** → Check browser console
4. **Panel widget recreation** → Callback needs to be re-wired (I've added this)

## Next Steps

Share what you see in:
1. Terminal output when clicking the button
2. Browser console errors (if any)
3. Whether the button is enabled/disabled
4. Whether the filename shows in the button after selecting a file

