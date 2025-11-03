#!/usr/bin/env python
"""
Quick diagnostic script to check if the app can start.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("SURGE Visualizer - Diagnostic Check")
print("=" * 70)

# Check Python version
print(f"\n1. Python version: {sys.version}")

# Check if Panel is installed
print("\n2. Checking dependencies...")
try:
    import panel as pn
    print(f"   ✅ Panel {pn.__version__}")
except ImportError as e:
    print(f"   ❌ Panel: {e}")
    print("   → Install with: pip install panel")
    sys.exit(1)

try:
    import pandas as pd
    print(f"   ✅ Pandas {pd.__version__}")
except ImportError as e:
    print(f"   ❌ Pandas: {e}")
    print("   → Install with: pip install pandas")

try:
    import holoviews as hv
    print(f"   ✅ HoloViews {hv.__version__}")
except ImportError as e:
    print(f"   ⚠️  HoloViews: {e}")
    print("   → Install with: pip install holoviews (optional)")

# Check if app can be imported
print("\n3. Checking app imports...")
try:
    from surge_viz.components.controls import DatasetControls
    print("   ✅ DatasetControls imported")
except Exception as e:
    print(f"   ❌ DatasetControls: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from surge_viz.components.views import DataView
    print("   ✅ DataView imported")
except Exception as e:
    print(f"   ❌ DataView: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from surge_viz.app import SURGEVisualizerApp
    print("   ✅ SURGEVisualizerApp imported")
except Exception as e:
    print(f"   ❌ SURGEVisualizerApp: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if app can be created
print("\n4. Creating app instance...")
try:
    app = SURGEVisualizerApp()
    print("   ✅ App instance created")
except Exception as e:
    print(f"   ❌ App creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if panel can be created
print("\n5. Creating panel...")
try:
    panel = app.panel()
    print(f"   ✅ Panel created: {type(panel)}")
except Exception as e:
    print(f"   ❌ Panel creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All checks passed! The app should be able to start.")
print("=" * 70)
print("\nTo start the app, run:")
print("  python surge_viz/run.py")
print("or")
print("  panel serve surge_viz/app.py --dev --autoreload --port 5007")

