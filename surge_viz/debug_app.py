"""
Debug version of the app to help diagnose the 500 error.

Run this to see what's actually failing.
"""

import sys
from pathlib import Path

# Add parent directory to path
__file_path__ = Path(__file__).resolve()
__parent_dir__ = __file_path__.parent.parent
if str(__parent_dir__) not in sys.path:
    sys.path.insert(0, str(__parent_dir__))

print("=" * 70)
print("DEBUG: Loading SURGE Visualizer")
print("=" * 70)

try:
    print("\n1. Importing holoviews...")
    import holoviews as hv
    print(f"   ✅ HoloViews {hv.__version__}")
except Exception as e:
    print(f"   ❌ HoloViews: {e}")
    sys.exit(1)

try:
    print("\n2. Importing panel...")
    import panel as pn
    print(f"   ✅ Panel {pn.__version__}")
except Exception as e:
    print(f"   ❌ Panel: {e}")
    sys.exit(1)

try:
    print("\n3. Importing pandas...")
    import pandas as pd
    print(f"   ✅ Pandas {pd.__version__}")
except Exception as e:
    print(f"   ❌ Pandas: {e}")

try:
    print("\n4. Importing surge_viz components...")
    from surge_viz.components.controls import DatasetControls
    print("   ✅ DatasetControls")
except Exception as e:
    print(f"   ❌ DatasetControls: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n5. Creating DatasetControls...")
    controls = DatasetControls()
    print("   ✅ DatasetControls created")
except Exception as e:
    print(f"   ❌ Error creating DatasetControls: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n6. Creating SURGEVisualizerApp...")
    from surge_viz.app import SURGEVisualizerApp
    app = SURGEVisualizerApp()
    print("   ✅ App created")
except Exception as e:
    print(f"   ❌ Error creating app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n7. Creating panel...")
    panel = app.panel()
    print(f"   ✅ Panel created: {type(panel)}")
except Exception as e:
    print(f"   ❌ Error creating panel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n8. Marking as servable...")
    panel.servable(title="SURGE Visualizer")
    print("   ✅ Marked as servable")
except Exception as e:
    print(f"   ❌ Error marking as servable: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All checks passed! The app should work now.")
print("=" * 70)

