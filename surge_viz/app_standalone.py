"""
Standalone version of SURGE visualizer app.

This version ensures proper servable marking for Panel.
"""

import sys
from pathlib import Path

# Add parent directory to path
__file_path__ = Path(__file__).resolve()
__parent_dir__ = __file_path__.parent.parent
if str(__parent_dir__) not in sys.path:
    sys.path.insert(0, str(__parent_dir__))

from surge_viz.app import SURGEVisualizerApp
import panel as pn

# Create app and panel
app = SURGEVisualizerApp()
panel = app.panel()

# Mark as servable - CRITICAL for Panel to serve it
panel.servable(title="SURGE Visualizer")

