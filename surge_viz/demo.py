"""
Demo script for SURGE visualizer.

Demonstrates loading PwE_.pkl dataset and running inference.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import panel as pn
from surge_viz.app import SURGEVisualizerApp

if __name__ == "__main__":
    # Create app
    app = SURGEVisualizerApp()
    
    # Create panel
    panel = app.panel()
    
    # Serve
    panel.servable()
    
    print("🚀 SURGE Visualizer Demo")
    print("=" * 60)
    print("Starting Panel server...")
    print("Open your browser to: http://localhost:5007")
    print("=" * 60)
    print("\nTo run: panel serve surge_viz/demo.py --dev --autoreload")

