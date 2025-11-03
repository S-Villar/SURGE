"""
Simplified app for quick testing.

This version has better error handling and works even if some dependencies are missing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import panel as pn
    import holoviews as hv
    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    print("⚠️ Panel or HoloViews not installed. Install with: pip install panel holoviews")

if PANEL_AVAILABLE:
    pn.extension('tabulator', 'holoviews', sizing_mode='stretch_width')
    hv.extension('bokeh')
    
    # Create a simple demo
    def create_simple_app():
        """Create a simple demo app."""
        title = pn.pane.Markdown("# 🚀 SURGE Visualizer", sizing_mode='stretch_width')
        
        info = pn.pane.Markdown("""
        ## Welcome to SURGE Visualization Platform
        
        This is the Panel-based visualization interface for SURGE surrogate models.
        
        ### Features:
        - Load datasets (PwE_.pkl, CSV, etc.)
        - Explore statistics and correlations
        - Train SURGE models
        - Run interactive inference
        - Visualize predictions (points, profiles, images)
        
        ### To Use:
        1. Install dependencies: `pip install panel holoviews bokeh pandas numpy`
        2. Load dataset using the file input
        3. Select parameters and outputs
        4. Train a model
        5. Run inference with parameter sliders
        """, sizing_mode='stretch_width')
        
        # Create a simple layout
        app = pn.Column(
            title,
            info,
            pn.pane.Markdown("### Status: Application Ready", sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        )
        
        return app
    
    # Create app
    app = create_simple_app()
    app.servable()

