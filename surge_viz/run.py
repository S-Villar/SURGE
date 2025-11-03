#!/usr/bin/env python
"""
Launcher script for SURGE visualizer.

Run this script to start the Panel application.
"""

import sys
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import panel as pn

if __name__ == "__main__":
    print("🚀 Starting SURGE Visualizer...")
    print("=" * 60)
    print("📂 App file: surge_viz/app.py")
    print("🌐 Opening at: http://localhost:5007/app")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Get the app.py path relative to this file
    app_dir = Path(__file__).parent
    app_file = app_dir / "app.py"
    
    # Serve with Panel - use pn.serve() with the file path
    try:
        pn.serve(
            str(app_file),
            port=5007,
            dev=True,
            autoreload=True,
            show=True,
            start=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

