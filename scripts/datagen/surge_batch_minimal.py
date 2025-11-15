#!/usr/bin/env python3
"""Minimal wrapper to create a dataset batch using SURGE's DataGenerator.

This script simply delegates to surge_batch_setup.py. Prefer using that script
with a YAML config file.

Example:
  python surge_batch_minimal.py --config examples/batch_setup.yml
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path for imports (needed when run as script)
script_dir = Path(__file__).parent
surge_root = script_dir.parent.parent
if str(surge_root) not in sys.path:
    sys.path.insert(0, str(surge_root))

from scripts.datagen.surge_batch_setup import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
