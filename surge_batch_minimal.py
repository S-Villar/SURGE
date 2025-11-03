#!/usr/bin/env python3
"""Minimal wrapper to create a dataset batch using SURGE's DataGenerator.

This script simply delegates to surge_batch_setup.py. Prefer using that script
with a YAML config file.

Example:
  python surge_batch_minimal.py --config examples/batch_setup.yml
"""
from __future__ import annotations

import sys
from surge_batch_setup import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
