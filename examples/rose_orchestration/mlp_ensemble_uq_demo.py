# -*- coding: utf-8 -*-
"""Run ``../mlp_ensemble_uq/mlp_ensemble_uq_demo.py`` from this directory."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parents[1] / "mlp_ensemble_uq" / "mlp_ensemble_uq_demo.py"
if not _target.is_file():
    raise FileNotFoundError(f"Missing ensemble demo: {_target}")
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
