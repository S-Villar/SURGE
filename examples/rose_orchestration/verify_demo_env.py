#!/usr/bin/env python3
"""Verify that the SURGE+ROSE+Rhapsody demo environment is usable."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _check_import(name: str) -> None:
    importlib.import_module(name)
    print(f"ok import {name}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print(f"SURGE root: {root}")
    _check_import("surge")
    _check_import("rose")
    _check_import("radical.asyncflow")
    _check_import("rhapsody")
    _check_import("pandas")
    _check_import("pyarrow")

    examples_dir = Path(__file__).resolve().parent
    required = [
        examples_dir / "workflow_rf.yaml",
        examples_dir / "workflow_mlp.yaml",
        examples_dir / "surge_train.py",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required demo file: {path}")
        print(f"ok file {path.name}")

    print("Demo environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
