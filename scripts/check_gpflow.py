#!/usr/bin/env python3
"""Verify GPflow availability and SURGE registry status."""

import sys

def main():
    print("GPflow availability check")
    print("=" * 50)

    # 1. Raw GPflow import
    try:
        import gpflow as gpf
        print("GPflow: OK (version:", getattr(gpf, "__version__", "unknown") + ")")
    except Exception as e:
        print("GPflow: NOT AVAILABLE -", type(e).__name__, str(e)[:80])
        print("  Install with: pip install gpflow tensorflow")
        print("  Or: pip install surge-surrogate[gpflow]")
        return 1

    # 2. TensorFlow (GPflow dependency)
    try:
        import tensorflow as tf
        print("TensorFlow: OK (version:", tf.__version__ + ")")
    except Exception as e:
        print("TensorFlow: NOT AVAILABLE -", type(e).__name__, str(e)[:80])
        return 1

    # 3. SURGE registry (run from project root: python scripts/check_gpflow.py)
    from surge.models import adapters as _  # noqa: F401
    from surge.registry import MODEL_REGISTRY

    keys = [e.key for e in MODEL_REGISTRY.list_entries()]
    print("Registered models:", keys)
    gpflow_registered = "gpflow.gpr" in keys
    print("gpflow.gpr in registry:", gpflow_registered)

    if gpflow_registered:
        print("\nGPflow is ready to use. Add to config:")
        print('  - key: gpflow.gpr')
        print('    name: my_gp')
        print('    params:')
        print('      maxiter: 200')
        return 0
    else:
        print("\nGPflow is installed but not registered. Check SURGE adapter imports.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
