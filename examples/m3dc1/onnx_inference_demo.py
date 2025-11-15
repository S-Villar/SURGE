"""Demonstrate loading a SURGE model exported to ONNX and running inference."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from surge.inference import ONNX_AVAILABLE, load_session, infer

MODEL_PATH = Path('models/sparc_random_forest.onnx')


def main():
    if not ONNX_AVAILABLE:
        print("onnxruntime not installed; skipping demo.")
        return

    if not MODEL_PATH.exists():
        print(f"ONNX model {MODEL_PATH} not found. Export a model before running the demo.")
        return

    session = load_session(MODEL_PATH)

    # Example normalized input vector [R0, a, kappa, delta, q0, q95, qmin, p0]
    sample = np.array([[3.13, 1.12, 1.85, 0.35, 1.05, 4.2, 0.85, 0.7]], dtype=np.float32)
    inputs = {session.get_inputs()[0].name: sample}

    outputs = infer(session, inputs)
    for name, value in outputs.items():
        print(f"{name}: {value.squeeze()}")


if __name__ == "__main__":
    main()
