#!/usr/bin/env python3
"""
Export XGC SURGE model to ONNX and scaler params for C++ inference.

Usage:
  python scripts/xgc_export_onnx.py --run-dir runs/xgc_aparallel_set1_v3
  python scripts/xgc_export_onnx.py --run-dir runs/xgc_model1_61cols --model xgc_mlp_aparallel

Output:
  run_dir/onnx/xgc_mlp_aparallel.onnx
  run_dir/onnx/scaler_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export XGC model to ONNX")
    parser.add_argument("--run-dir", type=Path, required=True, help="SURGE run directory")
    parser.add_argument("--model", type=str, default="xgc_mlp_aparallel", help="Model name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: run_dir/onnx)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 1

    model_path = run_dir / "models" / f"{args.model}.joblib"
    if not model_path.exists():
        model_path = run_dir / "models" / f"{args.model}.pt"
    if not model_path.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    import joblib
    from surge.io.load_compat import load_model_compat

    with (run_dir / "workflow_summary.json").open() as f:
        summary = json.load(f)
    model_entry = next((m for m in summary.get("models", []) if m.get("name") == args.model), {})
    if not model_entry:
        model_entry = {"name": args.model}

    adapter = load_model_compat(model_path, model_entry)
    if not hasattr(adapter, "_model") or not hasattr(adapter._model, "model"):
        print("Only PyTorch MLP models supported for ONNX export", file=sys.stderr)
        return 1

    pt_model = adapter._model.model
    n_inputs = pt_model.input_size
    n_outputs = pt_model.output_size

    try:
        import torch
    except ImportError:
        print("PyTorch required for ONNX export", file=sys.stderr)
        return 1

    # Move to CPU and eval mode for export (avoids device mismatch)
    pt_model.network.cpu().eval()
    dummy = torch.randn(1, n_inputs)
    output_dir = args.output_dir or (run_dir / "onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{args.model}.onnx"

    torch.onnx.export(
        pt_model.network,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
        dynamo=False,  # Use legacy exporter for compatibility
    )
    print(f"Exported ONNX: {onnx_path}")

    # Export scaler params for C++
    scaler_params = {
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "input_mean": pt_model.scaler_X.mean_.tolist(),
        "input_scale": (pt_model.scaler_X.scale_ + 1e-10).tolist(),
        "output_mean": pt_model.scaler_y.mean_.tolist(),
        "output_scale": (pt_model.scaler_y.scale_ + 1e-10).tolist(),
    }
    scaler_path = output_dir / "scaler_params.json"
    with scaler_path.open("w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Exported scaler params: {scaler_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
