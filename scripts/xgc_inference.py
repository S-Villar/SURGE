#!/usr/bin/env python3
"""
XGC A_parallel inference using SURGE-trained models.

Loads models from a SURGE run and runs inference on OLCF hackathon data.
Complements Ah_prediction_NN1_ASV_Geometrical_inference.ipynb by using
SURGE's standardized model loading and scalers.

Usage:
  # Inference on set1 (same as training)
  python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025

  # Inference on set2_beta0p5 (cross-set evaluation)
  python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \\
      --eval-set set2_beta0p5 --sample 10000

  # Use specific model
  python scripts/xgc_inference.py --run-dir runs/xgc_aparallel_set1_v3 \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 \\
      --model xgc_rf_aparallel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="XGC inference with SURGE models")
    parser.add_argument("--run-dir", type=Path, required=True, help="SURGE run directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="OLCF hackathon data dir")
    parser.add_argument("--eval-set", type=str, default="set1", help="set1 or set2_beta0p5")
    parser.add_argument("--sample", type=int, default=10000, help="Sample size (None = full)")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: first available)")
    parser.add_argument("--row-range", type=str, default=None, help="start,end for row range")
    parser.add_argument("--out", type=Path, default=None, help="Save predictions to CSV")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 1

    # Load workflow summary for model list
    summary_path = run_dir / "workflow_summary.json"
    if not summary_path.exists():
        print(f"No workflow_summary.json in {run_dir}", file=sys.stderr)
        return 1

    with summary_path.open() as f:
        summary = json.load(f)

    model_entries = summary.get("models", [])
    if not model_entries:
        print("No models in run", file=sys.stderr)
        return 1

    # Pick model
    if args.model:
        model_entry = next((m for m in model_entries if m.get("name") == args.model), None)
        if not model_entry:
            print(f"Model {args.model} not found. Available: {[m.get('name') for m in model_entries]}", file=sys.stderr)
            return 1
    else:
        model_entry = model_entries[0]

    model_name = model_entry.get("name", "unknown")
    model_path = run_dir / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        model_path = run_dir / "models" / f"{model_name}.pt"
    if not model_path.exists():
        print(f"Model file not found for {model_name}", file=sys.stderr)
        return 1

    # Load scalers
    import joblib
    input_scaler_path = run_dir / "scalers" / "input_scaler.joblib"
    output_scaler_path = run_dir / "scalers" / "output_scaler.joblib"
    input_scaler = joblib.load(input_scaler_path) if input_scaler_path.exists() else None
    output_scaler = joblib.load(output_scaler_path) if output_scaler_path.exists() else None

    # Load dataset
    from surge.dataset import SurrogateDataset

    hints = {"set_name": args.eval_set}
    if args.row_range:
        start, end = map(int, args.row_range.split(","))
        hints["row_range"] = (start, end)
        sample = None
    else:
        sample = args.sample

    dataset = SurrogateDataset.from_path(
        args.data_dir,
        format="xgc",
        sample=sample,
        analyzer_kwargs={"hints": hints},
    )
    if dataset.df is None or len(dataset.df) == 0:
        print("Dataset empty", file=sys.stderr)
        return 1

    X = dataset.df[dataset.input_columns].values.astype(np.float64)
    y_true = dataset.df[dataset.output_columns].values.astype(np.float64)

    if input_scaler is not None:
        X_scaled = input_scaler.transform(X)
    else:
        X_scaled = X

    # Load model and predict
    from surge.io.load_compat import load_model_compat

    adapter = load_model_compat(model_path, model_entry)
    y_pred = adapter.predict(X_scaled)
    if hasattr(y_pred, "numpy"):
        y_pred = y_pred.numpy()
    y_pred = np.asarray(y_pred)

    if output_scaler is not None:
        y_pred = output_scaler.inverse_transform(y_pred)

    # Metrics (output_1 = A_parallel)
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    n_out = y_true.shape[1]
    print(f"Inference: {model_name} on {args.eval_set} ({len(X)} samples)")
    print("-" * 50)
    for i in range(n_out):
        out_name = "A_parallel" if i == 1 else f"output_{i}"
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f"  {out_name}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    if args.out:
        import pandas as pd
        out_df = pd.DataFrame(
            np.hstack([y_true, y_pred]),
            columns=[f"y_true_{i}" for i in range(n_out)] + [f"y_pred_{i}" for i in range(n_out)],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Saved: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
