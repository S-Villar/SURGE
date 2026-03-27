#!/usr/bin/env python3
"""
XGC A_parallel predictions animation: GT vs prediction across datastreamsets.

Creates a GIF showing how model predictions track ground truth across contiguous
datastreamsets. Each frame is one datastreamset with a scatter plot of GT vs pred.

Supports Model1 (61 cols), Model12 (61 cols), ModelS (20 cols), and set1_v3 (201 cols).

Usage:
  python scripts/xgc_predictions_animation.py --run-dir runs/xgc_aparallel_set1_v3 \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025

  python scripts/xgc_predictions_animation.py --run-dir runs/xgc_model1_61cols \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1

  python scripts/xgc_predictions_animation.py --run-dir runs/xgc_model12_finetune \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set2_beta0p5
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


def _get_effective_input_columns(spec_dict, input_cols):
    """Get input columns from spec (n_inputs or input_columns) or full list."""
    spec_input_cols = spec_dict.get("input_columns")
    spec_n_inputs = spec_dict.get("n_inputs")
    if spec_input_cols and len(spec_input_cols) > 0:
        return [c for c in spec_input_cols if c in input_cols]
    if spec_n_inputs and input_cols:
        return input_cols[: spec_n_inputs]
    return input_cols


def _load_olcf_npy(data_dir, set_name="set1", sample=None, random_state=42, row_range=None):
    """Load OLCF hackathon .npy files into DataFrame (standalone, no surge)."""
    import pandas as pd
    data_dir = Path(data_dir)
    prefix = f"data_nprev5_{set_name}"
    data_path = data_dir / f"{prefix}_data.npy"
    target_path = data_dir / f"{prefix}_target.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")
    data = np.load(data_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    n_samples, n_inputs = data.shape
    _, n_outputs = target.shape
    if row_range is not None:
        start, end = row_range
        data = np.asarray(data[start:end])
        target = np.asarray(target[start:end])
    elif sample is not None and 0 < sample < n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=sample, replace=False)
        data = np.asarray(data[idx])
        target = np.asarray(target[idx])
    input_cols = [f"input_{i}" for i in range(n_inputs)]
    output_cols = [f"output_{i}" for i in range(n_outputs)]
    df_data = pd.DataFrame(data, columns=input_cols)
    df_target = pd.DataFrame(target, columns=output_cols)
    df = pd.concat([df_data, df_target], axis=1)
    return df, input_cols, output_cols


def main():
    parser = argparse.ArgumentParser(
        description="XGC predictions animation: GT vs pred across datastreamsets"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="SURGE run directory")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="OLCF hackathon data dir",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default="set1",
        help="set1 or set2_beta0p5",
    )
    parser.add_argument(
        "--datastreamset-size",
        type=int,
        default=5000,
        help="Rows per datastreamset (default 5000 for faster animation)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Max frames (datastreamsets) in animation (default 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: first MLP)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename stem (default: xgc_predictions_animation_{eval_set})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for GIF (default 2)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 1

    # Load spec
    spec_file = run_dir / "spec.yaml"
    summary_file = run_dir / "workflow_summary.json"
    if not spec_file.exists() or not summary_file.exists():
        print("Need spec.yaml and workflow_summary.json", file=sys.stderr)
        return 1

    try:
        import yaml
        with spec_file.open() as f:
            spec_dict = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Failed to load spec: {e}", file=sys.stderr)
        return 1

    with summary_file.open() as f:
        summary = json.load(f)

    model_entries = summary.get("models", [])
    if not model_entries:
        print("No models in run", file=sys.stderr)
        return 1

    # Prefer RF (joblib loads reliably); else MLP
    if args.model:
        model_entry = next((m for m in model_entries if m.get("name") == args.model), None)
    else:
        model_entry = next(
            (m for m in model_entries if "rf" in m.get("name", "").lower()),
            model_entries[0],
        )
    if not model_entry:
        print("Model not found", file=sys.stderr)
        return 1

    model_name = model_entry.get("name", "unknown")
    model_path = run_dir / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        model_path = run_dir / "models" / f"{model_name}.pt"
    if not model_path.exists():
        print(f"Model file not found: {model_name}", file=sys.stderr)
        return 1

    # Scalers
    import joblib
    scalers_dir = run_dir / "scalers"
    input_scaler = None
    output_scaler = None
    if (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")
    elif (scalers_dir / "input_scaler.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "input_scaler.joblib")
    if (scalers_dir / "outputs.joblib").exists():
        output_scaler = joblib.load(scalers_dir / "outputs.joblib")
    elif (scalers_dir / "output_scaler.joblib").exists():
        output_scaler = joblib.load(scalers_dir / "output_scaler.joblib")

    # Load model: joblib for RF/sklearn (avoids full surge import)
    import joblib as _joblib
    if model_path.suffix == ".joblib":
        adapter = _joblib.load(model_path)
    else:
        from surge.io.load_compat import load_model_compat
        adapter = load_model_compat(model_path, model_entry)

    # Build frames
    frames = []
    hints = {"set_name": args.eval_set}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required", file=sys.stderr)
        return 1

    try:
        import imageio
    except ImportError:
        try:
            from PIL import Image
            imageio = None
        except ImportError:
            print("imageio or Pillow required for GIF. pip install imageio", file=sys.stderr)
            return 1

    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_idx in range(args.max_frames):
        start = ds_idx * args.datastreamset_size
        end = start + args.datastreamset_size
        row_range = (start, end)

        try:
            df, input_cols, output_cols = _load_olcf_npy(
                args.data_dir,
                set_name=args.eval_set,
                sample=None,
                row_range=row_range,
            )
        except Exception as e:
            print(f"Datastreamset {ds_idx}: load failed: {e}")
            break

        if df is None or len(df) == 0:
            break

        eff_cols = _get_effective_input_columns(spec_dict, input_cols)
        X = df[eff_cols].values.astype(np.float64)
        y_true = df[output_cols].values.astype(np.float64)

        if input_scaler is not None:
            X = input_scaler.transform(X)

        y_pred = adapter.predict(X)
        if hasattr(y_pred, "numpy"):
            y_pred = y_pred.numpy()
        y_pred = np.asarray(y_pred)

        if output_scaler is not None:
            y_pred = output_scaler.inverse_transform(y_pred)

        # output_1 = A_parallel
        out_idx = 1 if y_true.shape[1] > 1 else 0
        yt = y_true[:, out_idx]
        yp = y_pred[:, out_idx]

        # Frame: scatter GT vs pred
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hexbin(yt, yp, gridsize=40, cmap="plasma_r", mincnt=1)
        lim_lo = min(yt.min(), yp.min())
        lim_hi = max(yt.max(), yp.max())
        pad = (lim_hi - lim_lo) * 0.05 or 0.01
        ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], "k--", alpha=0.5)
        ax.set_xlim(lim_lo - pad, lim_hi + pad)
        ax.set_ylim(lim_lo - pad, lim_hi + pad)
        ax.set_xlabel("Ground Truth A_parallel")
        ax.set_ylabel("Prediction A_parallel")
        r2 = np.corrcoef(yt, yp)[0, 1] ** 2 if len(yt) > 1 else 0.0
        ax.set_title(f"Datastreamset {ds_idx} (rows {start}-{end})  R²={r2:.3f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()  # drop alpha for GIF
        frames.append(rgb)
        plt.close(fig)

    if not frames:
        print("No frames generated", file=sys.stderr)
        return 1

    out_stem = args.output_name or f"xgc_predictions_animation_{args.eval_set}"
    gif_path = output_dir / f"{out_stem}.gif"

    if imageio:
        imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)
    else:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / args.fps),
            loop=0,
        )

    print(f"Animation saved: {gif_path} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
