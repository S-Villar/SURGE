#!/usr/bin/env python3
"""
XGC spatial A_parallel animation: 2D contour GT vs inference over timesteps.

Creates a GIF with two panels per frame: left = ground truth A_parallel on R-Z
circular geometry, right = surrogate inference. Loops over timesteps.

Supports Model1 (61 cols), Model12 (61 cols), ModelS (20 cols), set1_v3 (201 cols).

Usage:
  python scripts/xgc_spatial_aparallel_animation.py --run-dir runs/xgc_model1_61cols \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set1

  python scripts/xgc_spatial_aparallel_animation.py --run-dir runs/xgc_model12_finetune \\
      --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025 --eval-set set2_beta0p5

  # Batch: run for all four models
  for r in runs/xgc_model1_61cols runs/xgc_model12_finetune runs/xgc_modelS_shap20 runs/xgc_aparallel_set1_v3; do
    python scripts/xgc_spatial_aparallel_animation.py --run-dir $r --data-dir /global/cfs/projectdirs/m499/olcf_ai_hackathon_2025
  done
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

N_FEATURES = 14  # per-node features for R-Z column offset


def _compute_nsize(tags: np.ndarray) -> int:
    """Compute rows per timestep from tags structure."""
    local_max_indices = np.where(
        (tags[1:, 1] == 0) & (tags[:-1, 1] > 0)
    )[0] + 1
    if len(local_max_indices) < 2:
        raise ValueError("Could not determine nsize from tags")
    return int(np.diff(local_max_indices)[0])


def _extract_timestep_slice(
    data: np.ndarray,
    target: np.ndarray,
    tags: np.ndarray,
    istep: int,
    nsize: int,
    iphi: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data for one timestep and iphi=0 mask.

    Ground truth: target[:,1] = A_parallel (dAh/dt) at this timestep — the quantity
    we infer from current+previous state. Display: z_gt = target[:,1] * 10**norm.

    Returns:
        data_tslice: (nsize, 201) full timestep data
        target_tslice: (nsize, 2) full timestep target
        rz: (n_points, 2) R-Z coords for 2D plot (iphi=0)
        norm: (n_points,) norm factor for display (iphi=0)
        z_gt: (n_points,) ground truth for display (target[:,1] * 10**norm)
        mask: (nsize,) boolean mask for iphi=0
    """
    start = istep * nsize
    end = (istep + 1) * nsize
    data_tslice = np.asarray(data[start:end])
    target_tslice = np.asarray(target[start:end])
    tags_tslice = tags[start:end]

    mask = (tags_tslice[:, 1] == iphi).flatten()
    if not np.any(mask):
        # Fallback: use first unique iphi if iphi=0 has no points (e.g. set1 structure)
        unique_iphi = np.unique(tags_tslice[:, 1])
        iphi_use = int(unique_iphi[0]) if len(unique_iphi) > 0 else 0
        mask = (tags_tslice[:, 1] == iphi_use).flatten()
    rz = data_tslice[mask, N_FEATURES * 4 : N_FEATURES * 4 + 2]
    norm = data_tslice[mask, N_FEATURES]
    dAhdt = target_tslice[mask, 1]
    z_gt = np.asarray(dAhdt, dtype=np.float64) * (10.0 ** np.asarray(norm, dtype=np.float64))

    return data_tslice, target_tslice, rz, norm, z_gt, mask


def _get_effective_input_columns(spec_dict: dict, n_inputs: int) -> list[int]:
    """Get column indices from spec (n_inputs or input_columns)."""
    spec_input_cols = spec_dict.get("input_columns")
    spec_n_inputs = spec_dict.get("n_inputs")
    if spec_input_cols and len(spec_input_cols) > 0:
        # Map names like input_0, input_61 to indices
        return [int(c.replace("input_", "")) for c in spec_input_cols]
    if spec_n_inputs and spec_n_inputs > 0:
        return list(range(spec_n_inputs))
    return list(range(n_inputs))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XGC spatial A_parallel animation: GT vs inference 2D contours"
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
        "--max-frames",
        type=int,
        default=10,
        help="Max timesteps (frames) in animation (default 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: xgc_mlp_aparallel)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename stem (default: xgc_spatial_aparallel_animation_{model_label})",
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
    if not spec_file.exists():
        print("Need spec.yaml", file=sys.stderr)
        return 1

    try:
        import yaml
        with spec_file.open() as f:
            spec_dict = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Failed to load spec: {e}", file=sys.stderr)
        return 1

    model_entries = []
    if summary_file.exists():
        with summary_file.open() as f:
            summary = json.load(f)
        model_entries = summary.get("models", [])
    if not model_entries:
        # Fallback: infer from models dir
        models_dir = run_dir / "models"
        if models_dir.exists():
            for p in sorted(models_dir.glob("*.joblib")):
                name = p.stem
                if name.endswith("_training_history"):
                    continue
                backend = "pytorch" if "mlp" in name.lower() else "sklearn"
                model_entries.append({"name": name, "backend": backend})
    if not model_entries:
        print("No models in run", file=sys.stderr)
        return 1

    # Prefer MLP for spatial animation
    if args.model:
        model_entry = next((m for m in model_entries if m.get("name") == args.model), None)
    else:
        model_entry = next(
            (m for m in model_entries if "mlp" in m.get("name", "").lower()),
            next((m for m in model_entries if "rf" in m.get("name", "").lower()), model_entries[0]),
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

    # Load model (use load_model_compat to handle PyTorch .joblib files saved with torch.save)
    from surge.io.load_compat import load_model_compat
    adapter = load_model_compat(model_path, model_entry)

    # Load data and tags
    data_dir = Path(args.data_dir)
    prefix = f"data_nprev5_{args.eval_set}"
    data_path = data_dir / f"{prefix}_data.npy"
    target_path = data_dir / f"{prefix}_target.npy"
    tags_path = data_dir / f"{prefix}_tags.npy"
    if not data_path.exists():
        print(f"Data not found: {data_path}", file=sys.stderr)
        return 1
    if not target_path.exists():
        print(f"Target not found: {target_path}", file=sys.stderr)
        return 1
    if not tags_path.exists():
        print(f"Tags not found: {tags_path}", file=sys.stderr)
        return 1

    data = np.load(data_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    tags = np.load(tags_path, mmap_mode="r")

    n_samples, n_inputs = data.shape
    nsize = _compute_nsize(tags)
    n_timesteps = n_samples // nsize
    max_frames = min(args.max_frames, n_timesteps)
    if max_frames <= 0:
        print("No timesteps available", file=sys.stderr)
        return 1

    eff_cols = _get_effective_input_columns(spec_dict, n_inputs)

    _used_robust_fallback = [False]  # mutable to allow closure to set

    def _robust_transform(scaler, X: np.ndarray) -> np.ndarray:
        """Transform X with scaler. Use per-batch z-score if scaler would blow up (near-zero variance)."""
        if scaler is None:
            return X
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale_ok = (scale > 1e-4) & (scale < 1e6)
        if np.all(scale_ok):
            return scaler.transform(X)
        # Fallback: per-batch z-score when scaler has pathological scale (avoids 1e20 blow-up)
        if not _used_robust_fallback[0]:
            _used_robust_fallback[0] = True
            print(
                "Note: Input scaler has near-zero or huge scale; using per-batch z-score for inference.",
                file=sys.stderr,
            )
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std_safe = np.where(std > 1e-8, std, 1.0)
        return (X - mean) / std_safe

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as mcm
        import matplotlib.tri as tri
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

    # First pass: compute global vmin/vmax from all frames
    all_z_gt = []
    all_z_pred = []
    for istep in range(max_frames):
        data_tslice, target_tslice, rz, norm, z_gt, mask = _extract_timestep_slice(
            data, target, tags, istep, nsize
        )
        X_full = np.asarray(data_tslice[:, eff_cols], dtype=np.float64)
        X_full = _robust_transform(input_scaler, X_full)
        y_pred_full = adapter.predict(X_full)
        if hasattr(y_pred_full, "numpy"):
            y_pred_full = y_pred_full.numpy()
        y_pred_full = np.asarray(y_pred_full)
        if output_scaler is not None:
            if y_pred_full.ndim == 1:
                y_pred_full = y_pred_full.reshape(-1, 1)
            y_pred_full = output_scaler.inverse_transform(y_pred_full)
        if y_pred_full.ndim == 2 and y_pred_full.shape[1] > 1:
            y_pred_masked = y_pred_full[mask, 1]
        else:
            y_pred_masked = y_pred_full.flatten()[mask]
        n_pts = rz.shape[0]
        if len(y_pred_masked) != n_pts:
            y_pred_masked = np.full(n_pts, np.nan)
        z_pred = y_pred_masked * (10.0 ** norm)
        all_z_gt.append(z_gt)
        all_z_pred.append(z_pred)
    # Use GT range for colorbar so it shows full gradient; pred outliers are clipped
    vmin = min(np.nanmin(z) for z in all_z_gt)
    vmax = max(np.nanmax(z) for z in all_z_gt)
    if vmax <= vmin:
        vmax = vmin + 1e-10

    print(
        f"GT source: {target_path} target[:,1] (A_parallel) × 10^norm, "
        f"range [{vmin:.4f}, {vmax:.4f}] across {max_frames} frames"
    )

    frames = []
    for istep in range(max_frames):
        data_tslice, target_tslice, rz, norm, z_gt, mask = _extract_timestep_slice(
            data, target, tags, istep, nsize
        )

        # Build X for inference (full slice, all rows)
        X_full = np.asarray(data_tslice[:, eff_cols], dtype=np.float64)
        X_full = _robust_transform(input_scaler, X_full)

        y_pred_full = adapter.predict(X_full)
        if hasattr(y_pred_full, "numpy"):
            y_pred_full = y_pred_full.numpy()
        y_pred_full = np.asarray(y_pred_full)

        if output_scaler is not None:
            # output_scaler expects (n, n_outputs)
            if y_pred_full.ndim == 1:
                y_pred_full = y_pred_full.reshape(-1, 1)
            y_pred_full = output_scaler.inverse_transform(y_pred_full)

        # A_parallel is output_1 (index 1)
        if y_pred_full.ndim == 2 and y_pred_full.shape[1] > 1:
            y_pred_masked = y_pred_full[mask, 1]
        else:
            y_pred_masked = y_pred_full.flatten()[mask]

        # Apply same display scaling as GT: z = pred * (10 ** norm)
        n_pts = rz.shape[0]
        if len(y_pred_masked) != n_pts:
            y_pred_masked = np.full(n_pts, np.nan)
        z_pred = y_pred_masked * (10.0 ** norm)

        x_rz = rz[:, 0].flatten()
        y_rz = rz[:, 1].flatten()

        triang = tri.Triangulation(x_rz, y_rz)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        cf1 = ax1.tricontourf(triang, z_gt, levels=20, cmap="plasma", vmin=vmin, vmax=vmax)
        ax1.set_title("Ground Truth A_parallel")
        ax1.set_xlabel("R")
        ax1.set_ylabel("Z")
        ax1.set_aspect("equal")

        cf2 = ax2.tricontourf(triang, z_pred, levels=20, cmap="plasma", vmin=vmin, vmax=vmax)
        ax2.set_title("Surrogate Inference")
        ax2.set_xlabel("R")
        ax2.set_ylabel("Z")
        ax2.set_aspect("equal")

        fig.suptitle(f"Timestep {istep} ({args.eval_set})", fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        # Use explicit ScalarMappable for fixed colorbar gradient (not tied to cf1)
        sm = mcm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="plasma")
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="A_parallel")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()
        frames.append(rgb)
        plt.close(fig)

    if not frames:
        print("No frames generated", file=sys.stderr)
        return 1

    model_label = run_dir.name
    out_stem = args.output_name or f"xgc_spatial_aparallel_animation_{model_label}"
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

    print(f"Spatial animation saved: {gif_path} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
