#!/usr/bin/env python3
"""Visualize M3D-C1 run predictions with SURGE.

Shows how model predictions match ground truth (M3D-C1) for a given run.
Uses SURGE's plot_inference_comparison_grid for publication-quality
ground-truth vs prediction density plots.

Layout: 3 rows (models) × 2 columns (train / test).

Usage:
    conda activate surge  # or your SURGE env
    python examples/viz_m3dc1_predictions.py --run-dir runs/m3dc1_aug_r75
    python examples/viz_m3dc1_predictions.py --run-dir runs/m3dc1_aug_r75 --axis-lim -0.01,0.25

Output:
    Plots saved to <run-dir>/plots/inference_comparison_grid.png

Axis limits for scattered data with negatives and concentrations at ~1e-3:
    - Log scale is not viable when values cross zero (log of negative is undefined).
    - Use linear scale with symmetric or asymmetric padding around data range.
    - For γ ∈ [-0.008, 0.24]: --axis-lim -0.01,0.25 keeps negatives visible.
    - For zoom on small values: --axis-lim -0.005,0.02 (loses high-γ detail).
    - 2D density heatmap (current) handles scatter well; log-scale histogram
      would require splitting positive/negative or using symlog (matplotlib
      SymmetricalLogScale) if you need to emphasize small magnitudes.
    - Units default to [s⁻¹] (growth rate); use --units to override.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from surge.viz import plot_inference_comparison_grid


def load_predictions(run_dir):
    """Load predictions from a SURGE workflow run."""
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return {}

    results = {}
    for pred_file in predictions_dir.glob("*.csv"):
        model_name = pred_file.stem.replace("_train", "").replace("_val", "").replace("_test", "")
        dataset_type = (
            "train"
            if "_train" in pred_file.name
            else ("val" if "_val" in pred_file.name else "test")
        )

        df = pd.read_csv(pred_file)
        if "y_true_00" in df.columns and "y_pred_00" in df.columns:
            if model_name not in results:
                results[model_name] = {}
            results[model_name][dataset_type] = (
                df["y_true_00"].values,
                df["y_pred_00"].values,
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize M3D-C1 predictions: ground truth vs model predictions"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/m3dc1_aug_r75",
        help="Path to SURGE workflow run directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: run_dir/plots)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output_gamma",
        help="Output variable name for axis labels (default: output_gamma)",
    )
    parser.add_argument(
        "--units",
        type=str,
        default=r"[s$^{-1}$]",
        help="Units for the output (default: [s⁻¹] for M3D-C1 growth rate)",
    )
    parser.add_argument(
        "--axis-lim",
        type=str,
        default=None,
        help="Axis limits as 'min,max' e.g. '-0.01,0.25' (default: auto from data)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SURGE M3D-C1 Prediction Visualization")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load predictions
    predictions = load_predictions(run_dir)
    if not predictions:
        print("No predictions found in", run_dir / "predictions")
        sys.exit(1)

    models = list(predictions.keys())
    print(f"Models: {models}")

    # Build results dict for plot_inference_comparison_grid
    # Structure: {output_name: {model_name: {dataset: (y_true, y_pred)}}}
    results_dict = {args.output_name: {}}
    for model_name in models:
        results_dict[args.output_name][model_name] = {}
        for dataset_type in ["train", "test"]:
            if dataset_type in predictions[model_name]:
                results_dict[args.output_name][model_name][dataset_type] = (
                    predictions[model_name][dataset_type]
                )

    if not results_dict[args.output_name]:
        print("Insufficient data (need train and/or test for at least one model)")
        sys.exit(1)

    # Model short names for titles
    model_display = {
        "random_forest_profiles": "Random Forest",
        "torch_mlp_mc_dropout": "MLP",
        "gpflow_gpr_profiles": "GPR",
    }
    # Output display (γ for growth rate)
    output_display = {args.output_name: r"$\gamma$"}

    # Optional axis limits (for data with negatives, log scale is not viable)
    axis_lim = None
    if args.axis_lim:
        lo, hi = args.axis_lim.split(",")
        axis_lim = (float(lo.strip()), float(hi.strip()))

    # Create inference comparison grid (3 rows = models, 2 cols = train/test)
    save_path = output_dir / "inference_comparison_grid.png"
    fig, axes, r2_results = plot_inference_comparison_grid(
        results_dict,
        units={args.output_name: args.units},
        save_path=save_path,
        dpi=150,
        xlabel_prefix=r"M3D-C1 $\gamma$",
        ylabel_prefix=r"M3D-C1-ML $\gamma$",
        ylabel_include_model=False,
        output_display_names=output_display,
        model_display_names=model_display,
        title_include_model_dataset=True,
        layout="models_rows",
        axis_lim=axis_lim,
    )

    print()
    print("R² scores (how well predictions match ground truth):")
    print(json.dumps(r2_results, indent=2))
    print()
    print("=" * 70)
    print(f"Plot saved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
