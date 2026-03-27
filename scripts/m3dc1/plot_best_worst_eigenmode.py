#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot best, median, and worst eigenmode shape predictions vs ground truth.

Finds the sample with lowest, median, and highest per-profile RMSE (using the
best model, typically MLP), then plots delta_p(psi_N) for GT vs both MLP and RFR
predictions.

Usage:
  python plot_best_worst_eigenmode.py runs/m3dc1_delta_p_per_mode_hpo
  python plot_best_worst_eigenmode.py runs/m3dc1_delta_p_per_mode_hpo --reference-model mlp_per_mode
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Legend labels
LABEL_GT = "M3D-C1"
LABEL_MLP = "M3D-C1-ML(MLP)"
LABEL_RFR = "M3D-C1-ML(RFR)"


def main():
    parser = argparse.ArgumentParser(description="Plot best/median/worst eigenmode predictions vs GT")
    parser.add_argument("run_dir", type=Path, help="Run directory (e.g. runs/m3dc1_delta_p_per_mode_hpo)")
    parser.add_argument("--reference-model", default="mlp_per_mode",
                        help="Model used to define best/median/worst by RMSE (default: mlp_per_mode)")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Output plot path")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset .pkl for n,m labels")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pred_dir = run_dir / "predictions"

    # Load both models
    mlp_path = pred_dir / "mlp_per_mode_test.csv"
    rf_path = pred_dir / "rf_per_mode_test.csv"
    if not mlp_path.exists() or not rf_path.exists():
        print(f"Not found: need both {mlp_path} and {rf_path}", file=sys.stderr)
        return 1

    df_mlp = pd.read_csv(mlp_path)
    df_rf = pd.read_csv(rf_path)

    y_true_cols = sorted([c for c in df_mlp.columns if c.startswith("y_true_")])
    y_pred_cols = sorted([c for c in df_mlp.columns if c.startswith("y_pred_")])
    if not y_true_cols or not y_pred_cols or len(y_true_cols) != len(y_pred_cols):
        print("Missing or mismatched y_true/y_pred columns", file=sys.stderr)
        return 1

    n_pts = len(y_true_cols)
    y_true = df_mlp[y_true_cols].values.astype(float)
    y_pred_mlp = df_mlp[y_pred_cols].values.astype(float)
    y_pred_rf = df_rf[y_pred_cols].values.astype(float)
    psi = np.linspace(0.0001, 1.0, n_pts)

    # Use reference model (MLP) RMSE to select best/median/worst
    rmse_per_row = np.sqrt(np.mean((y_true - y_pred_mlp) ** 2, axis=1))
    best_idx = int(np.argmin(rmse_per_row))
    worst_idx = int(np.argmax(rmse_per_row))
    median_rmse = np.median(rmse_per_row)
    median_idx = int(np.argmin(np.abs(rmse_per_row - median_rmse)))

    # Get n, m if dataset available
    n_best = m_best = n_med = m_med = n_worst = m_worst = None
    if args.dataset and args.dataset.exists() and "index" in df_mlp.columns:
        try:
            ds = pd.read_pickle(args.dataset)
            idx_best = df_mlp.iloc[best_idx]["index"]
            idx_med = df_mlp.iloc[median_idx]["index"]
            idx_worst = df_mlp.iloc[worst_idx]["index"]
            if idx_best in ds.index:
                n_best, m_best = ds.loc[idx_best, "n"], ds.loc[idx_best, "m"]
            if idx_med in ds.index:
                n_med, m_med = ds.loc[idx_med, "n"], ds.loc[idx_med, "m"]
            if idx_worst in ds.index:
                n_worst, m_worst = ds.loc[idx_worst, "n"], ds.loc[idx_worst, "m"]
        except Exception:
            pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def plot_panel(ax, idx, title_prefix, rmse_val, n_val, m_val):
        ax.plot(psi, y_true[idx], "k-", linewidth=2, label=LABEL_GT)
        ax.plot(psi, y_pred_mlp[idx], "b--", linewidth=1.5, label=LABEL_MLP)
        ax.plot(psi, y_pred_rf[idx], "r:", linewidth=1.5, label=LABEL_RFR)
        ax.set_xlabel(r"$\psi_N$")
        ax.set_ylabel(r"$|\delta p_{n,m}|$")
        title = f"{title_prefix} (RMSE={rmse_val:.2f})"
        if n_val is not None and m_val is not None:
            title += f"  n={n_val}, m={m_val}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Best
    plot_panel(axes[0], best_idx, "Best", rmse_per_row[best_idx], n_best, m_best)

    # Median
    plot_panel(axes[1], median_idx, "Median", rmse_per_row[median_idx], n_med, m_med)

    # Worst
    plot_panel(axes[2], worst_idx, "Worst", rmse_per_row[worst_idx], n_worst, m_worst)

    fig.suptitle(
        f"Eigenmode shapes: best/median/worst by {args.reference_model} RMSE (test set)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    out = args.out or run_dir / "plots" / "eigenmode_best_worst.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")
    print(f"Best:   row {best_idx}, RMSE={rmse_per_row[best_idx]:.4f}")
    print(f"Median: row {median_idx}, RMSE={rmse_per_row[median_idx]:.4f}")
    print(f"Worst:  row {worst_idx}, RMSE={rmse_per_row[worst_idx]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
