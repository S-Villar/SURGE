#!/usr/bin/env python3
"""
XGC datastreamset comparison: set1 model vs finetuned model on set1 and set2.

Runs four evaluations:
  1. Set1 model on set1 datastreamsets
  2. Set1 model on set2 datastreamsets (cross-set)
  3. Finetuned model on set2 datastreamsets
  4. Finetuned model on set1 datastreamsets (catastrophic forgetting test)

Creates per-scenario plots and a combined comparison plot.

Usage:
  python scripts/xgc_datastreamset_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_SET1 = PROJECT_ROOT / "runs/xgc_aparallel_set1_v3"
RUN_FINETUNE = PROJECT_ROOT / "runs/xgc_aparallel_set2_finetune"
DATA_DIR = Path("/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025")


def main():
    from surge.viz.run_viz import viz_datastreamset_evaluation

    scenarios = [
        ("set1_on_set1", RUN_SET1, None, RUN_SET1 / "plots/eval_set1"),
        ("set1_on_set2", RUN_SET1, "set2_beta0p5", RUN_SET1 / "plots/eval_set2"),
        ("finetune_on_set2", RUN_FINETUNE, None, RUN_FINETUNE / "plots/eval_set2"),
        ("finetune_on_set1", RUN_FINETUNE, "set1", RUN_FINETUNE / "plots/eval_set1_forgetting"),
    ]

    all_stats = {}
    for name, run_dir, eval_set, output_dir in scenarios:
        if not run_dir.exists():
            print(f"Skip {name}: run dir not found: {run_dir}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {name}...")
        try:
            result = viz_datastreamset_evaluation(
                run_dir=run_dir,
                output_dir=output_dir,
                datastreamset_size=50000,
                max_datastreamsets=10,
                eval_set_name=eval_set,
            )
            all_stats[name] = result
            print(f"  -> {output_dir}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Combined comparison plot
    if all_stats:
        _plot_combined_comparison(all_stats)
    return 0


def _plot_combined_comparison(all_stats: dict):
    """Create combined R² comparison across all four scenarios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = {
        "set1_on_set1": "Set1 model on Set1",
        "set1_on_set2": "Set1 model on Set2",
        "finetune_on_set2": "Finetuned model on Set2",
        "finetune_on_set1": "Finetuned model on Set1 (forgetting)",
    }
    colors = {"set1_on_set1": "C0", "set1_on_set2": "C1", "finetune_on_set2": "C2", "finetune_on_set1": "C3"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, result in all_stats.items():
        dsets = result.get("datastreamsets", {})
        if not dsets:
            continue
        # Get MLP output_1 (A_parallel) R² per datastreamset
        r2_list = []
        rmse_list = []
        indices = []
        def _key_idx(k):
            parts = k.split("_")
            return int(parts[1]) if len(parts) > 1 else 0
        for key in sorted(dsets.keys(), key=_key_idx):
            try:
                idx = _key_idx(key)
            except (IndexError, ValueError):
                idx = len(indices)
            models = dsets[key].get("models", {})
            mlp = models.get("xgc_mlp_aparallel", {})
            out1 = mlp.get("output_1", {})
            r2 = out1.get("r2", np.nan)
            rmse = out1.get("rmse", np.nan)
            r2_list.append(r2)
            rmse_list.append(rmse)
            indices.append(idx)

        if not indices:
            continue
        label = labels.get(name, name)
        ax1, ax2 = axes[0], axes[1]
        ax1.plot(indices, r2_list, "o-", label=label, color=colors.get(name, "gray"), linewidth=2, markersize=8)
        ax2.plot(indices, rmse_list, "o-", label=label, color=colors.get(name, "gray"), linewidth=2, markersize=8)

    axes[0].set_ylabel("R² (A_parallel)")
    axes[0].set_title("Datastreamset Evaluation: Set1 vs Finetuned Model")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_ylabel("RMSE (A_parallel)")
    axes[1].set_xlabel("Datastreamset index")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = PROJECT_ROOT / "runs/xgc_datastreamset_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "datastreamset_comparison_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined plot: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
