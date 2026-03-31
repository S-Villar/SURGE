#!/usr/bin/env python3
"""
Emit a Markdown table of CFS δp trial metrics from runs/<run_tag>/metrics.json.

Usage (repo root):
  python scripts/m3dc1/harvest_cfs_trial_metrics.py
  python scripts/m3dc1/harvest_cfs_trial_metrics.py --runs-dir /path/to/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Trial id, Slurm script, hardware, run_tag, description, model keys in metrics.json
TRIALS: List[Tuple[str, str, str, str, str, List[str]]] = [
    (
        "T1",
        "train_delta_p_per_mode_cfs.slurm",
        "CPU",
        "m3dc1_delta_p_per_mode_cfs",
        "RF + MLP (fixed hyp.)",
        ["rf_per_mode", "mlp_per_mode"],
    ),
    (
        "T2",
        "train_delta_p_per_mode_cfs_hpo.slurm",
        "CPU",
        "m3dc1_delta_p_per_mode_cfs_hpo",
        "RF + MLP HPO (TPE)",
        ["rf_per_mode", "mlp_per_mode"],
    ),
    (
        "T3",
        "train_delta_p_per_mode_cfs_mlp_hpo.slurm",
        "CPU",
        "m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible",
        "MLP-only HPO (BoTorch, flexible arch)",
        ["mlp_per_mode"],
    ),
    (
        "T4",
        "train_delta_p_per_mode_cfs_mlp_hpo_gpu.slurm",
        "GPU",
        "m3dc1_delta_p_per_mode_cfs_mlp_hpo_gpu",
        "MLP-only HPO (BoTorch, GPU node)",
        ["mlp_per_mode"],
    ),
    (
        "T5",
        "train_delta_p_per_mode_cfs_gpr_hpo.slurm",
        "CPU",
        "m3dc1_delta_p_per_mode_cfs_gpr_lin_matern52_botorch",
        "GPflow GPR Linear+Matern52 HPO (BoTorch), 1 output + sample_rows",
        ["gpr_lin_matern52"],
    ),
]


def _fmt_metric(m: Optional[Dict[str, Any]], key: str) -> str:
    if m is None:
        return "—"
    v = m.get(key)
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run_tag subdirs",
    )
    args = ap.parse_args()
    runs = args.runs_dir.resolve()

    lines = [
        "| Trial | Resource | Run tag | Models (test R² / RMSE) | Status |",
        "|-------|----------|---------|-------------------------|--------|",
    ]

    for tid, slurm, hw, tag, desc, model_keys in TRIALS:
        mj = runs / tag / "metrics.json"
        if not mj.is_file():
            lines.append(
                f"| **{tid}** | {hw} | `{tag}` | — | pending (`{slurm}`) |"
            )
            continue
        with mj.open() as f:
            data = json.load(f)
        parts = []
        for mk in model_keys:
            block = data.get(mk)
            if not block:
                parts.append(f"`{mk}`: —")
                continue
            te = block.get("test") or {}
            parts.append(
                f"`{mk}` R²={_fmt_metric(te, 'r2')} RMSE={_fmt_metric(te, 'rmse')}"
            )
        status = "complete"
        detail = "; ".join(parts)
        lines.append(
            f"| **{tid}** | {hw} | `{tag}` | {detail} | {status} |"
        )

    print("\n".join(lines))
    print()
    print(f"_Auto-generated from `metrics.json` under `{runs}`. Re-run after batch jobs finish._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
