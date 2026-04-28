# -*- coding: utf-8 -*-
"""Build synthetic Parquet if needed, then run SURGE workflow with deep-ensemble UQ."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
_ROSE = _EX.parent / "rose_orchestration"
_REPO = _EX.parent.parent


def _summarize_uq(blob: dict) -> None:
    """Print mean/max of ensemble std if arrays are present."""
    std = blob.get("std")
    if std is None:
        return
    flat: list[float] = []
    if isinstance(std, list):
        for row in std:
            if isinstance(row, list):
                flat.extend(float(x) for x in row)
            else:
                flat.append(float(row))
    else:
        return
    if not flat:
        return
    print(
        f"Ensemble std: mean={sum(flat)/len(flat):.6f}  max={max(flat):.6f}  (n={len(flat)} cells)",
        flush=True,
    )


def _maybe_plot(
    root: Path,
    uq_path: Path,
    out_png: Path,
    *,
    max_points: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas required for --plot; skipping figure.", flush=True)
        return

    pred_dir = root / "predictions"
    csv_files = sorted(pred_dir.glob("mlp_ens_demo_val.csv"))
    if not csv_files:
        print("No val prediction CSV found for plotting.", flush=True)
        return

    df = pd.read_csv(csv_files[0])
    y_true_cols = sorted([c for c in df.columns if c.startswith("y_true_")])
    if not y_true_cols or not any(c.startswith("y_pred_") for c in df.columns):
        print("Prediction CSV missing y_true_/y_pred_ columns.", flush=True)
        return

    blob = json.loads(uq_path.read_text(encoding="utf-8"))
    mean = np.asarray(blob.get("mean"), dtype=float)
    std = np.asarray(blob.get("std"), dtype=float)
    yt = df[y_true_cols[0]].to_numpy()

    n = min(len(yt), int(mean.size) if mean.size else 0)
    if mean.ndim == 2:
        mean = mean[:, 0]
    if std.ndim == 2:
        std = std[:, 0]
    n = min(n, len(mean), len(std))
    if n <= 0:
        print("Could not align UQ payload with predictions.", flush=True)
        return

    idx = np.arange(n)
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(42)
        pick = np.sort(rng.choice(n, size=max_points, replace=False))
        idx = pick
        yt = yt[pick]
        mean = mean[pick]
        std = std[pick]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        idx,
        mean,
        yerr=std,
        fmt="o",
        markersize=3,
        alpha=0.7,
        label="ensemble mean +/- std",
    )
    ax.scatter(idx, yt, s=8, c="C1", label="y_true", zorder=3)
    ax.set_xlabel("val row (subsampled index)")
    ax.set_ylabel("target (first output)")
    ax.legend(loc="best")
    ax.set_title("Deep-ensemble UQ on validation (synthetic demo)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote plot: {out_png.resolve()}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SURGE MLP deep-ensemble UQ demo.")
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PNG",
        help="If set, write a simple val mean +/- std figure (needs matplotlib).",
    )
    ap.add_argument(
        "--plot-max-points",
        type=int,
        default=80,
        metavar="N",
        help="Subsample at most N val points for the figure (default 80).",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(_REPO))
    os.chdir(_ROSE)
    sys.path.insert(0, str(_ROSE))
    from dataset_utils import build_synthetic_parquet

    build_synthetic_parquet(2)

    import yaml
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    spec_path = _EX / "workflow_mlp_ensemble_uq.yaml"
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec = SurrogateWorkflowSpec.from_dict(payload)
    summary = run_surrogate_workflow(spec)
    root = Path(summary["artifacts"]["root"])
    uq = root / "predictions" / "mlp_ens_demo_val_uq.json"
    print("Artifacts:", root)
    print("val_uq:", uq)
    if uq.is_file():
        blob = json.loads(uq.read_text(encoding="utf-8"))
        print("keys:", sorted(blob.keys()))
        _summarize_uq(blob)

    if args.plot is not None:
        _maybe_plot(root, uq, args.plot, max_points=max(1, args.plot_max_points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
