#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
if str(_EXAMPLE_DIR.parents[1]) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR.parents[1]))
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one GPU-aware SURGE trial.")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-path", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--trial-id", type=int, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hidden-layers", default="128,64")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hidden_layers = [int(piece.strip()) for piece in args.hidden_layers.split(",") if piece.strip()]
    payload = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_format": "auto",
        "metadata_path": str((_EXAMPLE_DIR.parents[1] / "data" / "datasets" / "M3DC1" / "sparc_m3dc1_D1_metadata.yaml").resolve()),
        "test_fraction": 0.2,
        "val_fraction": 0.2,
        "standardize_inputs": True,
        "standardize_outputs": True,
        "predictions_scope": ["val"],
        "predictions_format": "csv",
        "output_dir": str(Path(args.output_dir).resolve()),
        "run_tag": args.run_tag,
        "seed": int(args.random_state),
        "overwrite_existing_run": True,
        "resources": {
            "device": args.device,
            "num_workers": int(args.num_workers),
            "strict": False,
        },
        "models": [
            {
                "key": "pytorch.mlp",
                "name": f"torch_mlp_trial_{args.trial_id:02d}",
                "params": {
                    "hidden_layers": hidden_layers,
                    "dropout_rate": 0.1,
                    "learning_rate": 1e-3,
                    "n_epochs": int(args.n_epochs),
                    "batch_size": int(args.batch_size),
                    "log_progress": False,
                    "random_state": int(args.random_state),
                },
            }
        ],
    }
    spec = SurrogateWorkflowSpec.from_dict(payload)
    summary = run_surrogate_workflow(spec, console_output=args.verbose)
    model = summary["models"][0]
    val = model["metrics"]["val"]
    resources_used = model.get("resources_used", {})
    out = {
        "trial_id": int(args.trial_id),
        "run_tag": args.run_tag,
        "hidden_layers": hidden_layers,
        "val_r2": float(val.get("r2", float("nan"))),
        "val_rmse": float(val["rmse"]),
        "resources_used": resources_used,
        "artifacts_root": summary["artifacts"]["root"],
        "splits": dict(summary.get("splits", {})),
    }
    Path(args.summary_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
