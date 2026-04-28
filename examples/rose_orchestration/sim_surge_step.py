# -*- coding: utf-8 -*-
"""Simulation step: refresh labeled Parquet (subprocess demo)."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from dataset_utils import build_training_parquet, workspace_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--dataset", choices=("synthetic", "m3dc1"), default="synthetic")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = build_training_parquet(args.iteration, dataset=args.dataset)
    n = pd.read_parquet(path).shape[0]

    meta = {"step": "simulation", "iteration": args.iteration, "n_rows": int(n), "dataset": str(path)}
    ws = workspace_dir()
    with (ws / "last_simulation.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.verbose:
        print("=" * 72, flush=True)
        print(f"[SIM] dataset={args.dataset}  iteration={args.iteration}  rows={n}", flush=True)
        print(f"[SIM] path={path}", flush=True)
        print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
