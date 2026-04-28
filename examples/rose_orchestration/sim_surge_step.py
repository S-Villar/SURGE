# -*- coding: utf-8 -*-
"""Simulation step: refresh labeled Parquet (subprocess demo)."""
from __future__ import annotations

import argparse

import pandas as pd

from dataset_utils import build_training_parquet, training_row_plan, write_iteration_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--dataset", choices=("synthetic", "m3dc1"), default="synthetic")
    ap.add_argument("--growing-pool", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    use_full_dataset = args.dataset == "m3dc1" and not args.growing_pool
    path = build_training_parquet(args.iteration, dataset=args.dataset, use_full_dataset=use_full_dataset)
    n = pd.read_parquet(path).shape[0]
    plan = training_row_plan(args.iteration, dataset=args.dataset, use_full_dataset=use_full_dataset)

    meta = {
        "step": "simulation",
        "iteration": args.iteration,
        "n_rows": int(n),
        "n_rows_requested": plan["n_rows_requested"],
        "n_rows_total": plan["n_rows_total"],
        "row_policy": plan["row_policy"],
        "dataset": str(path),
    }
    write_iteration_state(args.iteration, "simulation", meta)

    if args.verbose:
        print("=" * 72, flush=True)
        print(
            f"[SIM] dataset={args.dataset}  iteration={args.iteration}  "
            f"rows={n} total_available={plan['n_rows_total'] or 'generated'}",
            flush=True,
        )
        print(f"[SIM] row_policy={plan['row_policy']}", flush=True)
        print(f"[SIM] path={path}", flush=True)
        print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
