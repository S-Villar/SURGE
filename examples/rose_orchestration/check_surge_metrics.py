"""Emit validation MSE for ROSE stop-criterion (stdout: single float)."""
from __future__ import annotations

import argparse
import sys

from dataset_utils import read_iteration_state, write_iteration_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    data = read_iteration_state(args.iteration, "surge_metrics")
    mse = float(data["val_mse"])
    write_iteration_state(args.iteration, "criterion", {"iteration": args.iteration, "metric": "val_mse", "value": mse})
    if args.verbose:
        # ROSE shell tasks parse stdout as a single float; keep logs on stderr.
        print(
            f"[CRITERION] val_mse={mse:.6f}  val_rmse={data.get('val_rmse')}",
            flush=True,
            file=sys.stderr,
        )
    print(mse, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
