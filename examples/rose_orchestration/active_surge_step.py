"""Active-learning hook: verbose placeholder; real acquisition would read SURGE UQ here."""
from __future__ import annotations

import argparse

from dataset_utils import read_iteration_state, write_iteration_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    m = read_iteration_state(args.iteration, "surge_metrics")

    if args.verbose:
        print("=" * 72, flush=True)
        print("[ACTIVE] Acquisition policy (demo: grow dataset next iteration via sim step).", flush=True)
        print(f"[ACTIVE] iteration={args.iteration}", flush=True)
        if m:
            print(
                f"[ACTIVE] last surrogate val_rmse={m.get('val_rmse')}  "
                f"run_tag={m.get('run_tag')}",
                flush=True,
            )
        print("=" * 72, flush=True)

    info = {"step": "active_learn", "iteration": args.iteration, "policy": "stub", "surge": m}
    write_iteration_state(args.iteration, "active", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
