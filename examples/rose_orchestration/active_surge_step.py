"""Active-learning hook: verbose placeholder; real acquisition would read SURGE UQ here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_utils import workspace_dir, write_json_atomic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ws = workspace_dir()
    mpath = ws / "last_surge_metrics.json"
    m = {}
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))

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

    info = {"step": "active_learn", "iteration": args.iteration, "surge": m}
    write_json_atomic(ws / "last_active.json", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
