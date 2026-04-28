"""Emit validation MSE for ROSE stop-criterion (stdout: single float)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    p = Path(__file__).resolve().parent / "workspace" / "last_surge_metrics.json"
    if not p.exists():
        raise SystemExit(f"Missing {p} — run training step first.")

    data = json.loads(p.read_text(encoding="utf-8"))
    mse = float(data["val_mse"])
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
