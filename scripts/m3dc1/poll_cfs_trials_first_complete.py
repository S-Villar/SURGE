#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report which of two CFS trial runs finished first (by ``metrics.json`` mtime).

Use this to monitor a **CPU vs GPU** pair (default **T3** vs **T4**) or any two
``run_tag`` directories under ``runs/``. Prints a Markdown snippet suitable for
``m3dc1/results.md``.

Examples::

  python scripts/m3dc1/poll_cfs_trials_first_complete.py --trial T3 --trial T4
  python scripts/m3dc1/poll_cfs_trials_first_complete.py \\
      --run-tag m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible \\
      --run-tag m3dc1_delta_p_per_mode_cfs_mlp_hpo_gpu

  # Poll every 120s until one completes (Perlmutter login node)
  python scripts/m3dc1/poll_cfs_trials_first_complete.py --trial T3 --trial T4 --watch 120
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Mirror of scripts/m3dc1/harvest_cfs_trial_metrics.py TRIALS (avoid importing that module on Py3.6).
TRIALS = [
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


def _trial_by_id(tid):
    for row in TRIALS:
        if row[0].upper() == tid.upper():
            return row
    raise SystemExit("Unknown trial id {!r}; use T1..T5".format(tid))


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return "{:.4g}".format(v)
    return str(v)


def _metrics_snippet(
    runs_dir: Path,
    tag: str,
    trial_id: str,
    hw: str,
    model_keys,
):
    mj = runs_dir / tag / "metrics.json"
    if not mj.is_file():
        raise ValueError("missing metrics")
    data = json.loads(mj.read_text(encoding="utf-8"))
    if not model_keys:
        model_keys = sorted(
            k
            for k, v in data.items()
            if isinstance(v, dict) and isinstance(v.get("test"), dict)
        )
    parts = []
    for mk in model_keys:
        block = data.get(mk)
        if not block:
            parts.append("`{}`: —".format(mk))
            continue
        te = block.get("test") or {}
        parts.append(
            "`{}` R²={} RMSE={}".format(mk, _fmt(te.get("r2")), _fmt(te.get("rmse")))
        )
    rel_run = "`runs/{}`".format(tag)
    body = (
        "| Property | Value |\n"
        "|----------|-------|\n"
        "| Trial | **{}** ({}) |\n"
        "| Resource | {} |\n"
        "| Run tag | `{}` |\n"
        "| Models (test) | {} |\n"
        "| Run directory | {} (see `metrics.json`) |\n"
        "| Completed (file mtime UTC) | {} |\n"
    ).format(
        trial_id,
        tag,
        hw,
        tag,
        "; ".join(parts),
        rel_run,
        datetime.utcfromtimestamp(mj.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    )
    return body, data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument(
        "--trial",
        action="append",
        dest="trials",
        default=[],
        help="Trial id (T1..T5); pass twice for a pair (default: T3 and T4)",
    )
    ap.add_argument(
        "--run-tag",
        action="append",
        dest="run_tags",
        default=[],
        help="Explicit run_tag (two entries); overrides --trial if two given",
    )
    ap.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SEC",
        help="If >0, poll every SEC seconds until one run produces metrics.json",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print when a winner is found (watch mode)",
    )
    args = ap.parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()

    entries = []  # type: List[Tuple[str, str, str, List[str]]]
    if args.run_tags:
        if len(args.run_tags) < 2:
            print("Need two --run-tag values (or use default T3/T4).", file=sys.stderr)
            return 2
        for tag in args.run_tags[:2]:
            entries.append(("?", str(tag), "custom", []))
    elif args.trials:
        if len(args.trials) != 2:
            print("Need exactly two --trial values (or two --run-tag).", file=sys.stderr)
            return 2
        for tid in args.trials:
            t, _slurm, hw, tag, _desc, mks = _trial_by_id(tid)
            entries.append((t, tag, hw, mks))
    else:
        for tid in ("T3", "T4"):
            t, _slurm, hw, tag, _desc, mks = _trial_by_id(tid)
            entries.append((t, tag, hw, mks))

    def check_once():
        # type: () -> Optional[Tuple[float, int, str, str, str, List[str]]]
        """Return (mtime, slot_index, trial_id, tag, hw, model_keys) for earliest completed."""
        found = []  # type: List[Tuple[float, int, str, str, str, List[str]]]
        for i, (tid, tag, hw, mks) in enumerate(entries):
            mj = runs_dir / tag / "metrics.json"
            if mj.is_file():
                found.append((mj.stat().st_mtime, i, tid, tag, hw, mks))
        if not found:
            return None
        found.sort(key=lambda x: x[0])
        return found[0]

    winner = None  # type: Optional[Tuple[float, int, str, str, str, List[str]]]
    if args.watch > 0:
        while True:
            winner = check_once()
            if winner is not None:
                break
            if not args.quiet:
                print(
                    "[{}] Waiting for metrics under {} …".format(
                        datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        runs_dir,
                    ),
                    file=sys.stderr,
                )
            time.sleep(max(1, args.watch))
    else:
        winner = check_once()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = "### CFS race: first completed ({} vs {})".format(
        entries[0][1], entries[1][1]
    )

    if winner is None:
        print(title)
        print()
        print(
            "_No `metrics.json` yet for either run under `{}`._ ".format(runs_dir)
            + "Re-run after Slurm jobs finish, or use `--watch SEC` on a login node."
        )
        print()
        print("_Poller checked at {}._".format(stamp))
        return 1

    _mtime, _i, tid, tag, hw, mks = winner
    if not mks:
        trow = _trial_by_id(tid) if tid != "?" else None
        if trow:
            mks = trow[5]
    body, _data = _metrics_snippet(runs_dir, tag, tid, hw, mks)
    other = entries[1] if winner[1] == 0 else entries[0]
    other_tag = other[1]

    print(title)
    print()
    print("**First finished:** `{}` (trial **{}**, {}).".format(tag, tid, hw))
    print()
    print(body)
    print()
    print(
        "_Other run (`{}`) was still pending at check time; update this page when it completes._".format(
            other_tag
        )
    )
    print()
    print("_Poller checked at {}._".format(stamp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
