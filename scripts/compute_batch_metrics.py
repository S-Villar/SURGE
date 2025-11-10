#!/usr/bin/env python3
"""
compute_batch_metrics.py

Reads JSON lines produced by scan_batch_cases.sh and prints summary metrics:
  - counts by status
  - per-run status counts (top few runs)
  - runtime statistics from started/finished timestamps
  - estimated throughput based on recent completions
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize batch case snapshot.")
    parser.add_argument("--input", required=True, help="Path to cases.jsonl")
    parser.add_argument(
        "--top-runs",
        type=int,
        default=10,
        help="Number of runs to display in per-run breakdown (default 10).",
    )
    return parser.parse_args()


def read_cases(path: Path) -> List[dict]:
    cases: List[dict] = []
    with path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            cases.append(json.loads(line))
    return cases


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()
    cases = read_cases(Path(args.input))
    if not cases:
        print("No cases found in snapshot.")
        return

    counts = Counter(case["status"] for case in cases)
    total = len(cases)

    print("=== Batch Summary ===")
    print(f"Total cases : {total}")
    for status in ["finished", "inflight", "started", "pending"]:
        print(f"{status:9s}: {counts.get(status, 0)}")

    # Per-run breakdown
    per_run = defaultdict(lambda: Counter())
    for case in cases:
        per_run[case["run"]][case["status"]] += 1

    print("\nTop runs by remaining (pending + started + inflight):")
    def remaining_count(run_counts: Counter) -> int:
        return (
            run_counts.get("pending", 0)
            + run_counts.get("started", 0)
            + run_counts.get("inflight", 0)
        )

    sorted_runs = sorted(per_run.items(), key=lambda kv: remaining_count(kv[1]), reverse=True)
    for run, status_counts in sorted_runs[: args.top_runs]:
        rem = remaining_count(status_counts)
        print(
            f"  {run:8s} | rem={rem:4d} | fin={status_counts.get('finished',0):4d} "
            f"| inq={status_counts.get('inflight',0):4d} | start={status_counts.get('started',0):4d} "
            f"| pend={status_counts.get('pending',0):4d}"
        )

    durations = []
    finish_times = []
    for case in cases:
        started = case.get("started_ts", 0) or 0
        finished = case.get("finished_ts", 0) or 0
        if started and finished and finished > started:
            durations.append(finished - started)
            finish_times.append(finished)

    if durations:
        avg_duration = mean(durations)
        print("\nRuntime stats (based on finished cases):")
        print(f"  samples : {len(durations)}")
        print(f"  avg     : {format_duration(avg_duration)}")
        print(f"  min     : {format_duration(min(durations))}")
        print(f"  max     : {format_duration(max(durations))}")

        finish_times.sort()
        now_ts = finish_times[-1]
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        print(f"  last finish: {now.isoformat()}")

        window_seconds = 6 * 3600  # 6h rolling throughput
        lower_bound = now_ts - window_seconds
        recent = [ts for ts in finish_times if ts >= lower_bound]
        if recent:
            throughput = len(recent) / (window_seconds / 3600)
            print(f"  ~throughput (last 6h): {throughput:.1f} cases/hour")
    else:
        print("\nRuntime stats: no finished cases with timestamps.")


if __name__ == "__main__":
    main()





