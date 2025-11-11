#!/usr/bin/env python3
"""
batch_quick_status.py

Fast diagnostic: report how many runs are complete, queued, and remaining.
Completion == all cases under run*/sparc_* have a finished marker or C1.h5.
Queued == any case index from SLURM job arrays is currently active.
"""

import argparse
import getpass
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Dict, Iterable


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[{timestamp()}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick batch diagnostics.")
    parser.add_argument(
        "-b",
        "--batch",
        required=True,
        help="Path to batch directory containing run*/sparc_* cases.",
    )
    parser.add_argument(
        "--user",
        help="User for squeue lookup (defaults to SUBMIT_CHUNKS_USER or current user).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages; only print final summary.",
    )
    parser.add_argument(
        "--show-run-counts",
        action="store_true",
        help="Include total case counts per category in summary.",
    )
    return parser.parse_args()


def collect_runs(batch_dir: Path, quiet: bool):
    run_dirs = sorted(
        (p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith("run")),
        key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else p.name,
    )
    log(f"Found {len(run_dirs)} run directories.", quiet)

    runs: List[Dict[str, object]] = []
    case_to_run: List[int] = []
    case_id = 0

    for run_dir in run_dirs:
        run_id = int(run_dir.name[3:]) if run_dir.name[3:].isdigit() else run_dir.name
        case_dirs = sorted(
            (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("sparc_")),
            key=lambda p: int(p.name.split("_", 1)[1]) if "_" in p.name and p.name.split("_", 1)[1].isdigit() else p.name,
        )
        case_count = len(case_dirs)
        if case_count == 0:
            runs.append(
                {
                    "path": run_dir,
                    "run_id": run_id,
                    "case_indices": [],
                    "case_count": 0,
                    "completed": False,
                }
            )
            continue

        completed = True
        case_indices: List[int] = []

        for case_dir in case_dirs:
            case_id += 1
            case_to_run.append(run_id)
            case_indices.append(case_id)

            finished = case_dir.joinpath("finished").is_file()
            output_exists = case_dir.joinpath("C1.h5").is_file()
            if not finished and not output_exists:
                completed = False

        runs.append(
            {
                "path": run_dir,
                "run_id": run_id,
                "case_indices": case_indices,
                "case_count": case_count,
                "completed": completed,
            }
        )

    log(f"Total cases indexed: {case_id}", quiet)
    return runs, case_to_run


def parse_queue_indices(raw: str) -> Set[int]:
    ids: Set[int] = set()
    for line in raw.splitlines():
        entry = line.strip()
        if not entry or entry == "-":
            continue
        for part in entry.split(","):
            token = part.strip().strip("[]")
            if not token:
                continue
            token = token.split("%", 1)[0]
            if "-" in token:
                start_str, end_str = token.split("-", 1)
                if start_str.isdigit() and end_str.isdigit():
                    start, end = int(start_str), int(end_str)
                    if start <= end:
                        ids.update(range(start, end + 1))
            else:
                if token.isdigit():
                    ids.add(int(token))
    return ids


def fetch_queue_indices(user: str, quiet: bool) -> Set[int]:
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "-o", "%i", "-h"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log(f"Warning: unable to query squeue ({exc}); treating queue as empty.", quiet)
        return set()
    ids = parse_queue_indices(result.stdout)
    log(f"Active array indices detected: {len(ids)}", quiet)
    return ids


def summarize(runs, case_to_run, queued_ids):
    run_lookup = {
        r["run_id"]: r for r in runs if r["case_count"] > 0
    }
    total_runs = len(run_lookup)
    completed_runs = sum(1 for r in run_lookup.values() if r["completed"])

    queued_run_ids: Set[int] = set()
    queued_case_count = 0
    for case_id in queued_ids:
        if 1 <= case_id <= len(case_to_run):
            run_id = case_to_run[case_id - 1]
            queued_run_ids.add(run_id)
            queued_case_count += 1

    queued_runs = len([rid for rid in queued_run_ids if rid in run_lookup])
    remaining_runs = max(total_runs - completed_runs - queued_runs, 0)

    total_cases = sum(r["case_count"] for r in run_lookup.values())
    completed_cases = sum(r["case_count"] for r in run_lookup.values() if r["completed"])

    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "queued_runs": queued_runs,
        "remaining_runs": remaining_runs,
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "queued_cases": queued_case_count,
        "runs_detail": runs,
        "run_lookup": run_lookup,
    }


def fmt_pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(100.0 * count / total):.1f}%"


def compress_ranges(ids: Iterable[int]) -> str:
    sorted_ids = sorted(set(ids))
    if not sorted_ids:
        return "none"
    ranges = []
    start = prev = sorted_ids[0]
    for current in sorted_ids[1:]:
        if current == prev + 1:
            prev = current
            continue
        ranges.append((start, prev))
        start = prev = current
    ranges.append((start, prev))
    parts = []
    for a, b in ranges:
        parts.append(f"{a}-{b}" if a != b else str(a))
    return ", ".join(parts)


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch).resolve()
    if not batch_dir.is_dir():
        raise SystemExit(f"Batch directory not found: {batch_dir}")

    user = (
        args.user
        or os.environ.get("SUBMIT_CHUNKS_USER")
        or os.environ.get("USER")
        or getpass.getuser()
    )

    log(f"Batch: {batch_dir}", args.quiet)
    log(f"Using user '{user}' for queue inspection.", args.quiet)

    runs, case_to_run = collect_runs(batch_dir, args.quiet)
    queue_ids = fetch_queue_indices(user, args.quiet)
    summary = summarize(runs, case_to_run, queue_ids)

    total_runs = summary["total_runs"]
    completed_runs = summary["completed_runs"]
    queued_runs = summary["queued_runs"]
    remaining_runs = summary["remaining_runs"]

    print("=== Batch Quick Status ===")
    print(f"Batch directory      : {batch_dir}")
    print(f"Total runs           : {total_runs}")
    print(
        f"Completed runs       : {completed_runs} ({fmt_pct(completed_runs, total_runs)})"
    )
    print(
        f"Queued/in-flight runs: {queued_runs} ({fmt_pct(queued_runs, total_runs)})"
    )
    print(
        f"Remaining runs       : {remaining_runs} ({fmt_pct(remaining_runs, total_runs)})"
    )

    run_lookup = summary["run_lookup"]
    runs_detail = summary["runs_detail"]
    run_numbers = sorted(run_lookup.keys())
    completed_run_numbers = sorted(rid for rid, r in run_lookup.items() if r["completed"])
    incomplete_run_numbers = [n for n in run_numbers if n not in completed_run_numbers]

    print()
    print(f"Completed run IDs    : {compress_ranges(completed_run_numbers)}")
    print(f"Incomplete run IDs   : {compress_ranges(incomplete_run_numbers)}")

    if args.show_run_counts:
        total_cases = summary["total_cases"]
        completed_cases = summary["completed_cases"]
        queued_cases = summary["queued_cases"]
        print()
        print(f"Total cases          : {total_cases}")
        print(
            f"Completed cases      : {completed_cases} ({fmt_pct(completed_cases, total_cases)})"
        )
        print(
            f"Queued cases         : {queued_cases} ({fmt_pct(queued_cases, total_cases)})"
        )


if __name__ == "__main__":
    main()

