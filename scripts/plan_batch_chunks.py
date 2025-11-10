#!/usr/bin/env python3
"""
plan_batch_chunks.py

Reads a case snapshot (JSON lines) and produces a plan JSON describing upcoming
chunks and their SLURM configurations.

Output JSON structure:
{
  "generated_at": "... UTC timestamp ...",
  "batch_dir": "/path/to/batch",
  "cases_file": "/path/to/cases.jsonl",
  "entries": [
     {
        "array_range": "33-64",
        "case_ids": [33, ..., 64],
        "runs": [
           {"run": "run7", "equilibria": ["sparc_1423", "sparc_1424"]}
        ],
        "config": {
            "label": "N2_T32",
            "nodes": 2,
            "ntasks_per_node": 32,
            "cpus_per_task": 1,
            "time": "00:25:00"
        }
     },
     ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class Config:
    label: str
    nodes: int
    tasks_per_node: int
    cpus_per_task: int
    time: str

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Config":
        required = ["label", "nodes", "tasks_per_node", "cpus_per_task", "time"]
        for key in required:
            if key not in data:
                raise ValueError(f"Config entry missing '{key}': {data}")
        return cls(
            label=str(data["label"]),
            nodes=int(data["nodes"]),
            tasks_per_node=int(data["tasks_per_node"]),
            cpus_per_task=int(data["cpus_per_task"]),
            time=str(data["time"]),
        )


def builtin_configs() -> List[Config]:
    return [
        Config("N1_T16", 1, 16, 1, "00:30:00"),
        Config("N2_T32", 2, 32, 1, "00:25:00"),
        Config("N4_T64", 4, 64, 1, "00:20:00"),
        Config("N8_T64", 8, 64, 1, "00:20:00"),
        Config("N16_T32", 16, 32, 1, "00:20:00"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan batch submissions.")
    parser.add_argument("--cases", required=True, help="Path to cases.jsonl")
    parser.add_argument("--output", required=True, help="Path to write plan JSON")
    parser.add_argument("--chunk-size", type=int, default=32, help="Cases per chunk")
    parser.add_argument("--max-chunks", type=int, default=1, help="Maximum chunks to plan")
    parser.add_argument("--start-config", type=int, default=0, help="Index to start cycling configs")
    parser.add_argument("--config-file", help="Optional JSON list of configuration objects")
    return parser.parse_args()


def load_cases(path: Path) -> List[dict]:
    cases = []
    with path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            case = json.loads(line)
            cases.append(case)
    return cases


def load_configs(path: Path | None) -> List[Config]:
    if path is None:
        return builtin_configs()
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Config file must contain a list of objects.")
    return [Config.from_dict(entry) for entry in data]


def run_summary(case_ids: Sequence[int], cases: Dict[int, dict]) -> List[Dict[str, object]]:
    summary: Dict[str, List[str]] = {}
    for case_id in case_ids:
        case = cases[case_id]
        summary.setdefault(case["run"], []).append(case["eq"])
    return [{"run": run, "equilibria": eqs} for run, eqs in summary.items()]


def main() -> None:
    args = parse_args()
    cases_list = load_cases(Path(args.cases))
    if not cases_list:
        raise SystemExit("No cases found in cases snapshot.")

    cases_by_id = {case["case_id"]: case for case in cases_list}
    total_cases = len(cases_by_id)

    configs = load_configs(Path(args.config_file) if args.config_file else None)
    if not configs:
        raise SystemExit("No configurations available.")
    if args.start_config < 0 or args.start_config >= len(configs):
        raise SystemExit(f"--start-config must be between 0 and {len(configs)-1}.")

    chunk_size = args.chunk_size
    if chunk_size <= 0:
        raise SystemExit("Chunk size must be positive.")

    max_chunks = args.max_chunks if args.max_chunks > 0 else 2**31 - 1

    # Determine first unassigned case: anything not finished/inflight/started is available.
    next_case = None
    for case_id in range(1, total_cases + 1):
        status = cases_by_id[case_id]["status"]
        if status == "pending":
            next_case = case_id
            break
    if next_case is None:
        print("All cases are already assigned; no plan generated.")
        Path(args.output).write_text(json.dumps({"entries": []}, indent=2))
        return

    entries = []
    case_pointer = next_case
    config_index = args.start_config
    chunks_planned = 0

    while case_pointer <= total_cases and chunks_planned < max_chunks:
        case_ids = list(range(case_pointer, min(case_pointer + chunk_size, total_cases + 1)))
        available_case_ids = [cid for cid in case_ids if cases_by_id[cid]["status"] == "pending"]
        if not available_case_ids:
            case_pointer += chunk_size
            continue

        config = configs[config_index]
        entry = {
            "array_range": f"{available_case_ids[0]}-{available_case_ids[-1]}",
            "case_ids": available_case_ids,
            "runs": run_summary(available_case_ids, cases_by_id),
            "config": {
                "label": config.label,
                "nodes": config.nodes,
                "ntasks_per_node": config.tasks_per_node,
                "cpus_per_task": config.cpus_per_task,
                "time": config.time,
            },
        }
        entries.append(entry)
        chunks_planned += 1
        case_pointer = available_case_ids[-1] + 1
        config_index = (config_index + 1) % len(configs)

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_dir": str(Path(cases_list[0]["path"]).parent.parent),
        "cases_file": str(Path(args.cases)),
        "entries": entries,
    }

    Path(args.output).write_text(json.dumps(plan, indent=2))
    print(f"Planned {len(entries)} chunk(s). Output written to {args.output}")


if __name__ == "__main__":
    main()





