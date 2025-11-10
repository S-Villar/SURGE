#!/bin/bash
# update_batch_logs.sh
#
# Consumes a plan JSON and the latest case snapshot to append records to
# batch_assignments.log and batch_metrics_summary.txt.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: update_batch_logs.sh --plan <planned_submissions.json> --metrics <cases.jsonl>
EOF
}

plan_file=""
metrics_file=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan) plan_file="$2"; shift 2 ;;
        --metrics) metrics_file="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -f "$plan_file" ]] || { echo "Plan file not found: $plan_file" >&2; exit 1; }
[[ -f "$metrics_file" ]] || { echo "Metrics snapshot not found: $metrics_file" >&2; exit 1; }

python3 - "$plan_file" "$metrics_file" <<'PYCODE'
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

plan_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])

plan = json.loads(plan_path.read_text())
entries = plan.get("entries", [])
if not entries:
    print("No entries to log.")
    sys.exit(0)

cases_snapshot = []
with metrics_path.open() as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        cases_snapshot.append(json.loads(line))

status_counts = Counter(case["status"] for case in cases_snapshot)

batch_dir = Path(plan.get("batch_dir", ".")).resolve()

assign_log = batch_dir / "batch_assignments.log"
metrics_log = batch_dir / "batch_metrics_summary.txt"

timestamp = datetime.now(timezone.utc).isoformat()

for entry in entries:
    runs = ";".join(
        f"{item['run']}[{','.join(item['equilibria'])}]"
        for item in entry["runs"]
    )
    config = entry["config"]
    line = (
        f"{timestamp} array={entry['array_range']} cases={len(entry['case_ids'])} "
        f"label={config['label']} nodes={config['nodes']} "
        f"tasks_per_node={config['ntasks_per_node']} cpus_per_task={config['cpus_per_task']} "
        f"time={config['time']} runs={runs}"
    )
    with assign_log.open("a") as fh:
        fh.write(line + "\n")

metrics_line = (
    f"{timestamp} finished={status_counts.get('finished',0)} "
    f"inflight={status_counts.get('inflight',0)} started={status_counts.get('started',0)} "
    f"pending={status_counts.get('pending',0)}"
)
with metrics_log.open("a") as fh:
    fh.write(metrics_line + "\n")

print(f"Updated {assign_log} and {metrics_log}")
PYCODE

