#!/bin/bash
# submit_batch_plan.sh
#
# Reads planned_submissions.json and executes (or dry-runs) sbatch commands.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_batch_plan.sh --plan <planned_submissions.json> --job-script <sbatch_script> [--dry-run]

The plan JSON is expected to match the structure emitted by plan_batch_chunks.py.
EOF
}

plan_file=""
job_script=""
dry_run=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan) plan_file="$2"; shift 2 ;;
        --job-script) job_script="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -f "$plan_file" ]] || { echo "Plan file not found: $plan_file" >&2; exit 1; }
[[ -f "$job_script" ]] || { echo "Job script not found: $job_script" >&2; exit 1; }

python3 - "$plan_file" "$job_script" "$dry_run" <<'PYCODE'
import json
import subprocess
import sys

plan_path = sys.argv[1]
job_script = sys.argv[2]
dry_run = sys.argv[3].lower() == "true"

with open(plan_path) as fh:
    plan = json.load(fh)

entries = plan.get("entries", [])
if not entries:
    print("No entries in plan; nothing to submit.")
    sys.exit(0)

batch_dir = plan.get("batch_dir")
if batch_dir is None:
    raise SystemExit("Plan missing 'batch_dir'.")

for idx, entry in enumerate(entries, start=1):
    config = entry["config"]
    array_range = entry["array_range"]
    label = config["label"]
    cmd = [
        "sbatch",
        f"--nodes={config['nodes']}",
        f"--ntasks-per-node={config['ntasks_per_node']}",
        f"--cpus-per-task={config['cpus_per_task']}",
        f"--time={config['time']}",
        f"--array={array_range}",
        f"--job-name=m3dc1_{label}",
        f"--chdir={batch_dir}",
        job_script,
    ]
    print(f"[{idx}/{len(entries)}] {' '.join(cmd)}")
    if dry_run:
        continue
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("  sbatch ->", result.stdout.strip())
PYCODE





