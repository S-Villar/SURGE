#!/bin/bash
# submit_batch_chunks.sh
#
# Orchestrates the modular batch workflow:
#   1. scan_batch_cases.sh           -> collect case metadata (JSON lines)
#   2. compute_batch_metrics.py      -> summarize status and timing
#   3. plan_batch_chunks.py          -> choose next chunks/configurations
#   4. submit_batch_plan.sh          -> run or dry-run sbatch commands
#   5. update_batch_logs.sh          -> append submissions to log files
#
# Keeping each step separate makes debugging easier and avoids long-running monoliths.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_batch_chunks.sh -b <batch_dir> [-c <chunk_size>] [-n <chunks>] [--dry-run]
                               [--start-config <idx>] [--config-file <json>] [--plan-only]
                               [--stages <list>]

Options:
  -b  Absolute path to batch directory (required).
  -c  Chunk size passed to planner (default: 32).
  -n  Max chunks to submit (default: 1, use -n 0 for no limit).
  --dry-run       Plan only; submission step runs in dry-run mode.
  --plan-only     Stop after generating the plan JSON.
  --start-config  Index into configuration matrix (default 0).
  --config-file   JSON file describing custom configs (optional).
  --stages        Comma-separated stages to run (scan,metrics,plan,submit,log).
                  Default: all stages in the order listed above.
  -h              Show this help.
EOF
}

batch_dir=""
chunk_size=32
max_chunks=1
dry_run=false
plan_only=false
config_start_index=0
config_file=""
stages_arg=""
tmp_dir=""

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --plan-only) plan_only=true; shift ;;
        --start-config) config_start_index="$2"; shift 2 ;;
        --config-file) config_file="$2"; shift 2 ;;
        --stages) stages_arg="$2"; shift 2 ;;
        --) shift; break ;;
        -*) args+=("$1"); shift ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"

while getopts ":b:c:n:h" opt; do
    case "$opt" in
        b) batch_dir="$OPTARG" ;;
        c) chunk_size="$OPTARG" ;;
        n) max_chunks="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
    esac
done

[[ -n "$batch_dir" ]] || { echo "Error: batch directory (-b) required." >&2; usage; exit 1; }
[[ -d "$batch_dir" ]] || { echo "Error: batch directory not found: $batch_dir" >&2; exit 1; }
[[ "$chunk_size" =~ ^[0-9]+$ && "$chunk_size" -gt 0 ]] || { echo "Chunk size must be positive integer." >&2; exit 1; }
[[ "$max_chunks" =~ ^[0-9]+$ ]] || { echo "Max chunks must be non-negative integer." >&2; exit 1; }
if [[ "$max_chunks" -eq 0 ]]; then max_chunks=$((2**31-1)); fi
[[ "$config_start_index" =~ ^[0-9]+$ ]] || { echo "--start-config must be non-negative integer." >&2; exit 1; }

batch_dir="$(realpath "$batch_dir")"
root_dir="/global/homes/a/asvillar/src/SURGE"
tmp_dir="${batch_dir}/tmp"
mkdir -p "$tmp_dir"

scan_script="${root_dir}/scripts/scan_batch_cases.sh"
metrics_script="${root_dir}/scripts/compute_batch_metrics.py"
plan_script="${root_dir}/scripts/plan_batch_chunks.py"
submit_script="${root_dir}/scripts/submit_batch_plan.sh"
log_script="${root_dir}/scripts/update_batch_logs.sh"
job_script="${root_dir}/templates/batchjob.perlmutter"

for tool in "$scan_script" "$metrics_script" "$plan_script" "$submit_script" "$log_script"; do
    [[ -f "$tool" ]] || { echo "Required tool not found: $tool" >&2; exit 1; }
done
[[ -f "$job_script" ]] || { echo "Job script not found: $job_script" >&2; exit 1; }

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log_info() { printf "[%s] %s\n" "$(timestamp)" "$*"; }

log_info "Batch directory : $batch_dir"
log_info "Chunk size      : $chunk_size"
log_info "Max chunks      : $max_chunks"
log_info "Dry run         : $dry_run"
log_info "Plan only       : $plan_only"
log_info "Config start    : $config_start_index"
log_info "Config file     : ${config_file:-<built-in>}"
log_info "Stages arg      : ${stages_arg:-<default>}"
log_info "Temp directory  : $tmp_dir"

case_snapshot="${tmp_dir}/cases.jsonl"
plan_file="${tmp_dir}/planned_submissions.json"

default_stage_order=(scan metrics plan submit log)
declare -a stages_to_run=("${default_stage_order[@]}")

if [[ -n "$stages_arg" ]]; then
    IFS=',' read -ra user_stages <<<"$stages_arg"
    stages_to_run=()
    declare -A valid_stage_set=(
        [scan]=1
        [metrics]=1
        [plan]=1
        [submit]=1
        [log]=1
    )
    for stage in "${user_stages[@]}"; do
        stage_lower=$(echo "$stage" | tr '[:upper:]' '[:lower:]')
        if [[ -z "${valid_stage_set[$stage_lower]:-}" ]]; then
            echo "Invalid stage '${stage}'. Valid stages: scan,metrics,plan,submit,log." >&2
            exit 1
        fi
        stages_to_run+=("$stage_lower")
    done
fi

# If plan-only ensure submit/log excluded
if [[ "$plan_only" == true ]]; then
    filtered=()
    for stage in "${stages_to_run[@]}"; do
        [[ "$stage" == "submit" || "$stage" == "log" ]] && continue
        filtered+=("$stage")
    done
    stages_to_run=("${filtered[@]}")
fi

# Deduplicate while preserving order relative to default stage order
ordered_stages=()
declare -A seen_stage=()
for default_stage in "${default_stage_order[@]}"; do
    for requested_stage in "${stages_to_run[@]}"; do
        if [[ "$requested_stage" == "$default_stage" && -z "${seen_stage[$requested_stage]:-}" ]]; then
            ordered_stages+=("$requested_stage")
            seen_stage[$requested_stage]=1
        fi
    done
done
stages_to_run=("${ordered_stages[@]}")

log_info "Stages to run    : ${stages_to_run[*]}"

run_stage() {
    local name="$1"
    shift
    log_info ">> Stage ${name}: $*"
    eval "$@"
    log_info "<< Stage ${name} completed."
}

contains_stage() {
    local target="$1"
    for stage in "${stages_to_run[@]}"; do
        [[ "$stage" == "$target" ]] && return 0
    done
    return 1
}

if contains_stage "scan"; then
    log_info ">> Stage scan: generating snapshot at $case_snapshot"
    "$scan_script" -b "$batch_dir" -o "$case_snapshot"
    log_info "<< Stage scan completed."
else
    log_info "Skip stage scan (requested stages: ${stages_to_run[*]})."
    [[ -f "$case_snapshot" ]] || { echo "cases.jsonl missing; run scan stage first." >&2; exit 1; }
fi

if contains_stage "metrics"; then
    log_info ">> Stage metrics"
    python3 "$metrics_script" --input "$case_snapshot"
    log_info "<< Stage metrics completed."
else
    log_info "Skip stage metrics."
fi

planner_args=(
    --cases "$case_snapshot"
    --output "$plan_file"
    --chunk-size "$chunk_size"
    --max-chunks "$max_chunks"
    --start-config "$config_start_index"
)
[[ -n "$config_file" ]] && planner_args+=(--config-file "$config_file")

if contains_stage "plan"; then
    log_info ">> Stage plan: writing plan to $plan_file"
    python3 "$plan_script" "${planner_args[@]}"
    log_info "<< Stage plan completed."
else
    log_info "Skip stage plan."
    [[ -f "$plan_file" ]] || { echo "Plan file missing; run plan stage first." >&2; exit 1; }
fi

if [[ "$plan_only" == true ]]; then
    log_info "Plan-only mode: skipping submission/log update."
    exit 0
fi

submit_args=(
    --plan "$plan_file"
    --job-script "$job_script"
)
[[ "$dry_run" == true ]] && submit_args+=(--dry-run)

if contains_stage "submit"; then
    log_info ">> Stage submit"
    "$submit_script" "${submit_args[@]}"
    log_info "<< Stage submit completed."
else
    log_info "Skip stage submit."
fi

if contains_stage "log"; then
    log_info ">> Stage log"
    "$log_script" --plan "$plan_file" --metrics "$case_snapshot"
    log_info "<< Stage log completed."
else
    log_info "Skip stage log."
fi

log_info "Workflow completed successfully."
