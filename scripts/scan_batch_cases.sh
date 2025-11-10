#!/bin/bash
# scan_batch_cases.sh
#
# Walks a batch directory and emits JSON Lines describing each case.
# Each line includes case_id, run, equilibrium, path, status, and timestamps.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scan_batch_cases.sh -b <batch_dir>

Emits JSON lines to stdout. Fields:
  case_id: 1-based index
  run:     run directory name (e.g., run7)
  eq:      equilibrium directory name (e.g., sparc_1423)
  path:    absolute path to the case directory
  status:  finished | inflight | started | pending
  started_ts: seconds since epoch (if started file exists)
  finished_ts: seconds since epoch (if finished file exists)
EOF
}

batch_dir=""
output_file=""
verbose=true
while getopts ":b:o:h" opt; do
    case "$opt" in
        b) batch_dir="$OPTARG" ;;
        o) output_file="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
    esac
done

[[ -n "$batch_dir" ]] || { echo "Error: batch directory (-b) required." >&2; usage; exit 1; }
[[ -d "$batch_dir" ]] || { echo "Error: batch directory not found: $batch_dir" >&2; exit 1; }

batch_dir="$(realpath "$batch_dir")"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() {
    if [[ "$verbose" == true ]]; then
        printf "[%s] %s\n" "$(timestamp)" "$*" >&2
    fi
}

log "Scanning batch directory: $batch_dir"

shopt -s nullglob
runs=( "$batch_dir"/run* )
[[ "${#runs[@]}" -gt 0 ]] || { echo "No run directories in $batch_dir." >&2; exit 1; }
IFS=$'\n' runs=($(printf "%s\n" "${runs[@]}" | sort -V))
shopt -u nullglob

user="${SUBMIT_CHUNKS_USER:-$(whoami)}"
mapfile -t active_indices < <(
    squeue -u "$user" -o "%i" -h \
        | grep -v '^-' \
        | tr ',' '\n' \
        | tr -d '[]' \
        | awk '
            NF {
                split($1,r,"-");
                if (length(r)==1) print r[1];
                else for(i=r[1]; i<=r[2]; ++i) print i;
            }' \
        | sort -n | uniq
)

declare -A inflight
for id in "${active_indices[@]}"; do
    inflight["$id"]=1
done

emit_case() {
    local case_id="$1" run="$2" eq="$3" path="$4" status="$5"
    if [[ -n "$output_file" ]]; then
        printf '{"case_id":%s,"run":"%s","eq":"%s","path":"%s","status":"%s"}\n' \
            "$case_id" "$run" "$eq" "$path" "$status" >>"$output_file"
    else
        printf '{"case_id":%s,"run":"%s","eq":"%s","path":"%s","status":"%s"}\n' \
            "$case_id" "$run" "$eq" "$path" "$status"
    fi
}

if [[ -n "$output_file" ]]; then
    mkdir -p "$(dirname "$output_file")"
    : >"$output_file"
    log "Writing snapshot to $output_file"
fi

case_id=0
for run_path in "${runs[@]}"; do
    [[ -d "$run_path" ]] || continue
    run=$(basename "$run_path")
    log "Processing $run"

    shopt -s nullglob
    eq_candidates=( "$run_path"/sparc_* )
    shopt -u nullglob
    (( ${#eq_candidates[@]} == 0 )) && continue

    tmp_sort=$(mktemp)
    for candidate in "${eq_candidates[@]}"; do
        [[ -d "$candidate" ]] || continue
        printf '%s\n' "$candidate" >>"$tmp_sort"
    done
    mapfile -t eq_dirs < <(sort -V "$tmp_sort")
    /bin/rm -f "$tmp_sort"

    for path in "${eq_dirs[@]}"; do
        eq=$(basename "$path")
        ((case_id++))

        started_file="${path}/started"
        finished_file="${path}/finished"
        c1_file="${path}/C1.h5"

        status="pending"
        if [[ -f "$finished_file" || -s "$c1_file" ]]; then
            status="finished"
        elif [[ -n "${inflight[$case_id]:-}" ]]; then
            status="inflight"
        elif [[ -f "$started_file" ]]; then
            status="started"
        fi

        log "    case_id=${case_id} eq=${eq} status=${status}"
        emit_case "$case_id" "$run" "$eq" "$path" "$status"
    done
done

log "Scan complete. Cases processed: $case_id"

