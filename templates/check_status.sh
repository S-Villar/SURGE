#!/bin/bash

set -euo pipefail
shopt -s nullglob

summary_only=false
run_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--summary)
            summary_only=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [-s|--summary] <run_directory>"
            exit 1
            ;;
        *)
            if [[ -z "$run_dir" ]]; then
                run_dir="$1"
            else
                echo "Error: Multiple run directories specified"
                echo "Usage: $0 [-s|--summary] <run_directory>"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$run_dir" ]]; then
    echo "Usage: $0 [-s|--summary] <run_directory>"
    exit 1
fi

if [[ ! -d "$run_dir" ]]; then
    echo "Run directory '$run_dir' not found"
    exit 1
fi

total=0
finished=0
queued=0
pending=0

queued_list=()
pending_list=()

for case_dir in "$run_dir"/sparc_*; do
    [[ -d "$case_dir" ]] || continue
    total=$((total + 1))

    if [[ -f "$case_dir/C1.h5" || -f "$case_dir/finished" ]]; then
        finished=$((finished + 1))
        continue
    fi

    if [[ -f "$case_dir/queued" ]]; then
        queued=$((queued + 1))
        queued_list+=("$case_dir")
        continue
    fi

    pending=$((pending + 1))
    pending_list+=("$case_dir")
done

echo "Run directory      : $run_dir"
echo "Total cases        : $total"
echo "Finished cases     : $finished"
echo "Queued cases       : $queued"
echo "Pending cases      : $pending"

percent() {
    local count="$1"
    local total="$2"
    if [[ "$total" -eq 0 ]]; then
        printf "0.0"
    else
        printf "%.1f" "$(bc -l <<< "scale=3; ($count * 100) / $total")"
    fi
}

echo
echo "Finished (%): $(percent "$finished" "$total")%"
echo "Queued   (%): $(percent "$queued" "$total")%"
echo "Pending  (%): $(percent "$pending" "$total")%"

if [[ "$summary_only" == false ]]; then
    if [[ ${#queued_list[@]} -gt 0 ]]; then
        echo
        echo "Queued case directories:"
        printf '  %s\n' "${queued_list[@]}"
    fi

    if [[ ${#pending_list[@]} -gt 0 ]]; then
        echo
        echo "Pending case directories (no finished/C1.h5/queued markers):"
        printf '  %s\n' "${pending_list[@]}"
    fi
fi


