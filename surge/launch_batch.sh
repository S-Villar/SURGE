#!/bin/bash
# Launch SLURM job array for batch processing
# Each array task processes one run directory (all sparc_* cases in that run)
#
# Usage: launch_batch.sh [BATCH_NAME] [--queue QUEUE] [--nodes N] [--tasks-per-node T] [--cpus-per-task C]
#
# Arguments:
#   BATCH_NAME          Batch directory name (default: batch_5)
#   --queue QUEUE       Queue name: "regular" or "debug" (default: debug)
#   --nodes N           Number of nodes (default: 8)
#   --tasks-per-node T  Tasks per node (default: 12)
#   --cpus-per-task C   CPUs per task (default: 1)

set -euo pipefail

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

BATCH_NAME="batch_5"
QUEUE="debug"
NODES=8
TASKS_PER_NODE=12
CPUS_PER_TASK=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --tasks-per-node)
            TASKS_PER_NODE="$2"
            shift 2
            ;;
        --cpus-per-task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [BATCH_NAME] [--queue QUEUE] [--nodes N] [--tasks-per-node T] [--cpus-per-task C]"
            echo ""
            echo "Arguments:"
            echo "  BATCH_NAME          Batch directory name (default: batch_5)"
            echo "  --queue QUEUE       Queue: regular or debug (default: debug)"
            echo "  --nodes N           Number of nodes (default: 8)"
            echo "  --tasks-per-node T  Tasks per node (default: 12)"
            echo "  --cpus-per-task C   CPUs per task (default: 1)"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            BATCH_NAME="$1"
            shift
            ;;
    esac
done

# Template batchjob file (full path)
# Default: use batch_0 template from SPARC datasets
if [ -n "${SPARC_DATASETS:-}" ]; then
    TEMPLATE="${SPARC_DATASETS}/batch_0/batchjob.perlmutter"
else
    TEMPLATE="/global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC/batch_0/batchjob.perlmutter"
fi

# Optional: Override with environment variables
if [ -n "${SPARC_DATASETS:-}" ]; then
    BATCH_DIR="${SPARC_DATASETS}/${BATCH_NAME}"
else
    BATCH_DIR="/global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC/${BATCH_NAME}"
fi

# =============================================================================
# SCRIPT - Don't edit below unless you know what you're doing
# =============================================================================

echo "============================================================"
echo "Launching batch job array"
echo "============================================================"
echo "Batch directory: $BATCH_DIR"
echo "Template: $TEMPLATE"
echo "Queue: $QUEUE"
echo "Nodes: $NODES"
echo "Tasks per node: $TASKS_PER_NODE"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Total tasks: $((NODES * TASKS_PER_NODE))"
echo ""

# Expand environment variables in paths
BATCH_DIR=$(eval echo "$BATCH_DIR")
TEMPLATE=$(eval echo "$TEMPLATE")

# Validate batch directory
if [ ! -d "$BATCH_DIR" ]; then
    echo "ERROR: Batch directory does not exist: $BATCH_DIR" >&2
    exit 1
fi

# Validate template
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template file does not exist: $TEMPLATE" >&2
    exit 1
fi

# Find all run directories
RUNS=()
for run_dir in "$BATCH_DIR"/run*/; do
    if [ -d "$run_dir" ]; then
        run_name=$(basename "$run_dir")
        RUNS+=("$run_name")
    fi
done

# Sort runs
IFS=$'\n' RUNS=($(sort <<<"${RUNS[*]}"))
unset IFS

TOTAL_RUNS=${#RUNS[@]}

if [ $TOTAL_RUNS -eq 0 ]; then
    echo "ERROR: No run directories found in $BATCH_DIR" >&2
    exit 1
fi

echo "Found $TOTAL_RUNS runs: ${RUNS[*]}"
echo ""

# Limit array size for debug queue
ARRAY_SIZE=$TOTAL_RUNS
if [ "$QUEUE" = "debug" ] && [ $TOTAL_RUNS -gt 5 ]; then
    ARRAY_SIZE=5
    echo "WARNING: Debug queue limited to 5 jobs"
    echo "  Processing first 5 runs only (out of $TOTAL_RUNS total)"
    echo ""
fi

# Read template
TEMPLATE_CONTENT=$(cat "$TEMPLATE")

# Replace placeholder values with actual configuration
TEMPLATE_CONTENT=$(echo "$TEMPLATE_CONTENT" | sed "s/PLACEHOLDER_NODES/$NODES/g")
TEMPLATE_CONTENT=$(echo "$TEMPLATE_CONTENT" | sed "s/PLACEHOLDER_TASKS_PER_NODE/$TASKS_PER_NODE/g")
TEMPLATE_CONTENT=$(echo "$TEMPLATE_CONTENT" | sed "s/PLACEHOLDER_CPUS_PER_TASK/$CPUS_PER_TASK/g")

# Update queue/QOS directive
if grep -q "#SBATCH.*--qos" <<< "$TEMPLATE_CONTENT"; then
    TEMPLATE_CONTENT=$(sed "s/#SBATCH.*--qos=[^ ]*/#SBATCH --qos=$QUEUE/" <<< "$TEMPLATE_CONTENT")
elif grep -q "#SBATCH.*-q" <<< "$TEMPLATE_CONTENT"; then
    TEMPLATE_CONTENT=$(sed "s/#SBATCH.*-q [^ ]*/#SBATCH -q $QUEUE/" <<< "$TEMPLATE_CONTENT")
else
    # Insert after time directive
    TEMPLATE_CONTENT=$(sed "/^#SBATCH.*time/a #SBATCH --qos=$QUEUE" <<< "$TEMPLATE_CONTENT")
fi

# For debug queue, limit nodes to 1 (user can override with --nodes)
if [ "$QUEUE" = "debug" ] && [ "$NODES" -gt 1 ]; then
    echo "WARNING: Debug queue typically uses 1 node. Current setting: $NODES nodes"
    echo "  Use --nodes 1 to match debug queue constraints, or keep current setting"
fi

# Update array directive
if grep -q "#SBATCH.*--array" <<< "$TEMPLATE_CONTENT"; then
    TEMPLATE_CONTENT=$(sed "s/#SBATCH.*--array=[0-9]*-[0-9]*/#SBATCH --array=1-$ARRAY_SIZE/" <<< "$TEMPLATE_CONTENT")
else
    # Insert after other #SBATCH directives (after time)
    TEMPLATE_CONTENT=$(sed "/^#SBATCH.*time/a #SBATCH --array=1-$ARRAY_SIZE" <<< "$TEMPLATE_CONTENT")
fi

# Update output pattern to include array task ID
TEMPLATE_CONTENT=$(sed 's/#SBATCH -o \([^ ]*\)$/#SBATCH -o \1.%A.%a/' <<< "$TEMPLATE_CONTENT")

# Write batchjob to batch directory
BATCHJOB="$BATCH_DIR/batchjob.perlmutter"
echo "$TEMPLATE_CONTENT" > "$BATCHJOB"
chmod +x "$BATCHJOB"

echo "Created: $BATCHJOB"
echo "Array size: --array=1-$ARRAY_SIZE"
echo "Queue: $QUEUE"
echo ""

# Submit job
cd "$BATCH_DIR"
echo "Submitting job..."
JOB_OUTPUT=$(sbatch "$BATCHJOB" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+' || echo "unknown")
    echo "============================================================"
    echo "Job submitted successfully!"
    echo "============================================================"
    echo "Job ID: $JOB_ID"
    echo "Array tasks: 1-$ARRAY_SIZE"
    echo "Queue: $QUEUE"
    echo ""
    echo "Task mapping:"
    for i in $(seq 1 $ARRAY_SIZE); do
        echo "  Task $i → ${RUNS[$((i-1))]}"
    done
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  squeue -u $USER"
    echo ""
    echo "Check logs:"
    echo "  ls -lt $BATCH_DIR/M3DC1log.o${JOB_ID}.*"
else
    echo "ERROR: Job submission failed" >&2
    echo "$JOB_OUTPUT" >&2
    exit 1
fi
