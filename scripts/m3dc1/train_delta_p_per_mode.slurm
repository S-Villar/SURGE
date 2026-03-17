#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH -A mp288
#SBATCH -o surge_train_delta_p.%j.log
#SBATCH -e surge_train_delta_p.%j.log
#SBATCH -J surge_delta_p

# Train M3DC1 delta p per-mode surrogate on Perlmutter (debug queue).
# Output goes to surge_train_delta_p.<jobid>.log
#
# Submit: sbatch scripts/m3dc1/train_delta_p_per_mode.slurm
#
# For regular queue (longer runs): sbatch --qos=regular --time=02:00:00 scripts/m3dc1/train_delta_p_per_mode.slurm

set -euo pipefail

SURGE_ROOT="${SURGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$SURGE_ROOT"

# Activate conda (adjust env name if needed)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate surge 2>/dev/null || conda activate surge-devel 2>/dev/null || true
fi

echo "=== SURGE delta p per-mode training ==="
echo "SURGE_ROOT: $SURGE_ROOT"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Step 1: Build dataset (if batch dir exists and dataset missing)
BATCH_DIR="/pscratch/sd/a/asvillar/mp288/jobs/batch_16"
DATASET_PKL="data/datasets/SPARC/delta_p_per_mode.pkl"

if [ ! -f "$DATASET_PKL" ] && [ -d "$BATCH_DIR" ]; then
    echo ">>> Building dataset from $BATCH_DIR"
    python scripts/m3dc1/build_delta_p_per_mode.py "$BATCH_DIR" --out "$DATASET_PKL"
    echo ""
else
    echo ">>> Using existing dataset: $DATASET_PKL"
fi

# Step 2: Analyze dataset (optional)
echo ">>> Dataset analysis:"
python -m surge.cli analyze configs/m3dc1_delta_p_per_mode.yaml 2>/dev/null || true
echo ""

# Step 3: Train
echo ">>> Training models..."
python -m surge.cli run configs/m3dc1_delta_p_per_mode.yaml

echo ""
echo "=== Done at $(date) ==="
