#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH -A mp288
#SBATCH -o surge_train_delta_p_hpo_patience.%j.log
#SBATCH -e surge_train_delta_p_hpo_patience.%j.log
#SBATCH -J surge_dp_hpo_pat

# M3DC1 delta p per-mode HPO with patience (debug queue, 30 min).
# Uses early stopping: patience=20 epochs.
#
# Submit: sbatch scripts/m3dc1/train_delta_p_per_mode_hpo_patience_debug.slurm

set -euo pipefail

SURGE_ROOT="${SURGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$SURGE_ROOT"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate surge 2>/dev/null || conda activate surge-devel 2>/dev/null || true
fi

echo "=== SURGE delta p per-mode HPO (patience=20) ==="
echo "SURGE_ROOT: $SURGE_ROOT"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

BATCH_DIR="/pscratch/sd/a/asvillar/mp288/jobs/batch_16"
DATASET_PKL="data/datasets/SPARC/delta_p_per_mode.pkl"

if [ ! -f "$DATASET_PKL" ] && [ -d "$BATCH_DIR" ]; then
    echo ">>> Building dataset from $BATCH_DIR"
    python scripts/m3dc1/build_delta_p_per_mode.py "$BATCH_DIR" --out "$DATASET_PKL"
    echo ""
fi

echo ">>> Running HPO workflow (patience suffix)..."
python -m surge.cli run configs/m3dc1_delta_p_per_mode_hpo.yaml --run-tag m3dc1_delta_p_per_mode_hpo_patience

echo ""
echo "=== Done at $(date) ==="
