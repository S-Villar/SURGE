#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH -A mp288
#SBATCH -o surge_train_delta_p_hpo_extended.%j.log
#SBATCH -e surge_train_delta_p_hpo_extended.%j.log
#SBATCH -J surge_dp_hpo_ext

# M3DC1 delta p per-mode extended HPO: 15 RF + 250 MLP trials (4 hr).
#
# Submit: sbatch scripts/m3dc1/train_delta_p_per_mode_hpo_extended.slurm

set -euo pipefail

SURGE_ROOT="${SURGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$SURGE_ROOT"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate surge 2>/dev/null || conda activate surge-devel 2>/dev/null || true
fi

echo "=== SURGE delta p per-mode extended HPO (15 RF + 250 MLP trials) ==="
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

echo ">>> Running extended HPO workflow..."
python -m surge.cli run configs/m3dc1_delta_p_per_mode_hpo_extended.yaml

echo ""
echo "=== Done at $(date) ==="
