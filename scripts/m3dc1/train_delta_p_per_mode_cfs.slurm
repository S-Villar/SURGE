#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH -C cpu
#SBATCH -A mp288
#SBATCH -o surge_train_delta_p_cfs.%j.log
#SBATCH -e surge_train_delta_p_cfs.%j.log
#SBATCH -J surge_deltap_cfs

# CFS bulk δp per-mode: build pickle from sdata_complex_v2.h5, then train RF + MLP.
# Expect large memory for pickle build (~2M rows); request a fat node if OOM.
#
# Submit: sbatch scripts/m3dc1/train_delta_p_per_mode_cfs.slurm
#
# Adjust --time / --qos for HPO config (train_delta_p_per_mode_cfs_hpo.yaml).

set -euo pipefail

# Slurm runs the script from a copied path; use submit directory for repo root.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SURGE_ROOT="${SURGE_ROOT:-$SLURM_SUBMIT_DIR}"
else
  SURGE_ROOT="${SURGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
cd "$SURGE_ROOT"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate surge 2>/dev/null || conda activate surge-devel 2>/dev/null || true
fi

CFS_M3DC1="/global/cfs/projectdirs/amsc007/data/m3dc1"
DATASET_FILE="data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet"
CONFIG="configs/m3dc1_delta_p_per_mode_cfs.yaml"

echo "=== SURGE δp per-mode (CFS sdata_complex_v2) ==="
echo "SURGE_ROOT: $SURGE_ROOT"
echo "Date: $(date)"
echo ""

if [ ! -f "$DATASET_FILE" ]; then
    echo ">>> Building $DATASET_FILE from $CFS_M3DC1"
    python scripts/m3dc1/build_delta_p_per_mode.py "$CFS_M3DC1" \
        --out "$DATASET_FILE" \
        --filename sdata_complex_v2.h5
    echo ""
else
    echo ">>> Using existing $DATASET_FILE"
fi

echo ">>> Training: $CONFIG"
python -m surge.cli run "$CONFIG"

echo "=== Done at $(date) ==="
