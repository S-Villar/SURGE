#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH -C cpu
#SBATCH -A mp288
#SBATCH -o surge_train_delta_p_cfs_hpo.%j.log
#SBATCH -e surge_train_delta_p_cfs_hpo.%j.log
#SBATCH -J surge_deltap_cfs_hpo

# CFS δp per-mode with HPO (same trial counts as m3dc1_delta_p_per_mode_hpo.yaml).
# Requires an existing pickle (build separately or use train_delta_p_per_mode_cfs.slurm first).
#
# Submit: sbatch scripts/m3dc1/train_delta_p_per_mode_cfs_hpo.slurm

set -euo pipefail

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

DATASET_FILE="data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet"
if [ ! -f "$DATASET_FILE" ]; then
    echo "ERROR: Missing $DATASET_FILE"
    echo "Build with: python scripts/m3dc1/build_delta_p_per_mode.py /global/cfs/projectdirs/amsc007/data/m3dc1 --out \"$DATASET_FILE\" --filename sdata_complex_v2.h5"
    exit 1
fi

echo ">>> HPO training: configs/m3dc1_delta_p_per_mode_cfs_hpo.yaml"
python -m surge.cli run configs/m3dc1_delta_p_per_mode_cfs_hpo.yaml
echo "=== Done at $(date) ==="
