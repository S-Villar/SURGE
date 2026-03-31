#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH -C cpu
#SBATCH -A mp288
#SBATCH -o surge_cfs_mlp_hpo.%j.log
#SBATCH -e surge_cfs_mlp_hpo.%j.log
#SBATCH -J cfs_mlp_hpo

# MLP-only flexible HPO + BoTorch on CFS δp per-mode Parquet.
# Needs: data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet
#        pip install botorch  (Optuna BoTorch sampler)
#
# Submit: cd $SURGE_ROOT && sbatch scripts/m3dc1/train_delta_p_per_mode_cfs_mlp_hpo.slurm

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SURGE_ROOT="${SURGE_ROOT:-$SLURM_SUBMIT_DIR}"
else
  SURGE_ROOT="${SURGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
cd "$SURGE_ROOT"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate surge 2>/dev/null || true
fi

DATA="data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet"
test -f "$DATA" || { echo "Missing $DATA"; exit 1; }

echo ">>> $(date)  Running flexible MLP HPO (BoTorch)"
python -m surge.cli run configs/m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible.yaml
echo ">>> $(date)  Done"
