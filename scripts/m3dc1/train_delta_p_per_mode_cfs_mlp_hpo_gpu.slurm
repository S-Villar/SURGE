#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH -A mp288
#SBATCH -o surge_cfs_mlp_hpo_gpu.%j.log
#SBATCH -e surge_cfs_mlp_hpo_gpu.%j.log
#SBATCH -J cfs_mlp_hpo_gpu

# Same as train_delta_p_per_mode_cfs_mlp_hpo.slurm but GPU node; distinct run_tag for results.
#
# Submit: sbatch scripts/m3dc1/train_delta_p_per_mode_cfs_mlp_hpo_gpu.slurm

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

echo ">>> $(date)  MLP HPO (BoTorch) on GPU — run_tag m3dc1_delta_p_per_mode_cfs_mlp_hpo_gpu"
python -m surge.cli run configs/m3dc1_delta_p_per_mode_cfs_mlp_hpo_flexible.yaml \
  --run-tag m3dc1_delta_p_per_mode_cfs_mlp_hpo_gpu
echo ">>> $(date)  Done"
