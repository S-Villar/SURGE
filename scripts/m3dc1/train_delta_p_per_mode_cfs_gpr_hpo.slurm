#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH -C cpu
#SBATCH -A mp288
#SBATCH -o surge_cfs_gpr_hpo.%j.log
#SBATCH -e surge_cfs_gpr_hpo.%j.log
#SBATCH -J cfs_gpr_hpo

# GPflow GPR: Linear + Matern52 kernel, HPO with Optuna BoTorch sampler.
# Uses sample_rows + single output (output_p_0) — see config header.
#
# Submit: cd $SURGE_ROOT && sbatch scripts/m3dc1/train_delta_p_per_mode_cfs_gpr_hpo.slurm

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

echo ">>> $(date)  GPR Linear+Matern52 HPO (BoTorch)"
python -m surge.cli run configs/m3dc1_delta_p_per_mode_cfs_gpr_linear_matern52_botorch.yaml
echo ">>> $(date)  Done"
