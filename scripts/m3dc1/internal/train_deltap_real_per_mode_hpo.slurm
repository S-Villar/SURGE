#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH -A mp288
#SBATCH -o surge_deltap_real_hpo.%j.log
#SBATCH -e surge_deltap_real_hpo.%j.log
#SBATCH -J deltap_real_hpo

# Re(delta p) per-mode recipe-based HPO on a capped case subset (see config).
# Best hyperparameters transfer to the full flagship run.
#
# Submit from the SURGE repo root:
#   sbatch scripts/m3dc1/internal/train_deltap_real_per_mode_hpo.slurm

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SURGE_ROOT="${SURGE_ROOT:-$SLURM_SUBMIT_DIR}"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  SURGE_ROOT="${SURGE_ROOT:-$(cd "$_HERE/../../.." && pwd)}"
fi
cd "$SURGE_ROOT"
if [[ ! -d surge ]]; then
  echo "ERROR: SURGE_ROOT=$SURGE_ROOT is not the SURGE repo root."
  exit 1
fi

export SURGE_CONDA_ENV="${SURGE_CONDA_ENV:-/global/cfs/projectdirs/m3716/software/asvillar/envs/surge}"
source "$SURGE_ROOT/scripts/m3dc1/surge_slurm_env.sh"
surge_slurm_setup_python

echo ">>> $(date)  Re(delta p) per-mode recipe HPO — run_tag m3dc1_deltap_real_per_mode_hpo"
python scripts/m3dc1/internal/run_workflow.py configs/internal/m3dc1_deltap_real_per_mode_hpo.yaml
echo ">>> $(date)  Done"
