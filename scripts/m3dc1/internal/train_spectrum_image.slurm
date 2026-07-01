#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH -A mp288
#SBATCH -o surge_spectrum_image.%j.log
#SBATCH -e surge_spectrum_image.%j.log
#SBATCH -J spec_image

# Whole-spectrum 2D-image surrogate for |delta p hat|(m, psi_N) with FNO2D + U-Net
# (conditioning-as-channels, incl. the m - n q(psi) resonance channel).
#
# Submit from the SURGE repo root:
#   sbatch scripts/m3dc1/internal/train_spectrum_image.slurm

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

# Overridable via env: SI_NCASES, SI_OUT, SI_EPOCHS, SI_GRID, SI_BATCH,
#   SI_MODELS, SI_TNORM (none|max), SI_TSPACE (log10|raw), SI_PATIENCE,
#   SI_CKPT_EVERY, SI_RESUME (path to a ckpt .pt to continue from).
SI_NCASES="${SI_NCASES:-3000}"
SI_OUT="${SI_OUT:-runs/spectrum_image}"
SI_EPOCHS="${SI_EPOCHS:-120}"
SI_GRID="${SI_GRID:-128}"
SI_BATCH="${SI_BATCH:-16}"
SI_MODELS="${SI_MODELS:-fno2d unet}"
SI_TNORM="${SI_TNORM:-none}"
SI_TSPACE="${SI_TSPACE:-log10}"
SI_PATIENCE="${SI_PATIENCE:-25}"
SI_CKPT_EVERY="${SI_CKPT_EVERY:-0}"
SI_RESUME="${SI_RESUME:-}"
# Peak-weighted loss (0 = plain MSE); up-weights high-|dp| ridge/peak pixels.
SI_PEAK_WEIGHT="${SI_PEAK_WEIGHT:-0}"
SI_PEAK_POW="${SI_PEAK_POW:-2}"

# Fewer fragmentation OOMs on shared A100s.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# SI_NCASES=0 (or "all") -> use the entire dataset (omit --n-cases).
NCASES_ARG=(--n-cases "$SI_NCASES")
if [[ "$SI_NCASES" == "0" || "$SI_NCASES" == "all" ]]; then
  NCASES_ARG=()
fi
RESUME_ARG=()
if [[ -n "$SI_RESUME" ]]; then RESUME_ARG=(--resume "$SI_RESUME"); fi

echo ">>> $(date)  spectrum-image surrogate ($SI_MODELS) on |delta p hat| — n_cases=$SI_NCASES out=$SI_OUT norm=$SI_TNORM space=$SI_TSPACE resume=${SI_RESUME:-none}"
# In an sbatch job the batch script already runs ON the compute node, so python
# uses the allocated GPU directly (no srun wrapper needed here).
python scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 \
  "${NCASES_ARG[@]}" --grid "$SI_GRID" --m-lo -80 --m-hi 20 \
  --models $SI_MODELS --epochs "$SI_EPOCHS" --batch-size "$SI_BATCH" \
  --target-norm "$SI_TNORM" --target-space "$SI_TSPACE" \
  --patience "$SI_PATIENCE" --ckpt-every "$SI_CKPT_EVERY" \
  --peak-weight "$SI_PEAK_WEIGHT" --peak-pow "$SI_PEAK_POW" \
  "${RESUME_ARG[@]}" --out "$SI_OUT"
echo ">>> $(date)  Done"
