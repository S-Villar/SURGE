#!/usr/bin/env bash
# Auto-resume 4h interactive field-loss training on qc_peak4 recipe.
# Idempotent: safe to re-run across sessions until early-stop or --epochs.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss}"
mkdir -p "$OUT"
LAST="$OUT/ckpt_fno2d_last.pt"
RESUME=()
if [[ -f "$LAST" ]]; then
  RESUME=(--resume "$LAST")
  echo "[run_field_loss_4h] resuming from $LAST"
else
  echo "[run_field_loss_4h] fresh start -> $OUT"
fi

# Request 4h interactive GPU (user must run from login node or existing allocation).
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Launching salloc -N1 --gpus-per-node=1 -t 04:00:00 ..."
  exec salloc -N1 --gpus-per-node=1 -t 04:00:00 -C gpu -q interactive \
    bash "$0" "$@"
fi

srun -n1 --gpus-per-node=1 python -u scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 \
  --grid 128 --m-lo -80 --m-hi 20 \
  --models fno2d \
  --target-norm max --target-space log10 \
  --target-floor 6 --target-smooth 1 \
  --exclude-list runs/quarantine/bad_cases.json \
  --peak-weight 4 \
  --fno-modes 48 --fno-hidden 32 \
  --loc-weight 2 --marg-weight 1 \
  --select-by field \
  --field-loss-weight 0.5 \
  --field-loss-warmup 20 \
  --field-select-n 64 \
  --field-select-every 5 \
  --time-budget-min 210 \
  --epochs 400 --patience 120 --batch-size 16 --seed 42 \
  --out "$OUT" \
  "${RESUME[@]}"

echo "[run_field_loss_4h] session finished; re-run this script to continue if needed."
