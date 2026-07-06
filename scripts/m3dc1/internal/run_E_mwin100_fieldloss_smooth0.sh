#!/usr/bin/env bash
# Run E: symmetric wide m-window + D' recipe (fieldloss, smooth0, select-by field).
# Single lever vs D': m in [-100,100] instead of [-80,20].
# Contrasts with failed qc_mhi100 (geom+composite+peak0, m in [-80,100]).
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OUT="${OUT:-runs/spectrum_fno48_floor6_smooth0_qc_peak4_fieldloss_mwin100-100}"
mkdir -p "$OUT"
RESUME=()
[[ -f "$OUT/ckpt_fno2d_last.pt" ]] && RESUME=(--resume "$OUT/ckpt_fno2d_last.pt")

srun -n1 --gpus-per-node=1 python -u scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 \
  --grid 128 --m-lo -100 --m-hi 100 \
  --models fno2d \
  --target-norm max --target-space log10 \
  --target-floor 6 --target-smooth 0 \
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
