#!/usr/bin/env bash
# Run E+F combined: m in [-100,100] + grid 201 + D' recipe (fieldloss, smooth0, select-by field).
# TWO levers vs D' — interpret only after single-lever Runs E and F are benchmarked.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OUT="${OUT:-runs/spectrum_fno48_floor6_smooth0_qc_peak4_fieldloss_g201_mwin100-100}"
mkdir -p "$OUT"
RESUME=()
[[ -f "$OUT/ckpt_fno2d_last.pt" ]] && RESUME=(--resume "$OUT/ckpt_fno2d_last.pt")

srun -n1 --gpus-per-node=1 python -u scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 \
  --grid 201 --m-lo -100 --m-hi 100 \
  --models fno2d \
  --target-norm max --target-space log10 \
  --target-floor 6 --target-smooth 0 \
  --exclude-list runs/quarantine/bad_cases.json \
  --peak-weight 4 \
  --fno-modes 64 --fno-hidden 32 \
  --loc-weight 2 --marg-weight 1 \
  --select-by field \
  --field-loss-weight 0.5 \
  --field-loss-warmup 20 \
  --field-select-n 64 \
  --field-select-every 5 \
  --time-budget-min 210 \
  --epochs 400 --patience 120 --batch-size 4 --seed 42 \
  --out "$OUT" \
  "${RESUME[@]}"
