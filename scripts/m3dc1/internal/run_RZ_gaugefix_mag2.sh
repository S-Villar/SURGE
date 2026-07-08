#!/usr/bin/env bash
# Gauge-fixed RZ — |δp|² target (Step 4). Single lever: --target-mode mag2
set -euo pipefail
cd /global/homes/a/asvillar/src/SURGE
source scripts/m3dc1/surge_slurm_env.sh && surge_slurm_setup_python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-runs/rz_field_gaugefix_mag2}"
mkdir -p "$OUT"
RESUME=()
[[ -f "$OUT/ckpt_fno2d_last.pt" ]] && RESUME=(--resume "$OUT/ckpt_fno2d_last.pt")

python -u scripts/m3dc1/internal/train_rz_field_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 \
  --pert-field p_phi0 --time-idx -1 --grid 201 \
  --models fno2d --fno-modes 64 --fno-hidden 32 \
  --gauge-fix --target-mode mag2 \
  --no-target-zscore \
  --peak-weight 4 --loc-weight 2 --marg-weight 1 \
  --target-floor 6 --target-smooth 0 \
  --exclude-list runs/quarantine/bad_cases.json \
  --select-by field --epochs "${EPOCHS:-1000}" --patience "${PATIENCE:-0}" \
  --batch-size 4 --lr 1e-3 --time-budget-min "${TIME_BUDGET_MIN:-235}" \
  --test-frac 0.2 --val-frac 0.1 --seed 42 \
  --out "$OUT" \
  "${RESUME[@]}"

echo "Done. Metrics: $OUT/rz_field_metrics.json"
