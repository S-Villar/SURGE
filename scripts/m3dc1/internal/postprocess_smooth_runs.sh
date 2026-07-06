#!/usr/bin/env bash
# Full §9-style postprocessing for Run D (smooth05) and D' (smooth0).
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python

REF_CACHE="runs/spectrum_fno48_floor6_smooth1_qc/predictions_cache.npz"
ASSETS="docs/m3dc1/assets"
SPARC1530_CASES=(1904 1327 2729)  # run79, run43, run99 cache indices

post_one() {
  local OUT="$1"
  local TAG="$2"
  local SHORT="$3"
  local BENCH="field_bench/with_${TAG}"

  echo ""
  echo "========== ${OUT} =========="

  if [[ ! -f "${OUT}/predictions_cache.npz" ]]; then
    python scripts/m3dc1/internal/export_predictions_cache.py --run "$OUT" --device cuda
  else
    echo "cache exists: ${OUT}/predictions_cache.npz"
  fi

  python scripts/m3dc1/internal/metric_gallery.py \
    --cache "${OUT}/predictions_cache.npz" \
    --ref-cache "$REF_CACHE" \
    --split test --metric r2_pattern \
    --out "${ASSETS}/metric_reality_check_qc_peak4_${SHORT}_refqc_combined.png"

  python scripts/m3dc1/internal/metric_gallery.py \
    --cache "${OUT}/predictions_cache.npz" \
    --ref-cache "$REF_CACHE" \
    --split test --metric r2_pattern --field \
    --out "${ASSETS}/metric_reality_check_qc_peak4_${SHORT}_refqc_field.png"

  python scripts/m3dc1/internal/plot_case_field_recon.py \
    --run "$OUT" --split test --out-dir "$ASSETS" --tag "${SHORT}_test"

  for ci in "${SPARC1530_CASES[@]}"; do
    python scripts/m3dc1/internal/plot_case_field_recon.py \
      --run "$OUT" --split test --out-dir "${ASSETS}/sparc1530_smooth_ablation" \
      --tag sparc1530 --cases "$ci"
  done

  python scripts/m3dc1/internal/field_recon_compare.py \
    --run "$OUT" --model fno2d --device cuda \
    --out "${ASSETS}/${SHORT}_field_recon"
  cp "${ASSETS}/${SHORT}_field_recon/field_recon_fno2d.png" \
     "${ASSETS}/field_recon_${SHORT}.png"

  if [[ ! -f "${BENCH}/leaderboard.json" ]]; then
    python scripts/m3dc1/internal/field_bench.py \
      --runs runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss \
             runs/spectrum_fno48_floor6_smooth1_qc_peak4 \
             "$OUT" \
      --split test --device cuda --out "$BENCH"
  else
    echo "field_bench exists: ${BENCH}"
  fi
}

mkdir -p "$ASSETS/sparc1530_smooth_ablation"

post_one \
  "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05" \
  "fieldloss_smooth05" \
  "fieldloss_smooth05"

post_one \
  "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0" \
  "fieldloss_smooth0" \
  "fieldloss_smooth0"

python scripts/m3dc1/internal/postprocess_smooth_ablation_assets.py

echo ""
echo "=== All postprocessing complete ==="
echo "Assets in ${ASSETS}/ and ${ASSETS}/sparc1530_smooth_ablation/"
