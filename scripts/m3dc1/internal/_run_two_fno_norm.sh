#!/bin/bash
# Runs two full-dataset FNO2D spectrum-image trainings back-to-back inside an
# already-allocated GPU node (launched via salloc). Config A: max-normalized
# magnitude in RAW space; Config B: max-normalized magnitude in LOG10 space.
set -uo pipefail
cd /global/homes/a/asvillar/src/SURGE
export SURGE_ROOT="$PWD"
export SURGE_CONDA_ENV="${SURGE_CONDA_ENV:-/global/cfs/projectdirs/m3716/software/asvillar/envs/surge}"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python

# Reduce fragmentation-related OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# IMPORTANT: salloc runs THIS script on the login node; the actual training must
# be launched with `srun` so it executes on the allocated compute-node GPU.
SRUN=(srun --ntasks=1 --gpus-per-task=1)
echo ">>> $(date)  compute-node GPU:"
"${SRUN[@]}" bash -c 'echo "node=$(hostname)"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv' || true

BATCH=/pscratch/sd/a/asvillar/mp288/jobs/batch_16
# --n-cases 0 -> use the ENTIRE dataset (build_dataset treats 0 as "no limit").
COMMON=(--batch-dir "$BATCH" --filename csdata_deltap_b_ver.h5 --grid 128 \
        --m-lo -80 --m-hi 20 --models fno2d --epochs 150 --batch-size 16 \
        --n-cases 0 --target-norm max)

echo ">>> $(date)  [B] max-normalized LOG10 (full dataset)"
"${SRUN[@]}" python scripts/m3dc1/internal/train_spectrum_image.py "${COMMON[@]}" \
  --target-space log10 --out runs/spectrum_image_full_maxnorm_log10
echo ">>> $(date)  [B] done"

echo ">>> $(date)  [A] max-normalized RAW (full dataset)"
"${SRUN[@]}" python scripts/m3dc1/internal/train_spectrum_image.py "${COMMON[@]}" \
  --target-space raw --out runs/spectrum_image_full_maxnorm_raw
echo ">>> $(date)  [A] done"
echo ">>> $(date)  ALL DONE"
