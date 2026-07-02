#!/bin/bash
# Launch the DeepONet spectrum-image surrogate as a GPU batch job.
# Matches the unet_loc / fno16_loc config (same target/window/loss) so the
# four architectures (fno48, fno16, unet, deeponet) can be compared head to head.
#
#   bash scripts/m3dc1/internal/launch_deeponet_loc.sh
#
set -euo pipefail
SURGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
cd "$SURGE_ROOT"

OUT="${OUT:-runs/spectrum_deeponet_loc}"
EPOCHS="${EPOCHS:-250}"
PATIENCE="${PATIENCE:-100}"
BATCH="${BATCH:-16}"
GRID="${GRID:-128}"

sbatch -N 1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1 \
  --time=06:00:00 --qos=regular -C gpu -A mp288 \
  -J deeponet_loc -o "surge_deeponet_loc.%j.log" -e "surge_deeponet_loc.%j.log" \
  --wrap "cd $SURGE_ROOT && \
export SURGE_CONDA_ENV=/global/cfs/projectdirs/m3716/software/asvillar/envs/surge && \
source scripts/m3dc1/surge_slurm_env.sh && surge_slurm_setup_python && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 \
  --grid $GRID --m-lo -80 --m-hi 20 \
  --models deeponet --target-norm max --target-space log10 \
  --loc-weight 2 --marg-weight 1 --loc-beta 8 \
  --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 \
  --epochs $EPOCHS --patience $PATIENCE --ckpt-every 25 \
  --batch-size $BATCH --out $OUT"
