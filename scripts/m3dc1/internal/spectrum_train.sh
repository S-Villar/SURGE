#!/bin/bash
# One-command launcher for the full-dataset FNO2D |delta p| spectrum surrogate on
# an interactive Perlmutter GPU node. Handles the salloc + srun dance for you.
#
# Usage:
#   bash scripts/m3dc1/internal/spectrum_train.sh [SPACE] [MODE] [EPOCHS]
#     SPACE  : log10 (default) | raw          -> target space (after max-norm)
#     MODE   : fresh (default) | resume       -> resume continues from last ckpt
#     EPOCHS : total epochs (default 300)
#
# Examples:
#   bash scripts/m3dc1/internal/spectrum_train.sh                 # fresh log10
#   bash scripts/m3dc1/internal/spectrum_train.sh raw            # fresh raw
#   bash scripts/m3dc1/internal/spectrum_train.sh log10 resume   # continue log10
#
# Output goes to runs/spectrum_image_full_maxnorm_<SPACE>/ ; monitor with:
#   python -m surge.check_training --run runs/spectrum_image_full_maxnorm_<SPACE>
set -uo pipefail

SPACE="${1:-log10}"
MODE="${2:-fresh}"
EPOCHS="${3:-300}"
# Peak-weighted loss (0 = plain MSE). Set e.g. PEAK_WEIGHT=8 to force the model to
# reproduce the high-amplitude ridge/peak instead of the noise floor. Runs whose
# PEAK_WEIGHT>0 go to a separate _peak<W> dir so they don't clobber the baseline.
PEAK_WEIGHT="${PEAK_WEIGHT:-0}"
PEAK_POW="${PEAK_POW:-2}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # SURGE repo root
OUT="runs/spectrum_image_full_maxnorm_${SPACE}"
if [[ "$PEAK_WEIGHT" != "0" ]]; then OUT="${OUT}_peak${PEAK_WEIGHT}"; fi
mkdir -p "$OUT"

# Resume: prefer the rolling "last" ckpt, fall back to the best-val ckpt.
RESUME=""
if [[ "$MODE" == "resume" ]]; then
  if   [[ -f "$OUT/ckpt_fno2d_last.pt" ]]; then RESUME="$OUT/ckpt_fno2d_last.pt"
  elif [[ -f "$OUT/ckpt_fno2d.pt"      ]]; then RESUME="$OUT/ckpt_fno2d.pt"
  else echo "!! no checkpoint in $OUT to resume from; running fresh"; fi
fi

export SURGE_ROOT="$PWD"
export SURGE_CONDA_ENV="${SURGE_CONDA_ENV:-/global/cfs/projectdirs/m3716/software/asvillar/envs/surge}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TRAIN="python scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 --grid 128 --m-lo -80 --m-hi 20 \
  --models fno2d --target-norm max --target-space ${SPACE} \
  --epochs ${EPOCHS} --patience 40 --ckpt-every 25 --batch-size 16 \
  --peak-weight ${PEAK_WEIGHT} --peak-pow ${PEAK_POW} --out ${OUT}"
if [[ -n "$RESUME" ]]; then TRAIN="$TRAIN --resume $RESUME"; fi

echo ">>> requesting interactive GPU node (salloc); training runs via srun ON the node"
echo ">>> out=$OUT space=$SPACE mode=$MODE epochs=$EPOCHS resume=${RESUME:-none}"

# salloc runs THIS block on the login node; srun puts the training on the GPU node.
salloc -A mp288 -C gpu -q interactive -N 1 --gpus-per-node=1 -t 04:00:00 bash -c "
  cd '$PWD'
  source scripts/m3dc1/surge_slurm_env.sh
  surge_slurm_setup_python
  echo '>>> compute node:' \$(hostname)
  srun --ntasks=1 --gpus-per-task=1 $TRAIN
"
echo ">>> done. Inspect: python -m surge.check_training --run $OUT"
