#!/bin/bash
#SBATCH -A mp288
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 12:00:00
#SBATCH -J fno48_geom
#SBATCH -o runs/spectrum_fno48_floor6_smooth1_qc_geom/train_%j.out
#SBATCH -e runs/spectrum_fno48_floor6_smooth1_qc_geom/train_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/global/homes/a/asvillar/src/SURGE}"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=runs/spectrum_fno48_floor6_smooth1_qc_geom
mkdir -p "$OUT"

srun -n1 --gpus-per-node=1 python -u scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 --grid 128 --m-lo -80 --m-hi 20 \
  --models fno2d --target-norm max --target-space log10 \
  --target-floor 6 --target-smooth 1 \
  --exclude-list runs/quarantine/bad_cases.json \
  --geom-channels \
  --fno-modes 48 --fno-hidden 32 \
  --loc-weight 2 --marg-weight 1 --select-by composite \
  --epochs 400 --patience 120 --batch-size 16 \
  --out "$OUT"
