#!/bin/bash
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu
#SBATCH --account=amsc007
#SBATCH --job-name=surge_rose
#SBATCH --output=surge_rose_%j.out
#SBATCH --error=surge_rose_%j.err

# Batch one SURGE+ROSE example.
#
# Usage:
#   sbatch --export=ALL,EXAMPLE=example_01_rose_inprocess_verbose.py,ARGS="--max-iter 2" run_rose_example_batch.sh
#   sbatch --time=04:00:00 --export=ALL,EXAMPLE=example_03_rose_mlp_random_r2_stop.py,ARGS="--max-iter 12 --r2-threshold 0.9" run_rose_example_batch.sh
set -euo pipefail

SURGE_ROOT="${SURGE_ROOT:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/SURGE}"
SURGE_ROSE_VENV="${SURGE_ROSE_VENV:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv}"
M3DC1_SOURCE="${M3DC1_SOURCE:-/global/homes/a/asvillar/src/SURGEROSE/data/datasets/M3DC1/sparc-m3dc1-D1.pkl}"

EXAMPLE="${EXAMPLE:-example_01_rose_inprocess_verbose.py}"
ARGS="${ARGS:---max-iter 2}"

source "${SURGE_ROSE_VENV}/bin/activate"
export SURGE_ROOT
export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SURGE_ROOT}/data/datasets/M3DC1"
if [[ -e "${M3DC1_SOURCE}" ]]; then
  ln -sf "${M3DC1_SOURCE}" "${SURGE_ROOT}/data/datasets/M3DC1/sparc-m3dc1-D1.pkl"
fi

cd "${SURGE_ROOT}/examples/rose_orchestration"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Running: python ${EXAMPLE} ${ARGS}"
python "${EXAMPLE}" ${ARGS}
