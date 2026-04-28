#!/bin/bash
# Allocate an interactive Slurm node and run one SURGE+ROSE example.
#
# Usage:
#   ./run_rose_example_interactive.sh example_01_rose_inprocess_verbose.py --max-iter 2
#   TIME=04:00:00 CPUS_PER_TASK=16 ./run_rose_example_interactive.sh example_03_rose_mlp_random_r2_stop.py --max-iter 12
set -euo pipefail

if [[ "${1:-}" == "--inside-allocation" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 EXAMPLE.py [example args...]" >&2
    exit 2
  fi

  SURGE_ROOT="${SURGE_ROOT:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/SURGE}"
  SURGE_ROSE_VENV="${SURGE_ROSE_VENV:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv}"
  M3DC1_SOURCE="${M3DC1_SOURCE:-/global/homes/a/asvillar/src/SURGEROSE/data/datasets/M3DC1/sparc-m3dc1-D1.pkl}"

  source "${SURGE_ROSE_VENV}/bin/activate"
  export SURGE_ROOT
  export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  mkdir -p "${SURGE_ROOT}/data/datasets/M3DC1"
  if [[ -e "${M3DC1_SOURCE}" ]]; then
    ln -sf "${M3DC1_SOURCE}" "${SURGE_ROOT}/data/datasets/M3DC1/sparc-m3dc1-D1.pkl"
  fi

  cd "${SURGE_ROOT}/examples/rose_orchestration"
  echo "Running on $(hostname): python $*"
  exec python "$@"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXAMPLE.py [example args...]" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec "$0" --inside-allocation "$@"
fi

ACCOUNT="${ACCOUNT:-amsc007}"
QOS="${QOS:-interactive}"
TIME="${TIME:-02:00:00}"
NODES="${NODES:-1}"
CONSTRAINT="${CONSTRAINT:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
JOB_NAME="${JOB_NAME:-surge_rose}"

exec salloc \
  --nodes "${NODES}" \
  --qos "${QOS}" \
  --time "${TIME}" \
  --constraint "${CONSTRAINT}" \
  --account "${ACCOUNT}" \
  --job-name "${JOB_NAME}" \
  --cpus-per-task "${CPUS_PER_TASK}" \
  "$0" --inside-allocation "$@"
