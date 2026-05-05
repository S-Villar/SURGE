#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURGE_ROOT="${SURGE_ROOT:-$(cd "${HERE}/../.." && pwd)}"
VENV_PATH="${SURGE_ROSE_VENV:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv}"
M3DC1_SOURCE="${M3DC1_SOURCE:-/global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl}"

run_inside() {
  if [[ -d "${VENV_PATH}" ]] && [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
  fi

  export SURGE_ROOT
  export M3DC1_SOURCE
  export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  cd "${HERE}"
  python verify_demo_env.py

  python example_01_rose_inprocess_verbose.py --max-iter 2
  python example_02_rose_subprocess_shell.py --dataset synthetic --workflow-sequence rf,mlp,gpr,gpflow_gpr --quiet
  python example_03_rose_mlp_random_r2_stop.py --max-iter 12 --r2-threshold 0.9 --r2-operator ">="
  python example_04_parallel_model_race.py --dataset m3dc1 --max-iter 1 --candidates rf,mlp
  python example_05_uq_guided_mlp_ensemble.py --dataset m3dc1 --max-iter 1 --ensemble-n 3
}

if [[ "${1:-}" == "--inside-allocation" ]]; then
  shift
  run_inside "$@"
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec "$0" --inside-allocation "$@"
fi

ACCOUNT="${ACCOUNT:-amsc007}"
QOS="${QOS:-interactive}"
TIME="${TIME:-04:00:00}"
NODES="${NODES:-1}"
CONSTRAINT="${CONSTRAINT:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
JOB_NAME="${JOB_NAME:-surge_rose_examples}"

exec salloc \
  --nodes "${NODES}" \
  --qos "${QOS}" \
  --time "${TIME}" \
  --constraint "${CONSTRAINT}" \
  --account "${ACCOUNT}" \
  --job-name "${JOB_NAME}" \
  --cpus-per-task "${CPUS_PER_TASK}" \
  "$0" --inside-allocation "$@"
