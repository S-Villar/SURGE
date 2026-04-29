#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--inside-allocation" ]]; then
  shift
  cd "${HERE}"
  exec bash ./demo_04_gpu_aware_search.sh "$@"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec "$0" --inside-allocation "$@"
fi

ACCOUNT="${ACCOUNT:-amsc007}"
QOS="${QOS:-interactive}"
TIME="${TIME:-04:00:00}"
NODES="${NODES:-1}"
CONSTRAINT="${CONSTRAINT:-gpu&hbm80g}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
GPUS="${GPUS:-${GPUS_PER_NODE:-1}}"
JOB_NAME="${JOB_NAME:-surge_rose_gpu_demos}"

exec salloc \
  --nodes "${NODES}" \
  --qos "${QOS}" \
  --time "${TIME}" \
  --constraint "${CONSTRAINT}" \
  --account "${ACCOUNT}" \
  --job-name "${JOB_NAME}" \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --gpus "${GPUS}" \
  "$0" --inside-allocation "$@"
