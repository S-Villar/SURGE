#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == "--inside-allocation" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DEMO_SCRIPT [demo args...]" >&2
    exit 2
  fi
  DEMO_SCRIPT="$1"
  shift
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${HERE}"
  echo "Running on $(hostname): bash ${DEMO_SCRIPT} $*" >&2
  exec bash "${DEMO_SCRIPT}" "$@"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DEMO_SCRIPT [demo args...]" >&2
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
JOB_NAME="${JOB_NAME:-surge_rose_demo}"

exec salloc \
  --nodes "${NODES}" \
  --qos "${QOS}" \
  --time "${TIME}" \
  --constraint "${CONSTRAINT}" \
  --account "${ACCOUNT}" \
  --job-name "${JOB_NAME}" \
  --cpus-per-task "${CPUS_PER_TASK}" \
  "$0" --inside-allocation "$@"
