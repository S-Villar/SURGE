#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--inside-allocation" ]]; then
  shift
  cd "${HERE}"
  exec bash demos/demo_03_resource_aware_search.sh "$@"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec "$0" --inside-allocation "$@"
fi

ACCOUNT="${ACCOUNT:-amsc007}"
QOS="${QOS:-regular}"
TIME="${TIME:-02:00:00}"
NODES="${NODES:-1}"
CONSTRAINT="${CONSTRAINT:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
JOB_NAME="${JOB_NAME:-surge_demo03}"
LOG_DIR="${LOG_DIR:-${HERE}/output/demo_03/slurm}"

mkdir -p "${LOG_DIR}"

exec sbatch \
  --nodes "${NODES}" \
  --qos "${QOS}" \
  --time "${TIME}" \
  --constraint "${CONSTRAINT}" \
  --account "${ACCOUNT}" \
  --job-name "${JOB_NAME}" \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --output "${LOG_DIR}/%x_%j.out" \
  --error "${LOG_DIR}/%x_%j.err" \
  --wrap "cd '${HERE}' && bash demos/run_demo_03_slurm.sh --inside-allocation $*"
