#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--inside-allocation" ]]; then
  shift
  cd "${HERE}"
  bash ./demo_01_better_surrogate_selection.sh
  bash ./demo_02_smarter_campaign_control.sh
  bash ./demo_03_resource_aware_search.sh --cpus-per-trial "${CPUS_PER_TRIAL:-4}" --max-trials "${MAX_TRIALS:-4}"
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
JOB_NAME="${JOB_NAME:-surge_rose_cpu_demos}"

exec salloc \
  --nodes "${NODES}" \
  --qos "${QOS}" \
  --time "${TIME}" \
  --constraint "${CONSTRAINT}" \
  --account "${ACCOUNT}" \
  --job-name "${JOB_NAME}" \
  --cpus-per-task "${CPUS_PER_TASK}" \
  "$0" --inside-allocation "$@"
