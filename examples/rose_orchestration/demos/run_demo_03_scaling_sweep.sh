#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CPU_SWEEP="${CPU_SWEEP:-8 16 32}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-4}"
TIME="${TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-amsc007}"
QOS="${QOS:-regular}"
CONSTRAINT="${CONSTRAINT:-cpu}"
NODES="${NODES:-1}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-surge_demo03_scale}"
BASE_OUT="${BASE_OUT:-${HERE}/output/demo_03_scaling}"

mkdir -p "${BASE_OUT}"

for alloc_cpus in ${CPU_SWEEP}; do
  label="cpus_${alloc_cpus}"
  run_dir="${BASE_OUT}/${label}"
  log_dir="${run_dir}/slurm"
  mkdir -p "${log_dir}"

  sbatch \
    --nodes "${NODES}" \
    --qos "${QOS}" \
    --time "${TIME}" \
    --constraint "${CONSTRAINT}" \
    --account "${ACCOUNT}" \
    --job-name "${JOB_NAME_PREFIX}_${alloc_cpus}" \
    --cpus-per-task "${alloc_cpus}" \
    --output "${log_dir}/%x_%j.out" \
    --error "${log_dir}/%x_%j.err" \
    --wrap "cd '${HERE}' && source /global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv/bin/activate && export SURGE_ROOT=/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/SURGE && export PYTHONPATH=\"\$SURGE_ROOT\${PYTHONPATH:+:\$PYTHONPATH}\" && mkdir -p '${run_dir}' && python demos/demo_03_resource_aware_search.py --cpus-per-trial '${CPUS_PER_TRIAL}' --output-prefix '${run_dir}'"
done

cat <<EOF
Submitted Demo 3 scaling sweep.

CPU allocations: ${CPU_SWEEP}
CPUs per trial: ${CPUS_PER_TRIAL}

When jobs finish, aggregate with:
  python demos/demo_03_scaling_report.py --input-root "${BASE_OUT}"
EOF
