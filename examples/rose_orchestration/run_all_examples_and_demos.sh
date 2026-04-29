#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURGE_ROOT="${SURGE_ROOT:-$(cd "${HERE}/../.." && pwd)}"
VENV_PATH="${SURGE_ROSE_VENV:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv}"
M3DC1_SOURCE="${M3DC1_SOURCE:-/global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl}"
MODE="${MODE:-print}"
INCLUDE_GPU="${INCLUDE_GPU:-0}"
INCLUDE_SCALING_SWEEP="${INCLUDE_SCALING_SWEEP:-0}"

if [[ -d "${VENV_PATH}" ]] && [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export SURGE_ROOT
export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SURGE_ROOT}/data/datasets/M3DC1"
if [[ -e "${M3DC1_SOURCE}" ]]; then
  ln -sf "${M3DC1_SOURCE}" "${SURGE_ROOT}/data/datasets/M3DC1/sparc-m3dc1-D1.pkl"
fi

cd "${HERE}"

run_cmd() {
  echo
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    "$@"
  fi
}

echo "SURGE_ROOT=${SURGE_ROOT}"
echo "MODE=${MODE}  (use MODE=run to execute)"
echo "INCLUDE_GPU=${INCLUDE_GPU}"
echo "INCLUDE_SCALING_SWEEP=${INCLUDE_SCALING_SWEEP}"

run_cmd python verify_demo_env.py

echo
echo "## Examples"
run_cmd python example_01_rose_inprocess_verbose.py --max-iter 2
run_cmd python example_02_rose_subprocess_shell.py --dataset synthetic --workflow-sequence rf,mlp,gpr,gpflow_gpr --quiet
run_cmd python example_03_rose_mlp_random_r2_stop.py --max-iter 12 --r2-threshold 0.9 --r2-operator ">="
run_cmd python example_04_parallel_model_race.py --dataset m3dc1 --max-iter 1 --candidates rf,mlp
run_cmd python example_05_uq_guided_mlp_ensemble.py --dataset m3dc1 --max-iter 1 --ensemble-n 3

echo
echo "## Demos"
run_cmd bash demos/demo_01_better_surrogate_selection.sh
run_cmd bash demos/demo_02_smarter_campaign_control.sh
run_cmd bash demos/demo_03_resource_aware_search.sh --cpus-per-trial 4 --max-trials 4

if [[ "${INCLUDE_GPU}" == "1" ]]; then
  run_cmd bash demos/demo_04_gpu_aware_search.sh --allow-cpu-fallback --max-trials 1
fi

echo
echo "## Interactive launchers"
run_cmd bash -lc 'TIME=02:00:00 CPUS_PER_TASK=8 bash demos/launch_demo_01_interactive.sh'
run_cmd bash -lc 'TIME=02:00:00 CPUS_PER_TASK=8 bash demos/launch_demo_02_interactive.sh'
run_cmd bash -lc 'TIME=02:00:00 CPUS_PER_TASK=16 bash demos/run_demo_interactive.sh demos/demo_03_resource_aware_search.sh --cpus-per-trial 4'

if [[ "${INCLUDE_GPU}" == "1" ]]; then
  run_cmd bash -lc 'TIME=02:00:00 CPUS_PER_TASK=8 GPUS_PER_NODE=1 bash demos/launch_demo_04_gpu_interactive.sh'
fi

echo
echo "## Batch / Slurm"
run_cmd bash -lc 'TIME=02:00:00 CPUS_PER_TASK=16 bash demos/run_demo_03_slurm.sh --cpus-per-trial 4'

if [[ "${INCLUDE_SCALING_SWEEP}" == "1" ]]; then
  run_cmd bash -lc 'CPU_SWEEP="8 16 32" CPUS_PER_TRIAL=4 TIME=02:00:00 bash demos/run_demo_03_scaling_sweep.sh'
  run_cmd python demos/demo_03_scaling_report.py
  run_cmd python demos/plot_demo_03_scaling.py
fi

echo
echo "## Visualization"
run_cmd python viz_examples_and_demos.py

echo
echo "Done. This script printed the full command set."
echo "Use MODE=run to execute it."
