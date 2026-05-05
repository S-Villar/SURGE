#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
SURGE_ROOT="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
VENV_PATH="${SURGE_ROSE_VENV:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv}"
M3DC1_SOURCE="${M3DC1_SOURCE:-/global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl}"

if [[ -d "${VENV_PATH}" ]] && [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export SURGE_ROOT
export M3DC1_SOURCE
export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${EXAMPLE_DIR}"
exec python demos/demo_04_gpu_aware_search.py "$@"
