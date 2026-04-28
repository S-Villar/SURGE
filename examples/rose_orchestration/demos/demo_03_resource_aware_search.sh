#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
SURGE_ROOT="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
VENV_PATH="${SURGE_ROSE_VENV:-/global/cfs/projectdirs/amsc007/asvillar/radical-integration-smoke/venv}"
M3DC1_SOURCE="${M3DC1_SOURCE:-/global/homes/a/asvillar/src/SURGEROSE/data/datasets/M3DC1/sparc-m3dc1-D1.pkl}"

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

cd "${EXAMPLE_DIR}"
exec python demos/demo_03_resource_aware_search.py "$@"
