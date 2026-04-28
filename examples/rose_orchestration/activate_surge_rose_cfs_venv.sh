#!/usr/bin/env bash
# shellcheck source=/dev/null
# Activate the project-venv used for ROSE+SURGE demos (CFS quota, not $HOME/.local).
# Usage: source ./activate_surge_rose_cfs_venv.sh
# Optional: export SURGE_ROOT=/path/to/SURGE before sourcing if not under ~/src.

set -e
SURGE_PY="${SURGE_PY:-/global/common/software/m3716/asvillar/envs/surge/bin/python}"
VENV_DIR="${SURGE_ROSE_DEMO_VENV:-/global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos}"
SURGE_ROOT="${SURGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "$SURGE_ROOT" && -d "$HOME/src/SURGE" ]]; then
  SURGE_ROOT="$HOME/src/SURGE"
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/global/cfs/projectdirs/amsc007/asvillar/pip_cache}"
export SURGE_ROOT
export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Creating venv at $VENV_DIR (one-time)..." >&2
  "$SURGE_PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "Active: $VENV_DIR" >&2
echo "PYTHONPATH=$SURGE_ROOT" >&2
echo "cd to examples/rose_orchestration and run: python example_01_rose_inprocess_verbose.py ..." >&2
