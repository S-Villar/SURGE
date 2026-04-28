#!/usr/bin/env bash
# Create or refresh a dedicated venv for the SURGE+ROSE+Rhapsody demos.
set -euo pipefail

EX_DIR="$(cd "$(dirname "$0")" && pwd)"
SURGE_ROOT="${SURGE_ROOT:-$(cd "$EX_DIR/../.." && pwd)}"
ROSE_ROOT="${ROSE_ROOT:-/global/homes/a/asvillar/src/ROSE}"
SURGE_PY="${SURGE_PY:-/global/common/software/m3716/asvillar/envs/surge/bin/python}"
VENV_DIR="${SURGE_ROSE_DEMO_VENV:-/global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/global/cfs/projectdirs/amsc007/asvillar/pip_cache}"

if [[ ! -x "$SURGE_PY" ]]; then
  echo "Missing base Python: $SURGE_PY" >&2
  exit 1
fi
if [[ ! -d "$SURGE_ROOT" ]]; then
  echo "Missing SURGE_ROOT: $SURGE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ROSE_ROOT" ]]; then
  echo "Missing ROSE_ROOT: $ROSE_ROOT" >&2
  echo "Set ROSE_ROOT to your ROSE checkout before running this script." >&2
  exit 1
fi

mkdir -p "$PIP_CACHE_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Creating venv at $VENV_DIR" >&2
  "$SURGE_PY" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel
python -m pip install --no-cache-dir -e "$SURGE_ROOT"
python -m pip install --no-cache-dir -r "$SURGE_ROOT/requirements-rose-demo.txt"
python -m pip install --no-cache-dir -e "$ROSE_ROOT"

export SURGE_ROOT
export PYTHONPATH="$SURGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$EX_DIR/verify_demo_env.py"

echo "" >&2
echo "Environment ready." >&2
echo "Activate with:" >&2
echo "  source \"$VENV_DIR/bin/activate\"" >&2
echo "  export SURGE_ROOT=\"$SURGE_ROOT\"" >&2
echo "  export PYTHONPATH=\"\$SURGE_ROOT\${PYTHONPATH:+:\$PYTHONPATH}\"" >&2
