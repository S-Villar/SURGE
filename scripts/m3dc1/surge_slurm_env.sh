#!/usr/bin/env bash
# Shared Python/Conda setup for SURGE Slurm jobs on HPC (e.g. Perlmutter).
# Usage (repo root or any cwd):
#   module load conda    # Perlmutter login/batch: load NERSC conda first
#   source "$SURGE_ROOT/scripts/m3dc1/surge_slurm_env.sh"
#   surge_slurm_setup_python
#
# Slurm batch scripts call surge_slurm_setup_python after sourcing this file.
#
# Environment:
#   SURGE_CONDA_ENV — optional override: env *name* or absolute *prefix* path for `conda activate`.
#   SURGE_SKIP_MODULE_CONDA — set to 1 to skip `module load conda`.

# NERSC CFS env used for SURGE on Perlmutter (compute nodes see this path); `conda activate <prefix>` is valid.
SURGE_CONDA_PREFIX_DEFAULT="/global/cfs/projectdirs/m3716/software/asvillar/envs/surge"

surge_slurm_setup_python() {
  # Conda/env activate.d hooks (e.g. Intel MKL) sometimes expand unset vars; Slurm scripts use `set -u`.
  export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"
  export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

  # Perlmutter: always load the conda module when `module` exists unless skipped.
  # Batch jobs often have no usable conda until this runs (even if a stale `conda` is on PATH).
  if [[ "${SURGE_SKIP_MODULE_CONDA:-0}" != "1" ]] && command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
  fi

  if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found on this node."
    echo "On NERSC Perlmutter, ensure the job script runs: module load conda"
    echo "Or initialize conda in ~/.bashrc and set SURGE_SKIP_MODULE_CONDA=1 if conda is already on PATH."
    return 1
  fi

  eval "$(conda shell.bash hook)"

  if [[ -n "${SURGE_CONDA_ENV:-}" ]]; then
    conda activate "${SURGE_CONDA_ENV}" || {
      echo "ERROR: conda activate \"${SURGE_CONDA_ENV}\" failed."
      return 1
    }
  elif [[ -d "${SURGE_CONDA_PREFIX_DEFAULT}" ]] && conda activate "${SURGE_CONDA_PREFIX_DEFAULT}" 2>/dev/null; then
    :
  elif conda activate surge 2>/dev/null; then
    :
  elif conda activate surge-devel 2>/dev/null; then
    :
  else
    echo "ERROR: No usable conda env."
    echo "Tried: \${SURGE_CONDA_ENV:-unset}, ${SURGE_CONDA_PREFIX_DEFAULT} (if dir exists), surge, surge-devel"
    echo "Set SURGE_CONDA_ENV to your prefix, e.g.: export SURGE_CONDA_ENV=/path/to/envs/surge"
    return 1
  fi

  echo "Using Python: $(command -v python)"
  python --version

  # Parquet workflows need pyarrow; do not rely on user-site packages from login installs.
  if ! python -c "import pyarrow" 2>/dev/null; then
    echo "Installing pyarrow into the active env (needed for pd.read_parquet)..."
    python -m pip install -q "pyarrow>=10.0.0" || {
      echo "ERROR: pip install pyarrow failed. Install manually in this conda env."
      return 1
    }
  fi
  python -c "import pyarrow as pa; print('pyarrow OK', pa.__version__)"
  if ! python -c "import torch" 2>/dev/null; then
    echo "NOTE: torch not importable; install torch in this env for pytorch.mlp workflows."
  else
    python -c "import torch; print('torch OK', torch.__version__)"
  fi
}
