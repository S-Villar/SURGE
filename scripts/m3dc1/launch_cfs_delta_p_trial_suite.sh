#!/usr/bin/env bash
# Launch all CFS δp per-mode trial jobs (CPU + GPU) via Slurm.
#
# Usage:
#   cd /path/to/SURGE
#   ./scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh
#
# Requires: Parquet dataset present (see train_delta_p_per_mode_cfs.slurm).
# Override account:  SURGE_SLURM_ACCOUNT=myproj ./scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh
#
# Job IDs are appended to logs/cfs_trial_suite_jobids.txt

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

echo "Using SURGE root: $ROOT (run from repo root so SLURM_SUBMIT_DIR matches)" >&2

ACCT="${SURGE_SLURM_ACCOUNT:-mp288}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/cfs_trial_suite_jobids_${STAMP}.txt"

if ! command -v sbatch &>/dev/null; then
  echo "sbatch not found — run these scripts on Perlmutter (or your Slurm cluster)." >&2
  exit 1
fi

submit_one() {
  local script="$1"
  local name="$2"
  echo "=== Submit $name ($script) ===" | tee -a "$LOG"
  # Replace account line in-flight if custom account (optional)
  if [[ "$ACCT" != "mp288" ]] && [[ -f "$script" ]]; then
    sed "s/^#SBATCH -A .*/#SBATCH -A ${ACCT}/" "$script" > "${ROOT}/logs/_tmp_${name}.slurm"
    jid="$(sbatch "${ROOT}/logs/_tmp_${name}.slurm" | awk '{print $4}')"
    rm -f "${ROOT}/logs/_tmp_${name}.slurm"
  else
    jid="$(sbatch "$script" | awk '{print $4}')"
  fi
  echo "${name}	${jid}	${script}" | tee -a "$LOG"
}

echo "CFS trial suite launch $STAMP (account=${ACCT})" | tee "$LOG"
echo "" | tee -a "$LOG"

submit_one "$ROOT/scripts/m3dc1/train_delta_p_per_mode_cfs.slurm" "T1_cpu_rf_mlp_baseline"
submit_one "$ROOT/scripts/m3dc1/train_delta_p_per_mode_cfs_hpo.slurm" "T2_cpu_rf_mlp_hpo"
submit_one "$ROOT/scripts/m3dc1/train_delta_p_per_mode_cfs_mlp_hpo.slurm" "T3_cpu_mlp_flexible_botorch"
submit_one "$ROOT/scripts/m3dc1/train_delta_p_per_mode_cfs_mlp_hpo_gpu.slurm" "T4_gpu_mlp_flexible_botorch"
submit_one "$ROOT/scripts/m3dc1/train_delta_p_per_mode_cfs_gpr_hpo.slurm" "T5_cpu_gpr_linear_matern52_botorch"

echo "" | tee -a "$LOG"
echo "Also symlink latest: logs/cfs_trial_suite_jobids_latest.txt" | tee -a "$LOG"
ln -sf "$(basename "$LOG")" logs/cfs_trial_suite_jobids_latest.txt

echo ""
echo "Track: squeue -u \$USER"
echo "Harvest metrics into docs: python scripts/m3dc1/harvest_cfs_trial_metrics.py"
