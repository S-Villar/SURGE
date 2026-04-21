---
skill_id: surge.slurm.bootstrap
title: Slurm SURGE_ROOT and submit directory
version: "1.0.0"
domain: [hpc, slurm, surge]
academy_blueprint_hint: local-hpc-batch-bootstrap (see docs/agents/blueprint/README.md)
---

## Objective

Ensure Slurm **batch jobs** execute SURGE with the **correct repository root** so Python, configs, and relative dataset paths resolve on compute nodes.

## When to apply (triggers)

- User submits or debugs **Slurm** training for SURGE (e.g. `train_delta_p_per_mode_cfs*.slurm`).
- Jobs **fail immediately** with missing `scripts/m3dc1/...`, missing Parquet under `data/`, or **`SURGE_ROOT`** pointing at spool (`/var/spool/...`).

## Preconditions

- Access to **Slurm stdout/stderr** logs (`#SBATCH -o ...`) in the **submit directory**.
- Batch scripts use **bash** and can **`cd`** to repo root.

## Procedure

1. Open the failing job’s **`.log`** file; confirm printed **`SURGE_ROOT`** equals the **git checkout** path, not spool.
2. In the `.slurm` script, verify **`SLURM_SUBMIT_DIR`**: if set, **`SURGE_ROOT="${SURGE_ROOT:-$SLURM_SUBMIT_DIR}"`** then **`cd "$SURGE_ROOT"`** before any `python`.
3. Verify **`sbatch`** is run from **repo root** (or use a launcher that **`cd`s** to repo root before `sbatch`), so **`SLURM_SUBMIT_DIR`** is the SURGE tree.
4. Re-submit and re-check **`SURGE_ROOT`** in the new log.

## Verification

- Log shows **`SURGE_ROOT: /.../SURGE`** (or your real path).
- **`python -m surge.cli`** or build scripts find **`configs/`** and **`data/`** without `FileNotFoundError`.

## Failure modes & diagnostics

| Symptom | Likely cause | Next step |
|---------|----------------|-----------|
| `SURGE_ROOT` under `/var/spool` | Only **`BASH_SOURCE`** resolution from Slurm’s script copy | Add **`SLURM_SUBMIT_DIR`** + **`cd`** |
| Parquet “missing” though file exists on login | Wrong cwd / wrong root | Fix submit dir + **`cd`** |
| Works interactively, fails in batch | Login vs batch cwd | Use launcher; document **`cd` before sbatch** |

## Artifacts & code references

- CFS Slurm scripts: `scripts/m3dc1/train_delta_p_per_mode_cfs*.slurm`
- Launcher: `scripts/m3dc1/launch_cfs_delta_p_trial_suite.sh`
- Human index: `m3dc1/results.md` (trial suite section)

## Future: Academy mapping

- **Tool:** `assert_repo_layout(root: Path) -> None`
- **Action:** `prepare_batch_environment()` run before launching training **tools** on a Slurm **executor**.
