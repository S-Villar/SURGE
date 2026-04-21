---
skill_id: surge.hpc.conda_parquet
title: Conda activation and Parquet (pyarrow) on compute nodes
version: "1.0.0"
domain: [hpc, conda, python, parquet, surge]
academy_blueprint_hint: local-hpc-python-environment-parity (see docs/agents/blueprint/README.md)
---

## Objective

Guarantee the **same** Python stack on **compute nodes** as intended (conda env with **`pyarrow`**) so **`pandas.read_parquet`** and SURGE Parquet workflows succeed.

## When to apply (triggers)

- **`ImportError: Unable to find a usable engine`** for Parquet / **`pyarrow`** in batch logs.
- Stack traces show **`~/.local/lib/python3.../pandas`** while **`conda activate`** was “optional” or silenced with **`|| true`**.

## Preconditions

- **`conda`** available after **`module load conda`** on the cluster (Perlmutter-style), or conda on `PATH` with **`SURGE_SKIP_MODULE_CONDA=1`**.
- Project may define a **prefix env**, e.g. **`/global/cfs/projectdirs/m3716/software/asvillar/envs/surge`**.

## Procedure

1. Read the traceback: if **`~/.local`** appears, treat **conda activation as failed**.
2. Ensure batch script **sources** [`scripts/m3dc1/surge_slurm_env.sh`](../../../scripts/m3dc1/surge_slurm_env.sh) and runs **`surge_slurm_setup_python`** (or equivalent strict logic).
3. Confirm order: **`module load conda`** (if needed) → **`eval "$(conda shell.bash hook)"`** → **`conda activate`** (**`$SURGE_CONDA_ENV`**, else default prefix if directory exists, else **`surge`** / **`surge-devel`**).
4. Require **`python -c "import pyarrow"`**; if missing, **`pip install 'pyarrow>=10.0.0'`** into the **active** env (when policy allows network on compute).
5. For **GPR** jobs only: ensure **GPflow** is installed; **`GPflow not available`** is unrelated to Parquet but blocks GPR training.

## Verification

- Log prints **`Using Python: ...`** under the conda env.
- **`pyarrow OK <version>`** from the setup helper (or manual import).
- SURGE proceeds past **`[1/x] Loading dataset...`** for Parquet configs.

## Failure modes & diagnostics

| Symptom | Likely cause | Next step |
|---------|----------------|-----------|
| No engine for Parquet | **`pyarrow`** not in active interpreter | Fix conda activate; install pyarrow |
| Works on login, fails on node | **`module load conda`** missing in batch | Add module step or init script |
| Wrong env | Name vs prefix | Use **`conda activate /path/to/env`**; set **`SURGE_CONDA_ENV`** |

## Artifacts & code references

- `scripts/m3dc1/surge_slurm_env.sh`
- `requirements.txt` (`pyarrow>=10.0.0`)
- `m3dc1/results.md` — compute environment note

## Future: Academy mapping

- **Tool:** `ensure_python_stack(spec: EnvSpec) -> EnvReport`
- **Action:** run once per remote batch **before** dataset load **tools**.
