# ROSE + SURGE examples — M3DC1 dataset only

The default smoke commands use **`--dataset synthetic`** (no extra files). This page is **only** for runs that train on the **SPARC M3DC1** table (`--dataset m3dc1`).

**Recorded run output** (commands, timings, summary tables from a real session): [README.md — Reference: M3DC1 example runs](README.md#reference-m3dc1-example-runs).

## What you get with `m3dc1`

- **Source:** a pickle of M3DC1 rows; metadata YAML defines input/output columns for SURGE.
- **Per simulation step** the demo takes an increasing row count from a **shuffled pool** and writes:
  - `examples/rose_orchestration/workspace/sparc_m3dc1_active_learning.parquet`
- **Workflows** alternate RF / MLP via `demo_common.workflow_for_iteration` using `workflow_m3dc1_rf.yaml` and `workflow_m3dc1_mlp.yaml`.

## Step 0 — Environment (same as synthetic)

Use Python 3.10+ with ROSE + deps and **`PYTHONPATH`** pointing at the **SURGE repo root** (see [quickstart.md](quickstart.md)). Example: activate your `surge-rose-demos` venv and `export SURGE_ROOT=...` / `export PYTHONPATH=...` as there.

## Step 1 — Set the M3DC1 pickle path

`dataset_utils.m3dc1_pkl_path()` resolves the dataset in this order:

```text
1. $M3DC1_SOURCE
2. /global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl
3. data/datasets/M3DC1/sparc-m3dc1-D1.pkl
```

**Metadata (usually already in git):**

```text
data/datasets/M3DC1/sparc_m3dc1_D1_metadata.yaml
```

Recommended shared path:

```bash
export M3DC1_SOURCE=/global/cfs/projectdirs/amsc007/data/m3dc1/sparc-m3dc1-D1.pkl
```

You can still point `M3DC1_SOURCE` somewhere else if needed.

**Verify:**

```bash
test -f "$M3DC1_SOURCE" && echo "PKL OK"
```

## Step 2 — `cd` to the orchestration directory

```bash
cd "$SURGE_ROOT/examples/rose_orchestration"
```

## Step 3 — Run examples 1–3 (M3DC1)

Use **`--dataset m3dc1`** on every command. Tuning **`--max-iter`**: M3DC1 uses larger row chunks than synthetic; keep **`max-iter` small** for a quick test.

**Example 1 — in-process**

```bash
python example_01_rose_inprocess_verbose.py --dataset m3dc1 --max-iter 2
```

**Example 2 — subprocess**

```bash
python example_02_rose_subprocess_shell.py --dataset m3dc1 --max-iter 2
```

**Example 3 — random MLP + R² stop**

```bash
python example_03_rose_mlp_random_r2_stop.py --dataset m3dc1 --max-iter 6 --r2-threshold 0.85
```

Adjust `--r2-threshold` / `--r2-operator` / `--max-iter` to trade runtime vs chance of early stop.

## Step 4 — Optional: smoke script

From `examples/rose_orchestration`:

```bash
./run_m3dc1_smoke.sh
```

This checks for the PKL, then runs small **m3dc1** smokes for examples 1–2 and a short example 3 (fails with a clear message if data is missing).

## Artifacts

- Growing Parquet: `workspace/sparc_m3dc1_active_learning.parquet`
- Metrics handoff: `workspace/last_surge_metrics.json`
- SURGE runs: `runs/rose_*` (tags depend on iter / workflow / example)
