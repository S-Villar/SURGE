# ROSE + SURGE orchestration — quickstart (m3716 `surge` env)

This file mirrors the [complete recipe in README.md](README.md#complete-recipe-from-scratch) but **targets the shared conda environment** on NERSC project **m3716** and uses **runnable** commands (not commented out).

Adjust **`SURGE_ROOT`** and **`ROSE_ROOT`** if your clones live elsewhere; defaults below match a typical layout.

| Variable | Default in this quickstart |
|----------|----------------------------|
| Conda `surge` (Python 3.11) | `/global/common/software/m3716/asvillar/envs/surge` |
| `SURGE_ROOT` | `/global/homes/a/asvillar/src/SURGE` or a fresh `radical-integration` clone |
| `ROSE_ROOT` | `/global/homes/a/asvillar/src/ROSE` |

---

## Run the demos now (use a venv on CFS — **recommended** when `surge` or `~/.local` hits quota)

**Why:** `pip install` into the **shared** m3716 `surge` env *or* into **`~/.local`** can fail with `Errno 122` (different filesystems / limits). A **dedicated venv** under your AMSc project space keeps all extra wheels off home and off the global env.

**One-time setup** (uses the m3716 `surge` **Python** only to create the venv; packages install **inside the venv** on CFS):

```bash
export PIP_CACHE_DIR=/global/cfs/projectdirs/amsc007/asvillar/pip_cache
export VENV=/global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos
export SURGE_ROOT=/global/homes/a/asvillar/src/SURGE
export ROSE_ROOT=/global/homes/a/asvillar/src/ROSE

/global/common/software/m3716/asvillar/envs/surge/bin/python -m venv "$VENV"
source "$VENV/bin/activate"

pip install -U pip wheel
pip install --no-cache-dir -r "$SURGE_ROOT/requirements-rose-demo.txt"
pip install --no-cache-dir -e "$ROSE_ROOT"
pip install --no-cache-dir 'pyarrow>=14'
```

**SURGE** in this repo can be installed with `pip install -e "$SURGE_ROOT"` or used directly by setting `PYTHONPATH` to the clone root.

**Every new shell** — either:

```bash
source /global/homes/a/asvillar/src/SURGE/examples/rose_orchestration/activate_surge_rose_cfs_venv.sh
cd /global/homes/a/asvillar/src/SURGE/examples/rose_orchestration
```

or by hand:

```bash
source /global/cfs/projectdirs/amsc007/asvillar/conda_envs/surge-rose-demos/bin/activate
export SURGE_ROOT=/global/homes/a/asvillar/src/SURGE
export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$SURGE_ROOT/examples/rose_orchestration"
```

**Run:**

```bash
python example_01_rose_inprocess_verbose.py --dataset synthetic --max-iter 2
```

**M3DC1 only** (real SPARC pickle + same examples with `--dataset m3dc1`): [M3DC1_EXAMPLES.md](M3DC1_EXAMPLES.md). Default quickstart commands use **synthetic** data; they do **not** require the M3DC1 PKL.

---

### Using project disk (AMSc007) instead of `$HOME` for repos and pip cache

If **`$HOME` quota is full**, keep clones and large caches on **CFS project space** (higher limit than home). A typical layout (your project has `src/`, `pip_cache/`, and `conda_envs/` under `asvillar`):

| Path | Suggested use |
|------|----------------|
| `/global/cfs/projectdirs/amsc007/asvillar/src` | `git clone` **SURGE** and **ROSE** here (same as other repos: `BaseSim_Framework`, `fusionbench`, …). |
| `/global/cfs/projectdirs/amsc007/asvillar/pip_cache` | Set **`export PIP_CACHE_DIR=.../pip_cache`** so `pip` does not fill `$HOME/.cache/pip`. |
| `/global/cfs/projectdirs/amsc007/asvillar/conda_envs` | Optional: **`python -m venv`** a dedicated venv here if you do not use the m3716 `surge` conda. |

**Move or re-clone (pick one):**

- **Re-clone** (safest, clean history of paths):

  ```bash
  export PROJ=/global/cfs/projectdirs/amsc007/asvillar/src
  cd "$PROJ"
  git clone --branch radical-integration https://github.com/S-Villar/SURGE.git SURGE
  git clone <ROSE-upstream-URL> ROSE
  ```

- **Move an existing tree** (keeps git state; use if you already have `SURGE` under `~/src` and want one copy on project disk):

  ```bash
  mv /global/homes/a/asvillar/src/SURGE /global/cfs/projectdirs/amsc007/asvillar/src/SURGE
  mv /global/homes/a/asvillar/src/ROSE /global/cfs/projectdirs/amsc007/asvillar/src/ROSE 2>/dev/null
  ```

  After moving, set **`SURGE_ROOT`** and **`ROSE_ROOT`** to the new paths, update any IDE / `PYTHONPATH`, and re-run section **1** `pip install -e` from there.

- **Pip in every session** (optional) when working from project space:

  ```bash
  export PIP_CACHE_DIR=/global/cfs/projectdirs/amsc007/asvillar/pip_cache
  export SURGE_ROOT=/global/cfs/projectdirs/amsc007/asvillar/src/SURGE
  export ROSE_ROOT=/global/cfs/projectdirs/amsc007/asvillar/src/ROSE
  ```

  Note: the **m3716** `surge` conda under `/global/common/software/...` is still valid to activate; you only change where the **repositories** and **pip cache** live.

---

## 0) Activate the m3716 `surge` environment

```bash
# Python 3.10+; after activation, `python` is the 3.11 interpreter from this env
source "$(conda info --base)/etc/profile.d/conda.sh"   # skip if conda is already working in your shell
conda activate /global/common/software/m3716/asvillar/envs/surge
```

---

## 1) One-time (or when dependencies change)

Editable SURGE (this repo), ROSE extra deps, **ROSE** (brings `radical.asyncflow`, `rhapsody`, `rose`), and **pyarrow** (Parquet).

The **`surge` conda image does not include ROSE**; without `pip install -e "$ROSE_ROOT"` you get `ModuleNotFoundError: No module named 'radical'`.

If **`pip install` fails with `OSError: [Errno 122] Disk quota exceeded`** when writing into the env’s `site-packages`, install the same packages **into your user site** with `--user` (keeps the conda interpreter; packages land in `~/.local/...`):

```bash
export SURGE_ROOT=/global/homes/a/asvillar/src/SURGE
export ROSE_ROOT=/global/homes/a/asvillar/src/ROSE

pip install -e "$SURGE_ROOT" || pip install --user -e "$SURGE_ROOT"
pip install -r "$SURGE_ROOT/requirements-rose-demo.txt" || pip install --user -r "$SURGE_ROOT/requirements-rose-demo.txt"
pip install -e "$ROSE_ROOT" || pip install --no-cache-dir --user -e "$ROSE_ROOT"

# Parquet: remove a broken empty env pyarrow first if needed, then (see troubleshooting):
pip install --no-cache-dir 'pyarrow>=14' || pip install --no-cache-dir --user 'pyarrow>=14'
```

Use **either** plain `pip install` **or** the `--user` fallbacks, depending on which succeeds (not both unless you re-run only failed steps).

---

## 2) Every new shell session

```bash
conda activate /global/common/software/m3716/asvillar/envs/surge

export SURGE_ROOT=/global/homes/a/asvillar/src/SURGE
export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SURGE_ROOT/examples/rose_orchestration"
```

---

## 3) Example 1 — in-process ROSE loop (no separate OS processes for stages)

- **ROSE:** `SequentialActiveLearner` → sim → train → active_learn → criterion.
- **Rhapsody:** `ConcurrentExecutionBackend(ThreadPoolExecutor)`.
- **Radical Asyncflow:** `WorkflowEngine.create(rhapsody_backend)` (what ROSE drives).
- **SURGE:** `run_one_surge()` in `surge_train.py` (in-process).
- **Stop rule:** MSE on val from `last_surge_metrics.json`; threshold `0` → effectively run until `--max-iter` (default **3**).

```bash
python example_01_rose_inprocess_verbose.py
python example_01_rose_inprocess_verbose.py --dataset m3dc1 --max-iter 3
python example_01_rose_inprocess_verbose.py --quiet
```

---

## 4) Example 2 — same logical loop, each stage is a subprocess

- **ROSE:** same four stages as Ex1.
- **Rhapsody:** `ProcessPoolExecutor` (not threads).
- **Tasks:** command strings to `sim_surge_step.py`, `surge_train.py`, `active_surge_step.py`, `check_surge_metrics.py`.
- **SURGE:** `surge_train.py` in child processes; `main()` sets `PYTHONPATH` to `SURGE` root for children.
- **Stop rule:** same MSE/0.0 style via `check_surge_metrics.py`.

```bash
python example_02_rose_subprocess_shell.py
python example_02_rose_subprocess_shell.py --dataset synthetic --max-iter 3
```

---

## 5) Example 3 — random MLP + optional early stop on val R²

- **ROSE:** same stages; stop criterion on `"val_r2"` with `>` or `>=`.
- **Rhapsody:** thread pool (like Ex1).
- **SURGE:** `run_mlp_surge_random_arch()` (YAML patched in memory).
- **Default `max-iter`:** if you omit it, the script may use a **48**-iteration default; pass `--max-iter` for shorter runs.

```bash
python example_03_rose_mlp_random_r2_stop.py --dataset synthetic --max-iter 12
python example_03_rose_mlp_random_r2_stop.py --dataset m3dc1 --r2-threshold 0.85
python example_03_rose_mlp_random_r2_stop.py --max-iter 60 --r2-threshold 0.9 --r2-operator ">="
```

---

## Troubleshooting

### `AttributeError: module 'pyarrow' has no attribute '__version__'` (when importing `pandas`)

**Cause:** The environment has a **broken** `pyarrow` install: `import pyarrow` “succeeds” (often as an almost empty *namespace* under `site-packages/pyarrow`), but the real `pyarrow` package with `__version__` and binary extensions is missing. Pandas then crashes in `pandas.compat.pyarrow` (that path only falls back on **ImportError**, not a half-installed module).

**Check:**

```bash
conda activate /global/common/software/m3716/asvillar/envs/surge
python -c "import pyarrow as pa; print(pa.__file__); print(getattr(pa, '__version__', None))"
```

A healthy install prints a `.so`-backed `__file__` like `.../pyarrow/__init__.py` and a version string. If `__version__` is missing or `__file__` is `None`, fix as below.

**Fix — step 1:** Remove the broken `pyarrow` **directory in the env** (otherwise a useless namespace still wins on `import` and `pip install --user` will not help):

```bash
conda activate /global/common/software/m3716/asvillar/envs/surge
PY11=/global/common/software/m3716/asvillar/envs/surge/lib/python3.11/site-packages
rm -rf "$PY11/pyarrow" "$PY11"/pyarrow-*.dist-info 2>/dev/null
```

**Fix — step 2:** Install a real `pyarrow` (pick **one** that works with your **quota**):

```bash
# Prefer installing into the env (needs enough quota on the filesystem that holds site-packages)
pip install --no-cache-dir 'pyarrow>=14'
```

If you get **`OSError: [Errno 122] Disk quota exceeded`** on that command, the env’s prefix may be full. **Install into your user site** instead (often counts against a different limit than the project area):

```bash
pip install --no-cache-dir --user 'pyarrow>=14'
```

Do not set `PYTHONNOUSERSITE=1` or Python will ignore `~/.local` and `pyarrow` will not be found. Clearing `~/.cache/pip` only frees *cache*; it does not remove the broken `site-packages/pyarrow` folder — you still need the `rm` + install above.

**Verify:**

```bash
python -c "import pyarrow as pa; print('pyarrow', pa.__version__)"
python -c "import pandas as pd; print('pandas', pd.__version__)"
```

The demos use **Parquet** (`dataset_utils`, `read_parquet` / `to_parquet`); a working **pyarrow** (or, alternatively, `fastparquet` with the right pandas engine) is required for I/O, not only for the import fix.

### `ModuleNotFoundError: No module named 'radical'`

**Cause:** **ROSE** (and its dependency **`radical.asyncflow`**) are not installed in the active environment. The base `surge` conda stack includes SURGE-related libs but not ROSE.

**Fix:**

```bash
conda activate /global/common/software/m3716/asvillar/envs/surge
export ROSE_ROOT=/global/homes/a/asvillar/src/ROSE   # your clone
pip install -e "$ROSE_ROOT"
# if Errno 122 (quota) on the env prefix:
# pip install --no-cache-dir --user -e "$ROSE_ROOT"
```

Verify:

```bash
python -c "from radical.asyncflow import WorkflowEngine; import rose; print('ok')"
```

---

## See also

- [README.md](README.md) — datasets, `ROSE` / `Rhapsody` / `SURGE` roles, and the full comparison table.
- Shared env path on m3716: [envs/README.md](../../envs/README.md).
