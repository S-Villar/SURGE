# SURGE + ROSE orchestration demos

Branch **`radical-integration`**: Examples 1–3 share **`demo_common`** (CLI, schedules) and optional **`--dataset`**.

| Dataset | Description |
|---------|-------------|
| **synthetic** (default) | Growing `workspace/synthetic_active_learning.parquet`; workflows `workflow_rf.yaml` / `workflow_mlp.yaml`. |
| **m3dc1** | Growing `workspace/sparc_m3dc1_active_learning.parquet`; requires PKL + metadata (see [data/datasets/M3DC1/README.md](../../data/datasets/M3DC1/README.md)). Uses `workflow_m3dc1_rf.yaml` / `workflow_m3dc1_mlp.yaml`. **Step-by-step commands** (only for `m3dc1`): [M3DC1_EXAMPLES.md](M3DC1_EXAMPLES.md) and [`run_m3dc1_smoke.sh`](run_m3dc1_smoke.sh). |

**ROSE** schedules: simulation → SURGE training → active-learning placeholder → criterion.

## Demo details: what each example does and which functions it uses

All three examples use the same **ROSE** high-level object, **`rose.al.active_learner.SequentialActiveLearner`**, and the same **outer loop**: build a **Radical Asyncflow** `WorkflowEngine` on top of a **Rhapsody** `rhapsody.backends.ConcurrentExecutionBackend`, then drive **`async for state in acl.start(max_iter=..., initial_config=...)`** until ROSE stops or the iteration cap is reached. They differ in **how each stage is executed** (Python callables vs shell commands) and in **SURGE training** (fixed RF/MLP alternation vs random MLP search) and **stop criterion** (MSE with a zero threshold vs validation R² with a user threshold).

**Shared local modules (this directory):**

| Module | Role |
|--------|------|
| [`demo_common.py`](demo_common.py) | `workflow_for_iteration()` (RF/MLP alternation by dataset), `inprocess_task_kwargs` / `shell_task_kwargs` / `mlp_search_task_kwargs` → per-iteration kwargs for ROSE, plus `add_demo_cli` / `add_reporting_cli`. |
| [`dataset_utils.py`](dataset_utils.py) | `build_training_parquet()` grows the training Parquet; `workspace_dir()` is `examples/rose_orchestration/workspace/`. |
| [`surge_train.py`](surge_train.py) | `run_one_surge()` and `run_mlp_surge_random_arch()` wrap **`surge.workflow.spec.SurrogateWorkflowSpec.from_dict`** and **`surge.workflow.run.run_surrogate_workflow`**, then write **`workspace/last_surge_metrics.json`**. |
| [`orch_report.py`](orch_report.py) | `RunTimer`, `print_run_header`, `print_run_report`, `progress_label`, `snapshot_iteration` (logging / summary tables, not part of ROSE). |

**SURGE API used in every training path:** `run_surrogate_workflow` loads YAML workflow specs in this folder (`workflow_*.yaml`), trains, and returns a summary dict; metrics are written for ROSE to read from disk.

---

### Example 1 — `example_01_rose_inprocess_verbose.py`

| Topic | Details |
|--------|---------|
| **Intent** | Minimal **in-process** orchestration: simulation, training, and criterion run as **async Python functions** in one interpreter (no per-stage subprocesses). |
| **Executor** | `concurrent.futures.ThreadPoolExecutor` inside `ConcurrentExecutionBackend`. |
| **ROSE wiring** | `TaskConfig` kwargs from `inprocess_task_kwargs(max_iter, dataset)` → `LearnerConfig(simulation=..., training=..., active_learn=...)`. Stages are **`as_executable=False`** (they are coroutines, not shell strings). |
| **simulation** | `build_training_parquet(it, dataset=...)`, `pd.read_parquet` for row count, writes `workspace/last_simulation.json`. |
| **training** | **`run_one_surge(wf, it, verbose=...)`** (`surge_train.py`) with `wf` from `workflow_for_iteration` (`rf` / `mlp` or `m3dc1_*`). |
| **active_learn** | No acquisition; logs last metrics (stub) and returns a small bundle. |
| **Stop criterion** | `@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.0, ...)`; handler reads **`val_mse`** from `last_surge_metrics.json` (metric constant from **`rose.metrics`**). With threshold `0.0` and a positive MSE, ROSE does **not** stop early on the metric; the loop is effectively bounded by **`--max-iter`** (default 3). |
| **Lifecycle** | `await acl.start(...)` then **`await acl.shutdown()`**. |
| **Entry** | `main()` → `asyncio.run(_demo(...))`; CLI from `add_demo_cli` + `add_reporting_cli`. |

---

### Example 2 — `example_02_rose_subprocess_shell.py`

| Topic | Details |
|--------|---------|
| **Intent** | Same **logical** loop as Example 1, but each stage is a **separate process** running a small script (closer to batch/HPC: one job per step). |
| **Executor** | `concurrent.futures.ProcessPoolExecutor` inside `ConcurrentExecutionBackend`. |
| **ROSE wiring** | `shell_task_kwargs(max_iter, dataset)` → `TaskConfig` with string keys `--iteration` and `--workflow` (ROSE passes these to the task functions). Stages are **default shell/executable** mode: each handler returns a **command string** to run. |
| **simulation** | Spawns **`python sim_surge_step.py --iteration ... --dataset ...`**. |
| **training** | Spawns **`python surge_train.py --workflow ... --iteration ...`**, which ends up in **`run_one_surge()`** inside `surge_train.py`. |
| **active_learn** | Spawns **`python active_surge_step.py`** (placeholder). |
| **Stop criterion** | `@acl.as_stop_criterion(..., MEAN_SQUARED_ERROR_MSE, 0.0)`; command runs **`check_surge_metrics.py`**; same “non-early” MSE / max-iter behavior as Ex1. |
| **Environment** | **`main()`** sets **`os.environ["PYTHONPATH"]`** to include **`demo_common.SURGE_ROOT`** so child processes can **import `surge`**. |
| **Entry** | `main()` → `asyncio.run(_demo(...))`. |

---

### Example 3 — `example_03_rose_mlp_random_r2_stop.py`

| Topic | Details |
|--------|---------|
| **Intent** | **Architecture search** (random sklearn MLP `hidden_layer_sizes` each outer iteration) plus **metric-driven early stopping** on validation **R²**. |
| **Executor** | `ThreadPoolExecutor` again (same pattern as Ex1). |
| **ROSE wiring** | `mlp_search_task_kwargs(max_iter, base_seed)` so each iteration carries **`iteration`** and **`base_seed`**. |
| **simulation** | Same as Ex1: `build_training_parquet`, `last_simulation.json`. |
| **training** | Local **`sample_hidden_layers(rng)`** (NumPy) picks depth and widths; then **`run_mlp_surge_random_arch(..., arch, dataset=...)`** patches the MLP section of `workflow_mlp.yaml` / `workflow_m3dc1_mlp.yaml` in memory and calls **`run_surrogate_workflow`**. Metrics JSON includes **`hidden_layer_sizes`**. |
| **active_learn** | Stub; logs `val_r2` and chosen arch. |
| **Stop criterion** | `@acl.as_stop_criterion(metric_name="val_r2", threshold=..., operator= '>' or '>=')`; handler returns **`val_r2`** from `last_surge_metrics.json` so ROSE can set **`state.should_stop`** when the comparison holds. If the threshold is never met, the run stops at **`--max-iter`**; if you omit `--max-iter`, the script defaults to **48** iterations (see `main()`). |
| **Extra CLI** | `--seed`, `--r2-threshold`, `--r2-operator`, `--sklearn-mlp-max-iter`. |
| **Entry** | `main()` → `asyncio.run(_demo(...))`. |

---

## What each example does and how they differ (summary)

| | **Example 1** | **Example 2** | **Example 3** |
|---|----------------|-----------------|---------------|
| **Execution** | Python callables in the same process (async) | **Subprocesses**: `sim_surge_step.py`, `surge_train.py`, … | Python callables (async), like Ex1 |
| **Rhapsody pool** | `ThreadPoolExecutor` | `ProcessPoolExecutor` | `ThreadPoolExecutor` |
| **SURGE** | `run_one_surge()` | Same via `surge_train.py` in child processes | `run_mlp_surge_random_arch()` with **random** MLP shape each iter |
| **Active learning** | Stub (no acquisition) | Stub script | Stub |
| **Stopping** | MSE; threshold `0` → **max_iter** only | Same, via `check_surge_metrics.py` | **Early stop** on `val_r2` vs threshold (else `max-iter`) |

All three: simulation grows/refreshes data → training runs SURGE → placeholder active-learning → read `workspace/last_surge_metrics.json` for the stop criterion. Artifacts: SURGE under `runs/rose_*` (run tags differ by example), ROSE handoff in `last_surge_metrics.json`.

## Reference: M3DC1 example runs

*Structured record of command output (timings, tables, early-stop line) from a real session.*

The following is a **structured record** of successful **`--dataset m3dc1`** runs (Parquet: `workspace/sparc_m3dc1_active_learning.parquet`; source PKL: `data/datasets/M3DC1/sparc-m3dc1-D1.pkl`). Environment: NERSC login node, Python from project venv `surge-rose-demos` (see [quickstart.md](quickstart.md)), `PYTHONPATH` including the SURGE clone. Numbers are **illustrative** of one session; your metrics will differ slightly with the same data.

### Example 1 — in-process, `max-iter 2`

```bash
python example_01_rose_inprocess_verbose.py --dataset m3dc1 --max-iter 2
```

| | |
|--|--|
| **Wall time** | ~7.8 s |
| **Workflow schedule** | Iter 0 → `m3dc1_rf` (600 rows); iter 1 → `m3dc1_mlp` (1200 rows) |

| # | rows | workflow | val_rmse | val_r2 | run tag (extra) |
|---|------|----------|----------|--------|-----------------|
| 0 | 600 | m3dc1_rf | 0.008901 | 0.8312 | `rose_m3dc1_rf_iter_0` |
| 1 | 1200 | m3dc1_mlp | 0.010673 | 0.7020 | `rose_m3dc1_mlp_iter_1` |

In **verbose** mode, the log can show one more SURGE block after this summary (e.g. `workflow_rf.yaml` / `rf_demo`) as ROSE finishes the active learner; **treat the printed “Example 1 summary” table as the authoritative list of outer iterations.**

### Example 2 — subprocess, `max-iter 3`

```bash
python example_02_rose_subprocess_shell.py --dataset m3dc1 --max-iter 3
```

| | |
|--|--|
| **Wall time** | ~25.2 s |
| **Phases** | Each outer iteration: `sim_surge_step.py` → `surge_train.py` → `active_surge_step.py` → `check_surge_metrics.py` |

| # | rows | workflow | val_rmse | val_r2 | run tag (extra) |
|---|------|----------|----------|--------|-----------------|
| 0 | 600 | m3dc1_rf | 0.008901 | 0.8312 | `rose_m3dc1_rf_iter_0` |
| 1 | 1200 | m3dc1_mlp | 0.010673 | 0.7020 | `rose_m3dc1_mlp_iter_1` |
| 2 | 1800 | m3dc1_rf | 0.009489 | 0.7913 | `rose_m3dc1_rf_iter_2` |

As with Example 1, the log may continue with a partial spawn line after the completed iterations; the **summary table** is the intended result.

### Example 3 — random MLP + early stop on `val_r2 >= 0.9`

```bash
python example_03_rose_mlp_random_r2_stop.py --dataset m3dc1 --max-iter 12
```

| | |
|--|--|
| **Wall time** | ~23.8 s |
| **Stop rule** | `val_r2` **>= 0.9** (default threshold); `max-iter` capped at 12 but loop **exits early** when the criterion is met. |
| **Outcome** | Criterion met at **outer iteration 7** (0-based row 7 in the table): `val_r2 ≈ 0.9004` for architecture `[96, 48, 16]` at **4800** training rows. |

| # | rows | workflow | val_rmse | val_r2 | `hidden_layer_sizes` (extra) |
|---|------|----------|----------|--------|------------------------------|
| 0 | 600 | m3dc1_mlp | 0.007867 | 0.8681 | [64, 96, 96, 64] |
| 1 | 1200 | m3dc1_mlp | 0.012155 | 0.6135 | [96] |
| 2 | 1800 | m3dc1_mlp | 0.007838 | 0.8576 | [128, 64, 64, 64] |
| 3 | 2400 | m3dc1_mlp | 0.009984 | 0.7362 | [128, 64, 128] |
| 4 | 3000 | m3dc1_mlp | 0.010404 | 0.7465 | [128, 96, 32, 32] |
| 5 | 3600 | m3dc1_mlp | 0.009956 | 0.8092 | [48, 96] |
| 6 | 4200 | m3dc1_mlp | 0.009077 | 0.8431 | [32, 48, 48] |
| **7** | **4800** | m3dc1_mlp | **0.006707** | **0.9004** | **[96, 48, 16]** ← stop (`should_stop=True`) |

ROSE line: `stop criterion metric: val_r2 is met with value of: 0.9003767675107067 . Breaking the active learning loop`.

## Prerequisites

- Python **3.10+** (ROSE).
- `pip install -r requirements-rose-demo.txt` from SURGE root; `pip install -e /path/to/ROSE`.
- **`PYTHONPATH`** must include the **SURGE repo root**:

```bash
export PYTHONPATH=/path/to/SURGE:$PYTHONPATH
```

Run from **this directory** (`examples/rose_orchestration`).

On shared systems, a pre-built **conda** `surge` env may already exist, for example:  
`conda activate /global/common/software/m3716/asvillar/envs/surge`  
(see [envs/README.md](../../envs/README.md)). Replace `SURGE_ROOT` and `ROSE_ROOT` below with your actual clone paths.

**m3716 quick copy-paste (runnable commands, that env + default clone paths):** [quickstart.md](quickstart.md).

---

## Complete recipe (from scratch)

Copy-paste friendly; lines starting with `#` are comments (not executed).

```bash
# ---------------------------------------------------------------------------
# 0) Open a shell and activate Python 3.10+ (ROSE requires it).
#    On NERSC m3716, a typical choice is the shared surge env (Python 3.11):
# ---------------------------------------------------------------------------
# source "$(conda info --base)/etc/profile.d/conda.sh"   # if conda is not initialized
# conda activate /global/common/software/m3716/asvillar/envs/surge

# ---------------------------------------------------------------------------
# 1) One-time (or when deps change): editable SURGE + ROSE helper deps + ROSE.
#    SURGE_ROOT = path to this repository (parent of the `surge/` package).
#    ROSE_ROOT      = your ROSE git clone (sibling to SURGE is common).
# ---------------------------------------------------------------------------
# export SURGE_ROOT="$HOME/src/SURGE"
# export ROSE_ROOT="$HOME/src/ROSE"
# pip install -e "$SURGE_ROOT"
# pip install -r "$SURGE_ROOT/requirements-rose-demo.txt"
# pip install -e "$ROSE_ROOT"

# ---------------------------------------------------------------------------
# 2) Every session: put SURGE on PYTHONPATH and cd to this folder
#    (SURGE is imported as the `surge` package from repo root).
# ---------------------------------------------------------------------------
# export SURGE_ROOT="${SURGE_ROOT:-$HOME/src/SURGE}"
# export PYTHONPATH="$SURGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# cd "$SURGE_ROOT/examples/rose_orchestration"

# ---------------------------------------------------------------------------
# 3) EXAMPLE 1 — In-process ROSE loop (no separate OS processes for stages).
#    - ROSE: SequentialActiveLearner → stages sim → train → active_learn → criterion.
#    - Rhapsody: ConcurrentExecutionBackend(ThreadPoolExecutor) runs async work.
#    - Radical Asyncflow: WorkflowEngine.create(rhapsody_backend) is what ROSE drives.
#    - SURGE: called directly in Python: run_one_surge() in surge_train.py.
#    - Stop rule: MSE on val from last_surge_metrics.json; threshold 0 so loop runs
#      until --max-iter (default 3) unless you change the criterion in code.
# ---------------------------------------------------------------------------
# python example_01_rose_inprocess_verbose.py
# python example_01_rose_inprocess_verbose.py --dataset m3dc1 --max-iter 3
# python example_01_rose_inprocess_verbose.py --quiet

# ---------------------------------------------------------------------------
# 4) EXAMPLE 2 — Same ROSE *logical* loop as Ex1, but each stage is a SHELL / subprocess
#    (closer to HPC: separate Python per step).
#    - ROSE: same SequentialActiveLearner + same four stages.
#    - Rhapsody: ConcurrentExecutionBackend(ProcessPoolExecutor) — process pool
#      instead of threads (contrast with Ex1).
#    - Tasks return command strings: sim_surge_step.py, surge_train.py, active_surge_step.py,
#      check_surge_metrics.py (ROSE/rhapsody execute them in workers).
#    - SURGE: same training logic as Ex1, but via surge_train.py child processes;
#      main() prepends SURGE_ROOT to PYTHONPATH for those children.
#    - Stop rule: same MSE/0.0 style as Ex1, via the check_surge_metrics script.
# ---------------------------------------------------------------------------
# python example_02_rose_subprocess_shell.py
# python example_02_rose_subprocess_shell.py --dataset synthetic --max-iter 3

# ---------------------------------------------------------------------------
# 5) EXAMPLE 3 — In-process like Ex1, but: random sklearn MLP hidden_layer_sizes
#    each iteration and ROSE can STOP EARLY when validation R² passes a user threshold.
#    - ROSE: same learner + stages; criterion uses metric "val_r2" with > or >=.
#    - Rhapsody: ThreadPoolExecutor backend again (like Ex1).
#    - SURGE: run_mlp_surge_random_arch() patches YAML in memory; writes metrics.
#    - If R² never hits the threshold, the loop ends at --max-iter (default 48
#      when --max-iter is not passed; pass e.g. --max-iter 12 for a shorter run).
# ---------------------------------------------------------------------------
# python example_03_rose_mlp_random_r2_stop.py --dataset synthetic --max-iter 12
# python example_03_rose_mlp_random_r2_stop.py --dataset m3dc1 --r2-threshold 0.85
# python example_03_rose_mlp_random_r2_stop.py --max-iter 60 --r2-threshold 0.9 --r2-operator ">="
```

### Shared CLI (all three examples)

| Option | Effect |
|--------|--------|
| `--dataset synthetic` \| `m3dc1` | Which growing Parquet + workflow family (M3DC1 needs data files). |
| `--max-iter N` | Outer ROSE iterations (defaults: 3 for Ex1/2; Ex3 uses 48 if omitted). |
| `--workers K` | Thread or process pool size for Rhapsody (default 4). |
| `--quiet` | Less console output. |

Ex3 also: `--seed`, `--r2-threshold`, `--r2-operator` (`>` or `>=`), `--sklearn-mlp-max-iter`.

*(High-level comparison and per-demo function list: [Demo details: what each example does and which functions it uses](#demo-details-what-each-example-does-and-which-functions-it-uses) above.)*

---

## How ROSE, Rhapsody, and Radical Asyncflow fit together (and where SURGE sits)

**Stack (bottom = hardware/scheduling, top = your loop):**

1. **Rhapsody** — you build `ConcurrentExecutionBackend` with a **standard library executor** (threads in Ex1/3, **processes** in Ex2). That backend is the **execution engine** (how tasks run in parallel in the pool).

2. **Radical Asyncflow** — `await WorkflowEngine.create(engine)` **wraps** the Rhapsody backend so an **async** workflow can be driven from Python.

3. **ROSE** — `SequentialActiveLearner(asyncflow)` is the **orchestrator** for *simulation → training → active learn → stop criterion*. You register coroutine tasks with `@acl.simulation_task`, `@acl.training_task`, `@acl.active_learn_task`, and `@acl.as_stop_criterion`, then `async for state in acl.start(max_iter=..., initial_config=...)`.

4. **SURGE** — **not** part of Rhapsody. It is invoked **inside** your `training` task (in-process) or in **child processes** (Ex2) and writes **`workspace/last_surge_metrics.json`**, which the stop-criterion task reads. ROSE only needs the **metric values** to decide whether to stop, not a live import of SURGE.

**One-line summary:** *Rhapsody* supplies the **pool**; *Asyncflow* supplies the **async workflow engine**; *ROSE* supplies the **active-learning control loop** on top; *SURGE* is the **trainer** you call from the training stage. Rhapsody is wired to ROSE by passing the backend into **`WorkflowEngine.create(...)`** as shown in each example’s `_demo` coroutine.

## SURGE-only deep-ensemble UQ

- Core: `sklearn.mlp` with `ensemble_n > 1` and `request_uncertainty: true` in a workflow spec ([surge/model/sklearn.py](../../surge/model/sklearn.py)).
- Demo: [examples/surge_uq_demo/run_uq_demo.py](../surge_uq_demo/run_uq_demo.py) (from repo root: `python examples/surge_uq_demo/run_uq_demo.py` after `export PYTHONPATH=.`).
- Optional figure (requires matplotlib): `python examples/surge_uq_demo/run_uq_demo.py --plot /tmp/uq_val.png`
- Manual multi-seed loop: [examples/mlp_ensemble_uq/mlp_ensemble_uq_demo.py](../mlp_ensemble_uq/mlp_ensemble_uq_demo.py) or launcher `mlp_ensemble_uq_demo.py` in this folder.
- Tests: [tests/test_sklearn_mlp_ensemble.py](../../tests/test_sklearn_mlp_ensemble.py)

## ROSE / Rhapsody / SURGE (quick reference)

| Piece | Role |
|-------|------|
| **SURGE** (this repo) | Tabular ingest, training, metrics, `predict` / `predict_with_uncertainty`, `runs/` artifacts. |
| **ROSE** | Active-learning loop: which stage runs next and when to stop (uses Radical Asyncflow). |
| **Rhapsody** | Execution backends (e.g. `ConcurrentExecutionBackend` with thread or process pools) — [rhapsody](https://github.com/radical-cybertools/rhapsody). |

**How they connect in these demos** is described under [How ROSE, Rhapsody, and Radical Asyncflow fit together](#how-rose-rhapsody-and-radical-asyncflow-fit-together-and-where-surge-sits) above. These examples compose **ROSE + Rhapsody backend + Asyncflow + SURGE training**.

## Artifacts

- SURGE: `runs/rose_*`
- ROSE handoff: `workspace/last_surge_metrics.json`

## Standalone SURGE CLI (this directory)

```bash
python sim_surge_step.py --iteration 1 --dataset synthetic --verbose
python surge_train.py --workflow m3dc1_mlp --iteration 1 --verbose
```

Workflow keys: `rf`, `mlp`, `m3dc1_rf`, `m3dc1_mlp`.
