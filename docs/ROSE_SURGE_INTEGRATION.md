# ROSE and SURGE: roles and integration

## ROSE repository status

ROSE lives at `${HOME}/src/ROSE`. It was previously 61 commits behind `origin/main` and has been fast-forwarded to the current `main`, including UQ active-learning examples under `examples/active_learn/uq/` and tracking integrations.

---

## What each side is responsible for

### ROSE (README)

ROSE is an **async workflow orchestrator** on top of RADICAL Asyncflow / Rhapsody (`radical.asyncflow`, `rhapsody-py`). It does **not** implement your physics or surrogate math; it **schedules tasks** (often as shell strings you return from decorated coroutines), supports **sequential and parallel learners**, **UQ-based active learning** (`SeqUQLearner` / `ParallelUQLearner`), and **algorithm selection** (`AlgorithmSelector`). It requires **Python ≥ 3.10** (`pyproject.toml`).

### SURGE

SURGE is a **tabular surrogate training stack**: datasets → `SurrogateEngine` / YAML workflows → registered models, metrics, artifacts. It already has a small ensemble primitive for UQ (`FNNEnsemble` with mean/std over members) and separate data generation utilities (`surge.datagen`) for parameter batches—not an HPC workflow engine.

### Natural split

- **ROSE** = loop + HPC execution + task DAG  
- **SURGE** = training, prediction, and (optional) ensemble UQ in Python, **invoked from** ROSE’s training / prediction / scoring steps  

---

## Compatibility

| Aspect | Notes |
|--------|--------|
| **Python** | ROSE needs ≥3.10; SURGE’s `requirements.txt` doesn’t pin a floor but fits 3.10+ in practice. |
| **Dependencies** | Overlap is mostly NumPy. ROSE adds RADICAL/Rhapsody (substantial); SURGE adds PyTorch, sklearn, Optuna, etc. They don’t inherently conflict; combine them in one env if you need in-process imports, or keep a ROSE driver + subprocess calls into SURGE (simpler on HPC). |
| **Execution model** | SURGE is **synchronous**; ROSE is **async** and expects tasks (executable strings or registered units). The intended pattern in ROSE examples is: each stage returns e.g. `f"{python} …/training.py …"`—so SURGE fits as **CLI/scripts**, not as a library you `await` inside the engine. |

---

## How to run “SURGE with ROSE” (recommended shape)

**Driver:** A small ROSE `run_me.py` (pattern from `examples/active_learn/uq/run_me.py`) using `SequentialActiveLearner` or `ParallelUQLearner` and either `RadicalExecutionBackend` (PBS/Slurm/etc.) or `ConcurrentExecutionBackend` locally.

### Task contracts (four boundaries you implement once)

1. **`simulation_task`** — Submit/run the real code (XGC, M3DC1, etc.), write outputs under a fixed layout (e.g. `runs/iter_{k}/cases/…`).
2. **`training_task`** — Build/update the tabular dataset (your existing batch → parquet/CSV pipeline), then call `run_surrogate_workflow` or a thin wrapper around `SurrogateEngine` that reads a config path and writes one model dir per ensemble member (or one multi-model run—see UQ below).
3. **`prediction_task`** — Load trained artifact(s), run inference on a candidate pool of inputs, save per-member predictions or mean+std in files ROSE’s UQ step can read (mirror `predict.py` / `check_uq.py` in the UQ example).
4. **`active_learn_task`** (+ optional `@learner.uncertainty_quantification`) — Choose the next labeled points (using SURGE-exported uncertainty), update a queue file or case list consumed by the next `simulation_task`.

---

## Ensembles (UQ)

- **ROSE:** `ParallelUQLearner` / `SeqUQLearner` already treat `model_names` as an ensemble and run multiple trainings and multiple `num_predictions` for metrics like predictive entropy (`examples/active_learn/uq/run_me.py` uses `MODELS = ["MC_Dropout_CNN", "MC_Dropout_MLP"]` for `USECASE == "ENSEMBLE"`).
- **SURGE:** Either (a) multiple independent models in one YAML `models:` list (each saved separately), or (b) `FNNEnsemble` with *N* members and explicit export of member preds/std—your `prediction_task` must serialize what the ROSE UQ scorer expects.

**Gap to close:** SURGE’s main workflow is built around single-model artifacts; you’ll want a small script layer that trains *N* members (different seeds / bootstrap / distinct architectures) and writes a **normalized prediction bundle** for acquisition.

---

## Active learning and “which code for new samples”

- **Per-iteration simulation:** ROSE’s `LearnerConfig` supports `TaskConfig | dict[int, TaskConfig]` for simulation, training, etc., so you can change the command or kwargs by iteration (`SequentialActiveLearner` documents `set_next_config(...)` for mid-loop changes).
- **Choosing acquisition strategy:** Use `AlgorithmSelector` with multiple `@also.active_learn_task(name="…")` handlers (`examples/active_learn/algorithm_selector/run_me.py`) if the uncertainty side branches.
- **Choosing simulator fidelity (XGC vs cheaper code):** Put a **router** inside `simulation_task`: read a field written by AL (`--solver`, `--code_tag`, path to case template) and branch to the appropriate launcher. The interface is: agreed JSON/YAML **“next sample descriptor”** produced by AL, consumed by sim.

---

## Interfaces worth defining explicitly

| Area | Suggestion |
|------|------------|
| **Filesystem / manifest** | e.g. `pool.parquet`, `labeled.parquet`, `iter_manifest.json` (next cases + metadata + which code to run). |
| **Training I/O** | Path to dataset, model registry key, run output dir, optional `member_id` for ensembles. |
| **Prediction I/O** | Schema for pool features + outputs for each ensemble member (for entropy / max-std queries). |
| **Simulation launcher API** | Function or CLI that maps `(case_id, parameter dict, code_id)` → job submission + completion marker. |
| **Stopping / metrics** | Align SURGE’s validation metric with ROSE `as_stop_criterion(metric_name=…)` (ROSE supplies metric names like `MODEL_ACCURACY`, `MEAN_SQUARED_ERROR_MSE`; you can register custom metrics consistent with `rose.metrics`). |

---

## Practical takeaway

- **Pull status:** ROSE updated at `${HOME}/src/ROSE`.
- **Integration style:** Treat SURGE as the training/predict backend and ROSE as the HPC + AL control loop; connect them with **CLI scripts** and a **shared run directory contract**, not by merging asyncio into `surge.engine`.
- **Your goals:** UQ ensembles map cleanly to ROSE’s multi-model UQ learner + SURGE multi-member training/export; AL maps to ROSE’s simulation / train / predict / AL stages with `LearnerConfig` / `TaskConfig` (and optionally `AlgorithmSelector`) to encode which code or strategy applies to the next batch.

### Possible next step in this repo

A concrete artifact would be a `scripts/rose/` (or similar) trio: `rose_sim.py`, `rose_train_surge.py`, `rose_predict_uq.py` plus a single YAML describing paths—**without** changing SURGE core until the contract stabilizes.
