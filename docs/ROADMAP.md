# SURGE development roadmap

Planned and in-progress work beyond what the current release claims as stable. For a **prerelease** narrative, **public release / git history** steps, and deeper planning, see [PRERELEASE.md](PRERELEASE.md), [PUBLIC_OPEN_SOURCE_PLAN.md](PUBLIC_OPEN_SOURCE_PLAN.md), [REFACTORING_PLAN.md](REFACTORING_PLAN.md), and [dev/EXTENDED_ROADMAP.md](dev/EXTENDED_ROADMAP.md).

## Landed in v0.1.0 (reference, not planned work)

- **Resource policy.** `ResourceSpec` on `SurrogateWorkflowSpec` and `EngineRunConfig` lets users declare `device` (`auto`/`cpu`/`cuda`[`:N`]) and `num_workers` up front. Per-adapter `ResourceProfile` advertises what each backend can honour; the policy engine either warns and falls back, or (under `strict=True`) raises `ResourcePolicyError`. Every fit emits exactly one `[surge.fit] ...` banner, and the resolved fields are persisted to `workflow_summary.json` under `models[].resources_used`. See `surge/hpc/policy.py` and [`REFACTORING_PLAN.md`](REFACTORING_PLAN.md) §2.
- **Model profile in metrics.** Every run stores `model_size_bytes`, `parameter_count`, `inference_ms_per_sample`, and throughput at `models[].profile` in both `workflow_summary.json` and `metrics.json`. See `surge/model/profiling.py`.
- **Install hygiene.** `surge/__init__.py` no longer injects `scripts/m3dc1/` into `sys.path` at import time; `M3DC1Dataset` imports its loader lazily inside `_get_m3dc1_loaders()` so `pip install`-only users get a clean `import surge`.
- **CI regression gate.** A 3-job GitHub Actions workflow (`test`, `e2e-regression`, `lint`) runs `pytest -q` across 3.10/3.11 and a `sklearn → PyTorch → ONNX` round-trip on a tracked sample CSV; broken release paths now fail CI.

## Cross-validation in the unified workflow

- **Status:** `SurrogateWorkflowSpec.cv_folds` is accepted in YAML; values `> 0` currently **log a warning** and are **ignored**—training still uses one random train/val/test split.
- **Target behavior:** *k*-fold (or repeated *k*-fold) over the training pool, per-fold metrics, and an aggregate (mean ± std) for chosen metrics, written to run artifacts (e.g. `metrics_cv.json`), so users can check sensitivity to the split and reduce overfitting to a particular partition.
- **Reference:** `surge/legacy/engine_legacy.py` has older CV helpers; the goal is a clean reimplementation in `SurrogateEngine` + `run_surrogate_workflow`.

## Large datasets without a full in-memory `pandas.DataFrame`

- **Status:** The default path is tabular `DataFrame` + `SurrogateDataset` (good for research-scale and many workflows).
- **Target:** Optional backends—chunked Parquet, memory-mapped arrays, `torch.utils.data.IterableDataset` / sharded loaders—so training never materializes the full table in RAM. The engine would consume batches or an iterator where appropriate.

## Recipe-driven, semantics-aware visualization

- **Status:** Generic plots and analysis utilities exist under `surge/viz/`; full “one YAML recipe → multi-panel EDA (e.g. ICRF-style six-panel)” with per-column labels/units and agent-assisted flow is not wired as a first-class spec feature yet.
- **Target:** Enriched metadata (LaTeX labels, units, profile coordinate), recipe names, and optional agent/CLI to compose panels (violins, mean±σ profiles, heatmaps, etc.) from the same input layer the models use.

## Other items

- Unify the two model registry paths (`surge/model/registry.py` vs `surge/registry.py`).
- Public ONNX export CLI aligned with the C++/ONNX-Runtime path.
- Interactive / agent-assisted spec generation from dataset discovery and user intent.
