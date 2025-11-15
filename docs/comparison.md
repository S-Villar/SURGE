# SURGE vs skorch & SMT Capability Comparison

| Capability | SURGE (this branch) | skorch | SMT | Notes |
|-----------:|--------------------|:------:|:---:|-------|
| **C1** | End-to-end workflow via `surge/workflow/run.py`, structured artifacts (`surge/io/artifacts.py`), dataset loader + preprocessing integrated. | Partial – skorch covers training loop but not ingest/templating/artifacts. | Partial – SMT focuses on surrogate modeling; no ingest workflow. | SURGE now handles datagen→ingest→train→artifacts out of the box. |
| **C2** | `surge/dataset.py` auto-detects inputs/outputs across CSV/Parquet/pickle/HDF5/NetCDF with metadata overrides. | No – requires user-defined tensors/datasets. | Limited – SMT expects arrays supplied by user. | SURGE adds scientific naming heuristics + metadata grouping. |
| **C3** | Registry + adapters (`surge/registry.py`, `surge/models/adapters/*`) unify sklearn, PyTorch, GPflow APIs with consistent `fit/predict/predict_with_uncertainty/save/load`. | Similar API but PyTorch-only. | Supports multiple surrogate types but not a shared adapter interface. | SURGE also exposes registry summary + workflow integration. |
| **C4** | Built-in UQ via RandomForest variance, MC-Dropout in Torch adapter, GPflow mean/variance; standardized `predict_with_uncertainty`. | Requires custom heads; no unified UQ API. | Provides UQ for select surrogates, but no shared `predict_with_uncertainty`. | SURGE normalizes mean/variance payloads + stores them in artifacts. |
| **C5** | `SurrogateWorkflowSpec` + `run_surrogate_workflow` deliver one-call workflows, Optuna HPO, CV-ready splits, artifact emission (`metrics.json`, `workflow_summary.json`, predictions, scalers, spec, env, git). | Partial – training loop but no workflow spec or artifacts/HPO orchestrator. | No – SMT lacks workflow engine/HPO. | SURGE HPO via Optuna + optional BoTorch sampler. |
| **C6** | Analyzer emits profile groups + `surge/viz/profiles.py` for density/profile plots and per-profile metrics. | No built-in multi-output scientific profiling. | Provides multi-output Kriging but not profile visualization/metrics. | SURGE stores predictions + indices for downstream plotting. |
| **C7** | `runs/<tag>/` layout with models/scalers/predictions/metrics/spec/env/git rev; reproducibility metadata baked into workflow summary. | No artifact standardization. | No artifact management. | SURGE ensures each run is reproducible with config + environment snapshot. |
| **C8** | `surge/hpc/resources.py` detects schedulers/CPUs/GPUs and exposes HPC-aware metadata without scheduler deps; hooks ready for ROSE integration. | Not HPC-aware. | Not HPC-aware. | SURGE emits resource summary per workflow + keeps hooks scheduler-agnostic. |

> Evidence references: see `docs/status.md` for full before/after tables and file pointers. This comparison only reflects capabilities implemented directly in the repository—external ecosystems may offer similar features through separate tooling.

