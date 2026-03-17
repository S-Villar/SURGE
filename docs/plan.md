# Pre-ROSE Refactor Execution Plan

> Goal: deliver a minimal-yet-robust SURGE core, demonstrate an M3D-C1 stability surrogate, and document capability deltas before/after the refactor.

## Phase 1 — Recon & Capability Baseline
- Enumerate key modules (legacy `surge/`, notebooks, scripts, tests) to understand current coverage.
- Capture C1–C8 readiness in `docs/status.md`, citing concrete files/symbols and recording gaps.
- Finalize execution guidance (this file) so future contributors can track progress.

## Phase 2 — New `src/surge` Core Skeleton
- Bootstrap modern package layout (`src/surge/__init__.py`, `core/engine.py`, `core/registry.py`).
- Define `BaseModelAdapter`, `MODEL_REGISTRY`, and a trimmed `SurrogateEngine` that focuses on book-keeping and delegates backend-specific logic.
- Add structured logging hooks so later workflow stages can collect metrics/artifacts without touching legacy engine code.

## Phase 3 — Data Layer, Artifacts, and HPC Hooks
- Port dataset loading + analyzer logic into `src/surge/data/dataset.py` and `src/surge/data/preprocessing.py`, preserving CSV/Parquet/pickle/Excel/HDF5 support and extending to NetCDF if possible.
- Author `src/surge/io/artifacts.py` to enforce the standardized runs layout (`runs/<tag>/models|scalers|predictions`) and helper functions for metrics/spec/provenance files.
- Implement `src/surge/hpc/resources.py` with lightweight `ComputeResources` detection (scheduler env vars, CPU/GPU counts, GPU model, device string) without adding scheduler dependencies.

## Phase 4 — Workflow Spec, Runner, HPO, and UQ
- Define `SurrogateWorkflowSpec` (dataset paths, model specs, splits, standardization flags, CV/HPO knobs, artifact directories, seeds) in `src/surge/workflow/spec.py`.
- Build `src/surge/workflow/run.py::run_surrogate_workflow` to orchestrate dataset ingestion → analysis → splits/scaling → model training/eval → Optuna/BoTorch HPO → artifact emission.
- Implement adapters under `src/surge/models/adapters/` (sklearn, torch, optional GPflow) with consistent `fit/predict/predict_with_uncertainty/save/load` behavior and register them with the new registry.
- Provide MC-Dropout for the torch adapter so `predict_with_uncertainty` returns mean/variance consistently across backends.
- Export HPO results (`hpo_topk.json`, best-trial config) and standard metrics/predictions into the artifact layout.

## Phase 5 — Multi-output Profiles & Visualization
- Add `src/surge/viz/profiles.py` utilities for profile grouping, per-output metrics, GT vs Pred density plots with locked axes, and profile band visualizations.
- Extend dataset analysis to surface output profile metadata (e.g., `gamma_n*` groups) so workflow summaries can reference per-profile statistics automatically.

## Phase 6 — M3D-C1 Workflow Assets
- Supply `configs/m3dc1_demo.yaml` capturing a runnable workflow spec with at least two models (RFR + FNN, optional GPR) and deterministic splits/seeds.
- Offer `examples/m3dc1_workflow.py` CLI plus module entry point (`python -m surge.workflow.run --spec ...`) that prints a capability status bar upon completion.
- Author `notebooks/M3DC1_demo.ipynb` documenting data context, analyzer output, workflow spec creation, execution, GT vs Pred/profile plots, and artifact inspection. Use real M3D-C1 data if available, otherwise synthesize a representative dataset.

## Phase 7 — Documentation & Comparison
- Update the root `README.md` with install instructions, the new workflow quickstart, and artifact discovery tips.
- Create `docs/comparison.md` summarizing SURGE vs skorch/SMT capability coverage (C1–C8) using actual code references.
- Extend `docs/status.md` with post-work table + status bar snippet for PR descriptions; also update `M3DC1_CAPABILITIES_RESULTS.md` if needed.

## Phase 8 — Tests & CI
- Add smoke/unit coverage for registry (`tests/test_registry.py`), workflow runner (`tests/test_workflow.py` using synthetic multi-output data), dataset analyzer (`tests/test_dataset.py`), artifact layout, and MC-Dropout UQ.
- Ensure pytest picks up the new suites (update `pyproject.toml`/CI config as required) and document any heavy tests to skip by default.

## Phase 9 — Status Reporting Automation
- Provide a helper (script or workflow CLI flag) that prints both pre- and post-work capability tables with file pointers, so reviewers can drop them directly into the PR description.











