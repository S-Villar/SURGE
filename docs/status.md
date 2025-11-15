# SURGE Capability Status (Pre-work)

| Capability | Readiness (0–5) | Evidence (files/symbols) | Gaps / Next Steps |
|-----------:|:---------------:|--------------------------|-------------------|
| C1 | 2 | `surge/engine.py` (legacy), scattered scripts (`examples/unified_surrogate_engine_example.py`) | No declarative workflow spec or artifact layout; training tightly coupled to monolithic engine. |
| C2 | 3 | `surge/dataset.py`, `surge/preprocessing.py` | Regex-based detection only, limited metadata overrides, missing NetCDF + richer profiling info. |
| C3 | 3 | `surge/model/*`, `surge/models.py` shim | Registry exists but adapter API inconsistent; uncertainty/save/load behavior varies per backend. |
| C4 | 2 | `surge/model/gpflow.py`, `surge/model/pytorch_impl.py` ensembles | Only GP variants offer UQ; MC-Dropout/ensembles not wired into workflow, metrics ad-hoc. |
| C5 | 1 | `surge/engine.py` (Optuna hooks), assorted demos | No workflow spec, no standardized artifacts, Optuna integration not reusable. |
| C6 | 2 | `surge/visualization.py`, notebooks | Multi-output plots exist but no grouped profile metrics/plots or analyzer integration. |
| C7 | 1 | `runs/` samples, helper scripts | No canonical runs layout, provenance files, or spec/env snapshots. |
| C8 | 1 | `templates/*.perlmutter`, batch scripts | No Python-level resource detection; only shell helpers for schedulers. |

Legend: 0=Missing, 1=Stub, 2=Prototype, 3=Partial, 4=Ready, 5=Polished

---

# SURGE Capability Status (Post-work)

| Capability | Before | After | Evidence (links) | Notes |
|-----------:|:------:|:-----:|------------------|-------|
| C1 | 2 | 4 | `surge/engine.py`, `surge/workflow/run.py`, `surge/io/artifacts.py` | Unified engine + workflow spec produce reproducible runs and artifacts. |
| C2 | 3 | 4 | `surge/dataset.py`, `surge/preprocessing.py`, `data/datasets/SPARC/m3dc1_metadata.yaml` | Auto-detection with metadata overrides + multi-format loaders, profile grouping. |
| C3 | 3 | 4 | `surge/registry.py`, `surge/models/adapters/*` | Consistent adapter API across sklearn/torch/gpflow with registry summary + artifacts. |
| C4 | 2 | 4 | `surge/models/adapters/torch_adapter.py`, `surge/models/adapters/gpflow_adapter.py` | MC-Dropout + GP mean/variance exported via unified `predict_with_uncertainty`. |
| C5 | 1 | 4 | `surge/workflow/spec.py`, `surge/workflow/run.py`, `examples/m3dc1_workflow.py` | One-call workflow, Optuna/Botorch HPO, metrics/predictions/spec/env captured. |
| C6 | 2 | 3 | `surge/viz/profiles.py`, `notebooks/M3DC1_demo.ipynb` | Per-profile metrics + plots with consistent axes; ready for richer datasets. |
| C7 | 1 | 4 | `surge/io/artifacts.py`, `runs/<tag>/...` | Standard run layout (models/scalers/predictions/metrics/spec/env/git). |
| C8 | 1 | 3 | `surge/hpc/resources.py`, workflow summary | Scheduler/CPU/GPU detection recorded per run; still no submission hooks. |

Use this table in the PR description’s status bar.
