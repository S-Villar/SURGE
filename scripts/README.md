# `scripts/` — optional and application-specific

Most **reusable** library code lives under `surge/`. This folder holds **ad hoc**
utilities (batch jobs, M3D-C1 helpers, HPC launchers) that are **not** part of
the importable `surge` API.

- **M3D-C1 / SPARC** batch pipelines and interfaces live under `scripts/m3dc1/`.
- **XGC-oriented** one-off scripts were removed from the public-release tree;
  develop those privately or in a separate application repo if they contain
  collaboration-specific paths or data assumptions.

For installable, documented behavior, use **`surge`** from Python or the
**YAML workflow** (`run_surrogate_workflow`). See `docs/SURGE_OVERVIEW.md`.
