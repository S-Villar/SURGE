# Prerelease status (public)

SURGE is being prepared for a **citable, installable** public snapshot (e.g. **v0.1.x**). This page states **what is ready for early adopters** and **what is not promised yet**—without requiring ML-infrastructure expertise to get value from the project.

## What you *can* rely on for a prerelease

- **End-to-end tabular workflow:** load data with `SurrogateDataset`, split and scale with `SurrogateEngine`, train **registered** models, write **run artifacts** (metrics, models, scalers, predictions) via `run_surrogate_workflow` and YAML specs.
- **Pluggable models:** the adapter registry and documented extension path for new surrogate backends.
- **Documentation and examples** aimed at **scientific users** (datasets, M3DC1-style examples, config-driven runs).

## What is *not* guaranteed in the first public tag

- **k-fold in the unified workflow** — `cv_folds` in YAML is reserved; see [ROADMAP.md](ROADMAP.md).
- **Petabyte or thousand-GPU data paths** — full-table `DataFrame` + in-memory `numpy` splits are the default; **DDStore**, sharded loaders, and distributed training are **planned** and will appear behind the same *high-level* resource knobs when implemented—not as a separate “expert only” app.
- **Every backend at every scale** — e.g. **scikit-learn** and similar CPU estimators are appropriate for many surrogates but **do not** scale to massive multi-GPU data-parallel training; future versions will **validate and warn** when resource requests are inconsistent with the selected models.

Tutorials may **name** what runs under the hood (PyTorch `DataLoader`, future DDStore, etc.) for transparency; **using** the package should not require you to know those names.

## Two ways to work (both supported)

- **Config-first:** YAML + `run_surrogate_workflow` for quick comparisons and adoption.
- **Code-first:** `SurrogateEngine`, `ModelSpec`, and adapters **directly** for full control—the “easy” path is optional, not exclusive.

## Where the detailed plan lives

- **Single checklist** for a public URL, history cleanup (`git filter-repo`), and what was removed from the tree: [PUBLIC_OPEN_SOURCE_PLAN.md](PUBLIC_OPEN_SOURCE_PLAN.md).
- **Public priorities and honest gaps:** [ROADMAP.md](ROADMAP.md).
- **Deeper product planning** (DDStore, ezpz, resources): [dev/EXTENDED_ROADMAP.md](dev/EXTENDED_ROADMAP.md) — *optional reading for maintainers.*
