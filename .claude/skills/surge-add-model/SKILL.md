---
name: surge-add-model
description: Add a new model adapter to SURGE (backend + adapter + registration + tests + benchmark wiring). Use when asked to integrate a new ML model/architecture into the registry.
---

# Adding a model adapter

Layout convention (follow existing pairs, e.g. residual_mlp):
- `surge/model/backends/<name>.py` — the actual implementation (torch
  module + training loop, or wrapper around an external lib).
- `surge/model/adapters/<name>.py` — thin `BaseModelAdapter` subclass
  (`surge/model/base.py`): implement `fit`, `predict`, optionally
  `predict_with_uncertainty`, `save`/`load`; set `name`, `backend`,
  class-level defaults in `__init__` kwargs; call `self.mark_fitted()`.

Registration in `surge/model/__init__.py`:
- key format `<backend>.<model>` (e.g. `pytorch.tcn`), a few short aliases;
- optional heavy deps are guarded — keep guards to Import/OSError only and
  NEVER let a guard swallow a SURGE bug silently; verify the model appears:
  `python -c "from surge.model import list_models; print(sorted(list_models()))"`.

Checklist before claiming done:
1. `adapter.fit(X, y)` + `predict` round-trip on random arrays, 1D and 2D y.
2. save/load round-trip produces identical predictions (or document why
   serialization is unsupported).
3. Tests in `tests/` mirroring `tests/benchmarks/test_phase2_adapters.py`:
   registration test + fit/predict smoke, with
   `pytest.importorskip("<dep>")` guards so a missing optional dep SKIPS
   (never fails) — include an `OSError` guard for native-lib deps
   (lightgbm/xgboost-style dlopen failures).
4. If benchmark-eligible: add the key to the model lists in
   `surge/benchmarks/leaderboard.py` (`_TABULAR_MODELS` etc.) and, for HPO,
   a search space in `surge/benchmarks/hpo.py`.
5. `ruff check` clean on new files; run
   `pytest -q tests/test_models.py tests/benchmarks -k <name>`.
6. Resource semantics: declare a `ResourceProfile` if the backend has
   GPU/n_jobs behavior (see `surge/model/sklearn.py` examples).

Params reach the adapter via `MODEL_REGISTRY.create(key, **params)` — keep
every hyperparameter a keyword argument with a sensible default so HPO and
YAML specs can set them.
