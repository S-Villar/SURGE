# SURGE Model Zoo, Tests & Benchmarks — Implementation Plan

**Repository:** https://github.com/S-Villar/SURGE  
**Current version:** v0.1.0 (52 passed, 1 skipped, 0 failed in CI)  
**Grounded in:** `docs/ROADMAP.md`, `docs/REFACTORING_PLAN.md`, `docs/SURGE_OVERVIEW.md`

---

## 1. Where the Repo Actually Stands Today

### What exists and is stable
| Component | Status |
|---|---|
| `SurrogateEngine` + `SurrogateWorkflowSpec` | ✅ Stable, YAML-driven |
| `surge.registry.MODEL_REGISTRY` (canonical) | ✅ Stable |
| `sklearn.random_forest` adapter | ✅ Registered, tested |
| `pytorch.mlp` adapter (MC-Dropout UQ) | ✅ Registered, ONNX export |
| `gpflow.gpr` adapter | ✅ Registered |
| `ResourceSpec` + fit-banner logging | ✅ Landed in v0.1.0 |
| M3DC1 dataset (tokamak growth-rate, HDF5) | ✅ In repo under `data/` |
| Artifact layout (`runs/<tag>/`) | ✅ Stable |
| CI: test / e2e-regression / lint | ✅ 3-job GitHub Actions |

### Confirmed technical debt (from `REFACTORING_PLAN.md`)
| Issue | Section | Target |
|---|---|---|
| Two model registries (`surge/registry.py` vs `surge/model/registry.py`) | §1.1 | v0.2.0 |
| `pytorch.py` vs `pytorch_impl.py` naming | §1.2 | v0.2.0 |
| `surge/__init__.py` sys.path hack for M3DC1 loader | §1.5 | Before v0.1.0 tag |
| `surge/legacy/` excluded from wheel but still on disk | §1.4 | v0.2.0 |
| 38 tests currently skipped (legacy API signatures) | §1.9 | v0.2.0 |
| `cv_folds > 0` accepted but ignored | ROADMAP §1 | v0.2.0 |

**Policy: do not break CI green. Do not touch the two-registry issue until v0.2.0.  
Work within the current `surge.registry.MODEL_REGISTRY` + adapter pattern.**

---

## 2. Agent Roles — Mapped to Actual Repo

### Agent 1 — Architect / Planner  
**Owns:** `docs/`, naming decisions, acceptance criteria, PR review  
**Do not duplicate:** existing `docs/SURGE_OVERVIEW.md`, `REFACTORING_PLAN.md`, `ROADMAP.md`  
**New deliverables:**
- `docs/model_zoo_plan.md` — taxonomy table + acceptance checklist per model class
- `docs/benchmark_policy.md` — dataset ladder from `REFACTORING_PLAN.md §4`, smoke/standard/full tiers
- `surge/models/README.md` — input/output shape requirements per adapter
- `surge/benchmarks/README.md` — benchmark registry contract

### Agent 2 — Model Integrator  
**Constraint:** all new adapters must register via `surge.registry.MODEL_REGISTRY.register()` — the same singleton already used by RF, MLP, GPR.  
**Constraint:** do not rename `pytorch.py`/`pytorch_impl.py` until v0.2.0.  
**Adapter interface contract** (matches existing `BaseModelAdapter`):

```python
class BaseModelAdapter:
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    # Optional but preferred:
    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]: ...
    def export_onnx(self, path: Path, input_shape: tuple) -> None: ...
```

**Implementation order** (simple → complex):

| Priority | Key | File | Notes |
|---|---|---|---|
| P1 | `sklearn.gradient_boosting` | `surge/model/adapters/sklearn.py` | Add alongside RF; trivial |
| P1 | `sklearn.gaussian_process` | same | sklearn GP, not GPflow |
| P2 | `pytorch.residual_mlp` | `surge/model/backends/residual_mlp.py` | ResidualMLP on top of existing torch training loop |
| P2 | `pytorch.cnn1d` | `surge/model/backends/cnn.py` | 1D CNN for sequence inputs |
| P3 | `pytorch.lstm` | `surge/model/backends/rnn.py` | LSTM for temporal data |
| P3 | `pytorch.autoencoder` | `surge/model/backends/autoencoder.py` | For field reconstruction |
| P4 | `pytorch.fno` | `surge/model/backends/fno.py` | Fourier Neural Operator |
| P4 | `pytorch.deeponet` | `surge/model/backends/deeponet.py` | Trunk/branch architecture |
| P5 | `pytorch.transformer` | `surge/model/backends/transformer.py` | TransformerEncoder surrogate |

**Minimal adapter template** (use existing `pytorch.py` as the reference):

```python
# surge/model/adapters/residual_mlp.py
from surge.registry import MODEL_REGISTRY
from surge.model.base import BaseModelAdapter
from surge.model.backends.residual_mlp import ResidualMLPModel

class ResidualMLPAdapter(BaseModelAdapter):
    KEY = "pytorch.residual_mlp"
    BACKEND = "pytorch"
    TAGS = ["neural", "dense", "residual"]

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kw):
        self._emit_fit_banner(X_train)   # prints [surge.fit] line
        self._model = ResidualMLPModel(**self._params)
        self._model.fit(X_train, y_train, X_val, y_val)

    def predict(self, X):
        return self._model.predict(X)

    def save(self, path):
        self._model.save(path)

    def load(self, path):
        self._model = ResidualMLPModel.load(path)

MODEL_REGISTRY.register(ResidualMLPAdapter)
```

### Agent 3 — Benchmark Engineer  
**Dataset ladder** (from `REFACTORING_PLAN.md §4` — use exactly these rungs):

| Rung | Key | Shape | Source | Tier |
|---|---|---|---|---|
| 1 | `synthetic.regression_1d` | 1→1 | inline fixture | smoke |
| 2 | `tabular.diabetes` | 10→1 | `sklearn.datasets` | smoke |
| 3 | `tabular.california_housing` | 8→1 | `sklearn.datasets` | standard |
| 4 | `tabular.energy_efficiency` | 8→2 | UCI (fetch via `sklearn`) | standard |
| 5 | `fusion.m3dc1_sample` | 13→1 | `data/datasets/M3DC1/` | standard |
| 6 | `sequence.lorenz63` | synthetic | inline generator | smoke |
| 7 | `pde.burgers_1d` | synthetic | inline solver | standard |

**Benchmark YAML schema** (extend existing `configs/*.yaml` pattern):

```yaml
# surge/benchmarks/tabular/california_housing.yaml
benchmark:
  key: tabular.california_housing
  tier: standard          # smoke | standard | full
  runtime_budget_min: 10
  seed: 42
  dataset:
    loader: sklearn.california_housing
    inputs: all_except_last
    outputs: [MedHouseVal]
  splits:
    train: 0.6
    val: 0.2
    test: 0.2
  baseline_model: sklearn.random_forest
  expected_metrics:
    test_r2: {min: 0.75, max: 1.0}
    test_rmse: {max: 0.6}
  cli_command: >
    python -m surge.benchmarks.run
      --benchmark tabular.california_housing
      --model sklearn.random_forest
      --output benchmark_reports/california_housing/
```

**Benchmark registry** (new file `surge/benchmarks/registry.py`):

```python
# mirrors surge/registry.py pattern
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class BenchmarkEntry:
    key: str
    tier: str          # smoke | standard | full
    config_path: Path
    baseline_model: str
    expected_metrics: dict

class BenchmarkRegistry:
    def __init__(self):
        self._benchmarks: dict[str, BenchmarkEntry] = {}

    def register_from_yaml(self, path: Path) -> BenchmarkEntry: ...
    def get(self, key: str) -> BenchmarkEntry: ...
    def list(self, tier: str | None = None) -> list[BenchmarkEntry]: ...

BENCHMARK_REGISTRY = BenchmarkRegistry()
```

**CLI runner** (`surge/benchmarks/run.py`):
```
python -m surge.benchmarks.run \
    --benchmark tabular.california_housing \
    --model sklearn.random_forest \
    --output benchmark_reports/
```
Must write `benchmark_reports/<key>/<timestamp>/result.json` with:
```json
{
  "benchmark": "tabular.california_housing",
  "model": "sklearn.random_forest",
  "tier": "standard",
  "seed": 42,
  "metrics": {"test_r2": 0.82, "test_rmse": 0.47},
  "passed": true,
  "runtime_s": 14.3
}
```

### Agent 4 — Test / Verification Agent  
**Critical constraint:** do not re-enable the 38 skipped legacy tests — those require an API migration (v0.2.0 work per `REFACTORING_PLAN.md §1.9`).  
**Work only in `tests/unit/`, `tests/integration/`, `tests/benchmarks/`.**

**Unit test matrix per new adapter:**

```python
# tests/unit/test_new_adapters.py
import pytest, numpy as np
from surge.registry import MODEL_REGISTRY

ADAPTER_KEYS = [
    "pytorch.residual_mlp",
    "pytorch.cnn1d",
    "pytorch.lstm",
    # add as implemented
]

@pytest.fixture
def tiny_regression():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 8))
    y = rng.standard_normal((64, 1))
    return X[:50], y[:50], X[50:], y[50:]

@pytest.mark.parametrize("key", ADAPTER_KEYS)
def test_adapter_registered(key):
    assert MODEL_REGISTRY.get(key) is not None

@pytest.mark.parametrize("key", ADAPTER_KEYS)
def test_fit_predict_shape(key, tiny_regression):
    X_tr, y_tr, X_val, y_val = tiny_regression
    model = MODEL_REGISTRY.create(key)
    model.fit(X_tr, y_tr, X_val, y_val)
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape

@pytest.mark.parametrize("key", ADAPTER_KEYS)
def test_loss_decreases(key, tiny_regression):
    # verify training_history shows decreasing loss
    X_tr, y_tr, X_val, y_val = tiny_regression
    model = MODEL_REGISTRY.create(key, params={"epochs": 5})
    model.fit(X_tr, y_tr, X_val, y_val)
    hist = model.training_history()
    if hist:  # not all adapters expose this
        losses = [h["train_loss"] for h in hist]
        assert losses[-1] < losses[0], "Loss did not decrease"

@pytest.mark.parametrize("key", ADAPTER_KEYS)
def test_save_load_roundtrip(key, tiny_regression, tmp_path):
    X_tr, y_tr, X_val, y_val = tiny_regression
    model = MODEL_REGISTRY.create(key)
    model.fit(X_tr, y_tr)
    preds_before = model.predict(X_val)
    model.save(tmp_path / f"{key}.joblib")
    model2 = MODEL_REGISTRY.create(key)
    model2.load(tmp_path / f"{key}.joblib")
    preds_after = model2.predict(X_val)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)
```

**Smoke integration test** (must stay under 2 minutes on CPU):

```python
# tests/benchmarks/test_smoke_benchmarks.py
import pytest
from surge.benchmarks.run import run_benchmark

@pytest.mark.parametrize("benchmark_key,model_key", [
    ("synthetic.regression_1d", "sklearn.random_forest"),
    ("synthetic.regression_1d", "pytorch.mlp"),
    ("tabular.diabetes", "sklearn.random_forest"),
    ("sequence.lorenz63", "pytorch.lstm"),
])
def test_smoke_benchmark(benchmark_key, model_key, tmp_path):
    result = run_benchmark(benchmark_key, model_key, output_dir=tmp_path, seed=0)
    assert result["passed"] or result["metrics"]["test_r2"] > -1.0  # finite
    assert (tmp_path / "result.json").exists()
```

**Determinism test:**

```python
def test_deterministic_pytorch_mlp(tiny_regression):
    X_tr, y_tr, X_val, _ = tiny_regression
    results = []
    for _ in range(2):
        m = MODEL_REGISTRY.create("pytorch.mlp", params={"epochs": 3, "seed": 42})
        m.fit(X_tr, y_tr)
        results.append(m.predict(X_val))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-5)
```

### Agent 5 — Documentation / Examples Agent  
**Work in `docs/` and `examples/`. Do not modify README.md without co-author review.**

New deliverables:
- `docs/model_zoo_plan.md` — when to use each architecture (table format)
- `examples/model_zoo_quickstart.py` — run any registered model on California Housing
- `examples/benchmark_runner.py` — CLI wrapper showing smoke → standard progression
- `docs/benchmark_results_template.md` — blank table for filling in after runs

**Architecture comparison table** (to include in `docs/model_zoo_plan.md`):

| Architecture | Key | Input type | UQ | ONNX | When to use |
|---|---|---|---|---|---|
| Random Forest | `sklearn.random_forest` | tabular | tree variance | ❌ | fast baseline, any shape |
| MLP | `pytorch.mlp` | tabular | MC-Dropout | ✅ | nonlinear regression |
| Residual MLP | `pytorch.residual_mlp` | tabular | MC-Dropout | ✅ | deeper, skip connections |
| GPR | `gpflow.gpr` | tabular (<5k rows) | exact posterior | ❌ | small data, calibrated UQ |
| CNN1D | `pytorch.cnn1d` | 1D fields/sequences | — | ✅ | spatial 1D surrogates |
| LSTM | `pytorch.lstm` | time series | — | ✅ | temporal rollout |
| Autoencoder | `pytorch.autoencoder` | fields | — | ✅ | latent compression |
| FNO | `pytorch.fno` | 1D/2D fields | — | ✅ | PDE operator learning |
| DeepONet | `pytorch.deeponet` | function → function | — | ✅ | physics operator learning |
| Transformer | `pytorch.transformer` | patches / tokens | — | ✅ | attention over field data |

---

## 3. Recommended File Layout (additions only)

Files that do **not** yet exist and need to be created. Existing files in `surge/model/` are untouched.

```
surge/
  benchmarks/               ← NEW
    __init__.py
    registry.py             ← BenchmarkRegistry + BENCHMARK_REGISTRY singleton
    base.py                 ← BenchmarkResult dataclass, run_benchmark()
    run.py                  ← CLI entry point
    tabular/
      california_housing.yaml
      energy_efficiency.yaml
      diabetes.yaml
    sequence/
      lorenz63.yaml
    pde/
      burgers_1d.yaml
    fusion/
      m3dc1_sample.yaml     ← wraps existing data/datasets/M3DC1/

  model/
    backends/               ← NEW (v0.2.0 rename of *_impl.py files; stubs only now)
      residual_mlp.py
      cnn.py
      rnn.py
      autoencoder.py
      fno.py
      deeponet.py
      transformer.py
    adapters/               ← NEW stubs; full rename in v0.2.0
      residual_mlp.py       ← registers pytorch.residual_mlp
      cnn.py                ← registers pytorch.cnn1d, pytorch.cnn2d
      rnn.py                ← registers pytorch.lstm, pytorch.gru
      autoencoder.py        ← registers pytorch.autoencoder, pytorch.vae
      fno.py                ← registers pytorch.fno
      deeponet.py           ← registers pytorch.deeponet
      transformer.py        ← registers pytorch.transformer

tests/
  unit/
    test_new_adapters.py    ← parametrized over all new keys
    test_benchmark_registry.py
  integration/
    test_new_model_workflows.py  ← train/save/load/predict round-trips
  benchmarks/
    test_smoke_benchmarks.py

docs/
  model_zoo_plan.md
  benchmark_policy.md

examples/
  model_zoo_quickstart.py
  benchmark_runner.py
```

---

## 4. Benchmark Acceptance Checklist

Each new benchmark PR must include all of:

- [ ] YAML config under `surge/benchmarks/<tier>/<key>.yaml`
- [ ] Dataset loader that works offline (or via `sklearn.datasets` / small bundled fixture)
- [ ] `baseline_model` entry that already exists in `MODEL_REGISTRY`
- [ ] `expected_metrics` with realistic bounds (run once manually to calibrate)
- [ ] `tier` set to `smoke` if runtime < 2 min on CPU, else `standard`
- [ ] `seed` set to 42
- [ ] CLI command that runs end-to-end and writes `result.json`
- [ ] Passing entry in `tests/benchmarks/test_smoke_benchmarks.py`

---

## 5. Model Acceptance Checklist

Each new adapter PR must include all of:

- [ ] Registered via `MODEL_REGISTRY.register()` in the adapter module
- [ ] `fit(X_train, y_train, X_val=None, y_val=None)` — no model-specific training loop bypass
- [ ] `predict(X)` returning correct shape
- [ ] `save(path)` / `load(path)` round-trip
- [ ] `_emit_fit_banner()` called at the start of `fit()` (inherited from `BaseModelAdapter`)
- [ ] Unit test: registered, fit/predict shape, save/load round-trip, loss decreases
- [ ] At least one benchmark config that uses this model as baseline or candidate
- [ ] Docstring: expected input shape, output shape, hyperparameters
- [ ] ONNX export tested if `pytorch` backend (add to `tests/test_e2e_release_smoke.py`)

---

## 6. Implementation Phases (sequenced to stay CI-green)

### Phase A — Benchmark Infrastructure (no new models, no CI risk)
**Goal:** `python -m surge.benchmarks.run --benchmark tabular.diabetes --model sklearn.random_forest` works end to end.

Tasks:
1. Create `surge/benchmarks/registry.py` — `BenchmarkRegistry`, `BENCHMARK_REGISTRY`
2. Create `surge/benchmarks/base.py` — `BenchmarkResult`, `run_benchmark(key, model_key, output_dir, seed)`
3. Create `surge/benchmarks/run.py` — CLI using `argparse`
4. Add YAML configs: `diabetes.yaml`, `california_housing.yaml`, `synthetic_1d.yaml`
5. Add `tests/benchmarks/test_smoke_benchmarks.py` — 4 parametrized smoke cases
6. Add `tests/unit/test_benchmark_registry.py`

**Success:** `pytest tests/benchmarks/ tests/unit/test_benchmark_registry.py -q` → all green, no skips.

### Phase B — Dense Neural Adapters
**Goal:** `pytorch.residual_mlp` runs through the same `SurrogateWorkflowSpec` as `pytorch.mlp`.

Tasks:
1. `surge/model/backends/residual_mlp.py` — `ResidualBlock`, `ResidualMLP(nn.Module)`, reuse existing training loop from `pytorch_impl.py`
2. `surge/model/adapters/residual_mlp.py` — `ResidualMLPAdapter`, self-registers on import
3. Import in `surge/__init__.py` under `try/except ImportError` guard (same pattern as existing torch adapter)
4. `tests/unit/test_new_adapters.py` with `pytorch.residual_mlp`
5. Add `tabular/california_housing.yaml` benchmark with `candidate_models: [pytorch.residual_mlp]`

### Phase C — Temporal Adapters (LSTM / GRU)
**Goal:** `pytorch.lstm` runs on `sequence.lorenz63` benchmark.

Tasks:
1. `surge/model/backends/rnn.py` — `LSTMSurrogate(nn.Module)`, `GRUSurrogate(nn.Module)`
2. `surge/model/adapters/rnn.py` — registers `pytorch.lstm`, `pytorch.gru`
3. `surge/benchmarks/sequence/lorenz63.yaml` — inline Lorenz-63 data generator
4. Tests: shape, save/load, rollout error metric logged

### Phase D — Scientific Operator Adapters (FNO, DeepONet)
**Goal:** `pytorch.fno` runs on `pde.burgers_1d` benchmark with relative L2 metric.

Tasks:
1. `surge/model/backends/fno.py` — SpectralConv1d, FNO1d (minimal, self-contained)
2. `surge/model/backends/deeponet.py` — BranchNet + TrunkNet
3. Adapters: `pytorch.fno`, `pytorch.deeponet`
4. `surge/benchmarks/pde/burgers_1d.yaml` — inline Burgers solver fixture
5. Metrics: add `relative_l2` to `surge/metrics.py` (currently only R², RMSE, MAE, MAPE)
6. Tests: same checklist as Phase B

### Phase E — Documentation & Examples Pass
**Goal:** every architecture has a usage example and a table in docs.

Tasks:
1. `docs/model_zoo_plan.md` — architecture table, when-to-use, input/output shape guide
2. `docs/benchmark_policy.md` — tier definitions, acceptance checklist
3. `examples/model_zoo_quickstart.py` — loops over all registered models on California Housing, prints results table
4. `examples/benchmark_runner.py` — runs all smoke benchmarks, writes summary CSV

---

## 7. What NOT to Do

These are explicitly out of scope until v0.2.0, per `REFACTORING_PLAN.md`:

- Do not merge `surge/registry.py` and `surge/model/registry.py` — that is Phase 2 of the refactoring plan.
- Do not rename `pytorch.py` → `adapters/torch.py` or `pytorch_impl.py` → `backends/torch_mlp.py` — those are v0.2.0 renames.
- Do not re-enable or rewrite the 38 currently-skipped legacy tests — they need the API migration first.
- Do not add large datasets to the repo — use `sklearn.datasets` fetchers or small inline fixtures only.
- Do not hardcode dataset paths — use the existing `SurrogateDataset.from_path()` with environment-variable overrides.
- Do not bypass `MODEL_REGISTRY` — every model goes through `MODEL_REGISTRY.register()` and `MODEL_REGISTRY.create()`.
- Do not write model-specific training loops in adapter files — keep backends separate.

---

## 8. Definition of Done (per PR)

A PR adding a model + benchmark is complete when:

- `pytest -q` → 0 failed, 0 new skips beyond the existing 1
- At least one smoke benchmark for the new model runs in < 2 min on CPU
- `benchmark_reports/<key>/result.json` is written and `passed: true`
- The model is in `MODEL_REGISTRY` (verify with `python -c "from surge import MODEL_REGISTRY; print(MODEL_REGISTRY.list())"`)
- The benchmark is in `BENCHMARK_REGISTRY`
- The adapter has docstring with input/output shape requirements
- ONNX round-trip test passes (PyTorch adapters only)
- `docs/model_zoo_plan.md` table updated
