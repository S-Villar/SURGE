# SURGE Benchmark Expansion Plan
## Regression + Classification, Grounded in v0.1.0

---

## What Exists Today (v0.1.0)

SURGE has **no `surge/benchmarks/` module yet**. What exists is:

| Asset | Where | Notes |
|---|---|---|
| `configs/m3dc1_demo.yaml` | `configs/` | Workflow spec, not a benchmark definition |
| `configs/m3dc1_demo_augmented.yaml` | `configs/` | Same — spec, not benchmark |
| `tests/test_e2e_release_smoke.py` | `tests/` | E2E smoke on a tracked sample CSV |
| M3DC1 dataset (tokamak, 9,981 rows, 13→1) | `data/datasets/` | Regression only |
| Metrics: R², RMSE, MAE, MAPE | `surge/metrics.py` | **Regression only** |
| Models: RF, MLP (MC-Dropout), GPR | `surge/registry.py` | **Regression only** |

**Confirmed gap:** no benchmark registry, no classification metrics, no classification models, no benchmark runner CLI.

---

## Expansion Strategy

### Guiding principle for a scientific surrogate framework

Classification in SURGE serves two distinct roles:

1. **Direct surrogate classification** — the simulation output is itself categorical (e.g. "stable vs unstable plasma regime", "material phase", "flow regime label"). This is the primary use case.
2. **Auxiliary diagnostic classification** — classify surrogate quality or input regime before routing to a specialized regression surrogate. Secondary, but valuable.

Classification benchmarks should be grounded in **scientific or engineering datasets** wherever possible — not generic toy datasets.

---

## Part 1 — Regression Benchmarks (Complete the Existing Ladder)

From `REFACTORING_PLAN.md §4`, these rungs are specified but not yet implemented:

### Tier: smoke (< 2 min on CPU)

| Key | Dataset | Shape | Source | Expected R² | Notes |
|---|---|---|---|---|---|
| `synthetic.regression_1d` | inline `y = sin(x) + ε` | 1→1 | fixture | > 0.90 | Hermetic CI smoke |
| `tabular.diabetes` | sklearn Diabetes | 10→1 | `sklearn.datasets` | > 0.40 | Classical tabular |
| `synthetic.multioutput_2d` | inline `y = Ax + ε` | 8→2 | fixture | > 0.80 | Multi-output smoke |

### Tier: standard (< 30 min on CPU)

| Key | Dataset | Shape | Source | Expected R² | Notes |
|---|---|---|---|---|---|
| `tabular.california_housing` | California Housing | 8→1 | `sklearn.datasets` | > 0.75 | Canonical tabular |
| `tabular.energy_efficiency` | UCI Energy Efficiency | 8→2 | `sklearn.datasets` via fetch | > 0.90 | Multi-output, heating + cooling |
| `tabular.concrete_strength` | UCI Concrete | 8→1 | `sklearn.datasets` | > 0.80 | Nonlinear, good for neural models |
| `tabular.airfoil_noise` | NASA Airfoil | 5→1 | UCI | > 0.85 | Aerodynamics surrogate |
| `tabular.yacht_dynamics` | UCI Yacht Hydro | 6→1 | UCI | > 0.95 | CFD surrogate, highly nonlinear |
| `multioutput.scm20d` | SCM20d (20 targets) | 61→20 | OpenML | > 0.60 | Multi-target regression |
| `fusion.m3dc1_sample` | M3DC1 (tokamak) | 13→1 | `data/datasets/` | > 0.85 | Existing SURGE dataset |
| `sequence.lorenz63` | Lorenz-63 | 3→3 (rollout) | inline ODE solver | NRMSE < 0.1 | Temporal rollout |
| `pde.burgers_1d` | Burgers 1D | 64→64 (field) | inline solver | rel. L2 < 0.05 | PDE operator |

### Tier: full (GPU / HPC)

| Key | Dataset | Shape | Source | Notes |
|---|---|---|---|---|
| `pde.navier_stokes_2d` | NS vorticity | 64×64→64×64 | FNO paper data | Operator learning |
| `fusion.m3dc1_full` | M3DC1 full 9,981 rows | 13→1 | `data/datasets/` | Full HPO run |

---

## Part 2 — Classification Benchmarks (New Capability)

### What changes in the codebase

**New metrics needed in `surge/metrics.py`:**
```python
# Currently missing — must add
def accuracy(y_true, y_pred): ...
def f1_score(y_true, y_pred, average="macro"): ...
def auroc(y_true, y_prob): ...           # needs predict_proba
def log_loss(y_true, y_prob): ...
def confusion_matrix_summary(y_true, y_pred): ...
```

**New adapter capability needed:**
All classification adapters must support `predict_proba(X)` returning class probabilities, in addition to `predict(X)` returning class labels. This is needed for AUROC and log-loss.

**`task_type` field needed in benchmark YAML:**
```yaml
benchmark:
  key: classification.breast_cancer
  task_type: classification   # NEW — "regression" | "classification"
  ...
  expected_metrics:
    test_accuracy: {min: 0.92}
    test_f1_macro: {min: 0.91}
    test_auroc: {min: 0.97}
```

The `SurrogateEngine` must route to classification metrics when `task_type: classification`.

---

### Tier: smoke (< 2 min on CPU)

| Key | Dataset | Shape | Classes | Source | Expected Acc | Scientific relevance |
|---|---|---|---|---|---|---|
| `classification.synthetic_binary` | inline `make_classification` | 20→1 | 2 | fixture | > 0.85 | Hermetic CI smoke |
| `classification.iris` | Iris | 4→1 | 3 | `sklearn.datasets` | > 0.95 | Multiclass sanity check |
| `classification.breast_cancer` | Wisconsin BC | 30→1 | 2 | `sklearn.datasets` | > 0.95 | Medical binary classification |

### Tier: standard (< 30 min on CPU)

| Key | Dataset | Shape | Classes | Source | Expected Acc | Scientific relevance |
|---|---|---|---|---|---|---|
| `classification.wine_quality_binary` | UCI Wine (good/bad) | 11→1 | 2 | `sklearn.datasets` | > 0.78 | Materials characterization analog |
| `classification.plasma_stability` | UCI Electrical Grid Stability | 12→1 | 2 | UCI | > 0.97 | **Direct fusion / grid surrogate** |
| `classification.material_phase` | UCI Superconductor (binarized Tc) | 81→1 | 2 | UCI | > 0.80 | Materials science surrogate |
| `classification.fault_detection` | SECOM semiconductor | 590→1 | 2 | UCI | > 0.92 | Manufacturing process classification |
| `classification.flow_regime` | synthetic Mach/Re grid | 4→1 | 4 | inline fixture | > 0.90 | CFD regime classifier |
| `classification.digits` | Optical digits | 64→1 | 10 | `sklearn.datasets` | > 0.97 | Multiclass, dense features |
| `classification.covertype_binary` | Forest Covertype (2 classes) | 54→1 | 2 | `sklearn.datasets` | > 0.85 | Larger tabular, stress test |

### Tier: full

| Key | Dataset | Shape | Classes | Source | Notes |
|---|---|---|---|---|---|
| `classification.higgs_boson` | HIGGS (subset) | 28→1 | 2 | UCI / OpenML | Particle physics — natural SURGE use case |
| `classification.exoplanet_transit` | Kepler light curves (binarized) | 3197→1 | 2 | NASA / Kaggle | Astrophysics binary detection |

---

## Part 3 — Benchmark YAML Schema (with `task_type`)

```yaml
# surge/benchmarks/classification/plasma_stability.yaml
benchmark:
  key: classification.plasma_stability
  tier: standard
  task_type: classification          # routes to classification metrics
  runtime_budget_min: 5
  seed: 42

  dataset:
    loader: uci.electrical_grid_stability
    # fetched via: https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data
    # or: sklearn fetch_openml(name='electrical-grid-stability', version=1)
    inputs: [tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]
    outputs: [stabf]   # "stable" or "unstable" — binary string labels
    label_encoding: auto

  splits:
    train: 0.60
    val: 0.20
    test: 0.20

  baseline_model: sklearn.random_forest_classifier
  candidate_models:
    - sklearn.gradient_boosting_classifier
    - pytorch.mlp_classifier

  expected_metrics:
    test_accuracy: {min: 0.95}
    test_f1_macro: {min: 0.94}
    test_auroc:    {min: 0.98}

  cli_command: >
    python -m surge.benchmarks.run
      --benchmark classification.plasma_stability
      --model sklearn.random_forest_classifier
      --output benchmark_reports/plasma_stability/
```

---

## Part 4 — New Adapters Required for Classification

### Minimal set to add (all build on existing sklearn/pytorch infrastructure):

| Registry key | Class | Backend | Notes |
|---|---|---|---|
| `sklearn.random_forest_classifier` | `RandomForestClassifier` | sklearn | Add alongside existing RF regressor |
| `sklearn.gradient_boosting_classifier` | `GradientBoostingClassifier` | sklearn | |
| `sklearn.logistic_regression` | `LogisticRegression` | sklearn | Linear baseline |
| `pytorch.mlp_classifier` | MLP + softmax head | pytorch | Reuse existing training loop, change output activation + loss to CrossEntropy |

**Template: classification adapter wrapping existing sklearn adapter:**

```python
# surge/model/adapters/sklearn_classifiers.py
from sklearn.ensemble import RandomForestClassifier as _RFC
from surge.registry import MODEL_REGISTRY
from surge.model.base import BaseModelAdapter

class RandomForestClassifierAdapter(BaseModelAdapter):
    KEY = "sklearn.random_forest_classifier"
    BACKEND = "sklearn"
    TAGS = ["classical", "ensemble", "classification"]
    TASK = "classification"          # NEW field on BaseModelAdapter

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kw):
        self._emit_fit_banner(X_train)
        params = {k: v for k, v in self._params.items()}
        self._model = _RFC(**params)
        self._model.fit(X_train, y_train)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):          # required for AUROC, log-loss
        return self._model.predict_proba(X)

    def save(self, path):
        import joblib; joblib.dump(self._model, path)

    def load(self, path):
        import joblib; self._model = joblib.load(path)

MODEL_REGISTRY.register(RandomForestClassifierAdapter)
```

**PyTorch MLP classifier — minimal change from existing MLP:**

The existing `pytorch_impl.py` MLP uses `MSELoss`. Classification requires:
- Output layer: `nn.Linear(hidden, n_classes)` (no sigmoid — CrossEntropy handles it)
- Loss: `nn.CrossEntropyLoss()` for multi-class, `nn.BCEWithLogitsLoss()` for binary
- `predict()`: `argmax(logits, dim=1)`
- `predict_proba()`: `softmax(logits, dim=1)`

This is a **new backend** (`surge/model/backends/mlp_classifier.py`), not a change to the existing regression MLP.

---

## Part 5 — Metric Routing in `SurrogateEngine`

The engine needs to know which metric set to compute based on `task_type`. Proposed change to `EngineRunConfig`:

```python
@dataclass
class EngineRunConfig:
    # existing fields ...
    task_type: str = "regression"    # NEW: "regression" | "classification"
```

And in `SurrogateEngine.run()`:

```python
if self.run_config.task_type == "classification":
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
else:
    metrics = compute_regression_metrics(y_true, y_pred)  # existing path
```

`metrics.json` output for classification:
```json
{
  "sklearn.random_forest_classifier": {
    "test": {
      "accuracy": 0.972,
      "f1_macro": 0.971,
      "f1_weighted": 0.972,
      "auroc": 0.997,
      "log_loss": 0.094
    }
  }
}
```

---

## Part 6 — Implementation Sequence (CI-safe order)

### Step 1 — Add classification metrics (no model changes, no CI risk)
- Add `accuracy`, `f1_score`, `auroc`, `log_loss`, `confusion_matrix_summary` to `surge/metrics.py`
- Add `task_type` to `EngineRunConfig` (default `"regression"` — fully backward compatible)
- Unit test: metrics compute correctly on synthetic labels

### Step 2 — Add sklearn classification adapters
- `sklearn.random_forest_classifier`
- `sklearn.gradient_boosting_classifier`
- `sklearn.logistic_regression`
- Each self-registers on import, must not break existing RF regressor tests
- Unit tests: registered, fit/predict/predict_proba shape, save/load round-trip

### Step 3 — Benchmark registry + runner CLI
- `surge/benchmarks/registry.py` — `BenchmarkRegistry`, `BENCHMARK_REGISTRY`
- `surge/benchmarks/base.py` — `BenchmarkResult`, `run_benchmark()`
- `surge/benchmarks/run.py` — CLI entry point
- Smoke test: `synthetic.regression_1d` + `classification.synthetic_binary` both pass

### Step 4 — Regression benchmark YAML configs
- `synthetic/regression_1d.yaml`, `synthetic/multioutput_2d.yaml`
- `tabular/diabetes.yaml`, `tabular/california_housing.yaml`
- `tabular/energy_efficiency.yaml`, `tabular/concrete_strength.yaml`
- `tabular/airfoil_noise.yaml`

### Step 5 — Classification benchmark YAML configs
- `classification/synthetic_binary.yaml`, `classification/iris.yaml`
- `classification/breast_cancer.yaml` (smoke, uses sklearn.datasets)
- `classification/plasma_stability.yaml` (standard, scientific)
- `classification/wine_quality_binary.yaml` (standard)

### Step 6 — PyTorch MLP classifier
- `surge/model/backends/mlp_classifier.py`
- `surge/model/adapters/mlp_classifier.py` → registers `pytorch.mlp_classifier`
- Reuses training infrastructure from `pytorch_impl.py`; changes only loss + output head
- Add to `[project.optional-dependencies] torch` — no new dependency needed

### Step 7 — Documentation
- Update `docs/model_zoo_plan.md` with classification column
- Add `docs/benchmark_policy.md` with `task_type` field spec
- Add `examples/classification_quickstart.py`

---

## Part 7 — Complete Benchmark Inventory (target state)

### Regression (existing task)

| Key | Tier | Shape | Baseline R² |
|---|---|---|---|
| `synthetic.regression_1d` | smoke | 1→1 | > 0.90 |
| `synthetic.multioutput_2d` | smoke | 8→2 | > 0.80 |
| `tabular.diabetes` | smoke | 10→1 | > 0.40 |
| `tabular.california_housing` | standard | 8→1 | > 0.75 |
| `tabular.energy_efficiency` | standard | 8→2 | > 0.90 |
| `tabular.concrete_strength` | standard | 8→1 | > 0.80 |
| `tabular.airfoil_noise` | standard | 5→1 | > 0.85 |
| `tabular.yacht_dynamics` | standard | 6→1 | > 0.95 |
| `multioutput.scm20d` | standard | 61→20 | > 0.60 |
| `fusion.m3dc1_sample` | standard | 13→1 | > 0.85 |
| `sequence.lorenz63` | standard | 3→3 | NRMSE < 0.1 |
| `pde.burgers_1d` | standard | 64→64 | rel. L2 < 0.05 |
| `pde.navier_stokes_2d` | full | 64×64→64×64 | rel. L2 < 0.10 |
| `fusion.m3dc1_full` | full | 13→1 | > 0.88 |

### Classification (new task)

| Key | Tier | Shape | Classes | Baseline Acc |
|---|---|---|---|---|
| `classification.synthetic_binary` | smoke | 20→1 | 2 | > 0.85 |
| `classification.iris` | smoke | 4→1 | 3 | > 0.95 |
| `classification.breast_cancer` | smoke | 30→1 | 2 | > 0.95 |
| `classification.wine_quality_binary` | standard | 11→1 | 2 | > 0.78 |
| `classification.plasma_stability` | standard | 12→1 | 2 | > 0.95 |
| `classification.material_phase` | standard | 81→1 | 2 | > 0.80 |
| `classification.fault_detection` | standard | 590→1 | 2 | > 0.92 |
| `classification.flow_regime` | standard | 4→1 | 4 | > 0.90 |
| `classification.digits` | standard | 64→1 | 10 | > 0.97 |
| `classification.covertype_binary` | standard | 54→1 | 2 | > 0.85 |
| `classification.higgs_boson` | full | 28→1 | 2 | > 0.72 |
| `classification.exoplanet_transit` | full | 3197→1 | 2 | > 0.97 |

---

## Part 8 — What Does NOT Change

To stay CI-green and within the v0.1.0 → v0.2.0 boundary:

- The two registries (`surge/registry.py` vs `surge/model/registry.py`) are **not merged** — deferred to v0.2.0
- Existing regression adapters (RF, MLP, GPR) are **not modified** — classification adapters are additive
- The 38 skipped legacy tests are **not touched**
- No large datasets added to the repo — all standard-tier datasets fetched via `sklearn.datasets.fetch_openml` or UCI URL at runtime
- `task_type: "regression"` is the default everywhere — zero breaking changes

---

## Quick Reference: Datasets Available Without Downloads

All of these work offline after `pip install scikit-learn`:

```python
from sklearn.datasets import (
    load_diabetes,           # regression, 10→1
    fetch_california_housing,# regression, 8→1
    load_breast_cancer,      # classification, 30→2
    load_iris,               # classification, 4→3
    load_wine,               # classification, 13→3
    load_digits,             # classification, 64→10
)
# All others via:
from sklearn.datasets import fetch_openml
energy = fetch_openml(name="energy-efficiency", version=1, as_frame=True)
plasma = fetch_openml(name="electrical-grid-stability", version=1, as_frame=True)
concrete = fetch_openml(name="concrete-strength", version=1, as_frame=True)
covertype = fetch_openml(name="covertype", version=1, as_frame=True)
```

These require internet on first run, then cache locally — no data in the repo.
