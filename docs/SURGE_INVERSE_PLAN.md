# SURGE Inverse Design Module — Development Plan

**Module:** `surge.inverse`
**Status:** Planned
**Priority:** High (direct scientific value for fusion surrogate applications)
**Depends on:** `surge.model` (any trained adapter), `surge.uq` (optional)

---

## 1. Problem Statement

A surrogate model in SURGE approximates a forward map:

```
f : x ∈ R^d  →  y ∈ R^m
```

where `x` is a vector of plasma/equilibrium parameters (e.g. Miller geometry,
profiles, engineering limits) and `y` is one or more quantities of interest
(e.g. beta, gamma_MHD, stored energy W, bootstrap current fraction).

Once a surrogate is trained, three scientific questions arise that go **beyond
simple forward prediction**:

| Question | Name | Example |
|---|---|---|
| Which inputs drive the output most? | **Sensitivity** | Does delta or q95 control beta more? |
| What input change achieves a target output? | **Inverse design** | What delta gives beta = 0.08? |
| What is the minimum change from a known point to cross a threshold? | **Counterfactual** | Closest equilibrium where MHD becomes unstable? |

None of these require uncertainty quantification — they work with any trained
SURGE adapter. UQ (GP predictive variance, ensemble spread) can be layered on
top to report confidence in the suggested operating point.

---

## 2. Scientific Motivation

### 2.1 Fusion plasma context

Tokamak scenario design involves finding equilibrium parameters that
simultaneously satisfy performance targets (high beta, high bootstrap fraction)
and stability constraints (MHD stable, no disruption, q95 above limit). The
design space is high-dimensional, experiments are expensive, and no closed-form
inverse exists.

A trained surrogate + inverse solver allows a physicist to:

- **Scan**: "Show me all equilibria with beta > 0.06 and gamma < 0."
- **Optimize**: "Find the operating point that maximises beta subject to
  stability constraints."
- **Counterfactual**: "My current equilibrium has gamma = +0.02 (unstable). What
  is the smallest change to make it stable?"
- **Sensitivity**: "Rank the Miller parameters by their influence on beta at
  this specific operating point."

### 2.2 Generality

The same module applies to any SURGE benchmark — predicting concrete
compressive strength, airfoil noise, housing prices — but the fusion use case
is the primary scientific driver.

---

## 3. Requirements

### 3.1 Functional requirements

| ID | Requirement |
|---|---|
| F1 | Accept any trained SURGE adapter as the forward model |
| F2 | Support scalar and multi-output surrogates |
| F3 | Accept per-feature bounds `[(lo_1, hi_1), ..., (lo_d, hi_d)]` as constraints |
| F4 | Support fixing a subset of inputs (only optimize free parameters) |
| F5 | Return the optimized input vector and the predicted output value |
| F6 | Optionally return uncertainty bounds when the adapter supports `predict_with_uncertainty` |
| F7 | Provide local sensitivity: feature importance at a specific point (Shapley / gradient) |
| F8 | Provide global sensitivity: Sobol first-order and total-order indices across a domain |
| F9 | Provide counterfactual: minimum-norm input change to cross a scalar threshold |
| F10 | All functions should work without PyTorch, using only numpy/scipy by default |

### 3.2 Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | No mandatory new dependencies — scipy (already present) covers the default path |
| N2 | Optional dependencies: `SALib` for Sobol, `shap` for SHAP values, `botorch` for BO |
| N3 | Each function completes in < 60 s for typical surrogate + d ≤ 20 inputs |
| N4 | Functions return plain numpy arrays and dicts — no custom objects required |
| N5 | A worked example for the fusion use case ships in `examples/inverse_design_demo.py` |

### 3.3 Interface requirements

All public functions follow the same signature convention:

```python
result = surge_function(adapter, x0_or_bounds, target_or_query, **options)
```

where `result` is always a plain dict with documented keys so it can be
inspected, printed, and serialised without importing SURGE.

---

## 4. Module Structure

```
surge/
  inverse/
    __init__.py          ← public API exports
    sensitivity.py       ← local (gradient/SHAP) and global (Sobol) sensitivity
    optimize.py          ← inverse design via constrained optimization
    counterfactual.py    ← counterfactual / minimum-change search
    _helpers.py          ← shared utilities (numerical Jacobian, bounds handling)

examples/
  inverse_design_demo.py ← end-to-end worked example on fusion surrogate
```

---

## 5. Public API (target)

### 5.1 Sensitivity analysis

```python
from surge.inverse import sensitivity

# Local sensitivity at a specific point (gradient-based, works with any model)
result = sensitivity.local(adapter, x0)
# Returns:
# {
#   "feature_importance": array([0.45, 0.03, 0.31, ...]),  # |df/dx_i| * x_i_std
#   "gradient": array([...]),                               # df/dx at x0
#   "feature_names": ["delta", "kappa", "q95", ...],
# }

# Global sensitivity (Sobol indices, requires SALib)
result = sensitivity.global_sobol(adapter, bounds, n_samples=1024)
# Returns:
# {
#   "S1":  array([0.42, 0.02, 0.28, ...]),   # first-order indices
#   "ST":  array([0.51, 0.03, 0.35, ...]),   # total-order indices
#   "feature_names": [...],
# }

# SHAP values at a set of points (requires shap package)
result = sensitivity.shap_values(adapter, X_background, X_explain)
```

### 5.2 Inverse design (optimization)

```python
from surge.inverse import optimize

# Find inputs that achieve a scalar target output
result = optimize.inverse_design(
    adapter,
    x0      = current_equilibrium,    # starting point
    target  = 0.08,                   # desired beta
    bounds  = param_bounds,           # [(lo, hi), ...] per feature
    fixed   = {"q95": 5.0},           # inputs to hold constant
    method  = "L-BFGS-B",             # scipy optimizer (default)
)
# Returns:
# {
#   "x_opt":       array([0.50, 1.85, 5.0, ...]),  # suggested input
#   "y_pred":      0.0798,                          # surrogate prediction
#   "y_target":    0.08,
#   "delta_x":     array([+0.20, 0.0, 0.0, ...]),  # change from x0
#   "converged":   True,
#   "n_evals":     83,
#   "uq_interval": (0.074, 0.085),                  # if adapter has UQ
# }

# Multi-objective: maximise output subject to constraints
result = optimize.constrained(
    adapter,
    objective = "maximize",
    bounds    = param_bounds,
    constraints = [
        {"type": "ineq", "fun": lambda x: x[3] - 3.0},   # q95 >= 3
        {"type": "ineq", "fun": lambda x: 2.2 - x[1]},   # kappa <= 2.2
    ],
)
```

### 5.3 Counterfactual

```python
from surge.inverse import counterfactual

# Minimum-norm change from x0 to cross a threshold
result = counterfactual.find(
    adapter,
    x0        = unstable_equilibrium,
    threshold = 0.0,            # e.g. gamma < 0 means stable
    direction = "below",        # cross threshold going down
    bounds    = param_bounds,
    fixed     = ["R0", "a"],    # geometry is fixed
)
# Returns:
# {
#   "x_cf":         array([...]),    # counterfactual point
#   "delta_x":      array([...]),    # minimal change
#   "changed_features": ["delta", "beta_p"],
#   "y_before":     +0.023,          # gamma at x0 (unstable)
#   "y_after":      -0.004,          # gamma at x_cf (stable)
#   "norm_change":  0.18,
# }
```

---

## 6. Implementation Phases

### Phase 1 — Core optimization (high value, low effort)
- `optimize.inverse_design` using `scipy.optimize.minimize`
- Works with any adapter via numerical function evaluation
- No new dependencies
- Estimated effort: 1–2 days

### Phase 2 — Sensitivity analysis
- `sensitivity.local` via numerical Jacobian (finite differences)
- `sensitivity.global_sobol` via SALib (optional dependency)
- Estimated effort: 1 day

### Phase 3 — Counterfactual search
- `counterfactual.find` via constrained optimization (scipy)
- Estimated effort: 1 day

### Phase 4 — SHAP integration and Bayesian optimization path
- `sensitivity.shap_values` wrapping the `shap` library
- `optimize.bayesian` using BoTorch (already in SURGE) for expensive
  evaluations and constraint handling
- Estimated effort: 2 days

### Phase 5 — Worked fusion example
- `examples/inverse_design_demo.py` using the NSTX-U surrogate dataset
  already in `data/datasets/NSTX-U/`
- Demonstrates full workflow: train GP → sensitivity scan → inverse design →
  counterfactual stability boundary
- Estimated effort: 1 day

---

## 7. Example End-to-End Workflow (target UX)

```python
import numpy as np
from surge.model import MODEL_REGISTRY
from surge.inverse import sensitivity, optimize, counterfactual

# ── 1. Train a surrogate on NSTX-U equilibria ────────────────────────────
adapter = MODEL_REGISTRY.create("botorch.sparse_gp", n_train_iter=200)
adapter.fit(X_equil, y_beta)

# ── 2. Global sensitivity: which Miller params drive beta? ───────────────
bounds = {
    "delta":  (0.1, 0.8),
    "kappa":  (1.3, 2.4),
    "q95":    (3.0, 9.0),
    "beta_p": (0.3, 3.0),
    "li":     (0.4, 1.5),
}
sobol = sensitivity.global_sobol(adapter, bounds)
# >> delta:  S1=0.44  ST=0.52   ← dominant driver
# >> kappa:  S1=0.21  ST=0.29
# >> q95:    S1=0.08  ST=0.11

# ── 3. Local sensitivity at current operating point ──────────────────────
local = sensitivity.local(adapter, x0=current_eq)
# >> Most influential at this point: delta (+), beta_p (+), li (-)

# ── 4. Inverse design: find inputs for target beta = 0.08 ────────────────
design = optimize.inverse_design(
    adapter, x0=current_eq, target=0.08,
    bounds=list(bounds.values()),
    fixed={"q95": 5.0, "kappa": 1.9},
)
# >> Suggested: delta 0.30 → 0.51  (+0.21)
# >>            beta_p 1.2 → 1.6   (+0.40)
# >> Predicted beta: 0.0798  [95% CI: 0.072 – 0.087]

# ── 5. Counterfactual: minimum change to cross MHD stability boundary ────
cf = counterfactual.find(
    gamma_adapter, x0=current_eq,
    threshold=0.0, direction="below",
    bounds=list(bounds.values()),
    fixed=["R0", "a", "q95"],
)
# >> Smallest stabilising change: delta 0.30 → 0.44  (+0.14)
# >> gamma: +0.019 → -0.003
```

---

## 8. Testing Strategy

- Unit tests for each function with a simple analytical surrogate (`y = x1² + x2`)
  where the ground-truth inverse is known
- Integration test using `sklearn.ridge` trained on `tabular.california_housing`
- Fusion integration test using the NSTX-U dataset in `data/datasets/NSTX-U/`

---

## 9. References

- Wachter, S. et al. (2017). "Counterfactual Explanations Without Opening the Black Box." *HJLST*
- Saltelli, A. et al. (2010). "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index." *Computer Physics Communications*
- Wilson, A.G. et al. (2020). "Efficiently sampling functions from Gaussian process posteriors." *ICML* — for BO path
- Lundberg, S. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS* — SHAP
- Eriksson, D. et al. (2019). "Scalable Global Optimization via Local Bayesian Optimization." *NeurIPS* — TuRBO for high-d BO
