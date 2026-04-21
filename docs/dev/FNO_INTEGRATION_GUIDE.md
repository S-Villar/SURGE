# Integrating Fourier Neural Operators (FNO) into SURGE

## Overview

This guide outlines what would be necessary to extend SURGE with
**Fourier Neural Operators** (FNOs) — a family of **neural operators** that
learn maps between function spaces and are trained from simulation data on
structured grids. Adding FNO (and later DeepONet / neural operators more
broadly) is the natural next step in promoting SURGE from a
*data-driven surrogate framework for tabular I/O* to a *broader SciML framework
spanning regressors, Gaussian processes, ensembles, and neural operators*.

This document captures:

- FNO basics and why it matters for SURGE.
- Which parts of the existing SURGE stack carry over unchanged.
- Which parts must be generalized to accept field / mesh data.
- A concrete, phased implementation plan with effort estimates.
- Strategic implications for positioning and future adapters.

Target reader: a SURGE contributor who wants to scope, plan, or execute
the FNO integration.

---

## 1. What is a Fourier Neural Operator?

FNOs (Li et al., 2020, *"Fourier Neural Operator for Parametric Partial
Differential Equations"*) learn a mapping

```
G_θ : a(x)  ↦  u(x)
```

between function spaces, where `a(x)` is the input function (e.g. initial
condition, boundary data, coefficient field) sampled on a grid, and `u(x)`
is the PDE solution on the same grid. The core building block is a
**Fourier layer**:

```
FourierLayer(v) = σ( W · v + IFFT( R · FFT(v) ) )
```

- `FFT` / `IFFT` move signals between physical and spectral representations.
- `R` is a learned, truncated-frequency tensor acting only on the lowest
  `modes` Fourier coefficients (a global kernel parameterization).
- `W` is a pointwise linear map.
- `σ` is a nonlinearity (GELU / ReLU).

I/O shapes (2-D case):

```
X : (B, C_in,  H, W)   →   Y : (B, C_out, H, W)
```

Contrast with SURGE's current assumption:

```
X : (N_samples, N_features)  →   Y : (N_samples, N_outputs)
```

This shape mismatch is the **main integration cost** — the FNO architecture
itself is relatively small (~200–400 LOC of PyTorch).

---

## 2. Current SURGE Architecture (Tabular / Flat-I/O)

```
┌─────────────────────────────────────────┐
│         SurrogateDataset                │
│  - Loads CSV/Parquet/Pickle/HDF5/NetCDF │
│  - Auto-detects input/output columns    │
│  - Returns pandas DataFrame             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         SurrogateEngine                 │
│  - Splits data (train/val/test)         │
│  - Standardizes inputs/outputs (cols)   │
│  - Passes numpy (N, F) arrays to model  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         BaseModelAdapter                │
│  - fit(X: (N,F), y: (N,O))              │
│  - predict(X) -> (N, O)                 │
│  - predict_with_uncertainty(X)          │
└─────────────────────────────────────────┘
```

**Limitations for FNO**:

- Inputs are assumed to be a flat feature vector, not a field on a grid.
- Output is assumed to be a flat vector, not a field.
- Standardization is column-wise (`StandardScaler`), not channel-wise over
  spatial indices.
- Metrics (`R²`, `RMSE`, `MAE`, `MAPE`) are column-level, not field-level
  (no relative-L2 over a field, no spectral error, no per-channel metrics).
- Visualization defaults are scatter / profile plots, not field maps.

---

## 3. What SURGE Already Has That Helps

Much of the plumbing carries over:

| Existing capability | How it helps FNO |
|---|---|
| `BaseModelAdapter` contract (`fit` / `predict` / `predict_with_uncertainty`) | Clean slot for a new `FNOAdapter`. |
| PyTorch adapter scaffolding (`surge.model.pytorch_impl`) | Device handling, optimizer / scheduler patterns, training-loop skeleton. |
| Deep-ensemble pattern (`FNNEnsemble`) | Direct template for a deep-FNO-ensemble UQ adapter. |
| `MODEL_REGISTRY` | Registering `torch.fno` is a single line once the adapter exists. |
| `SurrogateWorkflowSpec` (YAML-driven) | `model.key: torch.fno` + a `params` dict are all the user needs to see. |
| M3D-C1 / XGC loaders (already yield R-Z fields, δp maps, profiles) | The data pipelines that produce FNO-appropriate tensors exist; they currently get flattened. |
| Artifact lineage + MLflow integration | Carries over unchanged. |
| Optuna HPO wiring | Reusable once the adapter exposes a search-space definition. |
| Field-style visualization (e.g. R-Z compare plots in `runs/.../plots/delta_p_2d_test/…`) | Template for a reusable `surge.viz` field-map panel keyed by adapter type. |
| ONNX export infrastructure | Partially useful; see caveats in §6. |

---

## 4. What Needs to Change

This is where the real work lives.

### 4.1 Field-aware dataset layer

Either extend `SurrogateDataset` with a "field mode" or introduce a
`FieldSurrogateDataset` subclass.

- **Input tensor shape**: `(N, C_in, *spatial_dims)` (1-D, 2-D, or 3-D).
- **Output tensor shape**: `(N, C_out, *spatial_dims)`.
- **Per-sample metadata**: grid spacing, coordinate arrays, boundary /
  mesh descriptor, simulation run ID (for provenance).
- **Source adapters**: extend the existing M3D-C1 / XGC loaders to emit
  field tensors directly rather than flattened columns. The raw data is
  already fielded — this is mostly re-shaping and typing.

### 4.2 Field-aware preprocessing

- Replace column-wise `StandardScaler` with a **field normalizer**:
  - Per-channel mean / std computed over `(N, *spatial_dims)`.
  - Or per-sample L2 normalization.
  - Or instance normalization (especially for heterogeneous fields).
- Keep the tabular normalizers available — the choice should be driven by
  adapter metadata or spec.

### 4.3 `FNOAdapter`

Implement in `surge/model/fno.py` (or `surge/model/pytorch/fno.py`):

- Core architecture: stack of Fourier layers (1-D, 2-D, 3-D variants).
  - Reuse an existing reference implementation (Li et al., or
    `neuraloperator`, or `modulus.models.fno`) rather than re-deriving.
- Training loop over batched field tensors.
- Honors `BaseModelAdapter` contract: `fit(X, y, X_val, y_val)`,
  `predict(X)`, optional `predict_with_uncertainty(X)`.
- Accepts model hyperparameters: `modes` (Fourier truncation), `width`
  (channel count), number of layers, dropout, lifting / projection sizes.

### 4.4 Field-appropriate loss and metrics

- **Loss**: relative L2 — `‖û − u‖₂ / ‖u‖₂` per sample, averaged over the
  batch (FNOs typically train much better with this than plain MSE).
- **Metrics** to add to `surge.metrics` / `surge.engine`:
  - Mean relative L2 (per split).
  - Per-channel R² / RMSE.
  - Spectral error (power-spectrum difference).
  - Optional boundary / region-of-interest error.

### 4.5 UQ strategy

Mirror the existing ensemble pattern:

- **Deep FNO ensemble** (simplest, cleanest — analogous to `FNNEnsemble`).
- **MC dropout FNO** (next).
- **Bayesian / variational FNO** (advanced; not a v1 target).

### 4.6 Visualization

Add to `surge.viz`:

- Field-map panel: predicted vs. ground truth vs. residual (re-usable from
  the existing `best_case_rz_compare_*` templates).
- Error-vs.-spectrum plot.
- Per-sample ranking (best / worst predictions) driven by relative L2.

### 4.7 HPO search space

Expose to Optuna (already wired into `SurrogateWorkflowSpec.hpo`):

- `modes` (int, log-uniform over `[4, 32]` typically).
- `width` (int, log-uniform over `[16, 128]`).
- `n_layers` (int, `[2, 6]`).
- `lr`, `weight_decay`, `batch_size`, `epochs`.
- Optional: `scheduler` type, `dropout`.

### 4.8 ONNX export (caveats)

- `torch.fft.fft2` / `ifft2` are supported from **ONNX opset ≥ 17**, but
  some runtimes still have partial coverage.
- Test early with `torch.onnx.export(..., opset_version=17)` on a trivial
  FNO; if it fails, retain PyTorch for serving and defer ONNX.

---

## 5. Phased Implementation Plan

| Phase | Scope | Effort |
|---|---|---|
| **0** | Decide: extend `SurrogateDataset` with a field mode, or introduce `FieldSurrogateDataset` subclass. | 0.5 day |
| **1** | Field-aware dataset + normalizer + splits + smoke test on existing δp-2D data. | 2–3 days |
| **2** | Minimal `FNOAdapter` (2-D, deterministic, MSE loss). Train on one M3D-C1 batch; confirm convergence. | 2–3 days |
| **3** | Register `torch.fno` in `MODEL_REGISTRY`; wire a YAML spec (`configs/m3dc1_fno_demo.yaml`); run end-to-end. | 1 day |
| **4** | Swap in relative L2 loss; add field metrics; add field-map viz panel. | 1–2 days |
| **5** | HPO search space + Optuna integration. | 1 day |
| **6** | UQ via deep FNO ensembles (and/or MC dropout). | 1–2 days |
| **7** | ONNX export probe; PyTorch-only fallback if needed. | 0.5–1 day |

**Total: ~8–12 focused person-days** for a polished, registry-complete
FNO adapter with UQ, viz, and HPO.

---

## 6. Suggested Starting Point (Throwaway Prototype)

Before investing in the dataset / preprocessing refactor, validate the
core feasibility in a prototype script:

1. Load one `delta_p_2d` batch from an existing run
   (e.g. `runs/.../delta_p_2d_test/…`).
2. Reshape to `(B, C_in, H, W)` with appropriate padding / cropping.
3. Train a 2-D FNO (`modes=12`, `width=32`, `4` layers) with relative L2
   loss for ~50 epochs on a single GPU.
4. Save predicted and reference R-Z maps alongside existing viz.
5. If convergence is reasonable and reconstructions are qualitatively
   correct, promote the prototype to a proper `FNOAdapter`.

Drop location for the prototype (clearly marked as exploratory):
`scripts/m3dc1/prototype_fno_delta_p.py`.

---

## 7. Strategic Implications

Adding FNO unlocks a broader SciML positioning for SURGE, and it creates
an obvious roadmap for further adapters:

| Adapter | Category | Comment |
|---|---|---|
| `torch.fno` | Neural operator (spectral) | This guide. |
| `torch.deeponet` | Neural operator (branch–trunk) | Similar effort profile; reuses the field-dataset machinery. |
| `torch.pinn` | Physics-informed network | Requires a residual-loss abstraction; larger effort. |
| `torch.graph_no` / MeshGraphNet | Mesh-based neural operator | Needs unstructured-mesh dataset support. |

Once one or more of these are in, the abstract can upgrade from
*"data-driven SciML framework"* to, for example:

> *"SURGE is a **Scientific Machine Learning (SciML) framework** for
> surrogate-model generation spanning **data-driven regressors,
> ensembles, Gaussian processes, and neural operators** (FNO, DeepONet,
> …), integrating AutoML features, uncertainty quantification (UQ), and
> MLOps-grade provenance in a single declarative pipeline."*

---

## 8. Open Questions / Decisions to Make

- **Dataset shape**: extend `SurrogateDataset` vs. add `FieldSurrogateDataset`?
  - Leaning toward a subclass, to keep tabular users on a simpler path.
- **Grid assumptions**: uniform rectangular for v1, or support masked /
  non-rectangular grids from the start?
  - v1: uniform rectangular (consistent with standard FNO).
- **Multi-resolution / discretization-invariant training**: defer to v2.
- **Loss configurability**: make loss function a first-class field on
  the adapter params dict (`loss: "relative_l2"`, `"mse"`, `"h1"`, …).
- **Checkpointing**: follow the existing adapter `save` / `load`
  conventions; confirm they work for `torch.fft`-bearing models.
- **Deterministic runs on GPU**: `torch.fft` is deterministic with fixed
  seeds on CUDA for supported shapes, but verify with a small test.

---

## 9. References

- Li, Z. et al., *"Fourier Neural Operator for Parametric Partial
  Differential Equations,"* ICLR 2021 — the original FNO paper.
- Kovachki, N. et al., *"Neural Operator: Learning Maps Between Function
  Spaces,"* JMLR 2023 — neural-operator theory & unification.
- `neuraloperator` (Python package) — reference implementations of FNO,
  DeepONet, and related architectures.
- NVIDIA Modulus (`modulus.models.fno`) — production-quality FNO
  implementation with HPC-scale training support.
