# Direct RZ-Field Surrogate Experiment — Full Technical Report

**Status:** 2026-07-07. Written for external review (e.g. Opus) of what we tried,
what we measured, what we ruled out, and what might still be wrong.

**Repo:** SURGE (`/global/homes/a/asvillar/src/SURGE`)  
**Script:** `scripts/m3dc1/internal/train_rz_field_image.py`  
**Runs:**
- `runs/rz_field_fno48_re_deltap_smooth0/` — g128, early recipe (failed)
- `runs/rz_field_fno64_g201_re_deltap_smooth0/` — g201, D′/E+F-adapted recipe (in progress / failed at ep ~30)

**Baseline comparison (spectrum path, works):**
- `runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0/` — D′ magnitude model
- Field bench median relL2 ≈ **0.47** with oracle true phase

---

## 1. Scientific question

**Hypothesis:** Can we learn Re(δp)(R,Z) **directly** on the RZ mesh from equilibrium
conditioning alone — **without** going through the poloidal Fourier spectrum
δp̂(m, ψ_N)?

**Motivation:**
- The user's notebook visualizes `p_hat` real part on (R,Z) with physical units (±700).
- Mathematically Re(p_hat) = p_phi0 = δp(R,Z, φ₀) when φ₀/φ_q samples are real.
- If equilibrium → RZ-field is learnable, we could skip spectrum + phase reconstruction.

**Null result so far:** val/test **relL2 ≈ 1.0** (same as predicting zero) across two
runs with different grids and loss recipes. Spectrum path achieves relL2 ≈ 0.47.

---

## 2. Data source

| Item | Value |
|------|-------|
| Batch | `/pscratch/sd/a/asvillar/mp288/jobs/batch_16` |
| Per-case file | `csdata_deltap_b_ver.h5` |
| Cases used | 9974 (2 quarantined via `runs/quarantine/bad_cases.json`) |
| Native mesh | 201×201 (R,Z) |
| Postprocess | `m3dc1ml` `build_sdata.py` |

### 2.1 Target field

- **HDF5 key:** `pertfields/p_phi0`
- **Physical meaning:** Re(δp) at toroidal angle φ₀ on the RZ grid.
- **Relation to notebook plot:** `p_hat` with `real` component is **identical** to
  `p_phi0` (since ef_hat = ef_phi0 − i·ef_phiq and Re(ef_hat) = ef_phi0).
- **Time index:** `time_idx=-1` (last time slice; typically t=1 when n_time=2).

### 2.2 Input channels (18 total)

Spatial fields (per-case max-normalized so peak |field| = 1):

| # | Channel | Source |
|---|---------|--------|
| 0 | `psin` | (ψ − ψ_axis)/(ψ_LCMS − ψ_axis) |
| 1 | `grad_psi` | \|∇ψ\| on RZ grid |
| 2 | `lcfs_prox` | 1 − psin |
| 3 | `q_on_rz` | q(ψ) profile interpolated to each (R,Z) via psin |
| 4 | `p_on_rz` | equilibrium p(ψ) interpolated to (R,Z) |
| 5 | `R_norm` | (R − R_min)/(R_max − R_min) per case |
| 6 | `Z_norm` | (Z − Z_min)/(Z_max − Z_min) per case |
| 7 | `inside_mask` | 1 inside LCFS (psin < 1), 0 outside |

Broadcast scalars (constant over the grid per case):

| # | Channel | Typical scale |
|---|---------|---------------|
| 8–17 | `kappa`, `delta`, `epsilon`, `pscale`, `batemanscale`, `ntor`, `q0`, `q95`, `qmin`, `p0` | O(1) except **`p0` ~ 10⁵–10⁶** |

**Critical:** inputs contain **only equilibrium information**. No δp, no spectrum,
no growth rate, no mode index in the input (unless implicitly in scalars).

### 2.3 What is NOT used as input

- δp̂(m, ψ_N) spectrum
- Phase φ(m, ψ)
- \|δp\| on RZ
- Magnetic perturbation components (BR, BZ, …)
- γ (growth rate) — available in scalars parquet but not wired in

---

## 3. Preprocessing pipeline

Implemented in `build_rz_dataset()` in `train_rz_field_image.py`.

### 3.1 Target normalization

```
raw field  →  y = field / max(|field|)     # per-case, signed, peak = ±1
           →  floor: |y| < 10^(-floor_dex) set to 0   # default floor_dex=6
           →  optional Gaussian smooth (σ=0 for D′-style runs)
           →  resize 201×201 → training grid (128 or 201)
           →  global z-score: Yn = (Y - y_mean_train) / y_std_train
```

Train stats (g201 run):
- `y_mean ≈ -4.1e-5`
- `y_std ≈ 0.103`

### 3.2 Input normalization

1. Per-case max-abs norm on spatial channels (`psin`, `grad_psi`, `q_on_rz`, `p_on_rz`).
2. Global z-score per channel using **train-split** mean/std (all 18 channels).
3. Scalars including `p0` are z-scored but **not** log-transformed.

### 3.3 Train / val / test split

- Seed 42, case-level shuffle
- test_frac=0.2 → 1994 test
- val_frac=0.1 → 997 val
- train → 6983

---

## 4. Model

- **Architecture:** FNO2D (`surge.model.backends.fno2d._FNO2dNet`)
- **Input:** (B, 18, H, W)
- **Output:** (B, 1, H, W) — single channel, z-scored target space
- **Default config (g201):** 64 modes, hidden 32, 4 layers, ~134M params @ g201
- **Optimizer:** Adam, lr=1e-3
- **Batch size:** 4 @ g201 (matches spectrum E+F recipe for memory)

No output activation on the final layer (linear projection).

---

## 5. Loss function (g201 recipe)

Composite loss on **z-scored** predictions vs **z-scored** targets:

```
L = peakMSE + loc_weight * L_loc + marg_weight * L_marg [+ grad_weight * L_grad]
```

### 5.1 peakMSE (peak_weight=4, peak_pow=1)

```python
w = 1 + peak_weight * (|target| / max(|target|))^peak_pow
loss = mean(w * (pred - target)^2)
```

Up-weights pixels near the global max of |δp| per case.

### 5.2 L_loc (loc_weight=2, loc_beta=8)

Soft-argmax ψ_N of |field| peak location:

```python
p = softmax(loc_beta * |field|.flatten())
psi_peak = sum(p * psin.flatten())   # expected ψ_N of peak
L_loc = mean((psi_peak_pred - psi_peak_true)^2)
```

Adapted from spectrum training where the coordinate was ψ_N along the spectrum columns.
Here `psin` at each (R,Z) pixel is the coordinate map.

**Observed:** `val_dpsi` improves (0.20 → 0.09 by epoch 20) — model learns radial
peak location without learning field shape.

### 5.3 L_marg (marg_weight=1)

MSE on 1D marginals (adapted from spectrum m- and ψ-marginals):

```python
L_marg = MSE(pred.mean(dim=2), target.mean(dim=2))   # Z-marginal
       + MSE(pred.mean(dim=3), target.mean(dim=3))   # R-marginal
```

### 5.4 NOT used (deliberately)

- **field_loss_weight / IFFT proxy** — not applicable; we predict the field directly.
- **select-by field with IFFT subset** — we use full-val relL2 each epoch instead.

### 5.5 Checkpoint selection

`--select-by field` → save best checkpoint by **lowest val relL2 median**.
Early stop patience=120 on no improvement of selection metric.

---

## 6. Evaluation metrics

### 6.1 relL2 (primary field metric)

Per case, on **max-normalized** physical fields (not z-scored):

```
pred_norm = pred_z * y_std + y_mean
relL2 = ||pred_norm - y_true||_2 / ||y_true||_2
```

- relL2 = **1.0** ⟺ prediction is **zero** (for max-norm target).
- relL2 < 1.0 ⟺ better than predicting zero.
- D′ oracle benchmark ≈ **0.47** median (spectrum + true phase).

### 6.2 Other logged metrics

| Metric | Space | Notes |
|--------|-------|-------|
| `val_loss` | z-scored MSE | Can decrease while relL2 worsens |
| `val_r2` | z-scored | Goes negative when worse than mean |
| `val_dpsi` | ψ_N units | Peak location error; improves |
| `val_comp` | composite loss | Training objective on val |
| `frac_relL2_gt_1` | — | Fraction of val cases worse than zero predictor |

---

## 7. Experimental results

### 7.1 Run A — g128, early recipe

**Out:** `runs/rz_field_fno48_re_deltap_smooth0/`

| Setting | Value |
|---------|-------|
| grid | 128 |
| fno_modes | 48 |
| peak_weight | 2 |
| loc/marg | off |
| target_floor | off (initially; NaN bug then fixed) |
| batch_size | 16 |

**Test results (completed):**

| Metric | Value |
|--------|-------|
| test relL2 median | **1.003** |
| test relL2 mean | 1.001 |
| test R² global | 0.0035 |
| test pattern R² | 0.0035 |
| frac relL2 > 1 | 59% |
| best epoch | 6 (early stop @ 86) |

Example plots show predictions ≈ zero; residuals ≈ true field.

### 7.2 Run B — g201, D′/E+F-adapted recipe

**Out:** `runs/rz_field_fno64_g201_re_deltap_smooth0/`

| Setting | Value |
|---------|-------|
| grid | 201 |
| fno_modes | 64 |
| peak_weight | 4 |
| loc_weight | 2 |
| marg_weight | 1 |
| target_floor | 6 dex |
| select_by | field (relL2) |
| batch_size | 4 |
| patience | 120 |

**Training curve (in progress at time of writing):**

| Epoch | train_loss | val_loss | val_r2 | relL2_med | dpsi | frac>1 |
|-------|------------|----------|--------|-----------|------|--------|
| 1 | 3.46 | 1.06 | −0.004 | **1.002** *best* | 0.200 | 0.75 |
| 10 | 3.15 | 1.11 | −0.044 | 1.021 | 0.098 | 0.72 |
| 20 | 2.55 | 1.18 | −0.116 | 1.058 | 0.089 | 0.71 |
| 30 | 2.03 | 1.21 | −0.147 | 1.068 | 0.086 | 0.68 |

**Pattern:** train composite loss **decreases**; val relL2 **increases**. Best
checkpoint remains epoch 1. Model learns peak ψ location (dpsi ↓) but not field shape.

---

## 8. Diagnostics we ran (not in training logs)

### 8.1 Baseline relL2 (200 cases, g201, floor=6)

| Predictor | relL2 median |
|-----------|--------------|
| Zero field | **1.0000** (exact) |
| Dataset mean field | 0.9996 |
| Plasma mean (inside mask) | 1.0000 |
| Oracle (true Y) | 0.0000 |

**Conclusion:** relL2 ≈ 1 at epoch 1 is **exactly** the zero-predictor baseline,
not evidence of mis-loaded data.

### 8.2 Target sparsity (after max-norm + floor=6)

| Statistic | Value |
|-----------|-------|
| Fraction pixels \|Y\| > 1e-8 | ~75% |
| Fraction pixels \|Y\| > 0.01 | ~**32%** |
| Per-case L2 norm of Y | ~19.85 |
| Per-case max \|Y\| | 1.0 (by construction) |

Most "nonzero" pixels are tiny (1e-6…0.01). Mode structure lives in ~32% of pixels.

### 8.3 Input channel sanity

- All channels finite after NaN sanitization.
- `p0` mean ≈ 6.6×10⁵ (not log-scaled).
- `R_norm`, `Z_norm` mean = 0.5 for every case (mesh geometry; weak case-specific signal).

### 8.4 Overfit test (32 cases, g201, 50 epochs, same architecture)

| Loss setup | relL2 median @ epoch 50 |
|------------|-------------------------|
| Plain z-scored MSE | **0.62** |
| Masked to inside_mask | ~1.02 (worse) |
| \|Y\| magnitude + peak weight | **0.41** |

**Conclusion:** FNO **can memorize** δp from equilibrium inputs on a tiny training
set. Failure on 9974 cases is **generalization**, not architecture inability to
represent the mapping on training data.

### 8.5 Bugs found and fixed during development

| Bug | Symptom | Fix |
|-----|---------|-----|
| NaN in equilibrium channels | y_mean=NaN, train_loss=NaN | `_finite_array()` + `nanmean` |
| `_loss_plot` log-scale on NaN | crash epoch 1 | `_rz_loss_plot()` with safe scaling |
| `tee` before `mkdir` | log file error | mkdir first |

No remaining known crash bugs as of g201 run.

---

## 9. Comparison to spectrum path (what works)

| Aspect | Spectrum D′ | Direct RZ (this work) |
|--------|-------------|----------------------|
| Target | log₁₀(max-norm \|δp̂\|(m,ψ)) | max-norm Re(δp)(R,Z) |
| Target domain | (m, ψ_N) — **dense ridge** | (R,Z) — **sparse signed lobes** |
| Target sign | positive (log magnitude) | signed (±) |
| Inputs | ψ, m, q, p, resonance, geometry | equilibrium on RZ only |
| val R² | ~0.84–0.92 | ~0 |
| field relL2 (oracle φ) | ~0.47 | ~1.0 (no oracle needed) |
| Phase | oracle or separate model | N/A |

---

## 10. Hypotheses: what could still be wrong?

Ranked for external review.

### H1 — Loss / target formulation (most likely, partially tested)

**Claim:** Pixel MSE on z-scored sparse signed 2D fields optimizes to predict ~0;
train loss can fall without improving relL2.

**Evidence:**
- Zero predictor relL2 = 1.0 exactly; epoch 1 already there.
- 32-case overfit reaches 0.62 (plain) / 0.41 (\|Y\|) — formulation matters.
- val relL2 worsens while train loss improves (overfitting to background).

**Proposed fixes:**
- Train on \|δp\| not signed Re(δp)
- Drop global z-score on Y; train in max-norm space
- Mask loss to \|Y\| > ε or top-k% amplitude pixels
- Stronger floor: zero below 1% of peak, not 1e-6 absolute

### H2 — Task is not identifiable from inputs (likely for generalization)

**Claim:** Equilibrium scalars + ψ(R,Z) do not determine the eigenmode δp structure
across 10k cases; spectrum compression is necessary.

**Evidence:**
- Overfit works (32 cases) but val relL2 ≈ 1 on full split.
- Spectrum with same equilibrium channels achieves high R² in (m,ψ) space.
- Different equilibria can host different mode numbers/structures with similar q, κ, δ.

**Counter:** Maybe with γ, n_tor mode matching, or richer inputs it becomes identifiable.

### H3 — Input preprocessing issues (possible, not primary)

**Claim:** Some input channels hurt more than help.

**Suspects:**
- `p0` ~ 10⁶ not log-scaled (after z-score OK but dynamic range extreme)
- `R_norm`, `Z_norm` identical mean across cases (only grid layout, not equilibrium)
- `q_on_rz`, `p_on_rz` are **equilibrium** profiles, not perturbation

### H4 — Resolution / grid (ruled out as sole cause)

g128 and g201 both fail at relL2 ≈ 1 from epoch 1. Resolution may matter for fine
detail but is not the root cause of total failure.

### H5 — Data alignment bug (mostly ruled out)

| Check | Status |
|-------|--------|
| p_phi0 vs Re(p_hat) | Mathematically identical |
| time_idx=-1 | Last slice |
| relL2 oracle = 0 | Pass |
| max-norm peak = 1 | Pass |
| quarantine applied | 2 cases excluded |

**Remaining doubt:** interpolation of q/p profiles using psin after per-case max-norm
of psin — ordering preserved, likely OK but not exhaustively verified.

### H6 — Evaluation bug (ruled out)

relL2 correctly denormalizes z-scored predictions before comparing to max-norm Y.
Zero baseline math checks out.

### H7 — Model selection / early stopping artifact

`select-by field` correctly picks epoch 1; later epochs are worse. Not a bug —
symptom of H1/H2.

---

## 11. Open questions for Opus

1. **Is signed Re(δp) the wrong target** for an FNO on a sparse RZ grid? Should we
   use \|δp\|, log\|δp\|, or a two-channel (mag, phase) representation on RZ?

2. **Is global z-score on Y harmful** when 68% of pixels are near-zero after floor?
   Spectrum uses it successfully on **dense** log-magnitude ridges.

3. **What is the right masked loss** for tokamak MHD modes on RZ — inside LCFS only,
   \|δp\| > 1% peak, or soft mask?

4. **Are we missing identifiability constraints** — e.g. n_tor matching via
   resonance channel `m − n·q(ψ)` on RZ (used in spectrum inputs but not RZ script)?

5. **Could the FNO see δp in equilibrium channels indirectly** (unlikely) or do we
   need explicit auxiliary inputs (γ, locator class, m_peak)?

6. **Why does loc loss improve dpsi but not relL2** — is soft-argmax ψ_N on RZ
   peak the wrong location coordinate vs poloidal angle structure?

7. **Is there a train/val distribution shift** in peak amplitude or mode type that
   makes epoch-1 zero predictor optimal on val?

8. **Should scalars be log10-transformed** before broadcast (especially p0)?

---

## 12. Command to reproduce g201 run

```bash
# On Perlmutter GPU node after salloc + surge env:
python -u scripts/m3dc1/internal/train_rz_field_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 \
  --n-cases 0 \
  --pert-field p_phi0 \
  --time-idx -1 \
  --grid 201 \
  --models fno2d \
  --fno-modes 64 \
  --fno-hidden 32 \
  --peak-weight 4 \
  --loc-weight 2 \
  --marg-weight 1 \
  --target-floor 6 \
  --target-smooth 0 \
  --exclude-list runs/quarantine/bad_cases.json \
  --select-by field \
  --epochs 400 \
  --patience 120 \
  --batch-size 4 \
  --lr 1e-3 \
  --time-budget-min 210 \
  --test-frac 0.2 \
  --val-frac 0.1 \
  --seed 42 \
  --out runs/rz_field_fno64_g201_re_deltap_smooth0
```

---

## 13. Key file paths

| File | Purpose |
|------|---------|
| `scripts/m3dc1/internal/train_rz_field_image.py` | Training script |
| `scripts/m3dc1/internal/train_spectrum_image.py` | Reference spectrum recipe |
| `m3dc1ml/src/m3dc1ml/postprocess/build_sdata.py` | HDF5 generation |
| `m3dc1ml/notebooks/explore_csdata_deltap_b_test.ipynb` | Field visualization |
| `runs/rz_field_fno48_re_deltap_smooth0/rz_field_metrics.json` | Run A metrics |
| `runs/rz_field_fno64_g201_re_deltap_smooth0/history_fno2d.jsonl` | Run B curve |
| `runs/rz_field_fno64_g201_re_deltap_smooth0/plots/fno2d_rz_examples.png` | Example panels |

---

## 14. Summary one-liner

We train an FNO to map **equilibrium-only** RZ channels → **max-normalized Re(δp)(R,Z)**;
the pipeline loads data correctly and relL2=1 is the **zero-predictor floor**; training
loss decreases but **val relL2 worsens** because the model learns background/peak-ψ
shortcuts without generalizable mode structure — while the **spectrum path on the same
dataset works** (relL2 ≈ 0.47 with oracle phase).
