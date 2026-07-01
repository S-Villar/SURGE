# m3dc1ml refactor plan — complex spectra surrogate pipeline

**Status:** draft (2026-06-09)  
**Goal:** one repo (`m3dc1ml`) with a clear, auditable pipeline from raw M3DC1 cases → complex HDF5 → QA → SURGE training → 2D field surrogates.

---

## Executive summary: why only 31 complex files?

| Question | Answer |
|----------|--------|
| **How many `sdata_pertfields_grid_complex_v2.h5` exist?** | **31** — all under `pscratch/.../batch_16/run1/sparc_*` |
| **How were they created?** | `batch_16/post/run_postprocess_batch.py` calling **`batch_16/post/postprocess_ndarray.py`** (complex-capable version) with `--full-fft` and output name `sdata_pertfields_grid_complex_v2.h5` |
| **Why only 31?** | **`batch_16/run1` has 103 `sparc_*` dirs (101 with `C1.h5` + `finished`)**, but the complex driver was a **pilot subset only**: roughly `sparc_1300`–`1319` and `sparc_1349`–`1358`. **`postprocess_timing_v2.csv` has 29 rows, all `exit_code=0`** — no crashes, ~70 run1 cases were simply **never queued**. Other runs/batches were also never postprocessed with the complex script |
| **Where did complex data go on CFS?** | **Nowhere — they were never copied.** CFS bulk postprocessing used a **different, older** `postprocess_ndarray.py` (under `amsc007/data/m3dc1/`) that stores **\|spec\|** (float64) as `sdata_complex_v2.h5` (~9859 cases) |

### Two postprocess pipelines (root cause)

```
PROTOTYPE (31 cases, complex128)                    PRODUCTION (9859 cases, float64)
─────────────────────────────────                   ─────────────────────────────────
pscratch/batch_16/post/postprocess_ndarray.py       amsc007/data/m3dc1/postprocess_ndarray.py
  _compute_complex_spectrum()                         m1.eigenfunction(..., fourier=True)
  ef_hat = ef_phi0 - 1j * ef_phiq                     then spec = |spec|  (magnitude)
  + reconstruction_check group                        no reconstruction_check
  + equilibrium group
  → sdata_pertfields_grid_complex_v2.h5               → sdata_complex_v2.h5
  batch_16/run1 only (Feb 16 2026)                    full CFS tree (Feb 17 2026)
```

**Verified on disk:**

| File | `spectrum/p/spec` dtype | `max|Im|` | Extra groups |
|------|-------------------------|-----------|--------------|
| pscratch `.../sdata_pertfields_grid_complex_v2.h5` | **complex128** | ~6e-8 | `equilibrium`, `reconstruction_check` |
| CFS `.../sdata_complex_v2.h5` | **float64** | 0 | none |

The name `sdata_complex_v2.h5` on CFS is **misleading** — it is magnitude/real, not complex.

---

## Data locations (current scatter map)

| Role | Path | Count / notes |
|------|------|---------------|
| Raw M3DC1 cases (CFS bulk) | `/global/cfs/projectdirs/amsc007/data/m3dc1/run*/sparc_*/` | 102 runs, ~9859 cases, `C1.h5` + `equilibrium.h5` |
| Real spectra (production) | same tree, `sdata_complex_v2.h5` | **9859** float64 |
| Complex spectra (prototype) | `/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run1/sparc_*/sdata_pertfields_grid_complex_v2.h5` | **31** complex128 |
| Staging duplicate (no complex) | `/global/cfs/projectdirs/amsc007/asvillar/data/m3dc1/` and `.../data/data/` | 113 runs, `sdata_pertfields_grid.h5` only |
| Postprocess tools (CFS) | `amsc007/data/m3dc1/postprocess_ndarray.py`, `postprocess`, `TOOLS.md` | **magnitude** pipeline |
| Postprocess tools (complex) | `pscratch/.../batch_16/post/postprocess_ndarray.py`, `run_postprocess_batch.py` | **complex** pipeline — **source of truth for step 1** |
| SURGE loaders | `scripts/m3dc1/dataset_complex_v2.py`, `loader.py` | reads either leaf; defaults `use_magnitude=True` |
| m3dc1 python | `notebooks/m3dc1/m3dc1_python_code/` **and** `~/src/M3DC1/unstructured/python/` | **divergent copies** (`eigenfunction.py` differs) |
| fusion-io | `~/.local/fusion-io/lib/{fpy.py, fio_py.so}` | required for postprocess + RZ plots |
| Scalar/profile analysis | `notebooks/analyze_sdata_10kruns_with_profiles.ipynb` | γ, Miller, q0/q95/p0 from aggregated `sdata03.h5` |
| Spectrum explorer | `notebooks/m3dc1_explore_spectra_cases.ipynb` | per-case CFS `sdata_complex_v2.h5` |
| Script backup | `amsc007/data/m3dc1/surge_postprocess_tools_backup_20260331/` | SURGE downstream tools only |

---

## Proposed `m3dc1ml` layout

```
m3dc1ml/
├── README.md
├── pyproject.toml
├── env/environment.yml
│
├── external/                        # pinned third-party physics code
│   ├── m3dc1/                       # ONE reconciled copy (from batch_16/post stack)
│   └── fusion_io/                   # fpy.py + fio_py.so (+ BUILD.md)
│
├── src/m3dc1ml/
│   ├── env.py                       # single sys.path + chdir-for-equilibrium.h5
│   ├── postprocess/
│   │   ├── build_sdata.py           # from batch_16/post/postprocess_ndarray.py (complex)
│   │   └── run_batch.py             # from run_postprocess_batch.py + Slurm array
│   ├── io/
│   │   ├── sdata.py                 # from dataset_complex_v2.py
│   │   └── loader.py
│   ├── dataset/
│   │   └── build_per_mode.py        # Parquet: amp, phase, re, im columns
│   ├── qa/
│   │   ├── validate_case.py         # schema, dtype, reconstruction_check, gamma
│   │   └── find_outliers.py         # non-converged, zero spectrum, bad γ
│   ├── viz/
│   │   ├── explore_case.py          # from m3dc1_case_viz.py
│   │   ├── scalar_profiles.py       # from 10k profiles notebook logic
│   │   └── mesh_field.py            # full mesh δp, not flux-surface-only
│   └── surrogate/                   # thin SURGE wrappers + eval
│
├── scripts/                         # CLI + Slurm launchers
├── configs/                         # SURGE YAML specs
├── notebooks/
│   ├── explore_scalars_profiles.ipynb   # ← analyze_sdata_10kruns_with_profiles
│   └── explore_spectra_complex.ipynb    # ← m3dc1_explore_spectra_cases (complex-aware)
└── data/                            # gitignored symlinks
    ├── raw_cfs → amsc007/data/m3dc1
    └── datasets/
```

**SURGE stays the training engine.** `m3dc1ml` owns M3DC1 physics I/O, postprocess, dataset build, and QA.

---

## Step 0 — Gather and refactor (inventory + copies)

**Deliverables**

1. **File manifest** — copy (not move) these into `m3dc1ml/external` and `src/m3dc1ml/`:

   | Source | Destination |
   |--------|-------------|
   | `pscratch/.../batch_16/post/postprocess_ndarray.py` | `src/m3dc1ml/postprocess/build_sdata.py` |
   | `pscratch/.../batch_16/post/run_postprocess_batch.py` | `src/m3dc1ml/postprocess/run_batch.py` |
   | `scripts/m3dc1/dataset_complex_v2.py` | `src/m3dc1ml/io/sdata.py` |
   | `scripts/m3dc1/loader.py` | `src/m3dc1ml/io/loader.py` |
   | `scripts/m3dc1/internal/build_delta_p_per_mode.py` | `src/m3dc1ml/dataset/build_per_mode.py` |
   | `scripts/m3dc1/internal/m3dc1_case_viz.py` | `src/m3dc1ml/viz/explore_case.py` |
   | `notebooks/m3dc1/m3dc1_python_code/m3dc1/` | `external/m3dc1/` (reconcile with `~/src/M3DC1/...`) |
   | `~/.local/fusion-io/lib/` | `external/fusion_io/` |

2. **`env.py`** — replace all hardcoded paths (`~/.local/fusion-io`, `~/src/M3DC1/...`, notebook-relative m3dc1 paths).

3. **Progress reporting** — every batch job writes:
   - `postprocess_status.jsonl` per case: `{case, status, elapsed_s, spec_dtype, err}`
   - `postprocess_summary.json`: `{total, ok, failed, skipped, started_at, finished_at}`

4. **Naming convention** (fix confusion):

   | Output | dtype | Use |
   |--------|-------|-----|
   | `sdata_spectrum_complex.h5` | complex128 | **canonical** phase-aware spectra |
   | `sdata_spectrum_magnitude.h5` | float64 | legacy / quick surrogates |

   Deprecate misleading `sdata_complex_v2.h5` name in new runs.

**Exit criteria:** `pip install -e m3dc1ml` + one-case postprocess + one-case load works from a clean env.

**Status (2026-06-09):** Step 0 **done** in `SURGE/m3dc1ml/` — manifest copied, `env.py` + `run_batch.py` progress logs, canonical `sdata_spectrum_complex.h5`, `pip install -e .` verified in surge conda env, prototype load (`sparc_1300` complex128) OK. Step 2 will postprocess remaining ~9859 CFS cases.

---

## Step 1 — Verify complex HDF5 architecture

**Goal:** prove the 31 prototype files are physically correct before scaling to 9859.

### 1.1 Document the math (`_compute_complex_spectrum`)

For toroidal mode `n` (`ntor` from parset), at flux coordinates `(θ, ψ_N)`:

1. Evaluate δfield at **φ = φ₀** and **φ = φ_q** (quadrature phase).
2. `ef_phi0 = δp(φ₀)`, `ef_phiq = δp(φ_q)`.
3. **Complex combination:** `ef_hat = ef_phi0 - i · ef_phiq` (preserves poloidal phase).
4. **Poloidal FFT** along θ → `spec[m, ψ_N]` (complex128).
5. **`reconstruction_check`:** verify `Re(ef_hat)` and `Re(ef_hat · e^{inφ_q})` match direct evaluations.

### 1.2 Validation notebook / script (`qa/validate_case.py`)

For each of the **31 reference cases** and **5 CFS float cases** (contrast):

| Check | Pass criterion |
|-------|----------------|
| `spec.dtype` | complex128 |
| `np.max(np.abs(np.imag(spec)))` | > 0 for converged cases |
| `reconstruction_check` errors | < 1e-8 (stored in HDF5) |
| `|spec|` vs CFS float `sdata_complex_v2` | correlate on same case (where both exist) |
| Time slice | slice 1 (or -1) has non-zero δp; slice 0 often zero |
| `m_modes` | full FFT: 200 modes, `-100…99` |
| IFFT round-trip | `recon_real_from_spectrum` vs `eval_field` on (R,Z) |

### 1.3 Compare postprocess script versions

- Diff `batch_16/post/postprocess_ndarray.py` vs `amsc007/data/m3dc1/postprocess_ndarray.py`.
- **Lock** the complex version as the only production script in `m3dc1ml`.
- Add unit test: synthetic `ef_phi0`, `ef_phiq` → known phase spectrum.

**Exit criteria:** written validation report on 31 cases; signed-off math doc; no ambiguity on dtype.

**Status (2026-06-09):** `m3dc1ml-validate-case` run on 31 prototypes → **30/31 pass** (`sparc_1359` HDF5 corrupt/truncated; re-postprocess). All readable files: complex128, full FFT m∈[-100,99], reconstruction_check < 1e-8, slice 1 active. CFS `sdata_complex_v2.h5` contrast: |spec|≈0 on same cases (confirms magnitude pipeline bug, not prototype).

---

## Step 2 — Queue full-dataset postprocessing

**Target:** all ~9859 CFS cases → `sdata_spectrum_complex.h5` (complex128) in place next to `C1.h5`.

### 2.1 Input

- Raw cases: `/global/cfs/projectdirs/amsc007/data/m3dc1/run*/sparc_*/`
- Require: `C1.h5`, `equilibrium.h5`, `time_001.h5` (or discover time slices)
- Skip if already postprocessed (resume-safe)

### 2.2 Slurm array design

```bash
# One array task per sparc_* case (or per run with inner loop)
# Example: 9859 tasks, 1 case each, ~5–15 min/case
python -m m3dc1ml.postprocess.run_batch \
  --root /global/cfs/projectdirs/amsc007/data/m3dc1 \
  --out-name sdata_spectrum_complex.h5 \
  --full-fft \
  --grid-mode grid --grid-res 200 \
  --status-jsonl postprocess_status.jsonl
```

- **Concurrency:** respect CFS I/O; start with 50–100 concurrent nodes.
- **Resume:** skip cases where output exists and `reconstruction_check` passes.
- **Failure log:** append to `postprocess_failures.txt` for manual retry.

### 2.3 Output audit (after job)

```bash
python -m m3dc1ml.qa.inventory \
  --root /global/cfs/projectdirs/amsc007/data/m3dc1 \
  --filename sdata_spectrum_complex.h5
# Expect: ~9859 complex128, 0 missing, dtype histogram
```

### 2.4 Build ML tables

```bash
python -m m3dc1ml.dataset.build_per_mode \
  --root /global/cfs/projectdirs/amsc007/data/m3dc1 \
  --filename sdata_spectrum_complex.h5 \
  --out data/datasets/delta_p_per_mode_complex.parquet \
  --columns amp,phase   # or re,im — see step 4
```

**Exit criteria:** ≥98% cases with complex128 HDF5; Parquet manifest with row count ~1.97M (9859 × 200 modes).

---

## Step 3 — Visualizations and QA statistics

Two complementary notebooks (merged from existing work):

### 3.1 Scalar / profile layer (`explore_scalars_profiles.ipynb`)

From `analyze_sdata_10kruns_with_profiles.ipynb`:

- γ, p0, q0, q95, qmin, Miller (δ, κ, R₀, a, ε)
- Input space: ntor, pscale, batemanscale
- Correlations, distributions, parameter sweeps

**Data source:** can use existing aggregated `sdata03.h5` / per-case `flux_average` from complex HDF5.

### 3.2 Spectrum / complex layer (`explore_spectra_complex.ipynb`)

From `m3dc1_explore_spectra_cases.ipynb`, extended for **complex**:

| Panel | Content |
|-------|---------|
| \|δp\|(m, ψ) | log-scale heatmap |
| phase(m, ψ) | new |
| Re(δp), Im(δp) on (R,Z) | from complex spectrum IFFT |
| γ, equilibrium line | already added |

**Global statistics dashboard:**

- Spectrum energy vs γ
- Dominant m vs equilibrium shape
- Fraction of zero / non-converged cases

### 3.3 Outlier / convergence filter (step 3.1)

Flag cases where:

| Flag | Rule |
|------|------|
| `no_growth` | γ ≈ 0 or missing |
| `zero_spectrum` | max \|spec\| < ε on all time slices |
| `bad_recon` | `reconstruction_check` > 1e-6 |
| `missing_equilibrium` | no `equilibrium.h5` |
| `single_time_slice` | only equilibrium slice available |

Write `qa/case_flags.parquet` keyed by `run/sparc` for downstream training masks.

**Exit criteria:** notebooks run on full index; `case_flags.parquet` generated; known outlier count documented.

---

## Step 4 — SURGE surrogate experiments

### 4.1 Spectrum-domain models (per-mode)

**Targets (try in order):**

1. `|δp(m, ψ)|` — reproduce current baseline
2. `phase(m, ψ)` — new, requires complex postprocess
3. `(Re, Im)` or `(amp, phase)` jointly — multi-output MLP

**Inputs:** `PER_MODE_INPUT_COLS` (12) unchanged.

**Configs:** `m3dc1ml/configs/surge_per_mode_{magnitude,phase,complex}.yaml`

### 4.2 Field-domain models (2D on M3DC1 mesh)

**Problem with current explorer:** `flux_coordinates` → `(rpath, zpath)` is a **flux-surface-aligned 200×200 grid** (LCFS topology), **not** the unstructured M3DC1 mesh.

**For true mesh-based learning:**

| Approach | Grid | Source |
|----------|------|--------|
| **A. Mesh vertices** | ~N_vertex points (irregular) | `postprocess` with `--grid-mode mesh` → `pertfields/p` in HDF5 |
| **B. Regular RZ grid** | 200×200 over bounding box | `postprocess --grid-mode grid` |
| **C. Graph / point cloud** | mesh connectivity from `fpy` | future: GNN on elements |

**Recommended path:**

1. Store **mesh-vertex δp** (complex) in HDF5: extend `build_sdata.py` to save `pertfields/p` as complex array on mesh nodes (from `ef_hat` evaluated per vertex).
2. Train surrogate: inputs → **vector of δp at mesh nodes** (or downsampled mesh).
3. Eval: compare predicted vs true on **same mesh nodes**, plot with `plot_mesh` — not flux-surface pcolormesh.

### 4.3 Experiment matrix

| ID | Target | Representation | Model |
|----|--------|----------------|-------|
| E1 | \|δp(m,ψ)\| | per-mode profile | MLP (baseline, trial 12) |
| E2 | phase(m,ψ) | per-mode profile | MLP |
| E3 | δp(R,Z) mesh | vertex field | MLP / conv on interpolated grid |
| E4 | δp(R,Z) mesh | vertex field (complex) | two-head MLP (Re, Im) |

**Exit criteria:** E1 reproduces trial-12 metrics on magnitude; E2/E4 show feasibility on held-out cases; mesh eval plots on ≥10 test cases.

---

## Suggested execution order

```
Step 0 (1 week)     → m3dc1ml repo skeleton, env.py, copy scripts
Step 1 (3–5 days)   → validate 31 complex cases, lock postprocess math
Step 2 (1–2 weeks)  → Slurm postprocess 9859 cases + Parquet build
Step 3 (1 week)     → notebooks + case_flags QA
Step 4 (ongoing)    → SURGE experiments magnitude → phase → mesh
```

---

## Immediate next actions

1. **Copy** `batch_16/post/postprocess_ndarray.py` into SURGE repo (or start `m3dc1ml`) — this is the **only** script that produced true complex128.
2. **Run validation** on 3 of the 31 cases: dtype, reconstruction_check, IFFT round-trip.
3. **Pilot postprocess** 10 CFS cases with complex script; confirm output lands as complex128 on CFS.
4. **Do not** use CFS `sdata_complex_v2.h5` for phase-aware training without reprocessing.

---

## References

- `docs/m3dc1/M3DC1_DELTA_P_SPECTRA_SURROGATE_WORKFLOW.md` — SURGE training/eval
- `amsc007/data/m3dc1/TOOLS.md` — legacy magnitude postprocess
- `amsc007/data/m3dc1/surge_postprocess_tools_backup_20260331/SDATA_COMPLEX_V2_WORKFLOW.md` — downstream only
- `pscratch/.../batch_16/post/postprocess_ndarray.py` — **complex spectrum source of truth**
