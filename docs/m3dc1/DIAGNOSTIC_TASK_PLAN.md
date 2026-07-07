# Diagnostic task plan — stability, gs_error, |δp̂|²

Three follow-up workstreams after the high-m tail scan (§4.5) and direct RZ
training failure. Status as of 2026-07-07.

---

## Task 1 — Compare γ for stable vs unstable; label cases in tail plots

### Convention (verified in codebase)

- **Unstable:** `growth_rate/0 > 0` in `csdata_deltap_b_ver.h5`
- **Stable:** `growth_rate/0 ≤ 0`
- Same rule in `curate_validate_mlm3dc1_predictions.ipynb`

### What the attached plots actually show

The three panels in `m_hi_tail_scan.png` are **spectral tail diagnostics**, not
stability labels:

| Plot | Quantity | Red threshold |
|------|----------|---------------|
| Histogram | Fraction of **\|δp̂\|²** energy at **m > 20** | `frac_hi ≥ 0.01` |
| peak m vs tail | Peak-m row vs `frac_hi` | vertical m=20, horizontal 0.01 |
| peak ψ_N vs tail | Radial peak vs `frac_hi` | horizontal 0.01 |

**Do not equate `frac_hi ≥ 0.01` with unstable.** High-m tail and γ are correlated
but distinct.

### Verified counts (9,976 csdata cases, Jul 2026 scan)

| Split | n | Share unstable |
|-------|---:|---:|
| **All cases** | 9,976 | 4,877 (48.9%) |
| **Stable** (γ≤0) | 5,099 | — |
| **Unstable** (γ>0) | 4,877 | — |
| **Lo tail** (`frac_hi < 0.01`) | 6,622 | 33.9% |
| **Hi tail** (`frac_hi ≥ 0.01`) | 3,354 | **78.4%** |

Cross-tab (hi-tail threshold 1%):

|  | stable | unstable | total |
|--|-------:|---------:|------:|
| lo tail | 4,374 | 2,248 | 6,622 |
| hi tail | 725 | 2,629 | 3,354 |

γ statistics:

| Group | median γ | p90 γ |
|-------|----------|-------|
| All unstable | +0.034 | +0.094 |
| All stable | −0.018 | −0.003 |
| Hi-tail unstable | +0.023 | +0.095 |
| Lo-tail unstable | +0.051 | +0.065 |

Interpretation:

- **Hi-m tail correlates with instability** (~78% of hi-tail cases are unstable vs
  ~34% of lo-tail), but **725 stable cases** still have ≥1% energy at m>20.
- The **upper-right cluster** (peak m > 20, `frac_hi` → 1) is only **31 cases**
  (`n_peak_m_gt_hi` in scan summary) — **all 31 are unstable** (γ > 0).
- The **ψ_N ≈ 1 vertical spike** mixes stable and unstable; edge localization alone
  does not imply γ > 0.

### Deliverables

- [x] Cross-tab script / numbers (this doc)
- [ ] Run `plot_stability_hi_m_tail.py` → `docs/m3dc1/assets/m_hi_tail_stability.png`
- [ ] Table: top-25 hi-tail cases with γ, stable/unstable, peak m, peak ψ_N
- [ ] Compare **median γ** unstable-in-hi-tail vs unstable-in-lo-tail (box/violin)
- [ ] Optional: export `runs/mwindow_scan/case_stability_tail.parquet` keyed by
  `run_id_eq_id` for explorer filters

### Commands

```bash
# Stability overlay on §4.5 plots (~2 min on login node)
python scripts/m3dc1/internal/plot_stability_hi_m_tail.py \
  --tail-json runs/mwindow_scan/m_hi_tail_full.json \
  --out-fig docs/m3dc1/assets/m_hi_tail_stability.png \
  --out-json runs/mwindow_scan/m_hi_tail_stability_summary.json

# Rebuild tail JSON if needed
python scripts/m3dc1/internal/scan_m_hi_tail.py --n-cases 0 --m-hi 20 \
  --min-frac 0.01 --out-json runs/mwindow_scan/m_hi_tail_full.json \
  --out-fig docs/m3dc1/assets/m_hi_tail_scan.png
```

---

## Task 2 — `gs_error` distribution and drivers

### Source found (2026-07-07): M3DC1 log files, not csdata

`gs_error` is **not** in `csdata_deltap_b_ver.h5`. It is extracted per case from
the M3DC1 stdout log:

```text
Final error in GS solution:    3.1104696235382447E-002
```

Older batches (`batch_11`, `batch_13`) have a `check.sh` that writes
`sparc_*/gs_error` via:

```bash
grep "Final error in GS solution" "$slurm" | awk '{ print $6 }' > $dir/gs_error
```

**`batch_16`:** `M3DC1log.o*` files contain the line, but `gs_error` text files
were **not** pre-extracted. Example:
`run12/sparc_1429/M3DC1log.o45115191` → GS error ≈ `3.11e-2`.

Harvest command for one case:

```bash
grep "Final error in GS solution" M3DC1log.o* | awk '{print $6}'
```

### Proxies available today

| Proxy | Location | Interpretation |
|-------|----------|----------------|
| `reconstruction_check/p_*_err_phi0` | csdata | FFT helical round-trip (≈0, not GS) |
| `growth_rate/0` | csdata | Linear stability eigenvalue |
| `output_gamma` | sdata03 CSV | Same γ, flat-file export |
| M3DC1 control errors | `C1.h5/scalars/*err*` | Time-step control, not equilibrium |

### Action items

1. **Bulk-harvest** `gs_error` from `M3DC1log.o*` across `batch_16/run*/sparc_*`
   (regex + fallback to `M3DC1log.e*`); write
   `data/datasets/SPARC/gs_error_ver.parquet` keyed by `(run_id, eq_id)`.

2. **Join** into `case_scalars_ver.parquet` and wire
   `gs_error_for_case()` in `curate_validate_mlm3dc1_predictions.ipynb`.

3. **Fallback proxies** if a case log is missing:
   - `reconstruction_check` max error (sanity / postprocess quality)
   - |γ| and hi-m tail fraction as confounders
   - Spectrum prediction relL2 from `predictions_cache.npz`

4. **Once sourced, build:**
   ```bash
   python scripts/m3dc1/internal/build_case_scalar_dataset.py ... \
     --out data/datasets/SPARC/case_scalars_with_gs.parquet
   ```
   Extend builder with `--extra-csv gs_error.csv` join.

5. **Plots (mirror NSTX notebook §6.6):**
   - Histogram of `gs_error`
   - Scatter: `gs_error` vs γ, vs q0/q95/p0, vs `frac_hi`
   - Scatter: `gs_error` vs spectrum / field prediction error
   - Correlation bar chart vs equilibrium scalars

### Blocker

Resolved: GS equilibrium residual from M3DC1 logs. Remaining work is bulk
extraction + plotting (no longer blocked on definition).

---

## Task 3 — Work on |δp̂|²

Several concrete meanings; pick one primary target:

### 3A — Spectral power target (training)

Current surrogate: **log₁₀|δp̂|(m, ψ)** (`train_spectrum_image.py`).

|δp̂|² alternatives:

| Target | Pros | Cons |
|--------|------|------|
| `log10(|δp̂|²)` = `2·log10|δp̂|` | Linear in power; emphasizes peaks | Dynamic range; redundant with mag |
| `\|δp̂|²` raw (max-norm) | Direct power image | Very sparse; same phase problem |
| `log10|δp̂|` (status quo) | Stable training | Underweights weak modes |

**Suggested experiment:** `--target-space power` → `log10(|δp̂|²+ε)` with same D′
recipe; compare patR² and field relL2. Implementation: ~20 lines in
`train_spectrum_image.py`.

### 3B — Integrated |δp̂|² metrics (analysis)

Already computed in `scan_m_hi_tail.py`:

```python
E = (mag ** 2).sum(axis=1)   # power vs m at fixed ψ sum
frac_hi = E[m > 20].sum() / E.sum()
```

Extensions:

- Per-case **total spectral energy** `Σ|δp̂|²`
- **Band powers:** m∈[-80,20] vs m>20 vs m<-80
- Correlate band powers with γ, gs_error (once sourced), prediction error
- Notebook section in `explore_csdata_deltap_b_test.ipynb`

### 3C — Direct RZ |δp|² (bypass spectrum)

Motivated by Run R failure on signed Re(δp):

- Target: `|δp(R,Z)|²` or `|δp|` at φ₀ from `pertfields/p_phi0` + `p_phiq`
- Drop global z-score on Y; masked loss on high-amplitude pixels
- Compare to spectrum+IFFT path

**Suggested order:** 3B (analysis, 1 day) → 3A (one ablation run) → 3C (only if
spectrum power target does not close field gap).

### Commands (3B analysis)

```bash
python scripts/m3dc1/internal/scan_m_hi_tail.py --n-cases 0 \
  --out-json runs/mwindow_scan/m_hi_tail_full.json

python scripts/m3dc1/internal/plot_hi_m_tail_scatter.py \
  --tail-json runs/mwindow_scan/m_hi_tail_full.json \
  --cache runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0/predictions_cache.npz:qc \
  --out docs/m3dc1/assets/hi_m_tail_r2_scatter.png
```

---

## Suggested priority

1. **Task 1** — run stability overlay plot; add γ column to explorer (unblocks
   interpretation of attached figure).
2. **Task 2** — resolve `gs_error` source (blocked on definition).
3. **Task 3B → 3A** — |δp̂|² band analysis, then optional power-target training run.

---

## Files

| File | Role |
|------|------|
| `scripts/m3dc1/internal/scan_m_hi_tail.py` | Tail scan (uses \|δp̂\|²) |
| `scripts/m3dc1/internal/plot_stability_hi_m_tail.py` | γ overlay (new) |
| `runs/mwindow_scan/m_hi_tail_full.json` | 9,974-case tail fractions |
| `docs/m3dc1/assets/m_hi_tail_scan.png` | Attached figure |
| `m3dc1ml/notebooks/curate_validate_mlm3dc1_predictions.ipynb` | γ vs prediction error |
