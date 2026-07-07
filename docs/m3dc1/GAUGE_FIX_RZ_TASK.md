# Gauge-fix RZ field training (Task 4)

Branch: `feat/m3dc1-gauge-fix`  
Base commit: `9b9f28f`

## Hypothesis

Signed Re(δp)(R,Z) is gauge-dependent (global phase θ₀ arbitrary). Max-norm
removes amplitude but not phase → dataset mean signed field is incoherent →
L2-optimal predictor ≈ 0 (relL2 ≈ 1).

## Step 2 diagnostic — PASS (500 cases)

| Metric | BEFORE | AFTER gauge-fix |
|--------|--------|-----------------|
| L2(mean signed field) | 0.910 | **2.389** |
| coherence ratio | 0.048 | **0.126** |
| peak(mean) | 0.047 | **0.266** |
| coherence gain | — | **2.61×** (≥ 2.0 required) |

Artifacts: `runs/gauge_diagnostic/gauge_diagnostic_report.json`,  
`runs/gauge_diagnostic/gauge_diagnostic_mean_fields.png`

## New modules / flags (all default OFF)

| File | Purpose |
|------|---------|
| `scripts/m3dc1/internal/gauge_fix.py` | θ_ref + rotation |
| `scripts/m3dc1/internal/gauge_diagnostic.py` | zero-GPU pass/fail gate |
| `train_rz_field_image.py` flags | `--gauge-fix`, `--complex-target`, `--no-target-zscore`, `--phase-align-eval` |

## Step 5 launch (Perlmutter)

```bash
salloc -N1 -C gpu -G1 -q interactive -A mp288 -t 04:00:00
cd /global/homes/a/asvillar/src/SURGE
bash scripts/m3dc1/internal/run_RZ_gaugefix_complex.sh
```

Output: `runs/rz_field_gaugefix_complex_g201/`

Compare against:
- Failed direct Re(δp): relL2 med ≈ **1.0** (`runs/rz_field_fno48_re_deltap_smooth0`)
- Spectrum + oracle phase: relL2 med ≈ **0.47** (D′ fieldloss)
