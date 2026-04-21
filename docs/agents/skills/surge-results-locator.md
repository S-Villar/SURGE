---
skill_id: surge.results.locator
title: Locate growth-rate vs delta-p metrics and figures
version: "1.0.0"
domain: [surge, m3dc1, documentation]
academy_blueprint_hint: null
---

## Objective

Resolve user questions about **“latest results,” R², RMSE, or plots** by routing to the correct **target** (growth rate **γ** vs **δ*p* per-mode**) and the right **files**.

## When to apply (triggers)

- User asks for metrics or figures without naming **γ** or **δ*p***.
- User conflates **`m3dc1_aug_r75`** artifacts with **`m3dc1_trial12_*`** (or other δ*p* run tags).

## Preconditions

- Repo checkout with optional local **`runs/`** (may be gitignored or symlinked).

## Procedure

1. **Disambiguate:** Ask or infer: **scalar γ** (`output_gamma`) vs **δ*p* profiles / Parquet** workflows.
2. **If γ:**
   - Authoritative metrics: **`runs/m3dc1_aug_r75/metrics.json`**
   - Human table: **`m3dc1/results.md`** (Growth-rate section)
   - Checked-in figures: **`m3dc1/figures/gamma_aug_r75_*.png`**, **`gamma_demo_*.png`**
3. **If δ*p*:**
   - Metrics: **`runs/<run_tag>/metrics.json`**
   - Configs: e.g. **`configs/m3dc1_delta_p_per_mode_cfs_mlp_trial12_single.yaml`**
   - Eval / maps: **`scripts/m3dc1/eval_test_delta_p_2d_maps.py`**, **`runs/.../plots/`**
4. Note **`runs/`** may be absent on a fresh clone; **`m3dc1/results.md`** may still hold **snapshots**.

## Verification

- User can open cited **`metrics.json`** or figures and see **consistent** model keys and run tags.

## Failure modes & diagnostics

| Symptom | Likely cause | Next step |
|---------|----------------|-----------|
| No `runs/m3dc1_aug_r75` | Not copied / ignored | Use doc snapshot or train |
| “Latest” unclear | Multiple δ*p* tags | List **`runs/`** by mtime or ask user |

## Artifacts & code references

- `m3dc1/results.md`
- `docs/agents/CONTEXT.md`
- `examples/viz_m3dc1_predictions.py` (γ viz helper)

## Future: Academy mapping

- **Tool:** `resolve_results_query(target: Literal["gamma","delta_p"], run_tag: str | None) -> ResultPointers`
- **Action:** read-only file resolution for user-facing summaries.
