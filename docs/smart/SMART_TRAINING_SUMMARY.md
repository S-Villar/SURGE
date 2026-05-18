# SMART surrogate training summary

This note consolidates **metrics**, **HPO**, **inference timing**, **plots**, **growth-rate scaling/sign**, and **input/output policy** for TokaMaker-style SMART runs in this repo.

## Real TokaMaker 10k bundle (full magnetics + VDE growth rate)

- **Curated pickle:** `data/datasets/SMART/smart_curated_10k_equil_magnetics.pkl` from  
  [`scripts/smart/build_curated_from_equil_csv.py`](../../scripts/smart/build_curated_from_equil_csv.py)  
  reading `Equil_data.csv`.
- **γ source:** the string column **`VDE growthrate`** (length‑3 vector per row), reduced to a **scalar** with `--gamma-vde-mode` (default **`first`** = first entry, large |γ|; **not** `VDE tau gamma`).
- **Sign:** `gamma_TOKAM = -gamma_VDE` (TokaMaker convention).
- **Inputs for new surrogates:** all nine A‑turn columns: **Solenoid, DIV1/2 up/lo, PF1/2 up/lo** (`I_SOL`, `I_DIV_*`, `I_PF*_up/lo`).

Workflows: `smart_magnetics_shaping_hpo.yaml`, `smart_magnetics_gamma_hpo.yaml`.

## Where the tables and figures live

- **Machine-readable metrics + HPO columns:**  
  [`data/datasets/SMART/reports/surrogate_pack/metrics_summary.csv`](../../data/datasets/SMART/reports/surrogate_pack/metrics_summary.csv)
- **Same table in Markdown:**  
  [`data/datasets/SMART/reports/surrogate_pack/metrics_summary.md`](../../data/datasets/SMART/reports/surrogate_pack/metrics_summary.md)
- **Figures (MLP HPO traces, test-split regression panels, RF feature importance):**  
  [`data/datasets/SMART/reports/surrogate_pack/figures/`](../../data/datasets/SMART/reports/surrogate_pack/figures/)

Regenerate after new runs:

```bash
cd /path/to/SURGE
PYTHONPATH=. python scripts/smart/generate_smart_surrogate_report.py
```

## Column definitions (metrics summary)

| Column | Meaning |
|--------|---------|
| `R2_train`, `R2_test` | sklearn R² on train / test (multi-output uses uniform average). |
| `t_tr_s` | Wall-clock **training** time for the final model after HPO (seconds), from `metrics.json`. |
| `t_I_us_per_row` | **Inference:** mean time per **test row** (one forward pass, all outputs), from `test_inference_per_sample` × 10⁶ → **microseconds**. |
| `HPO_metric`, `HPO_direction` | Optuna objective (e.g. `val_r2` / `maximize`). |
| `HPO_n_trials` | Number of Optuna trials completed. |
| `HPO_best_objective` | Best trial objective used to select hyperparameters before final retrain. |

Per-output timing: SURGE stores one timing per row; for 8 shaping targets, a single forward pass predicts all eight.

## Growth rate: why γ looks like −1…1, normalization, and sign

1. **No z-score normalization in curation**  
   [`scripts/smart/build_curated_smart.py`](../../scripts/smart/build_curated_smart.py) only copies the table and sets  
   **`gamma_TOKAM = −(raw_gamma_column)`**  
   (TokaMaker sign convention). It does **not** divide by σ or map to [-1, 1].

2. **Why ranges can sit in a band like [-1.5, 1]**  
   - TokaMaker / stability postprocessors may export γ in **normalized** or **model-specific** units.  
   - The **synthetic** pickle used in this repo was generated with a hand-built formula; its **γ** is not physical s⁻¹. On that table, **`gamma`** spans roughly **[-0.95, 1.52]** and **`gamma_TOKAM`** the negated range **[-1.52, 0.95]** — that is **not** proof of z-normalization; it is the range of the generator.  
3. **Checking the real export (`Equil_data.csv`)**  
   There is **no `Equil_data.csv` in this git tree**. You need to open the file **on the machine or share where you obtained `df_equil_database.pkl`**, compare the raw γ column (min/max, units in the TokaMaker docs), and confirm whether the code that wrote the CSV already scaled γ. SURGE does not rescale it beyond the **single minus sign** above.

## Surrogate I/O policy (shaping as **output**, not duplicated as causal **input**)

- **Currents + profiles → shaping**  
  Target **outputs** should be **`R0`, `Z0`, `a`, `A`, `kappa_u`, `kappa_l`, `delta_u`, `delta_l`** (or a subset). **Inputs** are **`I_PF*`** and **profile** scalars only — see [`metadata_currents_profiles_to_shaping.yaml`](../../configs/smart/metadata_currents_profiles_to_shaping.yaml). Do **not** add the same shape scalars as inputs when they are the quantities you are predicting.

- **γ regression without equilibrium leakage**  
  If γ must not depend on the **final** equilibrium shape (because those are unknown before the solve), use **actuators + profiles only** as inputs — see [`metadata_coils_profiles_to_gamma.yaml`](../../configs/smart/metadata_coils_profiles_to_gamma.yaml) and `smart_coils_profiles_gamma_hpo.yaml`.

- **Leakage warning**  
  [`metadata_shapes_to_gamma.yaml`](../../configs/smart/metadata_shapes_to_gamma.yaml) includes **R0, a, κ, δ** as *inputs*. That is fine for **diagnostics / interpolation** on a fixed database, but it is **circular** if those inputs are only known after the same equilibrium you are emulating. A comment was added at the top of that metadata file.

## Optimization metrics (HPO)

For each run with HPO, JSON under `runs/<tag>/hpo/<model>_hpo.json` contains:

- `metric`, `direction`, `best_trial` (value + params + trial id),
- full `trials` list with `value`, `params`, and user attrs `val_r2` / `val_rmse` when recorded.

**Plots:** MLP **per-trial objective** and **best-so-far trace** are under `reports/.../figures/*_mlp*_hpo_trace.png`.

## RF feature importance

Bar charts use **Gini importances** from the final `RandomForestRegressor` after HPO (`*_rf_*_feature_importance.png`). If `joblib` load fails in your environment (NumPy/GPflow/TF import side effects), rerun the report in a **clean `surge` conda env** with compatible NumPy.

## Run overview (representative)

Strong **γ** generalization on the local curated table is seen when shape/profile information is in the feature set (e.g. `smart_gamma_*_hpo`). **Currents+profiles → shaping** and **coils+profiles → γ** remain **poor on the synthetic generator** because currents and profiles were not coupled to geometry/γ in that synthetic build. Re-evaluate on **real TokaMaker** exports.

---

*Generated workflow artifacts: `runs/smart_*` directories; see CSV above for numeric comparison.*
