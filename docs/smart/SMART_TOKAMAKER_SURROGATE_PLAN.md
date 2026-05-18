# SMART TokaMaker – SURGE Surrogate Development Plan

**Database:** `SMART_10k_GR_Guess_-5E5` (~10⁴ equilibria)  
**Sources:** `df_equil_database.pkl` or `Equil_data.csv` (same logical table)

This plan defines three regression surrogates, one classifier, preprocessing for the TokaMaker γ sign convention, and expanded evaluation artifacts aligned with the VDE surrogate slide (feature importance, R² scatter, classifier metrics).

---

## 0. Repository status

After `git pull origin ai4fusion-dev`, run each surrogate family with **one config and one CLI invocation** (three runs total):

```bash
surge run configs/smart/smart_shapes_to_gamma.yaml --run-tag smart_shapes_gamma
surge run configs/smart/smart_currents_to_shaping.yaml --run-tag smart_currents_shaping
surge run configs/smart/smart_coils_to_gamma.yaml --run-tag smart_coils_gamma
```

Equivalent: `python -m examples.m3dc1_workflow --spec <same-yaml> --run-tag <tag>`.

Classifier training uses `scripts/smart/train_smart_classifier.py` (SURGE workflow is regression-oriented).

### Exploratory plots (equilibrium database)

After `build_curated_smart.py`, plot **gamma_TOKAM** (signed target) vs shaping:

1. **δ and ε = a/R₀:** scatter x = triangularity δ (`delta_l`, `delta_u`, or average — match your columns), y = **ε = a/R₀**, color = gamma_TOKAM.
2. **R₀ and a:** scatter x = R₀, y = a, color = gamma_TOKAM.

Implement as `scripts/smart/plot_smart_equil_exploratory.py` (curated pickle path + `--out` figures dir). Use hexbin or alpha if point overlap is high (~10⁴ points).

---

## 1. Sign convention (mandatory preprocess)

TokaMaker exports γ with opposite sign to the convention used elsewhere (e.g. TokaMaker / SMART stability plots).

Define:

```text
gamma_TOKAM = -gamma_dataset
```

**Implementation:** `scripts/smart/build_curated_smart.py` adds `gamma_TOKAM` and writes `smart_curated_<bundle>.pkl` plus optional CSV. All regression targets for “stability” use `gamma_TOKAM`.

---

## 2. Surrogate tasks

### Task A – Shaping + profiles → γ (primary benchmark)

**Goal:** Match slide targets: R² > 0.98 on VDE growth rate.

**Inputs (illustrative – reconcile with your CSV after `inspect_smart_data.py`):**

- Current / magnetic: `I_p`, `B_t`, `l_i`, `Phi_dia`, `Phi_tor` (or exact spellings in file)
- Geometry: `R_0`, `Z_0`, `a`, `A`, `kappa_u`, `kappa_l`, `delta_u`, `delta_l`, `V`
- Profiles / stability scalars: `q_0`, `q_95`, `p_0`, `W`, `beta_pol`, `beta_tor`

**Output:** `gamma_TOKAM`

**Workflow:** `configs/smart/smart_shapes_to_gamma.yaml`  
**Metadata:** `configs/smart/metadata_shapes_to_gamma.yaml` (edit column lists to match data)

### Task B – Plasma / circuit currents → shaping

**Goal:** Predict equilibrium shaping scalars (subset of geometry: e.g. `kappa_u`, `kappa_l`, `delta_u`, `delta_l`, `a`, `R_0`, `Z_0`, …) from coil / current drives.

**Inputs:** Columns identified as currents (e.g. PF coil currents, `I_*`, `Ic_*` – **discover via inspector**).

**Outputs:** Multi-output regression (one model per target or multi-output RF). Phase 1: single high-value shape parameter; expand to vector in Phase 2.

**Workflow:** `configs/smart/smart_currents_to_shaping.yaml` (after listing `input_cols` / `output_cols`)

### Task C – Coil currents → stability (γ)

**Goal:** Direct map from PF / coil currents to `gamma_TOKAM` (control-oriented path).

**Inputs:** Same current columns as Task B (possibly extended).

**Output:** `gamma_TOKAM`

**Workflow:** `configs/smart/smart_coils_to_gamma.yaml`

---

## 2b. MLP-focused SURGE workflows and uncertainty

**Preferred models:** Use `torch.mlp` in each SMART YAML (see `configs/m3dc1_demo.yaml` for a full parameter block). Tune `hidden_layers`, `epochs`, `batch_size`, `learning_rate`, `dropout`.

**UQ built into SURGE for a single MLP:** `request_uncertainty: true` with **`mc_dropout_passes`** (MC-Dropout at inference). Implemented in `surge/model/pytorch.py` → `predict_with_dropout`.

**Deep ensemble (K MLPs, different initial weights):** Not a single first-class adapter today. Options:

1. **Post-hoc:** Train K runs with different effective random seeds and combine prediction CSVs (mean & std across runs).
2. **YAML-only (after small code change):** Add `torch.manual_seed(...)` in `PyTorchMLPModel.fit` before instantiating `PyTorchMLP`, exposed as e.g. `params.torch_manual_seed`; then list K `torch.mlp` entries in one spec with different seeds.
3. **New adapter:** `torch.mlp_ensemble` that wraps K members and implements `predict_with_uncertainty` from ensemble variance.

For VDE benchmarks, **MC-Dropout + optional Random Forest** in the same workflow often suffices; add (2) or (3) if you need explicit deep-ensemble epistemic uncertainty.

---

## 3. Classifier – Diverted / configuration label

**Goal:** High precision diverted configuration (slide: precision ~0.995, AUROC ~0.9998 – use as stretch goal).

**Target column:** Binary `diverted` or equivalent (or derived rule: e.g. `is_limited` vs diverted). **Confirm column name in raw data.**

**Script:**

```bash
python scripts/smart/train_smart_classifier.py \
  --data data/datasets/SMART/smart_curated_shapes_gamma.pkl \
  --target diverted \
  --features shaping_profiles_or_coils \
  --out runs/smart_classifier_diverted
```

---

## 4. Execution sequence

1. **Obtain data:** Copy `SMART_10k_GR_Guess_-5E5` under `data/datasets/SMART/` (or set absolute paths in YAML).
2. **Inspect:**
   ```bash
   python scripts/smart/inspect_smart_data.py --path data/datasets/SMART/df_equil_database.pkl
   ```
3. **Curate:** Add `gamma_TOKAM`, drop NaNs, optional row filters.
   ```bash
   python scripts/smart/build_curated_smart.py \
     --input data/datasets/SMART/df_equil_database.pkl \
     --gamma-column <auto_or_name> \
     --out data/datasets/SMART/smart_curated_shapes_gamma.pkl
   ```
4. **Edit metadata YAMLs** with exact column names from step 2.
5. **Train regressions** (three configs, three `--run-tag` values).
6. **Train classifier** with `scripts/smart/train_smart_classifier.py`.
7. **Expanded figures:**
   ```bash
   python scripts/smart/plot_smart_regression_pack.py --run-dir runs/smart_shapes_to_gamma --model-prefix smart_rf_gamma
   ```
   Produces `plots/smart_gamma_scatter_test.png` and `plots/smart_feature_importance.png`. Add ROC/PR for the classifier with a small follow-on script if needed.

---

## 5. Metrics and “expanded” results pack

| Deliverable | Task A/B/C | Classifier |
|-------------|------------|------------|
| Train / val / test R², RMSE, MAE | ✓ | Accuracy, precision, recall, F1 |
| UQ (RF / GP / MC-dropout MLP) | ✓ optional | Probabilities + AUROC |
| Feature importance bar chart | ✓ | Optional permutation importance |
| Prediction vs truth scatter | ✓ | Confusion matrix |
| Wall-clock speedup vs TokaMaker | Document in README | N/A |

Store plots under `runs/<tag>/plots/` or `smart/figures/`.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Column names ≠ slide notation | Use `inspect_smart_data.py`; YAML lists are explicit |
| Multi-output shaping (Task B) | Start one output; extend with `MultiOutputRegressor` or separate specs |
| Class imbalance (diverted) | Stratified split; class weights in RF |
| Data not in repo | Plan assumes local path; `.gitignore` large pickles if needed |

---

## 7. Files added in SURGE

| Path | Role |
|------|------|
| `docs/smart/SMART_TOKAMAKER_SURROGATE_PLAN.md` | This plan |
| `data/datasets/SMART/README.md` | Expected layout |
| `scripts/smart/inspect_smart_data.py` | Column / dtype discovery |
| `scripts/smart/build_curated_smart.py` | `gamma_TOKAM` + curated pickle |
| `scripts/smart/train_smart_classifier.py` | Sklearn classifier + metrics |
| `configs/smart/metadata_*.yaml` | Input/output column lists |
| `configs/smart/smart_*.yaml` | SURGE workflow specs (edit paths) |

---

## 8. References

- TokaMaker SMART PT–NT equilibria, VDE growth-rate context (internal presentations).
- SURGE workflow: `examples/m3dc1_workflow.py`, `docs/SURGE_OVERVIEW.md` (if restored) / `surge/workflow/run.py`.
