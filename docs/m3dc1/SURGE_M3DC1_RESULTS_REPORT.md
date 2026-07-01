# SURGE × M3DC1 Surrogates — Workflow & Results Report

_Status snapshot: 2026‑07‑01. Repo: **SURGE-exp**, branch `feat/m3dc1-on-model-bench`._

This report documents (1) how the data was generated and where it lives, (2) how the
SURGE training workflow is run, (3) the model zoo and where each model lives, and
(4) results so far for the three tasks that show real progress — **growth rate γ**,
**δp̂ spectrum**, and **mode peak location (core/edge)** — including loss curves and
per‑case eigenmode visuals.

---

## 1. Data: source, generation, location

| Item | Value |
|---|---|
| Raw M3DC1 batch | `/pscratch/sd/a/asvillar/mp288/jobs/batch_16/` (10,201 case dirs: `run*/sparc_*/`) |
| Per‑case verification file | `csdata_deltap_b_ver.h5` (one per case dir) |
| Grid / spectrum | 201×201 (R,Z), poloidal harmonics `m ∈ [−100, 100]`, full FFT |
| Postprocess entry point | `m3dc1ml-run-batch` → [`m3dc1ml/src/m3dc1ml/postprocess/run_batch.py`](/global/homes/a/asvillar/src/SURGE/m3dc1ml/src/m3dc1ml/postprocess/run_batch.py) |
| Generation script | [`m3dc1ml/scripts/build_csdata_deltap_b_ver.sh`](/global/homes/a/asvillar/src/SURGE/m3dc1ml/scripts/build_csdata_deltap_b_ver.sh) (+ `.slurm`) |
| Scalars dataset (γ etc.) | [`data/datasets/SPARC/case_scalars_ver.parquet`](/global/homes/a/asvillar/src/SURGE/data/datasets/SPARC/case_scalars_ver.parquet) — **9,976 cases** |
| Dataset metadata | [`case_scalars_ver_metadata.yaml`](/global/homes/a/asvillar/src/SURGE/data/datasets/SPARC/case_scalars_ver_metadata.yaml) |

**What's inside each `csdata_deltap_b_ver.h5`** (`runs/run_0001/`):
`spectrum/{p,br,bz,bphi}` (complex `spec[t, m, ψ]` + `m_modes`, `psi_norm`),
`pertfields/{p,p_hat,p_phi0,p_phiq,...}` (the δp field on the 201×201 R,Z grid),
`equilibrium/` (ψ(R,Z), grad‑ψ, magnetic axis), `flux_average/{q,p,ne,...}`,
`miller/{R0,a,kappa,delta}`, `parset/{ntor,pscale,batemanscale}`, `growth_rate/`.

**How generation works (short version):** M3DC1 runs produce `C1.h5` + `equilibrium.h5`
per case. `m3dc1ml` postprocess uses the M3DC1 `fpy` bindings + `flux_coordinates`
(PEST) to evaluate the perturbed pressure/field on the R,Z grid, poloidally
Fourier‑transforms to `δp̂(m, ψ_N)`, and stores both the field and the spectrum
(with a reconstruction self‑check). Code: [`postprocess/build_sdata.py`](/global/homes/a/asvillar/src/SURGE/m3dc1ml/src/m3dc1ml/postprocess/build_sdata.py).

Repositories:
- SURGE (framework, experiments): **[github.com/S-Villar/SURGE-exp](https://github.com/S-Villar/SURGE-exp)** (branch `feat/m3dc1-on-model-bench`)
- M3DC1 postprocess/IO/viz: **[github.com/S-Villar/m3dc1ml](https://github.com/S-Villar/m3dc1ml)**

---

## 2. The SURGE training workflow

All tabular tasks share **one leakage‑free, case‑grouped split** so every model sees
exactly the same train/val/test partition. Grouping is by `run_id` + `eq_id`
(`GroupShuffleSplit`), so all rows from one physical case stay in one split.

- Engine / splitting: [`surge/engine.py`](/global/homes/a/asvillar/src/SURGE/surge/engine.py)
- Workflow spec (YAML): [`surge/workflow/spec.py`](/global/homes/a/asvillar/src/SURGE/surge/workflow/spec.py)
- Runner + artifacts: [`surge/workflow/run.py`](/global/homes/a/asvillar/src/SURGE/surge/workflow/run.py)
- HPO recipes (optuna): [`surge/benchmarks/hpo.py`](/global/homes/a/asvillar/src/SURGE/surge/benchmarks/hpo.py)

Each run writes: `metrics.json`, `workflow_summary.json`, `splits.json`
(+ `leakage_check`), `model_card_*.json`, `training_history_*.json`, and
`plots/training_dashboard_*.png`.

**Run a workflow (example — γ, all models):**
```bash
conda activate /global/cfs/projectdirs/m3716/software/asvillar/envs/surge
python scripts/m3dc1/internal/run_workflow.py configs/internal/m3dc1_gamma_ver_allmodels.yaml
```
Configs live in [`configs/internal/`](/global/homes/a/asvillar/src/SURGE/configs/internal). Full how‑to:
[`docs/m3dc1/TRAINING_RUNBOOK.md`](/global/homes/a/asvillar/src/SURGE/docs/m3dc1/TRAINING_RUNBOOK.md),
HPO details: [`docs/m3dc1/HPO_COMPARISON.md`](/global/homes/a/asvillar/src/SURGE/docs/m3dc1/HPO_COMPARISON.md).

---

## 3. Model zoo — what they are & where they live

| Model (key) | Type / architecture | Backend | Adapter |
|---|---|---|---|
| `ridge` | Linear regression, L2 penalty | sklearn | — |
| `random_forest` | Bagged decision‑tree ensemble | sklearn | — |
| `gradient_boosting` | Sequential boosted trees | sklearn | — |
| `sk_mlp` | sklearn feed‑forward MLP | sklearn | — |
| `torch_mlp` | PyTorch MLP (configurable depth/width) | — | [mlp](/global/homes/a/asvillar/src/SURGE/surge/model/adapters) |
| `residual_mlp` | MLP with residual/skip blocks | [`residual_mlp.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/residual_mlp.py) | [`residual_mlp.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/residual_mlp.py) |
| `geom_residual_mlp` | Residual MLP + geometry‑aware feature map | — | [`geometric_residual_mlp.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/geometric_residual_mlp.py) |
| `mlp_ensemble` | Deep ensemble of MLPs (mean + uncertainty) | [`mlp_ensemble.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/mlp_ensemble.py) | [`mlp_ensemble.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/mlp_ensemble.py) |
| `ft_transformer` | Feature‑Tokenizer Transformer (tabular attention) | [`ft_transformer.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/ft_transformer.py) | [`ft_transformer.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/ft_transformer.py) |
| `vae` | Variational auto‑encoder w/ regression head | [`vae.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/vae.py) | [`vae.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/vae.py) |
| **`fno2d`** | **2D Fourier Neural Operator** — spectral conv in Fourier space, resolution‑invariant operator learning | [`fno2d.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/fno2d.py) | [`fno2d.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/fno2d.py) |
| **`unet`** | **U‑Net** — encoder/decoder CNN with skip connections | [`unet.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/unet.py) | [`unet.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/unet.py) |
| `deeponet` | Operator net (branch/trunk) | [`deeponet.py`](/global/homes/a/asvillar/src/SURGE/surge/model/backends/deeponet.py) | [`deeponet.py`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters/deeponet.py) |

Full backend list: [`surge/model/backends/`](/global/homes/a/asvillar/src/SURGE/surge/model/backends) ·
adapters: [`surge/model/adapters/`](/global/homes/a/asvillar/src/SURGE/surge/model/adapters).

The **spectrum‑image FNO2D/U‑Net** driver (2D `|δp̂|(m,ψ)` prediction with
physics‑informed conditioning channels incl. the `m − n·q(ψ)` resonance channel):
[`scripts/m3dc1/internal/train_spectrum_image.py`](/global/homes/a/asvillar/src/SURGE/scripts/m3dc1/internal/train_spectrum_image.py).
Net config: `_FNO2dNet(in=11, out=1, hidden=32, n_modes=16, n_layers=4)`,
`_UNetNet(in=11, out=1, base=48, depth=4)`.

---

## 4. Results

### 4.1 Growth rate γ (regression, 9,976 cases)

Target `gamma`; 11 equilibrium/scalar inputs; case‑grouped split.
Run: `runs/m3dc1_gamma_ver_allmodels/` · HPO: `runs/m3dc1_gamma_ver_batch_hpo/`.

| Model | test R² | test RMSE | test MAE |
|---|---:|---:|---:|
| **torch_mlp (HPO)** | **0.868** | 0.0164 | 0.0065 |
| mlp_ensemble | 0.858 | 0.0171 | 0.0063 |
| random_forest | 0.852 | 0.0174 | 0.0057 |
| torch_mlp | 0.852 | 0.0174 | 0.0065 |
| vae | 0.843 | 0.0179 | 0.0084 |
| residual_mlp | 0.833 | 0.0185 | 0.0072 |
| sk_mlp | 0.822 | 0.0190 | 0.0075 |
| gradient_boosting | 0.803 | 0.0200 | 0.0090 |
| ft_transformer | 0.782 (→ 0.850 w/ HPO) | 0.0211 | 0.0097 |
| geom_residual_mlp | 0.390 (→ 0.743 w/ HPO) | 0.0353 | 0.0232 |
| ridge (linear baseline) | 0.531 | 0.0309 | 0.0186 |

![gamma model comparison](/global/homes/a/asvillar/src/SURGE/runs/report_assets/gamma_r2_bar.png)

Representative per‑model training dashboard (loss + pred‑vs‑true accuracy map):

![gamma torch_mlp dashboard](/global/homes/a/asvillar/src/SURGE/runs/m3dc1_gamma_ver_allmodels/plots/training_dashboard_torch_mlp.png)

**Takeaway:** γ is well‑predicted from equilibrium scalars — best **R² ≈ 0.87** (HPO
torch_mlp), with a tight cluster of NN/ensemble/RF models around 0.85. Linear ridge
(0.53) confirms the relationship is strongly nonlinear.

### 4.2 δp̂ spectrum

**Progress across approaches:**

![delta p progress](/global/homes/a/asvillar/src/SURGE/runs/report_assets/deltap_progress_bar.png)

**(a) Per‑mode MLP** (`|δp̂|` magnitude, low‑|m| band, signed‑log target) —
`runs/m3dc1_deltap_mag_residual_flex/`:

| Model | test R² |
|---|---:|
| residual_mlp_flex (HPO, per‑layer integer widths) | 0.358 |
| residual_mlp (baseline) | 0.316 |

> The **real part** `Re(δp̂)` is **ill‑posed** (arbitrary eigenmode phase/normalization):
> all models score R² ≈ 0 / negative (`runs/m3dc1_deltap_real_smoke/`). We therefore
> model the phase‑invariant **magnitude**.

**(b) Whole‑spectrum image (FNO2D)** — predicts the entire `log₁₀|δp̂|(m,ψ)` image per
case, conditioned on equilibrium channels + `m − n·q(ψ)`. Run: `runs/spectrum_image/`.

| Metric | Value |
|---|---:|
| Best **global** val R² (subset, 3,000 cases) | **0.654** (epoch 68) |
| Full‑data run (9,976 cases) — in progress | 0.605 (epoch 15, climbing) |
| Per‑case spectrum R² (subset val) | min −441 · median −0.87 · **max 0.948** |

**Important nuance:** the global R² (0.65) is inflated by *between‑case* amplitude
differences; **per‑case** R² is often poor because the **overall eigenmode
normalization is arbitrary** and low‑SNR edge modes are noise‑dominated. This is the
current plateau and the target of the next iteration (per‑case normalization).

Per‑case eigenmode visuals (2×3: **rows = field δp(R,Z) / spectrum |δp̂|(m,ψ)**,
**cols = truth / predicted / difference**). Predicted field = `|δp̂|_pred × true phase`
(the model predicts magnitude only), reconstructed on the `fpy` flux grid.

_Best case (R²=0.948) — ridge & RZ envelope captured:_
![best case](/global/homes/a/asvillar/src/SURGE/runs/spectrum_image/plots/case_best_fno2d.png)

_Median case (R²=−0.87) — low‑SNR edge mode, model smooths noise floor:_
![median case](/global/homes/a/asvillar/src/SURGE/runs/spectrum_image/plots/case_median_fno2d.png)

_Worst case (R²=−441) — anomalously large overall normalization (~10⁵), under‑predicted:_
![worst case](/global/homes/a/asvillar/src/SURGE/runs/spectrum_image/plots/case_worst_fno2d.png)

Regenerate/point at any case:
```bash
python scripts/m3dc1/internal/plot_spectrum_image_cases.py \
    --run runs/spectrum_image --model fno2d --n-cases 3000 --grid 128
# or specific validation-local indices:  --cases 12 45 130
```

### 4.3 Mode peak location → core vs edge

New, well‑posed target: radial location of the mode,
`ψ_peak = Σ(E·ψ)/ΣE` with `E(ψ)=Σ_m|δp̂|²`. Run: `runs/peak_location/`
(script [`peak_location.py`](/global/homes/a/asvillar/src/SURGE/scripts/m3dc1/internal/peak_location.py), 3,998 cases).

| Task | Model | Metric |
|---|---|---|
| ψ_peak regression | **random_forest** | **R² = 0.911**, MAE = 0.046 |
| ψ_peak regression | mlp | R² = 0.874 |
| ψ_peak regression | grad_boost | R² = 0.851 |
| **core vs edge** | random_forest (clf) | **acc = 0.954**, F1(edge) = 0.953 |

![peak location scatter](/global/homes/a/asvillar/src/SURGE/runs/peak_location/scatter_psi_peak_centroid.png)

**Takeaway:** where the mode sits radially (and core‑vs‑edge) is **highly predictable
(R² ≈ 0.91 / 95% accuracy)** from equilibrium scalars — a strong, deployable result.

---

## 5. Training progress (loss curves)

FNO2D, subset run (3,000 cases) — best val R² 0.654 @ epoch 68:
![fno2d subset loss](/global/homes/a/asvillar/src/SURGE/runs/spectrum_image/loss_fno2d.png)

FNO2D, full‑data run (9,976 cases, GPU‑resident cache) — live, epoch 16:
![fno2d full loss](/global/homes/a/asvillar/src/SURGE/runs/spectrum_image_full/loss_fno2d.png)

Live monitoring during a run:
```bash
# from a login node, regenerate curves from the streaming history
python scripts/m3dc1/internal/train_spectrum_image.py --plot-only --out runs/spectrum_image_full
```
Checkpoint policy: `ckpt_<model>.pt` is (over)written whenever **val loss beats the
previous best** (stores `epoch`, `val_loss`, `val_r2`, `state_dict`); the run then
auto‑loads the best checkpoint for final evaluation & figures.

---

## 6. Summary of "what shows progress"

| Task | Best result | Status |
|---|---|---|
| Growth rate γ | **R² 0.87** (HPO torch_mlp) | ✅ strong |
| Mode peak ψ_N (core/edge) | **R² 0.91 / 95% acc** | ✅ strong |
| δp̂ spectrum image (global) | **R² 0.65** (FNO2D) | 🟡 progressing; full‑data run live |
| δp̂ per‑mode magnitude | R² 0.36 | 🟡 baseline |
| Re(δp̂) | ≈ 0 | ❌ ill‑posed (use magnitude) |

**Next iteration (to break the δp plateau):** train the spectrum‑image model on
**per‑case‑normalized** targets (remove the arbitrary global amplitude) so the network
learns the *shape*; report per‑case pattern R². Full‑data FNO2D/U‑Net run
(`runs/spectrum_image_full/`) is still training.
