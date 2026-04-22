# Open-source release demo plan — end-to-end capability tour

Goal: the first public **v0.1.0** drop ships a **single, self-contained walkthrough**
that exercises SURGE from **data generation** to **ONNX-runtime inference**,
using only **redistributable** data (no HHFW / proprietary batches). This plan
complements [PUBLIC_OPEN_SOURCE_PLAN.md](PUBLIC_OPEN_SOURCE_PLAN.md) (history
hygiene) and [PRERELEASE.md](PRERELEASE.md) (scope promises).

## Canonical demo (what the README points a new user at)

The demo runs a **dummy but realistic** tabular surrogate task end-to-end so
viewers can see *every* advertised SURGE stage without needing HPC or
collaboration access. Keep it **one YAML + one notebook** (or one script) plus
helper configs per stage.

### Stage 0 — **Dataset generation (optional, documented)**

- **What it shows:** how SURGE can prepare a **batch of simulation inputs**
  (parameter grid → per-case directories → SLURM job scripts).
- **Where it lives today:** `surge/datagen/` (library) and
  `scripts/datagen/surge_batch_setup.py` / `examples/datagen_demo.py`.
- **Demo deliverable:** `examples/datagen_demo.py` running against a **toy**
  template (no real physics inputs) that produces a small directory tree and a
  mock SLURM script. Do **not** use proprietary `C1input` templates in the
  public tree.
- **Documentation:** `docs/DATAGEN_USER_MANUAL.md` + a short README in the
  example, noting that **generator templates are user-supplied**.

### Stage 1 — **Ingest + preanalysis + EDA**

- **What it shows:** `SurrogateDataset` loading (auto-detect I/O columns),
  `summary()` / `describe()` / `stats()`, and a small **viz panel** (violin
  plots, correlations, mean±σ profile — the “six-panel” style).
- **Demo deliverable:** one cell (or function) that loads
  `data/datasets/M3DC1/m3dc1_synthetic_profiles_sample.csv` (or a generated
  synthetic table) and saves a multi-panel PNG to the run directory.
- **Uses:** `surge.dataset.SurrogateDataset`, `surge.viz.*`.

### Stage 2 — **Standardize + split (deterministic)**

- **What it shows:** `SurrogateEngine.prepare()` producing deterministic
  **train/val/test** splits and `StandardScaler` bundles saved to the run.
- **Demo deliverable:** scalers + row-index JSON in `runs/<tag>/scalers/` and
  `runs/<tag>/data/`.
- **Uses:** `EngineRunConfig(test_fraction, val_fraction, standardize_*)`.

### Stage 3 — **Train four baseline models**

Required backends (must install cleanly with documented extras):

| Model key | Backend | Install extra |
|-----------|---------|---------------|
| `sklearn.random_forest` | scikit-learn RFR | core |
| `sklearn.mlp` | scikit-learn MLPRegressor | core |
| `torch.mlp` | PyTorch MLP (+ MC-Dropout for UQ) | `pip install surge-ml[torch]` |
| `gpflow.gpr` | GPflow GPR | `pip install surge-ml[gpflow]` (optional) |

- **What it shows:** the unified adapter/registry story — same `ModelSpec`
  interface drives all four.
- **Demo deliverable:** `configs/demo_release.yaml` with all four models listed;
  `run_surrogate_workflow(spec)` writes per-model metrics + predictions.

### Stage 4 — **Hyperparameter optimization (next phase)**

- **What it shows:** Optuna TPE (and optionally BoTorch) over a search space
  per model, with per-trial metrics + best config saved.
- **Demo deliverable:** a second config `configs/demo_release_hpo.yaml` with
  `hpo.enabled: true` on one or two models (small trial count — 5–10 — so the
  demo is minutes, not hours).

### Stage 5 — **Pick best → fine-tune**

- **What it shows:** selecting the best model from Stage 4 and running a
  **fine-tune** pass (lower LR, warm-started weights) to squeeze more
  performance.
- **Demo deliverable:** a config that sets `pretrained_run_dir:
  runs/demo_release_hpo` and `finetune_lr_scale: 0.1`; this is already
  supported by `SurrogateWorkflowSpec` + `_train_single_model` for the torch
  path. Report Δ-metrics before/after fine-tuning.

### Stage 6 — **Artifacts + metrics**

- **What it shows:** every run produces a **complete** `runs/<tag>/` layout:
  - `models/*.joblib` / `*.pt`
  - `scalers/*.joblib`
  - `predictions/<model>_{train,val,test}.{parquet|csv}` (+ UQ JSON)
  - `metrics.json`, `workflow_summary.json`
  - `hpo_trials.json`, `optuna_study.pkl` (when HPO on)
  - `spec.yaml`, `environment.json`, `git_revision.txt`
- **Demo deliverable:** a short **tree walk** (notebook cell or doc block) that
  opens each artifact and shows its content.

### Stage 7 — **Predict + report**

- **What it shows:** `load_model_compat(...)` (existing helper) loads an
  artifact and produces predictions on held-out data; full metrics recomputed.
- **Demo deliverable:** one script `examples/release_predict_demo.py` that
  takes `--run-dir` and `--data` and prints metrics.

### Stage 8 — **Export + ONNX-runtime inference**

- **What it shows:** export the (best) **PyTorch** surrogate to **ONNX**,
  then re-load with **onnxruntime** and verify outputs match.
- **Where it lives today:** `surge/inference/onnx_runtime.py` (load + infer
  helpers) and `examples/m3dc1/onnx_inference_demo.py`.
- **Demo deliverable:** generalize the existing example into
  `examples/release_onnx_demo.py` that (a) exports from an MLP adapter,
  (b) runs `onnxruntime`, (c) compares against the native prediction with a
  tolerance. **Call out** that sklearn / GPflow models are **not** part of this
  path (use their native `predict`).

---

## What you are likely missing (callouts)

1. **Data policy for the demo:** the **release tree** must not rely on HHFW,
   XGC, or private M3D-C1 batches. Use the tracked **sample CSV**
   (`data/datasets/M3DC1/*_sample.csv`) or a **synthetic** generator.
2. **UQ demonstration:** MC-Dropout (Torch) and ensemble (if present) should
   be exercised in at least one stage; add a `request_uncertainty: true` in
   the config and show `*_val_uq.json` is written.
3. **Cross-validation:** current `cv_folds` in the spec is **reserved** (not
   implemented). The demo should **not** imply k-fold is supported; reference
   `docs/ROADMAP.md` instead.
4. **Reproducibility:** `seed` in spec; show `spec.yaml` and `environment.json`
   in the artifact tour so others can re-run.
5. **Determinism across backends:** PyTorch determinism is platform-dependent;
   document this as a note.
6. **Lightweight CI** that runs the demo on a tiny synthetic dataset (1–2 min)
   so every PR proves the end-to-end story still works.
7. **Docs:** `README.md` should have a single “Run the demo” section that
   mirrors Stages 0–8 as one install + one command (e.g.
   `python examples/release_end_to_end.py --config configs/demo_release.yaml`).
8. **Naming clean-up:** keep `SurrogateDataset` / `SurrogateEngine` names;
   optional non-DataFrame backends stay roadmap work.
9. **Licence headers:** each shipped file should carry a short BSD-3 header
   (one-line `SPDX` is fine) to match the LICENSE/NOTICE.
10. **Citation:** add a `CITATION.cff` referencing the forthcoming DOE CODE
    record once the ID is known.

---

## Suggested new / updated files for the release

- `configs/demo_release.yaml`, `configs/demo_release_hpo.yaml`,
  `configs/demo_release_finetune.yaml` — three tight configs, tagged
  `demo_release*` runs.
- `examples/release_end_to_end.py` — driver that stitches stages (ingest →
  train → HPO → fine-tune → predict → ONNX) in under ~5 min on CPU with the
  sample dataset.
- `examples/release_predict_demo.py`, `examples/release_onnx_demo.py` —
  focused examples if users want one piece in isolation.
- `notebooks/release_tour.ipynb` — optional narrated variant.
- `docs/RELEASE_DEMO_PLAN.md` — this file.
- Minimal **GitHub Actions** workflow under `.github/workflows/ci.yml` (Python
  3.10, install `.[dev,torch]`, run `pytest -q` + the demo script).

---

## Acceptance checklist (what “done” looks like for v0.1.0)

- [ ] All four model backends train on the sample dataset.
- [ ] Artifacts written for every stage; tree verified by a test.
- [ ] HPO runs one small study (≤ 10 trials) in minutes.
- [ ] Fine-tune reduces test RMSE on the torch MLP (or documents that it did
      not, with reason).
- [ ] ONNX export + runtime match native predictions within tolerance.
- [ ] CI green on a fresh clone.
- [ ] `PUBLIC_OPEN_SOURCE_PLAN.md` checklist items satisfied (license, NOTICE,
      history scrub, public URL set).
