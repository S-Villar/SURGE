# Development backlog

Single place for pending developments, updated 2026-07-26. Detailed
designs live in the sibling docs (`ARCHITECTURE_RECOMMENDATIONS.md`
R1–R14, `RESOURCE_MANAGEMENT.md` R15–R18, `FIGURE_UPGRADE_PLAN.md`).

## Done recently (for orientation)

- Figure-upgrade plan phases 1–6: theme-aware README (`<picture>` +
  `docs/assets/readme/dark/`), mission-control leaderboard, ensemble
  retold, ConStellaration showcase, HPO mission-control dashboard,
  MLflow per-epoch curves + nested HPO trial runs, Gray-Scott study as
  honest 160-step forecast with persistence gate + residual-target
  variants; U-Net flat-input contract fixed.

## P1 — near term

- ~~R15 device resolution (GPU/MPS)~~ — DONE: `resolve_device` in all 19
  backends; MPS opt-in via `SURGE_DEVICE=auto` (LSTM unsafe on MPS).
  Follow-up: per-architecture MPS allow-list; record resolved device in
  run artifacts.
- ~~Parallel benchmark fan-out~~ — DONE: `surge bench --parallel N`
  (subprocess jobs, thread splitting, collision-safe result dirs).
- **`surge init` interactive wizard** — usability gap: new users must
  hand-copy a YAML spec. A stdlib-prompt wizard that inspects the data
  file (schema inference already exists), asks task-shape questions,
  suggests models, and writes a commented spec.yaml. YAML stays the
  source of truth; the wizard only *generates* it.
- **`surge validate <spec>`** — schema-check a spec without running it;
  publish a JSON Schema for editor autocomplete.
- **Publish 0.1.0rc1 → 0.1.0** (user-side: `uv publish`; then version
  bump, retag, Trusted Publishing on PyPI).
- **Verify README publication titles** (user-side).

- **PCA / POD training transforms (R3 slice)** — two distinct features:
  (i) input PCA: `preprocessing: {pca: {n_components: k|0.99}}` fitted on
  standardized train inputs, persisted with the scalers, applied to all
  splits; (ii) target PCA (POD surrogates): `target_pca: k` learns modal
  coefficients and inverse-transforms predictions before metrics and
  artifacts — the classic reduced-order-model path for field outputs
  (would let ANY tabular model attack the TheWell tasks via k modes).
  Inverse-transform plumbing goes through scalers + inference module.

## P2 — medium term

- **R16/R17 remaining** — spec-level `parallel_models` inside `surge
  run`, Optuna `n_jobs`, memory-tier enforcement, `--dry-run` placement
  report.
- ~~TheWell second dataset~~ — DONE: `turbulence_2d` study shipped
  (all neural operators beat persistence: U-Net 0.250 vs 0.355, MPS).
- **Helmholtz staircase full study** — feasibility CONFIRMED (smoke:
  U-Net rel-L2 0.077 vs persistence 1.38 on a quarter-period phase
  advance, 31 s on MPS); `examples/thewell_helmholtz_study.py` written,
  runs once the ~80 GB download completes.
- ~~Squeeze protocol~~ — RESULTS: constellaration was undertrained
  (60-epoch cap): wide [512,512,256] + 400 epochs lifts QI R2
  0.937 -> 0.951 (figure updated). Turbulence is CHAOS-LIMITED, not
  undertrained: 4-channel physics input, 2.5x epochs, and a 5x data
  scan (800/2000/4000 -> 0.250/0.258/0.254) all leave median rel-L2
  flat at ~0.25. At dt=8 the small scales decorrelate; pointwise rel-L2
  has a predictability floor. The right next steps are probabilistic
  forecasting (mlp_ensemble/CRPS), spectral or statistical evaluation
  metrics, and rollout-trained models — not bigger fits (P2).
- **DeepONet CNN branch** — the residual target halved its error but a
  convolutional branch is the real fix for field inputs.
- **Leaderboard preview special-cases** — constellaration benchmarks
  cache under a non-standard path; the leaderboard figure's data-preview
  inset falls back to "unavailable" for them (draw boundary
  cross-sections instead).
- **RTD dark mode** — README switches automatically; the Sphinx site is
  light-only (needs a theme with `prefers-color-scheme` support, e.g.
  furo, plus dark asset wiring in gallery.md).

## P3 — larger projects

- **R18 multi-GPU (DDP)** for large operator models.
- **3D operator backend** (FNO-3D / 3D U-Net) → unlocks TheWell `MHD_64`
  and other volumetric datasets.
- **Benchmark restructure** (`BENCHMARK_RESTRUCTURE.md`): suite/dataset/
  protocol split with status tracking.
- **PCA as training transform** (R3 pipeline work; analysis-only PCA
  shipped in preprocessing).
