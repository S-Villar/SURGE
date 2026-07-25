# SURGE `model-bench` Architecture and Product Audit

Date: 2026-07-24 · Audit branch: `audit/surge-2026-07` (created from `model-bench` @ e932dc2)
Comparison branch: `main` @ 3a0b6d6 · No branch was modified, merged, pushed, or rewritten.

---

## 1. Executive summary

`model-bench` should become the future baseline. `main` contains exactly one
commit not on `model-bench` (3a0b6d6, the strict-resource-mode fix), and it is
already cherry-picked into `model-bench` as 82c6cb5 (`git cherry` confirms the
patch is equivalent). Everything else — 76 commits, +51,335/−637 lines — is
`model-bench`-only: ~33 new model adapters, the entire benchmark subsystem
(37 registered benchmarks), classification metrics/plots, training/HPO
telemetry, multi-seed leaderboards, and CLI entry points.

The clean-environment verification largely validates the branch: fresh install
works, `287 passed / 26 skipped / 5 failed` with full optional dependencies
(all 5 failures are a missing macOS `libomp` runtime, not SURGE bugs), the
canonical workflow and benchmarks run end-to-end, and run artifacts are genuinely
reproducible (spec + git rev + env snapshot).

The four structural problems, in priority order:

1. **Silent optional-dependency handling is provably hiding errors.** 23
   `except Exception: pass` blocks around adapter registration (68
   `try-except-pass` and 179 blind excepts repo-wide per ruff). Demonstrated
   consequences: in a torch-less env, **14 tests fail** (registration tests
   assert torch models exist, with no skip guards); with a broken `libomp`,
   XGBoost vanishes from the registry with zero diagnostics while LightGBM
   registers and then crashes at `fit()`.
2. **The benchmark system is a second training framework**, not a consumer of
   the workflow engine. `surge/benchmarks/` contains zero references to
   `SurrogateEngine`; it reimplements splits, metrics, HPO, artifacts, and
   dataset loading, sharing only the model registry.
3. **Duplicated abstractions and dead weight**: two different `ModelRegistry`
   classes (`surge/registry.py` and `surge/model/registry.py`), a 2,300-line
   `surge/legacy/` tree, `surge/models.py` deprecation shim, `cv_folds` config
   accepted but silently ignored, no leakage-aware splitting.
4. **Repository hygiene and visual identity**: tracked `mlflow.db`, `mlruns/`,
   generated PDFs, nohup logs, ~30 MB of binary datasets, `README.md.backup`,
   `.cursor/` skill and two `cursor/canvas` TSX dashboards *inside the Python
   package*, with hard-coded (script-patched) leaderboard data; plotting has no
   shared theme (neon spider charts next to matplotlib defaults).

None of these block adoption of `model-bench` as baseline; all are addressable
in the small-PR roadmap (§12).

---

## 2. `main` versus `model-bench`

**Divergence.** Merge-base is e90c7f5 (2026-04-23). `model-bench` carries 76
unique commits authored in bursts 2026-05-17 → 2026-07-07. `main` advanced by a
single commit since divergence, already cherry-picked into `model-bench`.
**There is no content on `main` that `model-bench` lacks.**

**Capabilities added in `model-bench`** (all verified in code, not just docs):

| System | Detail |
|---|---|
| Model portfolio | ~7 → 40 registered adapters (live count with torch): full PyTorch family (MLP, residual MLP, ensemble, CNN1d, LSTM/GRU, FNO1d/2d, DeepONet, U-Net, LeNet-5, ResNet-20/56, AlexNet, ViT, KAN, FT-Transformer, VAE, DDPM, CGAN), sklearn additions (ridge, GBM, classifiers), LightGBM, CatBoost, XGBoost, BoTorch GP (exact + sparse), TabPFN |
| Benchmarks | `surge/benchmarks/` (registry 643 L, leaderboard 1,933 L, tasks 1,545 L, HPO 885 L, run 643 L) + loaders for PDEBench and TheWell |
| Metrics | classification metrics incl. ECE, relative-L2/NRMSE for fields |
| Viz | classification dashboards, training dashboards, benchmark spider plots, 2 Cursor canvas leaderboards |
| CLI | `surge` and `surge-benchmark` entry points |
| Telemetry | JSONL per-epoch training logs, HPO trial histories, MLflow logger |
| Tests | 2,759 → 5,692 test lines; portability and ConStellaration split tests |

**Only in `main`:** the `surge/__init__.py~` backup file (tracked!) — otherwise
nothing. Verdict: adopt `model-bench` lineage; treat `main` as stale.

**Source vs. clutter on `model-bench`** (of 393 tracked files, ~48 MB): the
functional source is `surge/` (115 files), `tests/`, `scripts/` (partly),
`docs/`, `examples/` (partly), packaging/CI. Clutter categories are itemized
in §7.

Note: a local branch `recovery/cnn-model-work-20260724` (= model-bench + 1
commit, "preserve CNN model adapter work", +1,179 lines incl.
`surge/model/pytorch_impl.py` CNN work and tests) exists in this clone and was
left untouched; it should be triaged as a candidate follow-on PR.

---

## 3. Implemented architecture (as-built)

```
                       config YAML/dict
                             │
              SurrogateWorkflowSpec (workflow/spec.py)
                             │
         run_surrogate_workflow (workflow/run.py, 933 L)
                             │
    ┌─ SurrogateDataset (dataset.py, 530 L; pandas-only core)
    │    formats: CSV, Parquet, Pickle, HDF5, NetCDF, JSON, Excel, XGC .npy dirs
    │    schema: metadata YAML/JSON or column-prefix inference (preprocessing.py)
    │    domain loaders: datasets/m3dc1.py, datasets/xgc.py
    │
    ├─ SurrogateEngine (engine.py, 700 L)
    │    splits: sklearn train_test_split ×2 (NO groups/stratify/CV;
    │            cv_folds accepted then ignored with a warning)
    │    scaling: StandardScaler fit on train
    │    resources: hpc/policy.py ResourcePolicy (strict → raises)
    │
    ├─ MODEL_REGISTRY ← surge/model/registry.py   ┐ two different
    │  (surge/registry.py also defines a registry) ┘ implementations
    │    registration: surge/model/__init__.py — 23× `except Exception: pass`
    │    adapters: model/adapters/* (thin) → model/backends/* (impl)
    │    base: BaseModelAdapter (fit/predict/save/load/predict_with_uncertainty)
    │
    ├─ HPO: _run_hpo_with_optuna (workflow/run.py) — Optuna TPE/BoTorch
    ├─ UQ: predict_with_uncertainty — GP adapters only; others NotImplemented
    ├─ metrics.py — regression + classification + ECE + relative-L2
    ├─ artifacts: io/artifacts.py → runs/<tag>/ (spec.yaml, metrics.json,
    │    models/, scalers/, predictions/*.parquet, git_rev, env, model card)
    ├─ viz/ (5,532 L, 9 modules, no shared theme) + visualization.py (legacy)
    └─ export: inference/onnx_runtime.py (torch → ONNX, parity-checked in CI)

  PARALLEL PATH (shares only MODEL_REGISTRY):
  surge/benchmarks/ — own dataset loading (_load_dataset), own splits
  (train_test_split), own metrics (_reg/_clf/_uq_metrics), own Optuna HPO
  (hpo.py, 885 L), own artifacts (benchmark_reports/**/result.json), own
  leaderboard rendering + canvas TSX. Zero references to SurrogateEngine.
```

**Hotspots**: `utils.py` 1,169 L (compute detection + IO + plotting helpers +
misc), `benchmarks/leaderboard.py` 1,933 L (loading + running + aggregation +
thresholds + printing), `viz/run_viz.py` 1,245 L, `workflow/run.py` 933 L.
Domain coupling in generic code: XGC group specs hard-coded in
`viz/importance.py`, NSTX-U default path in `utils.py`, M3D-C1 batch-mode
fields in the generic workflow spec, `fusion.m3dc1_sample` inside benchmark
`_load_dataset`.

---

## 4. Verified capability matrix

| Capability | Status | Evidence |
|---|---|---|
| Install (`pip install -e .[dev,benchmarks]`, py3.12, clean venv) | ✅ works | this audit |
| Import + registry population | ✅ 40 models (torch present) | live dump |
| Test suite, full deps | ✅ 287 pass / 26 skip / **5 fail (env: libomp)** | pytest |
| Test suite, no torch | ⚠️ **14 fail** — missing skip guards in `tests/benchmarks/test_new_phases.py` | pytest |
| Lint | ⚠️ ruff: 179 BLE001, 68 S110, 104 F401, 9 E722, 7 F821 | `ruff check .` |
| Workflow end-to-end (`examples.quickstart`, diabetes RF) | ✅ full artifact tree incl. model card | this audit |
| Benchmark smoke (RF + pytorch.mlp) | ✅ PASS lines with metrics | CLI runs |
| Leaderboard (3-model compare) | ✅ works; metric columns inconsistent per adapter (RF row lacks rmse/nrmse) | CLI run |
| ONNX export parity | ✅ covered by `test_e2e_release_smoke` (passed) | pytest |
| HPO (Optuna) | ✅ two implementations (workflow + benchmarks) | code + tests |
| UQ | ⚠️ GP-only; ensembles don't expose UQ | code |
| CV / grouped / stratified splits | ❌ `cv_folds` silently ignored; no group/stratify | engine.py |
| CLI `surge` | ⚠️ is an alias of the benchmark runner; **no `surge run spec.yaml` workflow command** | cli.py |

---

## 5. Model-adapter status matrix

Live registry with torch: **40 models**. Counts below from code + tests + live
probing (clean venv).

| Group | Models | Status |
|---|---|---|
| sklearn core | random_forest, mlp, gpr, ridge, logistic_regression, RF/GBM classifiers, gradient_boosting_regressor | Implemented; ridge + classifiers tested; several baselines lack dedicated tests |
| PyTorch tabular | mlp, mlp_classifier, residual_mlp, mlp_ensemble, geom_residual_mlp, ft_transformer(+classifier), kan(+classifier) | Implemented + tested (KAN needs optional `efficient_kan` for full path); mlp_ensemble & geom_residual_mlp lack save/load |
| PyTorch sequence | cnn1d, lstm, gru | Implemented; smoke-tested (fit/predict) |
| PyTorch field/operator | fno1d, fno2d, deeponet, unet | Implemented; smoke-tested |
| PyTorch vision | lenet5, resnet20, resnet56, alexnet, vit | Implemented; registration+fit smoke tests; benchmark-exercised (MNIST/CIFAR results exist) |
| PyTorch generative | vae, ddpm, cgan | Implemented; used as tabular regressors — role as *surrogate* models unverified beyond smoke tests |
| Boosted trees (optional) | lgbm.regressor/classifier, catboost ×2, xgboost ×2 | Implemented + tested when dep present. **xgboost/tabpfn disappear silently if import fails; lgbm registers then fails at fit** (lazy import) |
| GP (optional) | botorch.gp, botorch.sparse_gp | Implemented + tested (skipped without botorch) |
| GPflow (optional) | gpflow.gpr, gpflow.multi_kernel | Pre-existing; **no tests**; fragile TF dependency story (arm64 notes in requirements.txt) |
| TabPFN (optional) | tabpfn ×2 | Implemented; no save/load (library limitation); skipped silently without dep |
| Duplication | `pytorch.mlp` (pytorch_impl.py) vs `pytorch.residual_mlp`/backends MLPs — parallel MLP stacks; two registry classes; `models.py` shim; `legacy/` engine | consolidation targets |

**Broad-except verdict.** All 23 guards in `surge/model/__init__.py` are
*intended* as optional-dep guards, but `except Exception: pass` also swallows
SURGE's own bugs (an adapter refactor error would silently delete models from
the registry). Concretely observed: (a) no-torch env → 14 test failures that
*look* like registry bugs; (b) libomp-broken env → `xgboost.*` absent with no
message while `lgbm.*` present but broken. Required change: catch
`ImportError` only, log a one-line warning with the real cause, and expose
`surge.model.registration_report()` (name → registered | skipped(reason) |
error).

---

## 6. Benchmark-system assessment

**Verdict: a second, parallel training framework.** Reuse audit:

| Canonical piece | Reused by benchmarks? |
|---|---|
| Model registry | ✅ yes (`MODEL_REGISTRY.create`) — the only shared piece |
| `SurrogateDataset` | ❌ raw NumPy via own `_load_dataset` (leaderboard.py) |
| Split logic | ❌ own `train_test_split` calls |
| `SurrogateEngine` | ❌ zero references in `surge/benchmarks/` |
| HPO | ❌ own Optuna stack (`benchmarks/hpo.py`, 885 L) |
| Metrics | ❌ own `_reg/_clf/_uq_metrics` (not `surge.metrics`) |
| Resource policy | ❌ ad-hoc `tracemalloc` / CUDA peak-mem |
| Artifacts | ❌ own `BenchmarkResult` → `benchmark_reports/**/result.json` |
| Visualization | ❌ spider plots + canvas TSX + refresh script |

**What is genuinely good**: 33 implemented benchmarks across
smoke/tabular/image/field/plasma with real loaders (sklearn, OpenML, PDEBench
HDF5, TheWell, torchvision, HuggingFace ConStellaration, QLKNN); NPZ dataset
caching under `data/datasets/benchmarks/` (only `.gitkeep` tracked); multi-seed
runs with mean±std aggregation; runtime and (GPU) peak-memory capture;
thresholds with literature citations; per-run `result.json` with version +
timestamp. 6 registry entries are **proposed-only** (CTR-23 ×5,
`plasma.cmod_density_limit`: `None` runners) and must not be advertised as
implemented; docs like `SURGE_BENCHMARK_EXPANSION.md` are plans, not
capabilities.

**Weaknesses**: thresholds hard-coded in `leaderboard.py` (`_THRESHOLDS`)
rather than data files; leaderboard "publication" = regex-patching a
hard-coded `DATA` array inside a Cursor canvas TSX (`refresh_leaderboard_canvas.py`);
benchmark metadata (citations, URLs, tiers) duplicated by hand in the TSX;
per-adapter metric coverage inconsistent (visible in the CLI leaderboard);
no statistical comparison beyond mean±std; single shared HPO cache keyed by
benchmark×model without spec hashing.

---

## 7. Repository & data-hygiene findings

Tracked on `model-bench` (and mostly on `main`) — recommendation per category:

| Category | Items | Recommended destination |
|---|---|---|
| MLflow state | `mlflow.db` (640 KB), `mlruns/` (8 files) | untrack + ignore; artifact storage |
| Generated reports | `benchmark_reports/` 54 files incl. `.plots/` 32 PDFs, `.leaderboard_tables/` | untrack + ignore; publish curated snapshots as release assets or docs |
| Binary datasets | `data/datasets/NSTX-U/*.pkl/csv` (~28 MB, 13 files), `SMART/*.pkl` (2.6 MB) | external data storage or Git LFS + fetch script |
| Training logs | `data/datasets/NSTX-U/*.nohup.out` (440 KB) | delete from tracking; ignore |
| Backup/plan files | `README.md.backup`, `SURGE_IMPLEMENTATION_PLAN.md`, `SURGE_BENCHMARKS_VIZ_PLAN.md`, `SURGE_BENCHMARK_EXPANSION.md`, `surge/__init__.py~` (main) | untrack; move plans to docs/ or issues |
| Example outputs | `examples/*.png`, `examples/simple_optuna_demo_ruff_preview.html` | untrack; regenerate on demand |
| Big notebook | `notebooks/m3dc1/data_analysis.ipynb` (7.4 MB; outputs are stripped repo-wide, size is code/embedded) | split or LFS |
| History-only blobs | `data/datasets/HHFW-NSTX/Pw{IF,E}_.pkl` (43.4 MB each, deleted but in object DB) | future `git filter-repo` (already planned in docs/PUBLIC_OPEN_SOURCE_PLAN.md) |
| OK to keep | `data/logos/*.png`, `docs/m3dc1/assets/*` | keep (docs/brand) |
| Untracked local clutter (fine, but ignore-listed) | `runs/`, `logs/`, `dist/`, `*.egg-info` ×2, `catboost_info/`, `output*.png`, `Sketch*.png`, `hall_backup.yml`, `hall_pip_requirements.txt` | add ignore entries where missing (`mlflow.db` is currently NOT ignored) |

Empty `.pre-commit-config.yaml` (0 bytes) — either populate (ruff + nbstripout
+ large-file check) or remove.

---

## 8. Cursor-provenance findings

Scope is small and fully enumerable:

| Artifact | Where | Class |
|---|---|---|
| `.cursor/skills/pre-push-review/SKILL.md` (commit 8146a9c) | model-bench | future hygiene: untrack (keep locally if desired) |
| `surge/viz/leaderboard.canvas.tsx` (1,199 L), `surge/viz/benchmark_leaderboard.canvas.tsx` (786 L) — `import { … } from "cursor/canvas"` | model-bench, inside the shipped Python package | future hygiene: replace with the artifact-driven HTML report (§11) and remove |
| `scripts/refresh_leaderboard_canvas.py`, `scripts/refresh_datasets_canvas.py` (hard-codes `.cursor/projects/Users-asanche2-…` path) | model-bench | future hygiene: retire with the TSX files |
| Commits authored by `cursoragent@cursor.com`: ef01ef70 (on main + model-bench + 10 branches), 7ffde2f4 (on ai4fusion/experiment branches) | history | historical cleanup only |
| Cursor-named branches | none found | — |
| Cursor co-author trailers | none beyond the 2 authored commits | — |

**Future hygiene (normal PRs, no risk):** remove `.cursor/` from tracking, add
to `.gitignore`, delete both canvas TSX files + refresh scripts once the HTML
leaderboard replaces them, ensure MANIFEST/packaging never ships `.canvas.tsx`.

**Historical cleanup (separate decision, risky):** erasing
`cursoragent@cursor.com` authorship requires `git filter-repo`
(or GitHub support): all commit SHAs after ef01ef70 change on every branch;
forks/clones/PR references break; tags must be re-signed; DOE CODE / OSTI DOI
records that pin commit hashes could dangle. Since the same rewrite is already
planned to purge the 43 MB HHFW blobs, **batch both into one rewrite** at a
quiet moment: freeze pushes → `git filter-repo --mailmap` (rewrite the 2
author identities) `--strip-blobs-bigger-than 40M` (or path-based) → force-push
→ ask collaborators to re-clone. Until then the exposure is 2 commits, 0.4% of
history. Not performed in this audit.

---

## 9. Visualization & workflow-experience assessment

Current state (full inventory in agent report; ~7.5 k lines Python viz + 2 k
lines TSX):

- **No shared theme**: every module styles itself; fonts unspecified (renders
  differently per machine); colormaps mix `plasma_r`, `viridis`, `tab10`,
  Material pass/fail colors, and an isolated **neon dark spider theme**
  (`#050A14` bg / 8 neon hues) that clashes with everything else and is not
  CVD-validated.
- **Leaderboard dashboards depend on Cursor** and on regex-patched hard-coded
  data; the only machine-readable path is `result.json` → refresh script.
- **No HTML/PDF report generator**; a run produces loose PNGs; docs tables are
  hand-maintained.
- **CLI**: plain prints; the benchmark table is decent (best-marking asterisks)
  but metric columns vary per adapter; no `surge run <spec.yaml>`.
- **README**: good badges/structure; no visual evidence of outputs.
- **Artifact dirs are the strongest UX** already (spec, model card, git rev,
  quickstart's printed tree is excellent) — the right foundation for reports.
- Uncertainty is computed in places (multi-seed std, GP variance) but almost
  never *drawn*.

---

## 10. Proposed target architecture

Keep the working core; extract interfaces; make benchmarks a thin consumer.

```
surge/
  data/        DataSourceAdapter protocol (load→ DataBundle: arrays or frames
               + schema); adapters: csv, parquet, pickle, hdf5, netcdf,
               sim-dir, npz-cache, openml, torchvision, hf-datasets
               (plugin entry-point group: surge.data_sources)
  schema/      DatasetSchema (inputs/outputs/groups/shapes/units) — decouples
               tabular (DataFrame) from field/sequence tensors
  split/       SplitStrategy protocol: random, stratified, group-aware,
               temporal, k-fold, canonical (published splits, e.g.
               ConStellaration) — fixes leakage story, implements cv_folds
  model/       ONE registry (keep surge/model/registry.py API; delete
               surge/registry.py or make it a re-export); adapters+backends
               as today; registration_report(); ImportError-only guards
  engine/      SurrogateEngine as the single fit/eval path
  hpo/         one Optuna integration used by both workflow and benchmarks
  uq/          UQStrategy: native (GP), deep-ensemble, MC-dropout, conformal
               (wraps any adapter) + calibration metrics
  metrics/     registry of named metrics; benchmarks use it
  bench/       Benchmark = declarative spec (dataset source + canonical split
               + metric names + threshold provenance) executed BY the engine;
               results = artifacts in the standard run schema
  artifacts/   run manifest (JSON schema, versioned) — single source for
               reports, leaderboards, MLflow export
  viz/         theme.py (visual system) + task-type recipes (regression,
               classification, field, sequence, training, hpo, leaderboard)
  report/      offline HTML + md + figure bundles, generated from manifests
  cli/         surge run <spec.yaml> | surge bench … | surge report <run|dir>
               | surge models --verbose (shows skip reasons)
domain/        m3dc1, xgc, qlknn, tokamaker… live behind the adapter
               interfaces (or in a surge-fusion companion package) — core
               never mentions them
```

Migration is incremental: each protocol can be introduced under the current
call sites without breaking the API (§12 sequencing).

---

## 11. SURGE visual system & report architecture

Prototyped in `audit/prototypes/` (isolated; no production code touched):

- `surge_style.py` — single theme module: role-based palette (8-slot
  categorical in fixed order, CVD-validated in light *and* dark: worst adjacent
  CVD ΔE 9.1/8.4; sequential one-hue ramp; reserved status colors for
  PASS/FAIL), typography (Helvetica/DejaVu stack, 9 pt base), recessive grid,
  `surge_theme(mode)` context manager, deterministic `save_figure()`
  (PNG 300 dpi + SVG with fixed hashsalt + PDF, metadata stripped → byte-stable
  outputs for CI diffing), `fmt_metric()` for uniform metric formatting across
  CLI/plots/reports.
- `make_prototype_figures.py` — regression parity + residual diagnostic (from
  `runs/diabetes_rf` parquet + metrics.json) and a leaderboard (mean ± std over
  n runs, threshold line with provenance, runtime as a separate panel — never a
  dual axis) built **only** from `benchmark_reports/**/result.json`.
- `make_run_report.py` — single-file offline HTML run report (inline SVG, no
  JS/server, light+dark via `prefers-color-scheme`) from run artifacts:
  provenance header (run, git rev, version), dataset card, metrics table,
  diagnostics, re-runnable spec. Works over scp/HPC.

Rollout plan: adopt the theme in `surge/viz/theme.py`; convert recipes
per task type (regression parity/residual/calibration; classification
ROC/PR/confusion/reliability; field: truth/pred/error triptych + spectral
error + relative-L2 per sample; sequence: horizon curves + rollout error
growth; training: loss/metric/LR from JSONL; HPO: convergence + param
importance). Retire the neon spider theme and both canvas TSX dashboards; the
leaderboard becomes (a) a themed matplotlib figure and (b) a static HTML page,
both generated from `result.json` manifests, with mean±std, runtime, memory,
baseline/threshold provenance and per-benchmark citations pulled from the
benchmark spec (not hand-coded). CLI: adopt `rich`-style tables/progress with
a `--plain`/`NO_COLOR` HPC fallback. Optional interactivity: embed
sortable-table JS inline in the HTML (still a single file, no server).

---

## 12. Prioritized roadmap (small, reviewable PRs)

Phase 0 — baseline & hygiene
1. **PR-1 (recommended first, below):** registration transparency — replace the
   23 broad guards with `ImportError`-only + logged reason + `surge models
   --verbose` skip report; add torch-guards to `test_new_phases.py` so a
   torch-less env skips instead of failing; make optional-dep tests skip on
   `OSError` (libomp case).
2. PR-2: untrack generated/backup files (`mlflow.db`, `mlruns/`,
   `benchmark_reports/`, `README.md.backup`, plan MDs, example PNG/HTML,
   nohup logs) + `.gitignore` completion + populate or delete the empty
   pre-commit config.
3. PR-3: remove Cursor surface (`.cursor/`, both `.canvas.tsx`, refresh
   scripts) — merge after PR-9 replaces the leaderboard, or in the same PR.
4. PR-4: merge `model-bench` → `main` (fast-forward-ish; only 3a0b6d6 to
   reconcile, already cherry-picked), retire stale branches, protect `main`.
5. PR-5: dataset relocation — move NSTX-U/SMART pickles to external
   storage/LFS with a `surge.data fetch` helper; decide HHFW+author history
   rewrite as a separately-scheduled maintenance window (§8).

Phase 1 — single execution path
6. PR-6: delete `surge/registry.py` duplicate (re-export from
   `surge/model/registry.py`), delete `surge/legacy/` + `models.py` shim +
   `MLTrainer` aliases (one deprecation release if external users exist).
7. PR-7: split strategies (`surge/split/`) with group/stratified/temporal/
   canonical; implement or hard-reject `cv_folds` (no silent ignore).
8. PR-8: benchmarks-on-engine — `run_benchmark` builds an engine run from the
   benchmark spec; delete `benchmarks/hpo.py` in favor of the workflow HPO;
   route benchmark metrics through `surge.metrics`; thresholds move to a
   versioned `benchmarks/thresholds.yaml` with citations.

Phase 2 — visual system & reports
9. PR-9: `surge/viz/theme.py` (from prototype) + convert regression/
   classification/training recipes; deterministic export everywhere.
10. PR-10: `surge report` — offline HTML run report + benchmark leaderboard
    page from manifests; README gets one real themed figure.
11. PR-11: CLI polish (`surge run spec.yaml`, rich tables, `--plain`).

Phase 3 — extensibility & science depth
12. PR-12: DataSourceAdapter + DatasetSchema protocols; port existing loaders;
    entry-point plugin group; move M3D-C1/XGC behind it.
13. PR-13: UQ strategies (ensemble/MC-dropout/conformal) + uncertainty in all
    report recipes (bands, reliability, coverage).
14. PR-14: field/sequence diagnostics recipes; benchmark statistics upgrade
    (≥5 seeds default, CIs, paired comparisons vs. baseline).
15. PR-15: triage `recovery/cnn-model-work-20260724` CNN work onto the new
    baseline.

## 13. Recommended first implementation PR

**PR-1: "Make model registration transparent and dependency failures honest."**
Small (~200 lines), zero API break, immediately de-risks everything else:
ImportError-only guards with recorded skip reasons, `registration_report()`,
`surge models --verbose`, torch skip-guards in `test_new_phases.py`,
`importorskip`-hardening for OSError-style native failures. It converts the
audit's two demonstrated failure modes (14 misleading test failures; silently
missing/broken adapters) into visible, testable behavior — and every later
consolidation PR depends on trusting the registry.

---

## Appendix A — verification log

Environment: clean venv, Python 3.12.10 (Homebrew), macOS 15 (arm64), repo @ e932dc2.

| Command | Result |
|---|---|
| `pip install -e '.[dev,benchmarks]'` | OK |
| `pytest -q tests/` (no torch) | **14 failed**, 220 passed, 84 skipped — failures = missing skip guards (`test_new_phases.py` registration/fit tests) |
| `pip install torch torchvision onnx onnxruntime onnxscript datasets lightgbm xgboost catboost` | OK |
| `pytest -q tests/` (full) | **5 failed**, 287 passed, 26 skipped — all 5 = LightGBM `dlopen … libomp.dylib` (env, fixable with `brew install libomp`; tests should skip on OSError) |
| `ruff check . --statistics` | 179 BLE001 blind-except, 68 S110 try-except-pass, 104 F401, 9 E722, 7 F821, ~1,900 total |
| `python -m examples.quickstart --dataset diabetes` | OK — full artifact tree in `runs/diabetes_rf` |
| `surge-benchmark -b synthetic.regression_1d -m sklearn.random_forest/--no-save` | PASS r2=0.987 |
| `surge-benchmark -b synthetic.regression_1d -m pytorch.mlp` | PASS r2=0.991 |
| `surge-benchmark -b synthetic.multioutput_2d --compare-models rf,ridge,mlp` | OK — table renders; per-model metric coverage inconsistent |
| live registry probe | 40 models w/ torch; xgboost/tabpfn silently absent when import fails; lgbm registered but fit-time dlopen crash |
| prototype scripts (`audit/prototypes/`) | figures + HTML report generated from real artifacts, light+dark, PNG/SVG/PDF |

**Known uncertainties.** GPflow adapters unexercised (TF not installed —
heavyweight, arm64-fragile); PDEBench/TheWell/ConStellaration/vision benchmarks
not re-run end-to-end (large downloads) — loaders verified by reading + tracked
result.json evidence; XGC dataset path untested (no local data); MLflow logging
not exercised; `main`'s CI status not re-run (identical code minus 76 commits).

## Appendix B — files created by this audit (branch `audit/surge-2026-07`)

- `audit/AUDIT_REPORT.md` (this file)
- `audit/prototypes/surge_style.py`
- `audit/prototypes/make_prototype_figures.py`
- `audit/prototypes/make_run_report.py`
- `audit/prototypes/output/` (generated demos: parity/leaderboard ×{light,dark}×{png,svg,pdf}, `run_report.html`) — untracked by design
- untracked side effects of verification: `runs/diabetes_rf/`, `diabetes.csv` (quickstart), `/tmp/surge-audit-env`
