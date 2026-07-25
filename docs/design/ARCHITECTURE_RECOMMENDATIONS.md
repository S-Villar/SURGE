# SURGE architecture recommendations

Status: proposal, 2026-07. Companion to the repository audit
(`audit/AUDIT_REPORT.md` on branch `audit/surge-2026-07`). Written to be
implementable as small, independent PRs by any contributor or coding agent;
each section states the problem with file evidence, the proposed interface,
and acceptance criteria.

## Philosophy (the invariant to protect)

SURGE is a **generic surrogate-modeling workflow with adapters at every
seam**. The conceptual pipeline —

```
data source → schema → validation/preprocessing → leakage-aware split
→ model selection/training → HPO/UQ → evaluation → artifacts
→ visualization/report/export
```

— must be expressible for *any* scientific dataset without touching the
core. Concretely:

1. **One execution path.** Everything that trains a model goes through
   `SurrogateEngine`. Benchmarks, examples, and HPO are *callers* of the
   engine, never re-implementations of splitting/metrics/training.
2. **Artifacts are the contract.** Every downstream consumer (plots,
   reports, leaderboards, MLflow, exports) reads versioned machine-readable
   artifacts; nothing user-facing is hand-encoded.
3. **Domain code lives behind interfaces.** M3D-C1, XGC, QLKNN, TokaMaker,
   SMART, NSTX-U, PDEBench, TheWell, ConStellaration are *plugins/specs*,
   not branches in core modules.
4. **Optional dependencies degrade loudly.** A missing backend removes a
   capability with a recorded reason — it never silently changes behavior.
5. **Configuration is total.** Every accepted config key is either honored
   or rejected at parse time; silent ignoring is a bug.

The audit found each of these violated somewhere. The sections below close
the gaps.

---

## R1. Registration transparency (do this first)

**Problem.** `surge/model/__init__.py` wraps all 23 adapter registrations in
`except Exception: pass`. Demonstrated failures: a torch-less env fails 14
tests that expect registered models; with a broken `libomp`, `xgboost.*`
vanishes silently while `lgbm.*` registers and crashes at `fit()`.

**Proposal.**

```python
# surge/model/registry.py
@dataclass
class RegistrationRecord:
    key: str
    status: Literal["registered", "skipped", "error"]
    reason: str = ""            # e.g. "torch not importable: <msg>"

REGISTRATION_LOG: list[RegistrationRecord] = []

def try_register(key, loader: Callable[[], type[BaseModelAdapter]], *,
                 requires: tuple[str, ...] = (), aliases=()) -> None:
    """Import + register one adapter; ImportError => skipped(reason);
    any other exception => error(reason) and re-raise under SURGE_STRICT."""
```

- Guards catch `ImportError` (and `OSError` from native-lib loads) only,
  recording the true message. Any other exception is a SURGE bug: recorded
  as `error` and re-raised when `SURGE_STRICT_REGISTRY=1` (CI sets it).
- `surge models --verbose` prints the full table: registered / skipped
  (reason) / error (reason).
- Tests: torch-dependent registration tests gain skip guards; a unit test
  asserts a deliberately broken adapter surfaces as `error`, not silence.

**Acceptance.** No `except Exception: pass` remains in registration paths;
`pytest` in a no-torch env: 0 failures; skip reasons visible in CLI.

## R2. Single engine path for benchmarks

**Problem.** `surge/benchmarks/` (~6,000 lines) contains zero references to
`SurrogateEngine`; it re-implements dataset loading, splits
(`train_test_split` inline), metrics (`_reg/_clf/_uq_metrics` in
`tasks.py`), HPO (`benchmarks/hpo.py`, 885 lines), and artifacts.

**Proposal.** Make a benchmark a *declarative spec* executed by the engine:

```python
@dataclass(frozen=True)
class BenchmarkSpec:
    key: str                      # "tabular.california_housing"
    source: DataSourceRef         # loader name + params (R3)
    task: Literal["regression", "classification", "field", "sequence"]
    split: SplitSpec              # named strategy + params (R4)
    metrics: tuple[str, ...]      # names in the metric registry (R6)
    threshold: Threshold | None   # metric, value, citation, direction
    tier: int
    # descriptive fields join in from surge/benchmarks/metadata.yaml
```

`run_benchmark(spec, model_key, seed)` builds an `EngineRunConfig`, calls
the engine, and writes a **standard run manifest** (R7) to
`benchmark_reports/`. Delete `benchmarks/hpo.py` in favor of the workflow
HPO entry point; route metrics through `surge.metrics`; thresholds move
from the `_THRESHOLDS` dict in `leaderboard.py:1650` into
`surge/benchmarks/metadata.yaml` (already created; add `threshold_value`,
`threshold_metric`, `citation` fields as structured data).

**Migration order.** (a) thresholds → metadata.yaml; (b) metrics → registry;
(c) one benchmark category at a time onto the engine (start `smoke`, then
`tabular`); (d) delete the parallel HPO; (e) delete bespoke runners.

**Acceptance.** `grep -r "train_test_split" surge/benchmarks` → 0;
`benchmarks/` imports the engine; per-category parity runs reproduce
previous metrics within seed noise before each deletion lands.

## R3. Data-source adapters + dataset schema

**Problem.** `SurrogateDataset._read_file` hard-codes formats;
`dataset.py`/`engine.py` require a pandas DataFrame end-to-end
(`engine.py:129`), which cannot represent fields (PDEBench 128×128),
sequences, or out-of-core data. Domain loaders (M3D-C1, XGC) are inside the
package; `analyze_dataset_structure` infers schema from plasma-specific
column prefixes (`gamma_*`, `profile_*`) in generic code.

**Proposal.**

```python
class DataSourceAdapter(Protocol):
    name: str                                   # "csv", "hdf5", "sim_dir", ...
    def load(self, ref: DataSourceRef) -> DataBundle: ...

@dataclass
class DataBundle:
    X: np.ndarray | pd.DataFrame                # (n, ...) — tensors allowed
    y: np.ndarray | pd.DataFrame
    schema: DatasetSchema

@dataclass
class DatasetSchema:
    inputs: list[Feature]        # name, dtype, unit, shape, description
    outputs: list[Feature]
    groups: dict[str, list[str]] # e.g. profile channels
    sample_id: str | None        # provenance / group-split key
    task_hint: str | None
```

- Registry + entry-point group `surge.data_sources` so `surge-fusion` (or a
  `domain/` namespace) can ship `m3dc1`, `xgc`, `qlknn` adapters without
  core edits.
- The engine consumes `DataBundle`; the DataFrame path becomes the tabular
  special case. Column-prefix inference moves into a `tabular-heuristic`
  schema analyzer that domain packages can override; plasma prefixes move
  out of `preprocessing.py`.
- Explicit validation step: NaN policy, dtype/range checks, shape
  consistency — reported in the run manifest, not just dropped
  (`engine.configure_dataframe` currently drops NaNs silently).

**Acceptance.** PDEBench Darcy (2D field) flows through the engine without
flattening hacks; `grep -rn "gamma_\|NSTX\|m3dc1\|xgc" surge/{engine,preprocessing,dataset}.py surge/utils.py` → 0.

## R4. Split strategies (leakage-aware)

**Problem.** Only two chained `train_test_split` calls
(`engine.py:250-328`); no stratification, grouping, or temporal splits;
`cv_folds` is accepted then ignored with a warning (`workflow/run.py`);
ConStellaration's canonical split lives as ad-hoc NPZ logic in
`benchmarks/leaderboard.py`.

**Proposal.**

```python
class SplitStrategy(Protocol):
    name: str
    def split(self, bundle: DataBundle, cfg: SplitSpec) -> RawSplits: ...
# built-ins: random, stratified, group (by schema.sample_id),
# temporal (ordered, no shuffle), kfold (returns folds), canonical (published
# index files, hashed)
```

- `SplitSpec(strategy="group", key="shot_id", test_fraction=0.2, ...)` in
  the workflow spec; unknown strategy or missing key = parse-time error.
- `cv_folds` either becomes `strategy="kfold"` end-to-end (engine loops
  folds, metrics report mean±std) or is **rejected** — never ignored.
- Canonical splits are content-addressed (store the index-file SHA in the
  manifest) so published benchmark numbers are reproducible.

**Acceptance.** A group-split regression test proves no group straddles
train/test; specifying `cv_folds` either works or raises; ConStellaration
paper split is a named canonical strategy used by both workflow and
benchmark paths.

## R5. Uncertainty quantification as a strategy layer

**Problem.** UQ is GP-only; the base-class contract says
`predict_with_uncertainty -> Mapping{'mean','variance'}` but `GPRModel`
returns a `(mean, std)` tuple (`surge/model/sklearn.py:137-140`) — the
interface is already inconsistent between its only two implementers.
Ensembles exist but expose no UQ; multi-seed std is computed in benchmarks
but never plotted.

**Proposal.**

```python
@dataclass
class UQResult:
    mean: np.ndarray
    std: np.ndarray                  # 1σ, same shape as mean
    quantiles: dict[float, np.ndarray] | None = None
    method: str = ""                 # "native-gp", "deep-ensemble", ...

class UQStrategy(Protocol):
    def wrap(self, adapter: BaseModelAdapter) -> BaseModelAdapter: ...
# built-ins: native (passthrough), deep_ensemble(k), mc_dropout(n),
# conformal (split-conformal residual intervals — model-agnostic)
```

- Normalize every adapter to return `UQResult` (fix the tuple/mapping split
  once, with a deprecation shim).
- Conformal wrapper gives *every* model calibrated intervals — that is the
  scientifically defensible default for surrogates.
- Evaluation adds coverage/calibration metrics (PICP, interval width, ECE
  for classifiers) to the manifest; viz recipes draw bands automatically
  when `UQResult` is present (the theme gallery `uncertainty` figure is the
  reference rendering).

**Acceptance.** `predict_with_uncertainty` has one return type everywhere;
any registry model can produce 95% intervals via
`uq: {method: conformal}` in a workflow spec; coverage appears in
`metrics.json`.

## R6. Metric registry

**Problem.** Three metric implementations: `surge/metrics.py`, benchmark
`_reg/_clf/_uq_metrics` (`benchmarks/tasks.py:128-232`), and engine-side
tuples. Naming drifts (`test_r2` vs `test_r2_mean` vs `max_r2`).

**Proposal.** A tiny named registry:
`register_metric("r2", fn, direction="max", task="regression")`; engine and
benchmarks request metrics *by name*; manifest stores
`{name, value, direction, split}`. Field metrics (`relative_l2`, spectral
error) and sequence metrics (horizon-wise error) register the same way, so
`primary_metric` in benchmark metadata is a name, not a convention.

**Acceptance.** One module owns metric math; report/leaderboard code
resolves direction from the registry instead of substring heuristics
(current `_is_lower_better` string matching in `surge/report/leaderboard.py`).

## R7. Versioned run manifest (one artifact schema)

**Problem.** Workflow runs write `workflow_summary.json` + `metrics.json`;
benchmarks write a different `result.json`; MLflow gets a third shape. The
leaderboard/report layer must guess field names.

**Proposal.** `surge/artifacts/manifest.py` with a JSON-schema-validated
`RunManifest v1`:

```
run_id, kind: workflow|benchmark, spec (inline), dataset {source, schema
digest, n, split {strategy, seed, fold, index_sha}}, model {key, params,
n_params, artifact paths}, metrics [{name, split, value, std?, n_runs?}],
uq {method, coverage?}, resources {device, runtime_s, peak_mem_mb},
provenance {git_rev, surge_version, env_digest, timestamp}, schema_version
```

Both paths emit it; `result.json`/`workflow_summary.json` remain as thin
compatibility views for one release. Reports, leaderboards, MLflow export,
and future regression-gating CI all read manifests only.

**Acceptance.** `surge report` and `surge.report.leaderboard` consume
manifests exclusively; a schema-validation test rejects malformed manifests;
one release note documents the compatibility window.

## R8. Reports and visualization recipes (mostly done — finish the seam)

Done on the viz branches: `surge/viz/theme.py` (visual system),
artifact-driven HTML leaderboard + run report, publication gallery
(parity/training/HPO/classification/field/UQ recipes). Remaining:

- Promote gallery figure builders into `surge/viz/recipes/` keyed by task
  type, consuming manifests (R7) instead of ad-hoc paths.
- Delete the neon `_SPIDER_THEME` and restyle `viz/benchmark.py`,
  `run_viz.py`, `comparison.py` onto the theme; fold
  `viz/importance.py`'s XGC group table into the domain package (R3).
- `surge report <run_dir|reports_dir>` CLI wrapping run-report + leaderboard.

## R9. CLI truthfulness

**Problem.** `surge` is an alias of the benchmark runner; there is no
`surge run spec.yaml` even though the workflow is the product's core.

**Proposal.** `surge run <spec.yaml>` (workflow), `surge bench ...`
(benchmarks), `surge models --verbose` (R1), `surge report ...` (R8),
`surge data fetch <dataset>` (R10). Plain-text fallback honoring
`NO_COLOR`/`--plain` for HPC logs.

## R10. Data governance

**Problem.** ~30 MB of NSTX-U/SMART pickles tracked in git; benchmark
caches land under the repo (`data/datasets/benchmarks/`); two 43 MB blobs
remain in history.

**Proposal.** External storage (or LFS) + `surge data fetch` with
SHA-verified downloads into an XDG cache dir (`SURGE_DATA_HOME`,
default `~/.cache/surge`); repo keeps only fetch manifests
(name, url, sha256, size, license). Schedule the already-planned
`git filter-repo` window (HHFW blobs + the two `cursoragent@cursor.com`
author identities) as a single rewrite with collaborator re-clone notice.

---

## Missing-feature summary (quick matrix)

| Feature | Today | Target | Section |
|---|---|---|---|
| Registration diagnostics | silent `except: pass` ×23 | recorded skip/error + strict CI | R1 |
| Benchmarks on engine | parallel framework | declarative specs → engine | R2 |
| Non-tabular data path | DataFrame-only core | DataBundle + schema | R3 |
| Domain isolation | XGC/NSTX-U/M3D-C1 in core | plugin entry points | R3 |
| Data validation | silent NaN drop | validation report in manifest | R3 |
| Group/temporal/canonical splits | random only | SplitStrategy registry | R4 |
| Cross-validation | accepted, ignored | kfold strategy or hard error | R4 |
| UQ beyond GPs | 2 adapters, inconsistent API | UQResult + conformal default | R5 |
| Uncertainty in outputs | computed, rarely shown | bands/coverage in recipes | R5/R8 |
| Metric naming | 3 implementations | named registry with direction | R6 |
| Artifact schema | 3 shapes, unversioned | RunManifest v1 | R7 |
| Workflow CLI | missing | `surge run spec.yaml` | R9 |
| Dataset distribution | pickles in git | fetch manifests + cache | R10 |

## Suggested PR sequence (each independently reviewable)

1. R1 registration transparency (+ test skip guards) — unblocks trust in
   everything else; ~200 lines.
2. R6 metric registry (small, no behavior change, both paths adopt).
3. R7 manifest v1 emitted alongside existing files (additive).
4. R4 split strategies + `cv_folds` resolution.
5. R2 benchmarks-on-engine, one category per PR (smoke → tabular → field →
   plasma), deleting parallel code as each lands.
6. R3 DataBundle/schema + loader registry; then move domain loaders out.
7. R5 UQ layer (interface fix, conformal wrapper, coverage metrics).
8. R8 recipes/report consolidation; delete neon theme + legacy viz.
9. R9 CLI; R10 data fetch + history-rewrite maintenance window.

Ordering rationale: interfaces that only *add* (R1, R6, R7) go first so the
disruptive consolidations (R2, R3) land against stable contracts.

---

## Addendum (found while building the visual system)

**R11. Dataset characterization as a pipeline stage.** The audit sketch and
past publications treat pre-training data analysis (input distributions,
target distribution, SNR, input–target correlations, strongest
relationship) as a standard step, but today it exists only as ad-hoc
helpers in `surge/viz/analysis.py` plus one-off scripts. Make it a workflow
stage: after validation, emit a `characterization` block in the run
manifest (per-feature stats, correlations, class balance) and a themed
characterization figure in the run report (reference implementation:
`characterization` figure in `examples/viz_theme_gallery.py`). Acceptance:
every workflow run contains the characterization artifact without extra
config; reports render it.

**R12. One dataset cache root.** Benchmark loaders redirect sklearn/OpenML
caches into `data/datasets/benchmarks/**`, but at least one dataset
(California housing) resolves against the default `~/scikit_learn_data`
instead, so caches split across machines and "cached" checks disagree.
Fold into R10: one `SURGE_DATA_HOME` used by *every* loader, with a
`surge data status` command listing which benchmark caches are present.

**R13. HPO study artifacts (sampler comparisons).** The published HPO
figures compare samplers (Random vs Optuna TPE vs BoTorch) with running
bests and starred optima. Neither HPO path records a study-level artifact
that supports this — only per-model trial dumps. Add a `HPOStudy` manifest
(sampler, seed, trials, running best, wall-clock per trial) so sampler
comparisons and the starred-best figure are reproducible outputs, not
notebook one-offs.

**R14. Classifier probability contract.** Classification diagnostics
(ROC/PR/calibration/ECE) require `predict_proba`, which some adapters
expose informally and `BaseModelAdapter` does not declare. Add an optional
`predict_proba` to the adapter contract (capability-flagged like
`supports_uq`) so evaluation and viz can rely on it.
