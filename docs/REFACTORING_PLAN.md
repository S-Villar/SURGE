# SURGE refactoring plan (pre-v0.1.0 hygiene and v0.2 architecture)

This document is the single source of truth for **code-shape cleanup** ahead
of the open-source release. It complements:

- `docs/PUBLIC_OPEN_SOURCE_PLAN.md` — git-history, licensing, and public-repo
  hygiene.
- `docs/RELEASE_DEMO_PLAN.md` — the end-to-end user-facing demo.
- `docs/ROADMAP.md` / `docs/dev/EXTENDED_ROADMAP.md` — feature roadmap (k-fold,
  DDStore, sharded datasets, MLflow).

The goal here is different: **reduce surface area** so that first-time public
readers see one obvious way to do each thing, not three.

---

## 1. Code-smell inventory (why the code "looks too ample")

### 1.1 Two model registries

Both of these exist and **both are re-exported from `surge/__init__.py`**:

- `surge/registry.py` — newer, richer `ModelRegistry` / `RegistryEntry` with
  backend, tags, aliases, default params, `describe()`, `registry_summary()`.
- `surge/model/registry.py` — older `ModelRegistry` / `RegisteredModel` using
  `key + adapter_cls + aliases` only.

`surge/engine.py` imports **both**:

```python
from .registry import BaseModelAdapter, ModelRegistry
from .model.registry import MODEL_REGISTRY
```

**Plan:** promote `surge/registry.py` as the canonical implementation, keep
`MODEL_REGISTRY` as a single module-level singleton, and reduce
`surge/model/registry.py` to a thin re-export (with a `DeprecationWarning`) so
adapter modules that register themselves on import keep working. Public API
(`surge.MODEL_REGISTRY`, `surge.ModelRegistry`) stays stable.

Target: **one registry, one singleton, one import path**.

### 1.2 `pytorch.py` vs `pytorch_impl.py` (and the same for GPflow)

This is the question you asked directly. The layering is:

- `surge/model/pytorch.py`
  — `PyTorchMLPAdapter(BaseModelAdapter)`. Adapter that wires the PyTorch MLP
    into SURGE's registry + `SurrogateEngine`. Implements the contract
    (`fit`, `predict`, `predict_with_uncertainty`, `training_history`). ~60
    lines. No nn.Module code lives here.

- `surge/model/pytorch_impl.py`
  — `PyTorchMLPModel` plus the actual `nn.Module` classes, training loop,
    `_make_dataloader`, MC-Dropout sampling, early stopping, JSONL progress
    logging. ~500 lines. No registry knowledge, no engine knowledge.

The **layering is correct and worth keeping** (it's the same adapter/impl
separation used by `scikit-learn` wrappers). The **naming is the smell**:
`_impl` does not say *what*. The same layering exists for GPflow via
`gpflow.py` + `gpflow_impl.py`.

**Plan:** rename in this direction:

| Current                         | Proposed                        |
| ------------------------------- | ------------------------------- |
| `surge/model/pytorch.py`        | `surge/model/adapters/torch.py` |
| `surge/model/pytorch_impl.py`   | `surge/model/backends/torch_mlp.py` |
| `surge/model/gpflow.py`         | `surge/model/adapters/gpflow.py`    |
| `surge/model/gpflow_impl.py`    | `surge/model/backends/gpflow_gpr.py`|
| `surge/model/sklearn.py`        | `surge/model/adapters/sklearn.py`   |
| `surge/model/ensembles.py`      | `surge/model/backends/ensembles.py` |

Keep the old module paths as shim files that import from the new locations
and emit a `DeprecationWarning`, for one release cycle.

Rationale: `adapters/` are **contracts** (what SURGE sees), `backends/` are
**implementations** (what the ML library sees). The folder names carry the
meaning that `_impl` does not.

### 1.3 `surge/models.py` deprecated shim

Already tagged with `DeprecationWarning("surge.models is deprecated; use
surge.model package instead")`. Keep for v0.1.0 back-compat, **remove in
v0.2.0**. Track it in `CHANGELOG.md` under "Removed".

### 1.4 `surge/legacy/`

Contains `engine_legacy.py`, `dataset_legacy.py`, `preprocessing_legacy.py`.
These are not imported from `surge/__init__.py`. They are **not part of the
public API**. We ship two options:

- **Option A (preferred for 0.1.0):** keep on disk but exclude from the
  wheel via `[tool.setuptools.packages.find]` `exclude = ["surge.legacy*"]`.
  Source readers can still read them; `pip install` users cannot accidentally
  import them.
- **Option B (cleaner but risky pre-release):** delete entirely.

Mark Option A as the v0.1.0 plan.

### 1.5 `surge/__init__.py` tries to import from `scripts/m3dc1/loader.py`

```python
scripts_m3dc1 = Path(__file__).parent.parent / "scripts" / "m3dc1"
sys.path.insert(0, str(scripts_m3dc1))
from loader import load_m3dc1_hdf5, convert_to_dataframe, ...
```

This works **only** when SURGE is imported from the source tree. After
`pip install surge-ml` there is no `scripts/` next to the installed package,
the import silently fails, and `M3DC1_LOADER_AVAILABLE` becomes `False`.

**Plan:** move the functions used publicly (`load_m3dc1_hdf5`,
`convert_to_dataframe`, etc.) into `surge/datasets/m3dc1.py` and import them
from there unconditionally. Remove the `sys.path` hack. Scripts that live
under `scripts/m3dc1/` then import from `surge.datasets.m3dc1`, not the other
way around. This removes the last scripts-to-package back-dependency.

### 1.6 Visualization modules

Two overlapping layers today:

- `surge/visualization.py` (legacy flat module)
- `surge/viz/` (`analysis.py`, `comparison.py`, `hpo.py`, `importance.py`,
  `profiles.py`, `run_viz.py`)

**Plan:** treat `surge/viz/` as canonical, reduce `surge/visualization.py` to
a thin re-export with a deprecation notice (same pattern as `surge.models`),
delete in v0.2.0.

### 1.7 `scripts/` hygiene

Already documented in `scripts/README.md`: the public tree should contain
only **application-generic** scripts. Any XGC-specific file (already
removed from the working tree) must also be removed from git history via
`git filter-repo` (tracked in `PUBLIC_OPEN_SOURCE_PLAN.md`). No further
refactor needed here beyond what the open-source plan already specifies.

### 1.8 Type hints + `py.typed`

The package ships no `py.typed` marker, so type checkers treat it as
untyped. This is a **v0.2.0** task (not a release blocker), but note:

- Add `surge/py.typed` (empty file).
- Re-add the `[tool.setuptools.package-data]` entry that was removed from
  `pyproject.toml` when the file was missing.
- Run `mypy surge/` in CI as a **non-blocking** job first, tighten later.

### 1.9 Legacy pre-refactor tests (currently skipped)

`SurrogateEngine` was refactored to a keyword-only constructor
(`registry=`, `run_config=`) plus an explicit `configure_dataframe(df,
input_columns, output_columns)` step. The following tests predate that
refactor and still call the old signatures (e.g.
`SurrogateEngine(n_features=..., n_outputs=...)`, `dataframe=`,
`engine.F`, legacy-dict `run_surrogate_workflow` specs):

**Skipped at module level** (pending migration):

- `tests/test_helpers.py`
- `tests/test_dataset.py`
- `tests/test_engine.py`
- `tests/test_models.py`
- `tests/test_visualization.py`

**Skipped at function level** (within otherwise-healthy files):

- `tests/test_core.py::test_surrogate_engine_init`
- `tests/test_core.py::test_basic_operations`
- `tests/test_workflow_new.py::TestSurrogateWorkflow::test_run_workflow_creates_artifacts`
- `tests/test_workflow_new.py::TestSurrogateWorkflow::test_run_workflow_programmatic_model_config`

Each skip cites this section in its `reason=`. The remaining suite
(`pytest -q tests/`) is fully green: 40 passed, 38 skipped, 0 failed.

**v0.2.0 action:** rewrite each test against the current engine /
workflow API and remove the `pytestmark` / `@pytest.mark.skip`
decorators. The behaviour these tests describe is still valuable — what
broke is the wiring, not the scenarios.

---

## 2. Resources + worker policy (new API for v0.1.0)

You asked: *"for parallel gpu/cpu, can we indicate at this moment if we want
to use cpu or gpu and what number of workers we want?"* — today, **no**, not
in a first-class way. `surge/hpc/resources.py` only **observes** the machine.
Nothing in `SurrogateWorkflowSpec` lets the user say what to use.

### 2.1 New spec block

Add a `resources` section to `SurrogateWorkflowSpec` (and the equivalent
field on `EngineRunConfig`):

```python
@dataclass
class ResourceSpec:
    device: str = "auto"            # "auto" | "cpu" | "cuda" | "cuda:N"
    num_workers: int = 0            # DataLoader workers (torch) / n_jobs (sklearn)
    max_gpus: int = 1               # cap; 1 = single-device, >1 reserved for v0.2 DDP
    pin_memory: bool = True         # torch DataLoader
    persistent_workers: bool = True # torch DataLoader
    strict: bool = False            # if True: reject unsafe combos instead of warning
```

and on `SurrogateWorkflowSpec`:

```python
resources: ResourceSpec = field(default_factory=ResourceSpec)
```

YAML example:

```yaml
resources:
  device: cuda
  num_workers: 8
  max_gpus: 1
  strict: true
```

### 2.2 Per-adapter usage policy

Each adapter declares a **resource profile** (class-level metadata) and
implements `validate_resources(resources, dataset_shape)` that either:

- returns a (possibly adjusted) `ResourceSpec`, or
- raises `ResourcePolicyError` (when `strict=True`),

and always logs one line explaining the decision.

Profiles (proposed defaults):

| Adapter                      | Devices      | Workers policy                                                                |
| ---------------------------- | ------------ | ----------------------------------------------------------------------------- |
| `sklearn.random_forest`      | `cpu` only   | `n_jobs = min(num_workers or os.cpu_count(), 32)`; reject `device=cuda`       |
| `sklearn.mlp`                | `cpu` only   | same as RF; reject `device=cuda`                                              |
| `pytorch.mlp`                | `cpu`/`cuda` | DataLoader `num_workers`: warn if >16 without IterableDataset                 |
| `gpflow.gpr`                 | `cpu`/`cuda` | single-device in v0.1.0; multi-GPU explicitly not supported                   |
| `ensembles.fnn_ensemble`     | `cpu`/`cuda` | inherit torch policy; ensemble size multiplies memory, warn if > ~100k rows   |

Rejection examples (`strict: true`):

- `sklearn.random_forest` + `device: cuda` → `ResourcePolicyError("random_forest does not run on GPU; set device: cpu or switch model.")`
- `max_gpus: 4` for any model today → `ResourcePolicyError("Multi-GPU training is not yet implemented (see docs/ROADMAP.md §DDStore).")`

With `strict: false` (default) the spec is **adjusted** with a `UserWarning`
so the pipeline still makes progress.

### 2.3 "How many workers are being used" log statements

Every adapter's `fit()` must emit a single, consistent line before training
starts:

```
[surge.fit] model=sklearn.random_forest backend=sklearn device=cpu
            n_jobs=8 n_train=12345 n_features=6 n_outputs=4 seed=42
```

```
[surge.fit] model=pytorch.mlp backend=pytorch device=cuda:0
            dataloader_num_workers=4 pin_memory=True persistent_workers=True
            n_train=12345 n_features=6 n_outputs=4 epochs=200 batch_size=256
```

The same dict gets stored verbatim in `run_summary.json` under a new
`resources_used` key for provenance. This is the clear statement you asked
for.

### 2.4 Implementation steps (in order)

1. Add `ResourceSpec` dataclass (new module `surge/hpc/policy.py`).
2. Add `resources` field to `SurrogateWorkflowSpec`.
3. Add `.resources` on `EngineRunConfig`; plumb through `SurrogateEngine`.
4. Add `resource_profile` + `validate_resources()` on `BaseModelAdapter` with
   a sensible default (CPU-only, no restriction).
5. Override in each concrete adapter per the table above.
6. Emit the `[surge.fit] ...` log line from a base-class helper so every
   adapter gets it for free.
7. Persist `resources_used` to `run_summary.json` and add to
   `metrics.json`.
8. Update `configs/demo_release.yaml` to show the `resources:` block.
9. Unit tests: one test per adapter checking that a bad combo is rejected
   (strict) or warned (non-strict).

This lands in v0.1.0 and is a **user-visible** improvement (a headline feature
for the release notes).

---

## 3. Inference & metrics: model size + avg inference time

Today `metrics.json` records error metrics only. Add, per-model per-run:

- `model_size_bytes` — total bytes of the serialized artifact (joblib for
  sklearn, `state_dict` .pt for torch, GPflow checkpoint for GPflow).
- `n_parameters` — sum of numel over parameters (torch), `n_estimators *
  avg_tree_nodes` estimate (RF), `kernel_variance + lengthscales` count
  (GPR); documented per backend.
- `inference_ms_per_sample` — median over `k` warmups + `N` samples.
- `inference_throughput_samples_per_s` — reported at batch sizes
  `[1, 32, 256, 1024]` where memory allows.

Implementation: one helper `surge/metrics.py::measure_inference()` that
every adapter calls at the end of `fit()` (or that the engine calls after
training). Values are merged into the existing per-model `ModelRunResult`.

---

## 4. Dataset ladder (increasing complexity, open-source, leaderboard-aware)

You asked to test "single input/output first, then multi-multi." Ranked
ladder, all redistributable, all already reachable from `scikit-learn` or
`pmlb`:

| Rung | Task shape | Dataset                                      | Source               | Notes                          |
| ---- | ---------- | -------------------------------------------- | -------------------- | ------------------------------ |
| 1    | 1 → 1      | synthetic `y = f(x)+ε`                       | inline fixture       | hermetic CI smoke              |
| 2    | 10 → 1     | Diabetes                                     | `sklearn.datasets`   | classical tabular              |
| 3    | 8 → 1      | California housing                           | `sklearn.datasets`   | larger sample                  |
| 4    | **8 → 2**  | **UCI Energy Efficiency**                    | UCI                  | canonical multi-output         |
| 5    | 16 → 16    | M3D-C1 synthetic sample (in repo)            | `data/datasets/M3DC1` | current release demo           |
| 6    | MTR suite  | ATP7D, SCM1d/20d, EDM, OES, RF1, RF2         | OpenML / PMLB        | full multi-target benchmark    |

No central regression leaderboard matches our setup. Submit to
**Papers With Code (tabular regression)** and the **Mulan MTR benchmark**
after internal numbers are defensible. Internal tracking
(`metrics.json` + the new size/latency fields) is the first-pass leaderboard.

Action: ship **rungs 1, 4, 5** in the release tree (rung 1 via fixture,
rung 4 via a one-liner `sklearn_datasets` loader, rung 5 via the sample
CSV). Rungs 2/3/6 land as examples post-release.

---

## 5. CI regression (landed in this PR)

- `.github/workflows/ci.yml` now has three jobs: `test` (unit tests),
  `e2e-regression` (train → predict → ONNX on the sample CSV), `lint`.
- `tests/test_e2e_release_smoke.py` is the single source of truth for the
  "release E2E" story. Any PR that breaks the `import surge` contract, the
  `SurrogateEngine.run()` path, or `torch.onnx.export` + `onnxruntime`
  round-trip fails CI.

---

## 6. Phasing

### Phase 0 — in this PR (release blockers)
- [x] Three-job CI workflow (`test`, `e2e-regression`, `lint`).
- [x] `tests/test_e2e_release_smoke.py`.
- [x] This plan document.

### Phase 1 — before v0.1.0 tag
- [ ] Fix `surge/__init__.py` `from loader import …` (section 1.5).
- [ ] Add `ResourceSpec` + `resources` field + per-adapter profiles + fit log
      line (section 2).
- [ ] Add model-size + inference-time to `metrics.json` (section 3).
- [ ] Ship Energy Efficiency demo config (section 4, rung 4).
- [ ] Exclude `surge.legacy*` from the wheel (section 1.4 Option A).

### Phase 2 — v0.2.0
- [ ] Collapse the two registries into one (section 1.1).
- [ ] Rename adapter/backend modules (section 1.2).
- [ ] Delete `surge/models.py` and `surge/visualization.py` shims.
- [ ] `py.typed` + non-blocking `mypy` in CI (section 1.8).
- [ ] Multi-GPU (DDP / FSDP) lifting the current `max_gpus=1` restriction,
      backed by DDStore (see `docs/dev/EXTENDED_ROADMAP.md`).
