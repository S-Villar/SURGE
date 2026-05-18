# SURGE — Standard Benchmarks & Visualization Plan
## Literature-Standard Benchmarks · Classification + Regression · Loss Curves · ROC / PR / Calibration

---

## 0. What the Repo Has Today (Ground Truth)

From reading `REFACTORING_PLAN.md §1.6` and `ROADMAP.md`:

| Component | Status |
|---|---|
| `surge/viz/analysis.py` | ✅ Exists — generic analysis plots |
| `surge/viz/comparison.py` | ✅ Exists — model comparison plots |
| `surge/viz/hpo.py` | ✅ Exists — HPO history plots |
| `surge/viz/importance.py` | ✅ Exists — feature importance |
| `surge/viz/profiles.py` | ✅ Exists — ICRF-style profile plots |
| `surge/viz/run_viz.py` | ✅ Exists — run-level viz |
| `surge/visualization.py` | ⚠️ Legacy shim — deprecation warning, remove in v0.2.0 |
| Loss curve plotting | ❌ Not wired as first-class spec feature |
| ROC / PR / calibration | ❌ Not present anywhere |
| Classification metrics | ❌ Not present in `surge/metrics.py` |
| Standard image benchmarks | ❌ Not present (MNIST / CIFAR-10 / ImageNet) |
| Benchmark registry / runner | ❌ Not present |

The ROADMAP explicitly calls out: *"full 'one YAML recipe → multi-panel' with
per-column labels/units and agent-assisted flow is not wired as a first-class
spec feature yet."* This plan wires it.

---

## 1. Benchmark Taxonomy — Literature-Standard, Scalable

The benchmark ladder is organized by **scale tier**, not just dataset size.
Each tier is runnable on progressively more capable hardware.

### Tier 0 — Hermetic smoke (< 2 min, CPU only, no downloads)

Used in CI on every push. Inline data generators only.

| Key | Task | Shape | Literature reference |
|---|---|---|---|
| `synthetic.regression_1d` | regression | 1→1 | — |
| `synthetic.classification_binary` | classification | 20→2 | — |
| `synthetic.multioutput_2d` | regression | 8→2 | — |

### Tier 1 — Tabular classics (< 10 min, CPU, sklearn.datasets)

All offline after first fetch. The canonical "does the plumbing work" tier.

| Key | Task | Shape | n_samples | Literature reference |
|---|---|---|---|---|
| `tabular.diabetes` | regression | 10→1 | 442 | Efron et al. 2004 |
| `tabular.california_housing` | regression | 8→1 | 20,640 | Pace & Barry 1997 |
| `tabular.iris` | classification | 4→3 | 150 | Fisher 1936 |
| `tabular.breast_cancer` | classification | 30→2 | 569 | UCI / WDBC |
| `tabular.digits` | classification | 64→10 | 1,797 | Alpaydin 1998 |
| `tabular.wine` | classification | 13→3 | 178 | UCI |
| `tabular.energy_efficiency` | regression (multi) | 8→2 | 768 | Tsanas & Xifara 2012 |
| `tabular.concrete_strength` | regression | 8→1 | 1,030 | Yeh 1998 |

### Tier 2 — Standard ML benchmarks (< 30 min, CPU/GPU, torchvision)

The first tier where CNNs, ResNets, and ViTs are meaningful.

| Key | Task | Shape | n_samples | Standard metric | Literature reference |
|---|---|---|---|---|---|
| `vision.mnist` | classification | 28×28→10 | 70,000 | Top-1 accuracy | LeCun et al. 1998 |
| `vision.fashion_mnist` | classification | 28×28→10 | 70,000 | Top-1 accuracy | Xiao et al. 2017 |
| `vision.cifar10` | classification | 32×32×3→10 | 60,000 | Top-1 accuracy | Krizhevsky 2009 |
| `vision.cifar100` | classification | 32×32×3→100 | 60,000 | Top-1 / Top-5 accuracy | Krizhevsky 2009 |
| `tabular.covertype` | classification | 54→7 | 581,012 | Accuracy | Blackard & Dean 1999 |
| `tabular.higgs_boson` | classification | 28→2 | 11M (subset 100k) | AUROC | Baldi et al. 2014 |

**MNIST and CIFAR-10 are the minimum standard benchmarks** for any CNN or
image surrogate. They are universally recognized, have decades of published
results to compare against, and their loaders are two lines via `torchvision`.

### Tier 3 — Scientific surrogates (30 min – 4 hr, GPU recommended)

The SURGE-native tier. These are regression/classification problems from
actual simulation workflows.

| Key | Task | Shape | Source | Standard metric |
|---|---|---|---|---|
| `fusion.m3dc1_sample` | regression | 13→1 | `data/datasets/M3DC1/` | R², RMSE |
| `fusion.m3dc1_full` | regression | 13→1 | `data/datasets/M3DC1/` | R², RMSE |
| `pde.burgers_1d` | regression (operator) | 64→64 | inline solver | Relative L2 |
| `pde.darcy_2d` | regression (operator) | 64×64→64×64 | FNO paper data | Relative L2 |
| `sequence.lorenz63` | regression (rollout) | 3→3 | inline ODE | NRMSE horizon |
| `classification.plasma_stability` | classification | 12→2 | UCI | AUROC, F1 |
| `classification.material_phase` | classification | 81→2 | UCI Superconductor | AUROC, F1 |

### Tier 4 — Large-scale / ImageNet-class (GPU, multi-hour)

Not in CI. Run on scheduled HPC jobs or manual trigger. Results stored under
`benchmark_reports/`.

| Key | Task | Shape | n_samples | Standard metric | Notes |
|---|---|---|---|---|---|
| `vision.imagenet` | classification | 224×224×3→1000 | 1.28M | Top-1 / Top-5 | ResNet-50 baseline: 76.1% top-1 |
| `vision.imagenet_tiny` | classification | 64×64×3→200 | 100,000 | Top-1 accuracy | Scaled-down ImageNet; GPU in ~2hr |
| `pde.navier_stokes_2d` | regression (operator) | 64×64→64×64 | FNO paper data | Relative L2 | FNO baseline: ~1.8% |
| `tabular.higgs_full` | classification | 28→2 | 11M | AUROC | XGBoost baseline: 0.885 |

---

## 2. Published Baselines to Compare Against

Every SURGE benchmark result should be compared against the published number
for that dataset. These are the numbers from the original papers or widely
accepted leaderboard entries.

### Regression baselines

| Benchmark | Model | Published metric | Source |
|---|---|---|---|
| `tabular.california_housing` | Random Forest | R² ≈ 0.82 | sklearn docs |
| `tabular.concrete_strength` | Gradient Boosting | RMSE ≈ 4.5 | Yeh 1998 |
| `tabular.energy_efficiency` | Neural Net | R² ≈ 0.99 | Tsanas & Xifara 2012 |
| `pde.burgers_1d` | FNO | Rel. L2 ≈ 0.64% | Li et al. 2021 |
| `pde.darcy_2d` | FNO | Rel. L2 ≈ 0.90% | Li et al. 2021 |
| `pde.navier_stokes_2d` | FNO (T=20) | Rel. L2 ≈ 1.8% | Li et al. 2021 |

### Classification baselines

| Benchmark | Model | Published metric | Source |
|---|---|---|---|
| `vision.mnist` | LeNet-5 | 99.05% top-1 | LeCun 1998 |
| `vision.mnist` | ResNet-18 | 99.65% top-1 | Various |
| `vision.fashion_mnist` | ResNet-18 | ~95% top-1 | Xiao et al. 2017 |
| `vision.cifar10` | ResNet-20 | 91.25% top-1 | He et al. 2016 |
| `vision.cifar10` | ResNet-56 | 93.03% top-1 | He et al. 2016 |
| `vision.cifar100` | ResNet-56 | 72.3% top-1 | He et al. 2016 |
| `vision.imagenet` | ResNet-50 | 76.1% top-1 | He et al. 2016 |
| `vision.imagenet` | ViT-B/16 | 81.8% top-1 | Dosovitskiy et al. 2021 |
| `tabular.higgs_boson` | Deep NN | AUROC 0.881 | Baldi et al. 2014 |
| `classification.plasma_stability` | RF | Acc ≈ 97.8% | UCI leaderboard |

These numbers go into the `expected_metrics` block of each benchmark YAML and
into `docs/benchmark_results.md`.

---

## 3. New `surge/viz/` Modules Required

The existing `surge/viz/` modules handle EDA, HPO, feature importance, and
profile plots. Three new modules are needed:

```
surge/viz/
  analysis.py        ✅ exists
  comparison.py      ✅ exists
  hpo.py             ✅ exists
  importance.py      ✅ exists
  profiles.py        ✅ exists
  run_viz.py         ✅ exists
  classification.py  ← NEW: ROC, PR, confusion matrix, calibration
  training.py        ← NEW: loss curves, lr schedule, gradient norms
  benchmark.py       ← NEW: leaderboard tables, scaling curves, multi-model comparison
```

### 3.1 `surge/viz/classification.py`

All plots save to `runs/<tag>/plots/` alongside existing artifacts.

```python
# surge/viz/classification.py
"""
Classification diagnostic plots for SURGE.

All functions follow the same signature contract:
    plot_*(y_true, y_pred_or_prob, *, labels=None, title=None,
           save_path=None, ax=None) -> matplotlib.figure.Figure

save_path: if provided, saves PNG + PDF side by side (publication ready).
ax:        if provided, draws into existing axes (for subplot composition).
"""

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,         # shape (n_samples, n_classes) or (n_samples,) for binary
    *,
    labels: list[str] | None = None,
    per_class: bool = True,     # plot one curve per class + micro/macro average
    title: str = "ROC Curve",
    save_path: Path | None = None,
    ax=None,
) -> Figure:
    """
    Binary: one curve + AUC in legend.
    Multiclass: one curve per class (OvR) + micro-average + macro-average.
    Reference lines: random classifier diagonal.
    Shaded band: ±1 std if called with ensemble predictions.
    """

def plot_precision_recall_curve(
    y_true, y_prob, *,
    labels=None, title="Precision-Recall Curve",
    save_path=None, ax=None,
) -> Figure:
    """
    PR curve with AP (average precision) in legend.
    Multiclass: one curve per class + micro-average.
    Baseline: random classifier horizontal line at class prevalence.
    """

def plot_confusion_matrix(
    y_true, y_pred, *,
    labels=None,
    normalize: bool = True,     # show fractions, not counts
    title="Confusion Matrix",
    save_path=None, ax=None,
) -> Figure:
    """
    Annotated heatmap. normalize=True shows row-wise recall per class.
    normalize=False shows raw counts in lower triangle, fractions upper.
    """

def plot_calibration_curve(
    y_true, y_prob, *,
    n_bins: int = 10,
    strategy: str = "uniform",  # "uniform" | "quantile"
    title="Calibration Curve",
    save_path=None, ax=None,
) -> Figure:
    """
    Reliability diagram (fraction of positives vs mean predicted probability).
    Includes Expected Calibration Error (ECE) in subtitle.
    Perfectly calibrated diagonal shown as reference.
    """

def plot_classification_dashboard(
    y_true, y_pred, y_prob, *,
    labels=None,
    model_name: str = "",
    save_path: Path | None = None,
) -> Figure:
    """
    4-panel dashboard in a single figure:
      [ROC]  [Precision-Recall]
      [Confusion Matrix]  [Calibration]
    This is the one-call summary for any classification benchmark result.
    Saves as PNG + PDF.
    """
```

**Example output structure for a benchmark run:**
```
runs/my_run/
  plots/
    roc_curve.png
    roc_curve.pdf
    precision_recall.png
    confusion_matrix.png
    calibration.png
    classification_dashboard.png   ← 4-panel, publication ready
  metrics.json
  workflow_summary.json
```

---

### 3.2 `surge/viz/training.py`

For any model that exposes `training_history()` (currently `pytorch.mlp`
via `pytorch_impl.py`; extended to all new neural adapters).

```python
# surge/viz/training.py
"""
Training dynamics plots for SURGE neural backends.

Reads from:
  - adapter.training_history() → list[dict] with keys:
      epoch, train_loss, val_loss, [train_metric], [val_metric], [lr]
  - artifact: runs/<tag>/training_log.jsonl  (one JSON per line, same keys)

Both sources produce the same plot — pass history directly or point at
the artifact path.
"""

def plot_loss_curve(
    history: list[dict] | Path,
    *,
    metrics: list[str] | None = None,  # e.g. ["r2", "rmse"] or ["accuracy"]
    log_scale: bool = False,
    title: str = "Training History",
    save_path: Path | None = None,
    ax=None,
) -> Figure:
    """
    Top panel: train loss + val loss vs epoch.
    Bottom panel (if metrics given): one line per metric, train + val.
    Marks early-stopping epoch with vertical dashed line if present.
    Shaded region: gap between train and val (overfitting indicator).

    Designed to handle:
    - MSELoss curves (regression MLP)
    - CrossEntropyLoss curves (classification MLP)
    - ELBO curves (VAE)
    - Any future adapter that writes training_log.jsonl
    """

def plot_lr_schedule(
    history: list[dict] | Path,
    *,
    title: str = "Learning Rate Schedule",
    save_path: Path | None = None,
    ax=None,
) -> Figure:
    """Learning rate vs epoch. Only rendered if 'lr' key present in history."""

def plot_training_dashboard(
    history: list[dict] | Path,
    *,
    model_name: str = "",
    save_path: Path | None = None,
) -> Figure:
    """
    3-panel dashboard:
      [Loss curve (train + val)]
      [Primary metric (train + val)]
      [Learning rate schedule]
    Renders whatever panels have data; skips silently if lr not logged.
    """

def compare_training_curves(
    histories: dict[str, list[dict] | Path],
    *,
    metric: str = "val_loss",
    log_scale: bool = False,
    title: str | None = None,
    save_path: Path | None = None,
) -> Figure:
    """
    Overlay val_loss (or any logged metric) for multiple models/runs.
    Used in benchmark comparisons: "which model converges fastest?"
    histories = {"pytorch.mlp": [...], "pytorch.residual_mlp": [...]}
    """
```

**Training log contract** (what every neural adapter must write to disk):

```jsonl
{"epoch": 1, "train_loss": 0.342, "val_loss": 0.389, "train_r2": 0.61, "val_r2": 0.57, "lr": 1e-3}
{"epoch": 2, "train_loss": 0.291, "val_loss": 0.334, "train_r2": 0.68, "val_r2": 0.63, "lr": 1e-3}
...
{"epoch": 47, "train_loss": 0.089, "val_loss": 0.102, "train_r2": 0.91, "val_r2": 0.89, "lr": 2.5e-4, "early_stop": true}
```

Saved to `runs/<tag>/training_log.jsonl`. This file is the source of truth;
`adapter.training_history()` just parses it in memory.

---

### 3.3 `surge/viz/benchmark.py`

```python
# surge/viz/benchmark.py
"""
Cross-model, cross-benchmark visualization for SURGE benchmark runs.

Reads from: benchmark_reports/<key>/<timestamp>/result.json
"""

def plot_benchmark_leaderboard(
    results: list[BenchmarkResult] | Path,
    *,
    metric: str = "test_r2",
    title: str | None = None,
    baseline_key: str | None = None,   # highlight baseline bar
    save_path: Path | None = None,
) -> Figure:
    """
    Horizontal bar chart: one bar per model, sorted by metric.
    Annotates published baseline if provided.
    Error bars if result contains std (e.g. from k-fold).
    """

def plot_scaling_curve(
    results_by_size: dict[int, list[BenchmarkResult]],
    *,
    metric: str = "test_r2",
    title: str = "Performance vs Dataset Size",
    save_path: Path | None = None,
) -> Figure:
    """
    X-axis: training set size (log scale).
    Y-axis: metric.
    One line per model — shows who benefits most from more data.
    """

def plot_metric_table(
    results: list[BenchmarkResult],
    *,
    metrics: list[str] | None = None,
    save_path: Path | None = None,
) -> Figure:
    """
    Styled matplotlib table — publishable summary.
    Bold best-per-column. Red worst. Gray published baseline row.
    """
```

---

## 4. YAML Recipe for Visualization (extending ROADMAP item)

The ROADMAP says: *"one YAML recipe → multi-panel EDA"*. Extend it to cover
benchmark plots:

```yaml
# surge/benchmarks/vision/cifar10.yaml
benchmark:
  key: vision.cifar10
  tier: standard
  task_type: classification
  seed: 42

  dataset:
    loader: torchvision.cifar10
    root: ${SURGE_DATA_DIR:-~/.surge/data}   # env-var with default
    download: true

  splits:
    train: 0.8
    val: 0.1
    test: 0.1

  baseline_model: pytorch.resnet20
  candidate_models:
    - pytorch.resnet56
    - pytorch.mlp_classifier

  expected_metrics:
    test_accuracy: {min: 0.91}       # ResNet-20 paper number
    test_top5_accuracy: {min: 0.99}

  visualization:                     # ← NEW: recipe-driven viz
    on_run_complete:
      - classification_dashboard     # ROC + PR + confusion + calibration
      - training_dashboard           # loss + metric + lr
    on_benchmark_complete:
      - leaderboard                  # bar chart across candidate_models
      - scaling_curve                # perf vs n_train
    save_formats: [png, pdf]         # both for publications
    dpi: 150

  cli_command: >
    python -m surge.benchmarks.run
      --benchmark vision.cifar10
      --model pytorch.resnet20
      --output benchmark_reports/cifar10/
```

The `visualization.on_run_complete` list maps directly to functions in
`surge/viz/classification.py` and `surge/viz/training.py`. The benchmark
runner calls them automatically after each model finishes.

---

## 5. New Models Required for Standard Benchmarks

MNIST / CIFAR-10 / ImageNet require image-native architectures. These go in
`surge/model/backends/` and follow the same adapter pattern.

| Registry key | Architecture | MNIST | CIFAR-10 | ImageNet | Paper |
|---|---|---|---|---|---|
| `pytorch.lenet5` | LeNet-5 | ✅ primary | ❌ | ❌ | LeCun 1998 |
| `pytorch.resnet20` | ResNet-20 | ✅ | ✅ primary | ❌ | He et al. 2016 |
| `pytorch.resnet50` | ResNet-50 | ✅ | ✅ | ✅ primary | He et al. 2016 |
| `pytorch.vit_tiny` | ViT-Tiny (patch 4) | ✅ | ✅ | — | Dosovitskiy 2021 |
| `pytorch.mlp_classifier` | MLP + softmax | ✅ | ❌ | ❌ | baseline |

**Implementation rule:** all image models must accept both:
- Raw `numpy` arrays of shape `(N, H, W, C)` — the SURGE default
- `torch.utils.data.DataLoader` — for CIFAR-10/ImageNet-scale data that
  cannot fit in RAM; this requires the `IterableDataset` path from
  `ROADMAP.md §large datasets`

**LeNet-5 for MNIST** (simplest; implement first):

```python
# surge/model/backends/lenet.py
import torch.nn as nn

class LeNet5(nn.Module):
    """LeCun 1998. Input: (N, 1, 28, 28). Output: logits (N, 10)."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))
```

**ResNet-20 for CIFAR-10** (the He et al. 2016 reference implementation):

```python
# surge/model/backends/resnet_cifar.py
# Standard ResNet for CIFAR-10 (not the ImageNet ResNet — different stem).
# Input: (N, 3, 32, 32). Classes: 10 or 100.
# n=3  → ResNet-20   (0.27M params)
# n=9  → ResNet-56   (0.85M params)
# n=18 → ResNet-110  (1.7M params)
```

**ResNet-50 for ImageNet** — use `torchvision.models.resnet50` directly,
wrapped in a SURGE adapter. Do not reimplement.

---

## 6. `pyproject.toml` Changes Required

```toml
[project.optional-dependencies]
# existing
torch = ["torch>=1.9.0"]
onnx  = ["onnx>=1.14", "onnxscript>=0.1", "onnxruntime>=1.17"]
dev   = ["pytest>=6.0", "pytest-cov", "ruff>=0.1.0", "h5py>=3.0"]

# NEW
vision = [
    "torchvision>=0.12.0",   # MNIST, CIFAR-10, ImageNet loaders
]

benchmarks = [
    "torchvision>=0.12.0",
    "scikit-learn>=1.0.0",   # already a core dep — fetch_openml for tabular
]

viz = [
    "matplotlib>=3.5.0",     # already a core dep
    "seaborn>=0.12.0",       # NEW — heatmaps for confusion matrix
    # scikit-learn already provides roc_curve, precision_recall_curve
]

# Convenience bundle: everything needed to run all standard benchmarks
all = ["surge-ml[torch,onnx,vision,viz,dev]"]
```

`seaborn` is the only new dependency. Everything else is already present.

---

## 7. Implementation Order (CI-safe, additive)

### Step 1 — `surge/viz/training.py` (no new deps, no CI risk)

- Reads `training_log.jsonl` already written by `pytorch_impl.py`
- `plot_loss_curve()`, `plot_training_dashboard()`, `compare_training_curves()`
- Unit test: generate synthetic history dict, assert figure is returned, assert
  PNG saved to tmp_path
- Wire into `SurrogateEngine.run()`: after fit, if `training_log.jsonl` exists,
  call `plot_training_dashboard(save_path=run_dir/"plots/training.png")`
  automatically

### Step 2 — Classification metrics in `surge/metrics.py`

```python
# Additive additions — no existing function touched
def accuracy_score(y_true, y_pred) -> float
def f1_score(y_true, y_pred, average="macro") -> float
def auroc(y_true, y_prob, multi_class="ovr") -> float
def log_loss(y_true, y_prob) -> float
def top_k_accuracy(y_true, y_prob, k=5) -> float
def expected_calibration_error(y_true, y_prob, n_bins=10) -> float
```

All delegate to `sklearn.metrics` — no custom implementations.

### Step 3 — `surge/viz/classification.py`

- `plot_roc_curve()`, `plot_precision_recall_curve()`, `plot_confusion_matrix()`,
  `plot_calibration_curve()`, `plot_classification_dashboard()`
- Dependencies: `matplotlib` (existing), `seaborn` (new, optional import)
- Unit test: synthetic binary + multiclass labels, assert figures returned

### Step 4 — Benchmark registry + runner CLI

- `surge/benchmarks/registry.py` — `BenchmarkRegistry`, `BENCHMARK_REGISTRY`
- `surge/benchmarks/base.py` — `BenchmarkResult`, `run_benchmark()`
- `surge/benchmarks/run.py` — CLI
- Wire visualization: after `run_benchmark()`, call `plot_classification_dashboard()`
  or `plot_training_dashboard()` depending on `task_type`

### Step 5 — Tier 0 + Tier 1 benchmark YAMLs

All use `sklearn.datasets` — no new dependencies, work offline after first fetch.

### Step 6 — MNIST benchmark + LeNet-5 adapter

```
pip install surge-ml[torch,vision]
python -m surge.benchmarks.run --benchmark vision.mnist --model pytorch.lenet5
```

Expected: ~99.0% top-1, loss curve PNG + classification dashboard PNG in
`benchmark_reports/mnist/`.

### Step 7 — CIFAR-10 benchmark + ResNet-20 adapter

```
python -m surge.benchmarks.run --benchmark vision.cifar10 --model pytorch.resnet20
```

Expected: ~91.3% top-1 (matches He et al. 2016 exactly — this is the
validation that the implementation is correct).

### Step 8 — `surge/viz/benchmark.py`

- `plot_benchmark_leaderboard()`, `plot_scaling_curve()`, `plot_metric_table()`
- Reads `benchmark_reports/*/result.json`

### Step 9 — Tier 4 (ImageNet) — manual trigger only

```yaml
# .github/workflows/benchmark_full.yml
on:
  workflow_dispatch:    # manual trigger only — not in CI
  schedule:
    - cron: "0 2 * * 0"  # weekly Sunday 2am
```

---

## 8. Complete File Map (additions only)

```
surge/
  viz/
    classification.py    ← NEW: ROC, PR, confusion, calibration, dashboard
    training.py          ← NEW: loss curves, lr, training dashboard
    benchmark.py         ← NEW: leaderboard, scaling curves, metric table

  metrics.py             ← EXTEND: add accuracy, f1, auroc, log_loss, top_k, ECE

  model/
    backends/
      lenet.py           ← NEW: LeNet-5 (MNIST)
      resnet_cifar.py    ← NEW: ResNet-20/56/110 (CIFAR-10/100)
      resnet_imagenet.py ← NEW: thin wrapper around torchvision.models.resnet50
    adapters/
      lenet.py           ← NEW: registers pytorch.lenet5
      resnet.py          ← NEW: registers pytorch.resnet20, pytorch.resnet50

  benchmarks/
    __init__.py
    registry.py          ← NEW: BenchmarkRegistry, BENCHMARK_REGISTRY
    base.py              ← NEW: BenchmarkResult, run_benchmark()
    run.py               ← NEW: CLI entry point

    synthetic/
      regression_1d.yaml
      classification_binary.yaml

    tabular/
      diabetes.yaml
      california_housing.yaml
      energy_efficiency.yaml
      concrete_strength.yaml
      iris.yaml
      breast_cancer.yaml
      digits.yaml

    vision/
      mnist.yaml         ← NEW: standard MNIST benchmark
      fashion_mnist.yaml ← NEW
      cifar10.yaml       ← NEW: standard CIFAR-10 benchmark
      cifar100.yaml      ← NEW
      imagenet.yaml      ← NEW: tier 4, manual trigger only
      imagenet_tiny.yaml ← NEW: tier 3, GPU recommended

    sequence/
      lorenz63.yaml

    pde/
      burgers_1d.yaml
      darcy_2d.yaml
      navier_stokes_2d.yaml  ← tier 4

    fusion/
      m3dc1_sample.yaml
      m3dc1_full.yaml

tests/
  unit/
    test_viz_classification.py  ← NEW
    test_viz_training.py        ← NEW
    test_benchmark_registry.py  ← NEW
    test_classification_metrics.py ← NEW

  benchmarks/
    test_smoke_benchmarks.py    ← NEW

docs/
  benchmark_results.md          ← NEW: published baseline table
  benchmark_policy.md           ← NEW: tier definitions, acceptance checklist
```

---

## 9. What a Complete MNIST Benchmark Run Produces

```
benchmark_reports/mnist/2026-05-17T14:32:00/
  result.json                  ← machine-readable pass/fail + metrics
  training_log.jsonl           ← epoch-by-epoch loss + accuracy
  plots/
    training_dashboard.png     ← loss curve + accuracy curve + lr schedule
    training_dashboard.pdf
    roc_curve.png              ← 10-class OvR ROC + micro/macro average
    precision_recall.png
    confusion_matrix.png       ← 10×10 normalized heatmap
    calibration.png            ← reliability diagram + ECE
    classification_dashboard.png  ← 4-panel publication figure
    classification_dashboard.pdf
  model.pt                     ← saved weights
  metrics.json
```

`result.json` for a passing run:
```json
{
  "benchmark": "vision.mnist",
  "model": "pytorch.lenet5",
  "tier": "standard",
  "task_type": "classification",
  "seed": 42,
  "metrics": {
    "test_accuracy": 0.9912,
    "test_top5_accuracy": 0.9999,
    "test_auroc": 0.9998,
    "test_f1_macro": 0.9911,
    "test_log_loss": 0.031,
    "test_ece": 0.008
  },
  "published_baseline": {
    "model": "LeNet-5",
    "test_accuracy": 0.9905,
    "source": "LeCun et al. 1998"
  },
  "passed": true,
  "runtime_s": 187.4,
  "model_size_bytes": 431200,
  "parameter_count": 61706,
  "inference_ms_per_sample": 0.21
}
```
