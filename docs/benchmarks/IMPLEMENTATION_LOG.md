# Model benchmark plan — implementation log

Tracks [SURGE_BENCHMARKS_VIZ_PLAN.md](../../SURGE_BENCHMARKS_VIZ_PLAN.md) delivery on branch `model-bench`.

## Phase 1 — Training visualization (`surge/viz/training.py`)

**Summary**
- New module supports `load_training_history` from list, `.jsonl`, or JSON array file.
- `plot_loss_curve`, `plot_lr_schedule`, `plot_training_dashboard`, `compare_training_curves` (matplotlib Agg backend).
- Workflow writes `training_log_<model>.jsonl` next to `training_history_<model>.json` and auto-generates `plots/training_dashboard_<model>.png` after PyTorch (or any) training that returns `training_history`.
- `PyTorchMLP` now logs `lr` each epoch for the LR panel.

**Verification**
```bash
cd /path/to/SURGE
PYTHONPATH=. python -m pytest tests/unit/test_viz_training.py -q
```

**Showcase (optional, requires torch + tiny workflow)**
After any `torch.mlp` workflow run, check `runs/<tag>/plots/training_dashboard_*.png` and `training_log_*.jsonl`.

---

## Phase 2 — Classification metrics (`surge/metrics.py`)

**Summary**
- Added thin wrappers: `accuracy_score`, `f1_score`, `auroc`, `log_loss`, `top_k_accuracy_score`, `expected_calibration_error` (multiclass-safe max-prob binning for ECE).
- `__all__` documents the public metric API.

**Verification**
```bash
PYTHONPATH=. python -m pytest tests/unit/test_classification_metrics.py -q
```

---

## Phase 3 — Classification visualization (`surge/viz/classification.py`)

**Summary**
- ROC / PR (binary + multiclass OvR), confusion matrix, calibration (+ ECE subtitle), and a 2×2 `plot_classification_dashboard`.
- PNG writes optionally pair with PDF when saving `.png`.

**Verification**
```bash
PYTHONPATH=. python -m pytest tests/unit/test_viz_classification.py -q
```

**Showcase**
```bash
PYTHONPATH=. python -c "from pathlib import Path; import numpy as np; from surge.viz.classification import plot_roc_curve; plot_roc_curve(np.array([0,1,1]), np.array([0.1,0.8,0.9]), save_path=Path('/tmp/roc_demo.png'))"
```

---

## Phase 4–5 — Benchmark registry + Tier 0/1 runner

**Summary**
- New package `surge/benchmarks/`: `BenchmarkResult`, `REGISTRY`, `run_benchmark`, `list_benchmarks`.
- **Tier 0:** `synthetic.regression_1d`, `synthetic.classification_binary` (no downloads).
- **Tier 1:** `tabular.iris`, `tabular.diabetes` via `sklearn.datasets`.
- CLI: `python -m surge.benchmarks.run` (exit code 1 if `passed` is false). Console script: `surge-benchmark` after install.
- Policy: `docs/benchmarks/benchmark_policy.md`.

**Verification**
```bash
PYTHONPATH=. python -m surge.benchmarks.run --list
PYTHONPATH=. python -m surge.benchmarks.run --benchmark synthetic.regression_1d
PYTHONPATH=. python -m pytest tests/benchmarks/test_smoke_benchmarks.py -q
```

Vision / torchvision benchmarks remain future work (plan §6–7).
