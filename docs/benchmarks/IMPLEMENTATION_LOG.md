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

*(filled in next commit)*

---

## Phase 3 — Classification visualization (`surge/viz/classification.py`)

*(filled in next commit)*

---

## Phase 4–5 — Benchmark registry + Tier 0/1 runner

*(filled in next commit)*
