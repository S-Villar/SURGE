---
name: surge-scale
description: Train SURGE models at scale with resource management - GPU/MPS device selection, parallel benchmark fan-out (surge bench --parallel), thread budgeting, long-run monitoring. Use when asked to "train at scale", "use the GPU", "run many models/benchmarks in parallel", "speed up training", or when a study will take more than ~15 min on defaults.
---

# Training at scale with SURGE

Three levers, in order of payoff: pick the right device, parallelize
independent jobs, budget threads so workers don't fight.

## 1. Device selection (R15)

`surge.utils.resolve_device` governs every torch backend. Control it
with the `SURGE_DEVICE` env var or a `device=` model param:

| Setting | Meaning |
|---|---|
| (unset) | cuda if present, else cpu — deterministic default |
| `SURGE_DEVICE=auto` | cuda > **mps** > cpu (opt-in to Apple GPU) |
| `SURGE_DEVICE=mps` / `cuda:1` / `cpu` | forced |

**MPS safety table (measured):** conv/spectral models are safe and fast
— FNO-2D ~4x, U-Net ~2x faster with identical accuracy. **Recurrent
models (LSTM/GRU) are NOT safe on MPS** (lorenz63 R² collapses
0.99 → 0.32). Never set `SURGE_DEVICE=auto` for a run that includes
RNNs; force `device: cpu` on those model blocks instead. For
bit-reproducible results pin cpu.

Rule of thumb: operator-learning studies (fno2d/unet on ≥64² fields) →
always `SURGE_DEVICE=auto`; tabular MLP/GBM → cpu is usually fine.

## 2. Parallel fan-out (R16 first slice)

Independent (benchmark, model) jobs run as concurrent subprocesses:

```bash
surge bench -b plasma.qlknn_transport -m all --seeds 3 --parallel 4
```

- BLAS/OpenMP threads are split evenly (`cpu_count / N` each) so N
  workers don't oversubscribe; don't exceed ~cpu_count/2 workers for
  torch models.
- Requires saving (no `--no-save`); results aggregate automatically
  from `benchmark_reports/` and the combined table prints at the end.
- One GPU + many workers = contention: with `SURGE_DEVICE=auto`, keep
  `--parallel 2` max, or leave heavy operator models to a solo run.
- Exit code 1 from a job means "ran but below the pass gate" — that is
  a completed job, not a failure; real failures print a stderr tail.

## 3. Long runs: launch + monitor

- Run in the background and stream the log unbuffered:
  `python -u examples/<study>.py > /tmp/study.log 2>&1 &`
- Monitor: `tail -f /tmp/study.log`, per-epoch JSONL under
  `runs/<tag>/training_log_*.jsonl`, or render the dashboard:
  `python examples/viz_theme_gallery.py --only mission_control --hpo-run runs/<tag>`
- MLflow live curves: `mlflow_tracking: true` in the spec, then the
  trial pages show per-epoch train/val loss (nested HPO child runs).
- On OOM: halve `batch_size` first, then subsample training data;
  memory tiers per benchmark live in the registry
  (`resource_expectation`).
- Keep the machine awake for unattended runs: `caffeinate -is` (macOS).

## Not yet available (see docs/design/RESOURCE_MANAGEMENT.md)

Spec-level `parallel_models` inside one `surge run`, Optuna `n_jobs`,
memory-tier enforcement, multi-GPU DDP — R16–R18. Don't promise these.
