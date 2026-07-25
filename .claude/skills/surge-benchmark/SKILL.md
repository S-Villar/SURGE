---
name: surge-benchmark
description: Run SURGE benchmarks and leaderboards (surge-benchmark CLI), manage dataset caches, thresholds, seeds, and HPO. Use when asked to benchmark models, reproduce leaderboard numbers, or add benchmark results.
---

# SURGE benchmarks

CLI: `surge-benchmark` (alias: `surge run`). Registry keys look like
`tabular.california_housing`, `pde.burgers_1d`, `vision.mnist`,
`plasma.qlknn_transport`.

```bash
surge-benchmark --list                          # all keys by category
surge-benchmark --list-models                   # registered model keys
surge-benchmark -b synthetic.regression_1d -m sklearn.random_forest --no-save   # smoke (<1 s)
surge-benchmark -b tabular.california_housing -m all --seeds 5       # leaderboard, mean±std
surge-benchmark -b pde.burgers_1d --compare-models pytorch.fno1d,pytorch.residual_mlp
surge-benchmark -b tabular.diabetes -m pytorch.mlp --hpo --hpo-trials 20
```

Facts to rely on:
- Results persist to `benchmark_reports/<key>/<timestamp>/result.json`
  (benchmark_key, model_key, metrics incl. `runtime_s`, `passed`, seed) —
  these files are the ONLY source of truth for leaderboards; the directory
  is git-ignored.
- Seeds: default 42; `--seeds N` runs seeds 42..42+N-1 and aggregates.
- Pass/fail thresholds are hard-coded in `_THRESHOLDS`
  (surge/benchmarks/leaderboard.py); descriptive metadata (citations,
  tiers, IO docs) lives in `surge/benchmarks/metadata.yaml`.
- Dataset caches land under `data/datasets/benchmarks/<category>/` (NPZ,
  MNIST/CIFAR raw, sklearn/OpenML cache). First run of a network-backed
  benchmark downloads; afterwards it is offline. Never commit cache files.
- 6 registry entries are placeholders with no runner (ctr23.*,
  plasma.cmod_density_limit) — do not report them as runnable.
- The benchmark path currently bypasses `SurrogateEngine` (own splits,
  metrics, HPO). Treat benchmark and workflow numbers as separate
  pipelines until the consolidation lands.

After runs, regenerate the HTML leaderboard (see the surge-viz skill):
`python -m surge.report.leaderboard --out surge_leaderboard.html`.
