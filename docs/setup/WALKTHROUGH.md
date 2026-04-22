# First-run walkthrough (NERSC `$SCRATCH` example)

This is the exact copy-paste-ready flow we use to validate a fresh
checkout of SURGE on Perlmutter. It works without changes on any
workstation too — just replace `$SCRATCH` with the directory you want
the checkout, venv, and run artifacts to live in.

The walkthrough goes: clone → venv → install → verify imports →
tests → CLI quickstart (two public datasets) → artifact inspection.

> **Before you start:** remove any previous attempt to keep the state
> clean, e.g. `rm -rf $SCRATCH/surge-release-test`.

## 1. Clone into `$SCRATCH`

```bash
export TEST_DIR=$SCRATCH/surge-release-test
mkdir -p "$TEST_DIR" && cd "$TEST_DIR"
git clone git@github.com:S-Villar/SURGE.git
```

## 2. Install `uv` (one-time) and create a Python 3.11 venv

Putting the venv alongside (not inside) the repo keeps `git status`
noise-free and lets you nuke the venv without touching the source tree.
We use [`uv`](https://github.com/astral-sh/uv) because the
`torch + onnx + dev` extras resolve in ~30 s under `uv` vs 3-5 min
with bare `pip`, which matters a lot when `$SCRATCH` gets purged and
you have to rebuild the env.

### 2a. Install `uv` (skip if `uv --version` already works)

`uv` isn't on NERSC login nodes by default. Either of the two paths
below drops the binary in `~/.local/bin`, which is on the default
login-node `PATH`:

```bash
# Primary — the Astral installer script (fast, self-updating).
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Fallback — if curl is blocked on the login node during maintenance.
# python3.11 -m pip install --user uv
# export PATH="$HOME/.local/bin:$PATH"

# Make it survive new login shells (only append if not already there).
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

uv --version            # -> uv 0.x.y
```

`$HOME` is shared across Perlmutter login nodes, so this install is a
one-time cost per user account, not per session.

### 2b. Create and activate the venv

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

python --version      # -> Python 3.11.x
which python          # -> $TEST_DIR/.venv/bin/python
```

If you really can't install `uv`, `python3.11 -m venv .venv` is a
drop-in replacement; downstream `uv pip install` calls become plain
`pip install`.

## 3. Editable install with `torch + onnx + dev` extras

The `dev` extra includes `pytest`, `ruff`, and **`h5py`** (HDF5 /
M3DC1 batch tests).

```bash
cd "$TEST_DIR/SURGE"
uv pip install -e ".[torch,onnx,dev]"
```

## 4. Sanity-check the import surface

```bash
python - <<'PY'
import surge, torch, onnx, onnxruntime, sklearn
print(f"surge       {surge.__version__}")
print(f"torch       {torch.__version__}")
print(f"onnx        {onnx.__version__}")
print(f"onnxruntime {onnxruntime.__version__}")
print(f"sklearn     {sklearn.__version__}")
PY
```

A cosmetic `Warning: GPflow not available or incompatible` is expected
and harmless — GPflow is an optional backend.

## 5. Run the test suite

```bash
pytest -q
```

Current steady state with `.[torch,onnx,dev]` (which pulls **`h5py`**
for M3DC1 tests): **52 passed, 1 skipped** in under 30 s. The skip is a
legacy visualization scaffold in `test_model_comparison.py` pending
migration to `surge.viz`.

## 6. End-to-end CLI quickstart

The CLI is packaged under `examples/quickstart.py`. Run it **from
inside the SURGE checkout** so `python -m examples.quickstart` can
resolve the package, and so `runs/<tag>/` lands next to the source
tree (`--output-dir` defaults to the repo root).

```bash
cd "$TEST_DIR/SURGE"

# 6a. Diabetes + random forest (~5 s), with inference round-trip and plots.
python -m examples.quickstart --dataset diabetes --model rf --infer --viz

# 6b. California housing + PyTorch MLP (~90 s), inference + plots.
python -m examples.quickstart --dataset california --model mlp --infer --viz

# 6c. Optional: short Optuna HPO sweep (5 trials × 50 epochs each,
#     ~1-2 min on CPU). Each trial prints its own epoch loss history.
python -m examples.quickstart --dataset california --model mlp --n-trials 5 --viz
```

`--viz` calls `surge.viz.viz_run`, which reads the per-split parquet
predictions and writes a 2D-density "regression map" per output
(`plots/inference_comparison_output_<i>.png`) plus a combined grid
(`plots/inference_comparison_grid.png`). Each subplot includes the R²
annotation and the diagonal reference.

Expected tail of (6b):

```text
  train R²    = 0.849
  val   R²    = 0.816
  test  R²    = 0.810
  test  RMSE  = 0.499
  model       = 133.3 KB, parameter count n/a
  inference   = 0.04 ms/sample

[viz] parity plots (predictions vs ground truth):
        plots/inference_comparison_output_0.png
        plots/inference_comparison_grid.png

[artifacts] runs/california_mlp
  ├── hpo/
  ├── models/
  │   └── pytorch.mlp.joblib                  50.6 KB
  ├── plots/
  │   ├── inference_comparison_grid.png       61.4 KB
  │   └── inference_comparison_output_0.png   61.4 KB
  ├── predictions/
  │   ├── pytorch.mlp_test.parquet            63.4 KB
  │   ├── pytorch.mlp_train.parquet          201.3 KB
  │   └── pytorch.mlp_val.parquet             33.6 KB
  ├── scalers/
  │   └── inputs.joblib                         775 B
  ├── env.txt                                   355 B  — pip freeze at run time
  ├── git_rev.txt                                41 B  — repo HEAD or 'unknown'
  ├── metrics.json                             1.1 KB  — per-model train/val/test + timings
  ├── model_card_pytorch.mlp.json               619 B  — data + model provenance card
  ├── run.log                                   180 B  — stdout capture
  ├── spec.yaml                                1.2 KB  — workflow spec (re-runnable)
  ├── train_data_ranges.json                    778 B  — canonical input column order + min/max
  ├── training_history_pytorch.mlp.json       38.4 KB  — per-epoch loss / metric curves
  ├── training_progress_pytorch.mlp.jsonl     32.3 KB  — streaming JSONL progress log
  └── workflow_summary.json                    4.5 KB  — metrics + profile + resources_used

[infer] round-trip inference on the first 5 rows:
        model  : pytorch.mlp.joblib  (pytorch)
        inputs : ['AveBedrms','AveOccup','AveRooms','HouseAge',
                  'Latitude','Longitude','MedInc','Population']
        y_true : [4.53 3.58 3.52 3.41 3.42]
        y_hat  : [3.86 4.26 3.85 2.85 2.74]
```

## 7. Inspect the artifacts

`examples.quickstart` already prints the annotated tree shown above at
the end of each run. For a quick summary of test-set metrics:

```bash
python -c "import json; \
           d = json.load(open('runs/california_mlp/metrics.json')); \
           print(json.dumps(d['pytorch.mlp']['test'], indent=2))"
```

To (re)generate or refresh the parity plots for an existing run
without retraining, you can call the packaged CLI:

```bash
python -m surge.cli viz --run-dir runs/california_mlp
```

## Notes and gotchas

- **`runs/` location.** `examples/quickstart.py` writes the generated
  CSV and the `runs/<tag>/` directory under `--output-dir`, which
  defaults to the SURGE repo root. Pass `--output-dir /path/to/dir` to
  redirect them elsewhere (for example, a per-job Slurm scratch).
- **PyTorch checkpoint format.** `pytorch.mlp.joblib` is a `torch.save`
  zip archive, not a joblib pickle. Inference code should use
  `PyTorchMLPModel().load()` (as `_demo_inference` in
  `examples/quickstart.py` does); `joblib.load` errors out on it.
- **Canonical input column order.** SURGE sorts input columns
  alphabetically during preprocessing. Every run writes the final
  order to `runs/<tag>/train_data_ranges.json` under
  `inputs.columns` — read it back rather than trusting the CSV
  header order when scoring new inputs.
- **`internal/` is gitignored.** If you drop proprietary datasets,
  Slurm templates, or scratch notebooks into `internal/<whatever>/`
  under the checkout, they stay local and never show up in
  `git status`.

## Next steps

- README *Quickstart* — captured output transcripts and the full
  artifact listing.
- [`docs/quickstart.rst`](../quickstart.rst) — same content rendered by
  Sphinx, plus the sklearn and PyTorch inference snippets.
- [`docs/setup/INSTALLATION.md`](INSTALLATION.md) — longer install
  reference (extras, troubleshooting, NERSC tips).
