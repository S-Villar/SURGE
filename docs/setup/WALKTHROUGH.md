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

## 2. Create a clean Python 3.11 venv *next to* the checkout

Putting the venv alongside (not inside) the repo keeps `git status`
noise-free and lets you nuke the venv without touching the source tree.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

python --version      # -> Python 3.11.x
which python          # -> $TEST_DIR/.venv/bin/python
```

If `uv` is not available, plain `python3.11 -m venv .venv` is
equivalent; pip install times are just slower.

## 3. Editable install with `torch + onnx + dev` extras

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

Current steady state on a clean install: **~47 passed, ~41 skipped** in
under 10 s. The skipped tests target a pre-refactor API that is being
migrated (see `docs/REFACTORING_PLAN.md` §1.9); the passing set covers
the public API users actually touch (`SurrogateEngine`,
`SurrogateWorkflowSpec`, `run_surrogate_workflow`, the model registry,
ONNX round-trip, HPO).

## 6. End-to-end CLI quickstart

The CLI is packaged under `examples/quickstart.py`. Run it **from
inside the SURGE checkout** so `python -m examples.quickstart` can
resolve the package, and so `runs/<tag>/` lands next to the source
tree (`--output-dir` defaults to the repo root).

```bash
cd "$TEST_DIR/SURGE"

# 6a. Diabetes + random forest (~5 s), with inference round-trip.
python -m examples.quickstart --dataset diabetes --model rf --infer

# 6b. California housing + PyTorch MLP (~90 s), with inference round-trip.
python -m examples.quickstart --dataset california --model mlp --infer

# 6c. Optional: add a short Optuna HPO sweep (MLP only).
python -m examples.quickstart --dataset california --model mlp --n-trials 20
```

Expected tail of (6b):

```text
  train R²    = 0.849
  val   R²    = 0.816
  test  R²    = 0.810
  test  RMSE  = 0.499
  model       = 133.3 KB, parameter count n/a
  inference   = 0.04 ms/sample

[infer] round-trip inference on the first 5 rows:
        model  : pytorch.mlp.joblib  (pytorch)
        inputs : ['AveBedrms','AveOccup','AveRooms','HouseAge',
                  'Latitude','Longitude','MedInc','Population']
        y_true : [4.53 3.58 3.52 3.41 3.42]
        y_hat  : [3.99 4.48 4.02 2.62 2.57]
```

## 7. Inspect the artifacts

```bash
ls runs/california_mlp/
#   spec.yaml  env.txt  git_rev.txt  run.log  workflow_summary.json
#   metrics.json  train_data_ranges.json  model_card_pytorch.mlp.json
#   scalers/inputs.joblib
#   models/pytorch.mlp.joblib  models/pytorch.mlp.onnx
#   predictions/{pytorch.mlp_train.parquet,..._val.parquet,..._test.parquet}
#   hpo/                              # populated only when --n-trials > 0

python -c "import json; \
           d = json.load(open('runs/california_mlp/metrics.json')); \
           print(json.dumps(d['pytorch.mlp']['test'], indent=2))"
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
