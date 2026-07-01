# M3DC1 surrogate — session status

Snapshot of the M3DC1 δp/γ surrogate work so it is not lost when moving to the
`model-bench` model zoo. Training run artifacts live under `runs/` (gitignored)
and are backed up at `/pscratch/sd/a/asvillar/SURGE_runs_backup/`.

## Dataset (verification postprocess)

- Leaf file: `csdata_deltap_b_ver.h5` (201×201 grid, m ∈ [-100, 100],
  `--grid-res 201 --points 201 --full-fft`, fields `p B` + spectra `p br bz bphi`).
- Location: `/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run<N>/sparc_<ID>/`.
- Completion: **9976 / 9999** cases across run1–run99 (~99.77%).
- Known failures (2 equilibria):
  - `sparc_1400` — fails in 21 runs (flux-coordinate θ interpolation edge case at
    `--points 201`); succeeds in 78 runs.
  - `sparc_1593` — fails in run13, run15 (`get_field` NULL for `bphi` at t=1).

## New capability added this session (SURGE core)

Opt-in **case-grouped train/val/test split** so per-mode rows from the same
physical `(run_id, eq_id)` case never leak across splits; the exact partition is
persisted to `splits.json` for reuse by every model/eval task.

- `surge/engine.py` — `group_columns` in `EngineRunConfig` + `_build_grouped_splits()`.
- `surge/workflow/spec.py` — `group_columns` spec field.
- `surge/workflow/run.py` — wiring + `_save_splits_manifest()` (writes `splits.json`
  with per-split row indices, unique case groups, and a leakage check).

## Data builder (M3DC1 extraction kept in a script)

`scripts/m3dc1/internal/build_case_scalar_dataset.py` — reads the csdata files and
emits a plain per-case Parquet + metadata YAML (SURGE trains on the table with no
M3DC1 import at train time).

- Output: `data/datasets/SPARC/case_scalars_ver.parquet` (9976 rows, 11 scalar
  inputs, target `gamma`).

## Runs completed

### γ (growth rate) multi-model benchmark — `runs/m3dc1_gamma_ver_multimodel/`

Config: `configs/internal/m3dc1_gamma_ver_multimodel.yaml`.
One shared case-grouped split: train 5985 / val 1995 / test 1996 cases,
leakage overlaps all **0**.

| model | train R² | val R² | test R² | test RMSE |
|-------|---------:|-------:|--------:|----------:|
| RandomForest | 0.964 | 0.881 | 0.852 | 0.01737 |
| sklearn MLP  | 0.890 | 0.844 | 0.822 | 0.01905 |
| PyTorch MLP  | 0.937 | 0.865 | 0.860 | 0.01690 |

## Ready to run (not yet launched)

Flagship δp per-mode benchmark (~2M rows, GPU SLURM job):

- Config: `configs/internal/m3dc1_deltap_per_mode_ver_multimodel.yaml`
- Launcher: `scripts/m3dc1/internal/train_deltap_per_mode_ver_multimodel.slurm`

## Next steps

1. Port the `group_columns` / `splits.json` feature onto a branch off `model-bench`.
2. Learn **Re(δp)** per mode first (then Im), with profile inputs q(ψ), p(ψ) + shaping.
3. Reconstruct R–Z field from predicted modes
   (`scripts/m3dc1/internal/plot_best_case_rz_compare.py`, IFFT).
4. Extract `m3dc1ml/` into its own repo (github.com/S-Villar/m3dc1ml).
