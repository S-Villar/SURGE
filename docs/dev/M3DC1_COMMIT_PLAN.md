# M3DC1 Delta p Per-Mode – Commit Plan

Run from project root. **Excludes XGC codevelopment files** (scripts/xgc_datastreamset_generalization.py, configs/xgc_*, surge/viz datastreamset_eval_set, engine pretrained/finetune).

---

## Commit 1: M3DC1 delta p per-mode dataset and workflow

**Message:** `feat(m3dc1): delta p per-mode format, from_batch_dir_per_mode, m3dc1_batch_per_mode`

```bash
git add scripts/m3dc1/dataset_complex_v2.py
git add scripts/m3dc1/build_delta_p_per_mode.py
git add surge/datasets/m3dc1.py
git add surge/workflow/run.py
git add surge/workflow/spec.py
git add data/datasets/SPARC/delta_p_per_mode_metadata.yaml
git add configs/m3dc1_delta_p_per_mode.yaml
git add configs/m3dc1_delta_p_per_mode_from_batch.yaml
git commit -m "feat(m3dc1): delta p per-mode format, from_batch_dir_per_mode, m3dc1_batch_per_mode

- Per-mode: one row per (case, m), 12 inputs → 200 profile outputs
- build_delta_p_per_mode.py: CLI to build from batch dir
- M3DC1Dataset.from_batch_dir_per_mode()
- dataset_source: m3dc1_batch_per_mode in workflow
- Configs: per-mode from file and from batch dir"
```

---

## Commit 2: surge analyze CLI

**Message:** `feat(cli): add surge analyze command for dataset inspection`

```bash
git add surge/cli.py
git commit -m "feat(cli): add surge analyze command for dataset inspection

- surge analyze <spec|dataset.pkl>: show inputs, outputs, stats
- Resolve dataset path from spec or use path directly
- Add __main__ block for python -m surge.cli"
```

---

## Commit 3: PyTorch MLP early stopping with patience

**Message:** `feat(torch): MLP early stopping with configurable patience`

```bash
git add surge/model/pytorch_impl.py
git add surge/model/pytorch.py
git add surge/engine.py
git commit -m "feat(torch): MLP early stopping with configurable patience

- PyTorchMLP.fit() accepts X_val, y_val, patience
- Stop when val loss does not improve for patience epochs
- Engine passes X_val, y_val to torch adapters
- Params: patience, max_epochs (or epochs)
- Engine also adds pretrained_scalers and finetune for fine-tuning workflows"
```

**Note:** engine.py includes pretrained_scalers/finetune (XGC fine-tuning). If you want M3DC1-only, use `git add -p surge/engine.py` to stage only the X_val/y_val changes.

---

## Commit 4: Dataset summary fix

**Message:** `fix(dataset): summary returns n_rows when analysis is None`

```bash
git add surge/dataset.py
git commit -m "fix(dataset): summary returns n_rows when analysis is None

- SurrogateDataset.summary() always returns n_rows, n_inputs, n_outputs
- Fixes workflow summary for datasets loaded via load_from_dataframe"
```

---

## Commit 5: M3DC1 HPO configs and SLURM scripts

**Message:** `feat(m3dc1): HPO configs and Perlmutter SLURM scripts`

```bash
git add configs/m3dc1_delta_p_per_mode_hpo.yaml
git add configs/m3dc1_delta_p_per_mode_hpo_extended.yaml
git add scripts/m3dc1/train_delta_p_per_mode.slurm
git add scripts/m3dc1/train_delta_p_per_mode_hpo.slurm
git add scripts/m3dc1/train_delta_p_per_mode_hpo_extended.slurm
git add scripts/m3dc1/train_delta_p_per_mode_hpo_patience_debug.slurm
git commit -m "feat(m3dc1): HPO configs and Perlmutter SLURM scripts

- HPO config: 15 RF + 30 MLP trials, patience=20
- Extended HPO: 15 RF + 250 MLP trials
- SLURM: debug (30 min), regular (2 hr), extended (4 hr)"
```

---

## Commit 6: M3DC1 workflow mode tests

**Message:** `test(m3dc1): add workflow mode tests (from file, from batch, per-mode)`

```bash
git add tests/test_m3dc1_workflow_modes.py
git commit -m "test(m3dc1): add workflow mode tests (from file, from batch, per-mode)

- Mode A: from pre-built .pkl
- Mode B: dataset_source=m3dc1_batch (full spectrum)
- Mode C: dataset_source=m3dc1_batch_per_mode
- Fixture: fake complex_v2 HDF5 for CI"
```

---

## Commit 7: M3DC1 documentation

**Message:** `docs(m3dc1): per-mode training guide and README updates`

```bash
git add docs/m3dc1/DELTA_P_PER_MODE_GUIDE.md
git add docs/m3dc1/BATCH_DIRS_AND_DATA.md
git add docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md
git add docs/m3dc1/M3DC1_COMMITS_AND_SUMMARY.md
git add scripts/m3dc1/README.md
git add docs/dev/M3DC1_COMMIT_PLAN.md
git commit -m "docs(m3dc1): per-mode training guide and README updates

- DELTA_P_PER_MODE_GUIDE.md: quick start, configs, SLURM, patience
- BATCH_DIRS_AND_DATA.md: batch locations, per-mode usage
- scripts/m3dc1/README.md: training from scratch, SLURM"
```

---

## Excluded (XGC codevelopment)

- `scripts/xgc_datastreamset_generalization.py` (--eval-set)
- `configs/xgc_aparallel_set2_*.yaml`
- `surge/viz/run_viz.py` (datastreamset_eval_set)
- `surge/engine.py` pretrained_scalers, finetune (if splitting from X_val/y_val)
- `surge/io/*`, `surge/hpc/*`, `pyproject.toml`, `docs/dev/dev-plan.md`

---

## Verify before push

```bash
git status
git log --oneline -10
# Ensure no unintended files; then:
# git push origin ai4fusion-dev
```
