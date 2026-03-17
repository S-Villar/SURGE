# SURGE Commit Plan

Proposed logical commit sequence for current changes. Run from project root.

> **Note:** This file (`docs/dev/COMMIT_PLAN.md`) can be committed with the first docs commit or omitted.

---

## Commit 1: XGC documentation

**Message:** `docs(xgc): add AI_OCLF architecture, data mapping, and training guide`

```bash
git add docs/xgc/AI_OCLF_ARCHITECTURE.md
git add docs/xgc/XGC_APARALLEL_DATA.md
git add docs/xgc/XGC_TRAINING.md
git commit -m "docs(xgc): add AI_OCLF architecture, data mapping, and training guide"
```

---

## Commit 2: XGC config and dataset format support

**Message:** `feat(xgc): add XGC A_parallel config and SurrogateDataset format support`

```bash
git add configs/xgc_aparallel_set1.yaml
git add surge/dataset.py
git add tests/test_datasets_xgc.py
git commit -m "feat(xgc): add XGC A_parallel config and SurrogateDataset format support"
```

---

## Commit 3: Preprocessing input/output grouping fix

**Message:** `fix(preprocessing): correct input/output column grouping`

```bash
git add surge/preprocessing.py
git commit -m "fix(preprocessing): correct input/output column grouping"
```

---

## Commit 4: CLI and entry point

**Message:** `feat(cli): add surge run CLI and pyproject entry point`

```bash
git add surge/cli.py
git add setup.py
git add pyproject.toml
git commit -m "feat(cli): add surge run CLI and pyproject entry point"
```

---

## Commit 5: Engine validation and early stopping

**Message:** `feat(engine): pass validation data to torch adapters, early stopping support`

```bash
git add surge/engine.py
git commit -m "feat(engine): pass validation data to torch adapters, early stopping support"
```

---

## Commit 6: XGC run comparison plotting script

**Message:** `feat(scripts): add XGC run comparison plotting script`

```bash
git add scripts/xgc/plot_xgc_run_comparison.py
git commit -m "feat(scripts): add XGC run comparison plotting script"
```

---

## Commit 7: GPflow availability check script

**Message:** `feat(scripts): add GPflow availability check script`

```bash
git add scripts/check_gpflow.py
git commit -m "feat(scripts): add GPflow availability check script"
```

---

## Commit 8: Registry and workflow spec fixes

**Message:** `fix(registry): remove slots=True for Python 3.8/3.9 compatibility`

```bash
git add surge/registry.py
git add surge/workflow/spec.py
git commit -m "fix(registry): remove slots=True for Python 3.8/3.9 compatibility"
```

---

## Commit 9: M3DC1 / misc updates (optional, separate branch)

These changes are M3DC1/delta_p or other work; consider separate commits or branch:

- `data/datasets/SPARC/delta_p_spectra_metadata.yaml`
- `docs/dev/dev-plan.md`
- `docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md`
- `docs/m3dc1/BATCH_DIRS_AND_DATA.md`
- `configs/m3dc1_delta_p_batch16.yaml`
- `scripts/m3dc1/README.md`
- `scripts/m3dc1/dataset_complex_v2.py`
- `scripts/m3dc1/build_delta_p_dataset.py`
- `scripts/m3dc1/eigenmode_features.py`
- `surge/datasets/m3dc1.py`
- `surge/hpc/resources.py`
- `surge/io/artifacts.py`

---

## Quick apply (XGC-focused commits 1–8)

```bash
# From project root
git add docs/xgc/AI_OCLF_ARCHITECTURE.md docs/xgc/XGC_APARALLEL_DATA.md docs/xgc/XGC_TRAINING.md
git commit -m "docs(xgc): add AI_OCLF architecture, data mapping, and training guide"

git add configs/xgc_aparallel_set1.yaml surge/dataset.py tests/test_datasets_xgc.py
git commit -m "feat(xgc): add XGC A_parallel config and SurrogateDataset format support"

git add surge/preprocessing.py
git commit -m "fix(preprocessing): correct input/output column grouping"

git add surge/cli.py setup.py pyproject.toml
git commit -m "feat(cli): add surge run CLI and pyproject entry point"

git add surge/engine.py
git commit -m "feat(engine): pass validation data to torch adapters, early stopping support"

git add scripts/xgc/plot_xgc_run_comparison.py
git commit -m "feat(scripts): add XGC run comparison plotting script"

git add scripts/check_gpflow.py
git commit -m "feat(scripts): add GPflow availability check script"

git add surge/registry.py surge/workflow/spec.py
git commit -m "fix(registry): remove slots=True for Python 3.8/3.9 compatibility"
```
