# Night 1 / Night 2-morning disclosure audit — CLOSED

**Status:** CLOSED as of 2026-04-21 (commits 864d464 / ead0177 / 053389a
on `ai4fusion-dev`, CI green on all three). The disk-quota block that
halted Night 1 was cleared; the subsequent §1.2 + §2.1 passes were
executed in one sweep as part of Night 2 morning.

### Closing metrics (post-sweep)
- Tracked files containing NERSC absolute paths
  (`/global/homes/`, `/pscratch/`, `/global/cfs/`):
  **85 → 6**. All 6 remaining are intentional false-positives:
  - `docs/AUDIT_NIGHT1.md` (this file),
    `docs/OPEN_QUESTIONS.md`,
    `docs/RELEASE_SPRINT.md` — release-plan docs that *discuss*
    the removal;
  - `.github/pull_request_template.md`,
    `CONTRIBUTING.md` — contributor checklists that say
    "no new `/global/homes/...` paths" (anti-pattern warnings);
  - `scripts/m3dc1/surge_slurm_env.sh` — documented
    `SURGE_CONDA_ENV` default override point.
- Tracked files total: **338 → 214** (−124).
- M3D-C1 experiment tree (27 configs, 15 Slurm training scripts, 13
  batch-setup YAMLs, 19 auxiliary `.py` scripts, 5 HPC templates,
  3 notebooks, 9 research figures, 13 internal docs) moved to
  gitignored `docs/internal/`, `scripts/m3dc1/internal/`,
  `configs/internal/`, `examples/internal/`, `templates/internal/`,
  `notebooks/m3dc1/internal/`, `data/internal/`.
- Public-facing dataset API preserved: `XGCDataset`, `M3DC1Dataset`,
  `SurrogateDataset` still import cleanly; 3 kept-public
  `scripts/m3dc1/*.py` (`dataset_complex_v2`, `eigenmode_features`,
  `loader`) are the lazy-import dependencies.
- Two evidence-based internal reference docs drafted from the 27
  historical runs (gitignored, local-only):
  `docs/internal/M3DC1_RECIPES_AND_LESSONS.md`,
  `docs/internal/SURGE_SLURM_BEST_PRACTICES.md`.
- `.gitignore:190` blanket `models/` footgun (the one that caused
  the fresh-clone CI import error in commit 6f33d89) narrowed to
  `runs/**/models/` + `data/**/models/`, with `/surge/models/`
  explicitly ignored to preserve the `surge/models.py` canonical module.

### Remaining for Night 3
- Night 3 T3.1 `git filter-repo` on a fresh clone — strip the
  pre-sweep NERSC paths out of git history entirely (the sweep above
  only protects HEAD going forward).
- Night 3 T3.2 DOE-CODE registration + optional PyPI publish.

---

## Historical audit snapshot (as taken at Night 1 halt)

Everything below this line is the Night 1 scan as captured before
the sweep; kept for provenance.

---

## 1. Working-tree audit — proprietary / internal name tokens

Scan: `rg -i 'xgc|hhfw|c1input|olcf'`, across the entire tracked tree.

### 1.1 Expected / safe hits (no action)
- `docs/RELEASE_SPRINT.md`, `docs/PUBLIC_OPEN_SOURCE_PLAN.md`,
  `docs/REFACTORING_PLAN.md`, `docs/RELEASE_DEMO_PLAN.md` — release
  plans that *discuss* the removal. Intentional.
- `data/datasets/HHFW-NSTX/README.md` — disclaimer already added stating
  the .pkl binaries are not part of the release.
- `.gitignore` — HHFW pkls explicitly ignored. Intentional.
- `scripts/README.md` — documents removed XGC scripts. Intentional.
- `surge/datasets/xgc.py` — the generic stacked-NPY loader (already
  renamed from `_load_olcf_npy` to `_load_stacked_npy`, with
  back-compat alias). Docstrings generic. Intentional.
- `surge/dataset.py` — calls into `surge.datasets.xgc._load_stacked_npy`
  for the "xgc" format string. This is a **format identifier** not
  proprietary content. Consider renaming the format to `stacked_npy`
  in v0.2.0; leave as-is for v0.1.0.

### 1.2 Needs a closer pass (Night 1 T1.1 continuation)
Each of these has at least one line containing one of the tokens. Most
are probably incidental comments, but each needs a 10-second look:

- `surge/__init__.py` — already flagged separately (T1.2 `from loader`).
- `surge/cli.py`
- `surge/datasets/__init__.py`
- `surge/integrations/mlflow_logger.py`
- `surge/utils.py`
- `surge/verify_batch.py`
- `surge/datagen/generator.py`
- `surge/viz/run_viz.py`, `surge/viz/importance.py`
- `tests/test_datagen.py`
- `docs/dev/FNO_INTEGRATION_GUIDE.md`
- `docs/dev/M3DC1_COMMIT_PLAN.md`
- `docs/ROSE_SURGE_INTEGRATION.md`
- `docs/m3dc1/BATCH_DIRS_AND_DATA.md`
- `docs/DATAGEN_USER_MANUAL.md`, `docs/BATCH_GENERATION_WORKFLOW.md`,
  `docs/setup/DATA_ACCESS.md`
- `scripts/m3dc1/README.md`, `scripts/m3dc1/write.py`,
  `scripts/m3dc1/fix_inputs.py`, `scripts/m3dc1/collect_from_batch.py`,
  `scripts/m3dc1/collect_data.py`
- `notebooks/m3dc1/delta_p_spectra_demo.ipynb`,
  `notebooks/m3dc1/data_analysis.ipynb`
- `examples/unified_surrogate_engine_example.py`,
  `examples/pwe_minimal_demo.py`, `examples/datagen_demo.py`,
  plus all `examples/batch_setup*.yml` files
- `templates/singlebatchjob.perlmutter`

### 1.3 Action plan when quota is resolved
1. Read each file in §1.2, classify as:
   - (a) mere discussion → no change,
   - (b) hardcoded path/name → strip, or
   - (c) proprietary logic → delete from working tree, add to
     filter-repo target list.
2. Commit as `docs(sweep): remove residual XGC/HHFW references (non-code)`
   and `chore(strip): remove hardcoded paths and internal names`.

---

## 2. Working-tree audit — absolute NERSC paths

Scan: `rg '/global/homes|/pscratch|/global/cfs'`. Hits in ~80 tracked
files; the big concentrations are:

- `configs/*.yaml` — several hardcoded `/global/homes/...` and
  `/pscratch/...` paths. Cleanup: change to `${SURGE_DATA_ROOT}`-style
  placeholders and document the variable.
- `scripts/m3dc1/*.slurm`, `scripts/m3dc1/surge_slurm_env.sh` — same
  hardcoded paths. Keep the scripts (they're useful examples of a SLURM
  setup), but templatize the paths.
- `examples/batch_setup_*.yml` — same. Replace with `${DATA_ROOT}`.
- `scripts/m3dc1/build_*.py`, `scripts/m3dc1/eval_*.py`, a handful of
  plotting scripts in `scripts/m3dc1/interfaces/*.py` — check each for
  hardcoded paths; these are more likely to have genuine leaks than the
  configs.
- `docs/m3dc1/*.md`, `docs/agents/*.md`, `docs/ROSE_SURGE_INTEGRATION.md`
  — paths in prose. Replace with `/path/to/your/data` style.
- `m3dc1/results.md`, `H5_FILES_ANALYSIS.md` (top-level) — candidates
  for deletion; they look like personal notes that don't belong in the
  public tree.
- `templates/*.perlmutter` — hardcoded paths in job templates. Same
  templatize-and-document treatment as the SLURM scripts.

### 2.1 Action plan when quota is resolved
1. Introduce two environment-variable conventions and document them:
   - `SURGE_DATA_ROOT` — where input datasets live.
   - `SURGE_RUN_ROOT`  — where artifacts / runs are written.
2. Mass-substitute hardcoded paths in `configs/`, `examples/`, SLURM
   scripts, and templates.
3. Delete `m3dc1/results.md` and `H5_FILES_ANALYSIS.md` from the working
   tree (if you want to keep them, move to `docs/internal/` and add to
   `.gitignore` pattern — but simpler to just delete).
4. Commit as `chore(strip): replace hardcoded NERSC paths with
   environment variables`.

---

## 3. Git-history audit — sensitive files in past commits

Scan: `git log --all --oneline --name-only -- <sensitive paths>`.

### 3.1 Confirmed sensitive items **in history**
| Commit   | Path                                          |
| -------- | --------------------------------------------- |
| 3882a96  | `docs/xgc/XGC_CAPABILITIES.md`                |
| 3882a96  | `docs/xgc/XGC_SUMMARY.md`                     |
| 3882a96  | `scripts/xgc_datastreamset_comparison.py`     |
| 3882a96  | `scripts/xgc_datastreamset_generalization.py` |
| 3882a96  | `scripts/xgc_export_onnx.py`                  |
| 3882a96  | `scripts/xgc_inference.py`                    |
| 3882a96  | `scripts/xgc_predictions_animation.py`        |
| 3882a96  | `scripts/xgc_spatial_aparallel_animation.py`  |
| 3882a96  | `scripts/xgc_workflow_orchestrator.py`        |
| 3882a96  | `xgc_cpp/CMakeLists.txt`                      |
| 3882a96  | `xgc_cpp/README.md`                           |
| 3882a96  | `xgc_cpp/main_test.cpp`                       |
| 3882a96  | `xgc_cpp/xgc_surrogate_infer.cpp`             |
| 3882a96  | `xgc_cpp/xgc_surrogate_infer.h`               |
| 724560f  | `docs/xgc/COMMITS_TO_PUSH.md`                 |
| 724560f  | `docs/xgc/XGC_SUMMARY.md`                     |
| 705d9a0  | `scripts/xgc_export_onnx.py`                  |
| e6359fc  | `scripts/xgc_workflow_orchestrator.py`        |
| cc658ac  | `docs/xgc/XGC_TRAINING.md`                    |
| 2b80054  | `scripts/xgc_datastreamset_generalization.py` |
| e8d8f72  | `docs/xgc/CHANGES_XGC_VIZ_2025-03.md`         |
| e8d8f72  | `docs/xgc/XGC_APARALLEL_DATA.md`              |
| e8d8f72  | `docs/xgc/XGC_FEATURES_2025-03.md`            |
| e8d8f72  | `docs/xgc/XGC_INPUT_STRUCTURE.md`             |
| 0636f9b  | `scripts/xgc_datastreamset_generalization.py` |
| 3d5f8b7  | `docs/xgc/DRIFT_DETECTION_POLICY.md`          |
| 54feca0  | `docs/xgc/AI_OCLF_ARCHITECTURE.md`            |
| 54feca0  | `docs/xgc/XGC_APARALLEL_DATA.md`              |
| 54feca0  | `docs/xgc/XGC_TRAINING.md`                    |
| 566ac73  | `docs/xgc/XGC_APARALLEL_DATA.md`              |
| 5c3a0e4  | `data/datasets/HHFW-NSTX/PwE_.pkl`            |
| 5c3a0e4  | `data/datasets/HHFW-NSTX/PwIF_.pkl`           |
| f8e43b9  | `scripts/ml_utils.py`                         |

**Conclusion:** history rewrite via `git filter-repo` is mandatory before
any public push. Q5 in `OPEN_QUESTIONS.md` is therefore auto-resolved.

### 3.2 Scope to strip from history (paths for `git filter-repo --invert-paths`)
```
docs/xgc/
xgc_cpp/
scripts/xgc_*
scripts/ml_utils.py
data/datasets/HHFW-NSTX/PwE_.pkl
data/datasets/HHFW-NSTX/PwIF_.pkl
```
The exact command is already written in `docs/PUBLIC_OPEN_SOURCE_PLAN.md`;
Night 3 runs it against a fresh clone.

### 3.3 What I did **not** scan yet
- Commit **messages** for the same tokens (`git log --all --grep='XGC'`).
  Messages also need rewrite / `--replace-message` if they name things.
- Author/committer email strings across history.
- Commit-message-only mentions of collaborators/projects.
- Large-blob diff content beyond the path list (e.g. binary files
  embedded in a commit under a non-obvious path).

These are Night-3 preparation items, not Night-1 blockers.

---

## 4. Next steps (order)

1. **Unblock disk quota** (`OPEN_QUESTIONS.md` top entry).
2. Resume T1.1 by finishing §1.2 classification and §2.1 path cleanup.
3. Proceed to T1.2 (`from loader` fix).
4. Continue the sprint.
