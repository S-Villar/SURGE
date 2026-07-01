# M3DC1 δ*p* per-mode spectra surrogate — end-to-end workflow

This note documents how the **per-poloidal-mode δ*p* spectrum** surrogate is built, trained, checkpointed, evaluated on the test split, and plotted after **spectral reconstruction** back to a poloidal (R,Z) field. It matches the pipeline that produced figures such as:

`runs/m3dc1_trial12_e500_bs128_withval_0410_085853/plots/delta_p_2d_test/torch_mlp_profiles_deltap/best_case_rz_compare_run11_sparc_1319.png`

All **M3DC1-specific** automation lives under **`scripts/m3dc1/internal/`** (not `scripts/m3dc1/` at repo root). SURGE core training is **`surge.workflow.run.run_surrogate_workflow`**, invoked via **`python -m surge.cli run`**.

---

## 1. What is being modeled?

- **Inputs (per row):** equilibrium scalars, toroidal mode number `n`, poloidal mode `m`, and related controls — see `PER_MODE_INPUT_COLS` and `load_single_complex_v2_per_mode()` in **`scripts/m3dc1/dataset_complex_v2.py`** (imported by the builder below).
- **Outputs:** a fixed-length profile of **|δ*p*| vs ψ_N** for that `(case, n, m)`, stored as columns `output_p_0` … `output_p_{K-1}` (native resolution in the CFS build was 200 ψ points × 200 m modes per case in the aggregate dataset).
- **Registry:** the trained network is a **`pytorch.mlp`** adapter (`PyTorchMLPAdapter` in **`surge/model/pytorch.py`**; registered in **`surge/model/__init__.py`** with aliases `torch_mlp`, `torch.mlp`). The workflow YAML usually sets a human-readable **`name`** (e.g. `torch_mlp_profiles_deltap`) used for artifacts and for **`--model`** in eval scripts; the **`key`** must remain a registered key such as `pytorch.mlp`.

---

## 2. Build the Parquet dataset from M3D-C1 batch trees

**Script:** **`scripts/m3dc1/internal/build_delta_p_per_mode.py`**

**Role:** Streams HDF5 cases to a single Parquet file: one row per `(run, n, m)` with magnitude spectra from `sdata_complex_v2.h5` (or another leaf name via `--filename`).

**Important functions (via imports):**

- **`dataset_complex_v2.find_complex_v2_files`** — discover HDF5 paths under a batch directory.
- **`dataset_complex_v2.load_single_complex_v2_per_mode`** — extract rows for one case (`spectrum_field="p"`, `use_magnitude=True`, etc.).
- **`_write_parquet_chunked`** in the builder — chunked Parquet writer to avoid holding ~2M rows in RAM.

**Example shape (adjust paths):**

```bash
python scripts/m3dc1/internal/build_delta_p_per_mode.py /path/to/m3dc1/batch \
  --out data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet \
  --filename sdata_complex_v2.h5
```

**Metadata:** save or version a sidecar YAML (e.g. `delta_p_per_mode_metadata.yaml`) listing `input_columns` / `output_columns` for **`SurrogateWorkflowSpec`**; **`surge.cli analyze`** can help validate column sets.

---

## 3. Train with SURGE (workflow spec → checkpoints → best weights)

### 3.1 CLI entry

**Command:** `python -m surge.cli run <spec.yaml> [--run-tag …] [--output-dir …]`

**Implementation:** **`surge/cli.py`** → **`_run()`** loads YAML via **`SurrogateWorkflowSpec.from_dict`**, then calls **`run_surrogate_workflow`** (**`surge/workflow/run.py`**).

### 3.2 Training loop and artifacts

**Orchestration:** **`run_surrogate_workflow`** in **`surge/workflow/run.py`**

- Builds the dataset and **`SurrogateEngine`**, then for each model config calls **`engine.train_model`** (**`surge/engine.py`**, method **`train_model`**).
- **Checkpoint directory:** if the model’s backend is `pytorch` and **`checkpoint_every_n_epochs`** > 0 in the model **`params`**, the workflow sets **`os.environ["SURGE_CHECKPOINT_DIR"]`** to **`<run_dir>/checkpoints/<artifact_tag>`** for the duration of **`train_model`** (see the block around the `SURGE_CHECKPOINT_DIR` assignment in **`run_surrogate_workflow`**). **`artifact_tag`** is derived from the workflow model **`name`** (or **`key`**) via **`_safe_model_artifact_tag`**.
- **Persistence:** **`_persist_model_artifacts`** writes the saved **`models/<name>.joblib`** (wrapper), predictions under **`predictions/`**, optional **`training_history_<tag>.json`**, and records **`checkpoints_dir`** in **`workflow_summary.json`** when `epoch_*.pt` files exist.

### 3.3 PyTorch MLP: validation-based best weights and `epoch_*.pt`

**Implementation:** **`surge/model/pytorch_impl.py`** — class **`PyTorchMLPModel`** (and **`PyTorchMLP`** used inside it).

- **`restore_best_weights`** (default **true**): after training, weights are rolled back to the epoch with the **best validation loss** (early stopping / patience interact with this).
- **`checkpoint_every_n_epochs`**: when > 0 and **`SURGE_CHECKPOINT_DIR`** is set, periodic **`epoch_<n>.pt`** files are written; each checkpoint can include **`model_state_dict`**, **`input_size`**, **`output_size`**, and embed **`scaler_X` / `scaler_y`** when present.

**Re-implementing a historical run:** copy the **model `params`** from the archived **`spec.yaml`** / **`inputs/*.yaml`** under **`runs/<tag>/`**, and keep **`checkpoint_every_n_epochs`** and **`restore_best_weights`** aligned with that run so the **saved joblib** and **metrics** match expectations.

---

## 4. Test-split 2D maps (m × ψ_N), metrics, and “best case”

**Script:** **`scripts/m3dc1/internal/eval_test_delta_p_2d_maps.py`** — **`main()`**

**Split / scalers:** the script **imports the module** **`eval_per_case_delta_p_recon`** via **`importlib`** from **`scripts/m3dc1/internal/eval_per_case_delta_p_recon.py`** and reuses **`_load_spec`**, **`_raw_splits`**, **`_group_cols`**, **`_load_saved_scalers`**, etc., so **test rows match the original SURGE run** (same seed and fractions as in **`spec.yaml`**).

**Model loading (`main()`):**

1. **Default:** load the path recorded in **`workflow_summary.json`** for that **`--model`** name.
2. **Explicit checkpoint:** **`--checkpoint path/to/epoch_0123.pt`** (or a `.joblib`). For **`.pt`** files, **`_load_epoch_checkpoint_adapter`** rebuilds a **`surge.model.pytorch_impl.PyTorchMLP`**, infers **`hidden_layers`** from the state dict if needed, loads **`model_state_dict`**, and attaches scalers from the checkpoint when stored there.

**Per-case evaluation:** for each **test group** (same physical inputs except **`m`**), the script stacks predictions into **`Y_true`**, **`Y_pred`** with shape **`(n_modes, n_psi)`**, computes map-level metrics, and optionally renders heatmaps (`--plot-mode`, `--m-source case|range`, etc.).

**Best case:** with **`--save-best-case-data`**, writes **`best_case_data.pkl`** plus **`best_case_y_true.npy`**, **`best_case_y_pred.npy`**, **`best_case_m.npy`**, **`best_case_psi.npy`** under the plot root. **`summary_2d_maps.json`** records **`best_case`** (e.g. **`group_id`**, metric used: **`--best-case-metric`** `rmse` vs **`nrmse`**).

**Typical layout:**

`runs/<tag>/plots/delta_p_2d_test/<model_name>/`

**Example invocation pattern:**

```bash
python scripts/m3dc1/internal/eval_test_delta_p_2d_maps.py runs/m3dc1_trial12_e500_bs128_withval_0410_085853 \
  --model torch_mlp_profiles_deltap \
  --checkpoint runs/m3dc1_trial12_e500_bs128_withval_0410_085853/checkpoints/torch_mlp_profiles_deltap/epoch_XXXX.pt \
  --m-source case \
  --plot-mode quantiles \
  --save-best-case-data \
  --best-case-metric nrmse
```

(Replace **`epoch_XXXX.pt`** with the epoch you want to evaluate; this is how you **tie plots to a specific checkpoint** even if **`restore_best_weights`** already aligned the final joblib with the best val epoch.)

**Related (pooled reconstruction metrics):** **`scripts/m3dc1/internal/eval_per_case_delta_p_recon.py`** — **`main()`** — same split/scalers, but focuses on **combining modes** (`sum` / `rss`) into a single ψ profile per case and writing **`per_case_recon_metrics.csv`**.

---

## 5. R–Z plot: reconstructed field from predicted spectra

**Script:** **`scripts/m3dc1/internal/plot_best_case_rz_compare.py`** — **`main()`**

**Inputs:**

- **`--best-case-pkl`:** **`best_case_data.pkl`** from step 4.
- **`--sdata`:** `sdata_complex_v2.h5` (or the run’s HDF5) for **`parset` / `time_index`** metadata.
- **`--c1`:** **`C1.h5`** for **`fpy.sim_data`** and **`m3dc1.flux_coordinates`**.
- **`--run`:** HDF5 group such as **`run_0001`** (must match the case you want on the figure).
- **`--m3dc1-python-code`:** directory containing the **`m3dc1`** package and **`fpy`** bindings.

**Reconstruction logic (in `main()`):**

- **`_to_full_m_grid`** — embeds non-contiguous **`m`** rows into a dense integer **`m`** grid.
- **`_recon_zero_phase`** — **`np.fft.ifft`** along **`m`** (zero phase fallback) to get a real-space poloidal slice from the spectrum.
- Ground truth on (R,Z): **`m3dc1.eval_field.eval_field`** on a flux coordinate path from **`flux_coordinates`**, comparing equilibrium vs linear time slice (see script for **`phi0` / `phiq`** handling).

**Outputs:** PNG paths from **`--out`** and optional **`--out-shared-recon`** (shared colorbar for pred vs true-spectrum recon).

---

## 6. Slurm / environment (CFS)

**Reference:** **`scripts/m3dc1/surge_slurm_env.sh`** — conda, **`pyarrow`**, optional path exports for CFS.

Internal batch templates for δ*p* per-mode training live next to the builders, e.g. **`scripts/m3dc1/internal/train_delta_p_per_mode_cfs*.slurm`**.

---

## 7. Quick reference table

| Step | Script / module | Primary symbols |
|------|-------------------|-----------------|
| Build Parquet | `scripts/m3dc1/internal/build_delta_p_per_mode.py` | `_write_parquet_chunked`, `dataset_complex_v2.*` |
| Train | `python -m surge.cli run` | `surge/cli.py:_run`, `surge/workflow/run.py:run_surrogate_workflow`, `surge/engine.py:train_model` |
| Checkpoints / best val | `surge/workflow/run.py` + `surge/model/pytorch_impl.py` | `SURGE_CHECKPOINT_DIR`, `PyTorchMLPModel.fit`, `restore_best_weights` |
| 2D test maps + best case | `scripts/m3dc1/internal/eval_test_delta_p_2d_maps.py` | `main`, `_load_epoch_checkpoint_adapter` |
| Split helpers | `scripts/m3dc1/internal/eval_per_case_delta_p_recon.py` | `_raw_splits`, `_load_saved_scalers`, … |
| RZ figure | `scripts/m3dc1/internal/plot_best_case_rz_compare.py` | `main`, `_recon_zero_phase`, `_to_full_m_grid` |

---

## 8. Re-running on the latest SURGE

1. Use the **same dataset and YAML** semantics (`dataset_path`, `metadata_path`, `models[].key` / `name`, `params`, `seed`, `val_fraction`, `test_fraction`).
2. Confirm **`pytorch.mlp`** **`params`** still map to **`PyTorchMLPModel`** kwargs in **`surge/model/pytorch_impl.py`** (field names for patience, checkpoints, `restore_best_weights`).
3. Re-run **`eval_test_delta_p_2d_maps.py`** with **`--checkpoint`** if you need parity with a **specific** `epoch_*.pt`.
4. Regenerate RZ plots with **`plot_best_case_rz_compare.py`** once **`best_case_data.pkl`** exists.

For broader result context and caveats on per-mode δ*p* difficulty, see **`docs/internal/M3DC1_RECIPES_AND_LESSONS.md`** (internal memo).
