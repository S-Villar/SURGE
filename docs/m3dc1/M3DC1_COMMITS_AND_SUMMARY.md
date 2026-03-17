# M3DC1 Delta p Spectra Development – Commits and Summary

## Commits to Make (M3DC1-focused)

| # | Commit message | Files |
|---|----------------|-------|
| 1 | **M3DC1: batch dirs, dataset extraction, eigenmode integration** | `docs/m3dc1/BATCH_DIRS_AND_DATA.md`, `scripts/m3dc1/dataset_complex_v2.py`, `scripts/m3dc1/eigenmode_features.py`, `scripts/m3dc1/build_delta_p_dataset.py`, `surge/datasets/m3dc1.py` |
| 2 | **M3DC1: workflow from batch dir, profile training** | `surge/workflow/spec.py`, `surge/workflow/run.py`, `configs/m3dc1_delta_p_batch16.yaml`, `configs/m3dc1_delta_p_batch16_from_dir.yaml`, `configs/m3dc1_delta_p_profile_mode0.yaml`, `scripts/m3dc1/build_profile_dataset.py`, `scripts/m3dc1/debug_dataloader.py` |
| 3 | **M3DC1: fix eq_id in training inputs, torch.mlp alias** | `scripts/m3dc1/dataset_complex_v2.py`, `surge/datasets/m3dc1.py`, `surge/model/__init__.py` |
| 4 | **M3DC1: training docs and metadata** | `docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md`, `data/datasets/SPARC/delta_p_spectra_metadata.yaml`, `docs/dev/dev-plan.md` |

## What Was Done

### 1. Extraction
- `build_delta_p_dataset.py` – CLI to build and save delta p DataFrame from batch dir
- Extracted: `data/datasets/SPARC/delta_p_batch16.pkl` (30 rows, 2514 cols, 2500 spectrum outputs)
- Profile: `data/datasets/SPARC/delta_p_profile_mode0.pkl` (mode 0, 50 psi points)

### 2. Dataloader
- `iter_batches(batch_size=32)` – 1 batch per epoch with 29 samples
- `to_dataloader(batch_size=32)` – PyTorch DataLoader with correct shapes
- `debug_dataloader.py` – script to inspect batch flow

### 3. Workflow
- `dataset_source: m3dc1_batch` – load directly from batch dir
- `python -m surge.cli run configs/m3dc1_delta_p_batch16_from_dir.yaml`
- PyTorch MLP uses DataLoader with `batch_size=32`

### 4. Profile Training (delta_p vs psiN for specific n,m)
- Spectrum layout: `output_p_0..49` = mode 0 profile (50 psi points), `output_p_50..99` = mode 1, etc.
- `build_profile_dataset.py` – build dataset for a single mode’s profile
- Config: `configs/m3dc1_delta_p_profile_mode0.yaml`, run tag: `m3dc1_delta_p_profile_mode0`

## Run Results

| Run | Models | Notes |
|-----|--------|------|
| `runs/m3dc1_delta_p_batch16` | rf_delta_p, mlp_delta_p | Full spectrum (2500 outputs), 30 samples, R² poor (expected) |
| `runs/m3dc1_delta_p_profile_mode0` | rf_profile_mode0, mlp_profile_mode0 | Mode 0 profile (50 outputs). Train R² ~0.4 (RF), test R² negative (overfitting, n=30) |

## Profile Surrogate Concept

For a given (n, m) mode, the surrogate predicts **amplitude vs psiN**: the 1D profile of δp at each flux surface. With `mode_step=4`, `psi_step=4` we get 50×50. Mode 0 is `output_p_0..49` = δp(ψ₀)..δp(ψ₄₉).

```bash
# Build profile for mode 0
python scripts/m3dc1/build_profile_dataset.py /pscratch/.../batch_16 --mode 0 --out data/datasets/SPARC/delta_p_profile_mode0.pkl

# Train
python -m surge.cli run configs/m3dc1_delta_p_profile_mode0.yaml
```
