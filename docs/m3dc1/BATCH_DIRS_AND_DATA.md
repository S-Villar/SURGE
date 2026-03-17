# M3DC1 Batch Directories and sdata_pertfields_grid_complex_v2 Availability

This document records where M3DC1 batch data lives and which locations have `sdata_pertfields_grid_complex_v2.h5` files for delta p spectra surrogate training.

## Data Locations

| Location | Description | sdata_pertfields_grid_complex_v2.h5 |
|----------|-------------|-------------------------------------|
| `/global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC` | CFS project storage (batch config out_root) | **No** – run dirs contain raw outputs (C1input, geqdsk, sparc_fw1, etc.), not postprocessed complex_v2 |
| `${SURGE_SCRATCH}/mp288/jobs` | Scratch batch runs | **Yes** – batch_16 has 31 files; other batches vary |

## File Layout

Per-run delta p spectra live under:

```
batch_N/
  run1/
    sparc_1300/
      sdata_pertfields_grid_complex_v2.h5   ← one file per run/sparc
    sparc_1301/
      sdata_pertfields_grid_complex_v2.h5
    ...
  run2/
    ...
```

Each `sdata_pertfields_grid_complex_v2.h5` contains:
- `runs/<run_name>/`: miller (R0, a, kappa, delta), parset (ntor, pscale, batemanscale), flux_average (q0, q95, qmin, p0), growth_rate, spectrum/p (delta p modes)

## Batch Availability (pscratch)

| Batch | complex_v2 count | Notes |
|-------|------------------|-------|
| batch_1 | 0 | No postprocessed files |
| batch_2 | 0 | No postprocessed files |
| batch_16 | 31 | Has run1/sparc_*/sdata_pertfields_grid_complex_v2.h5 |

## Usage

```python
from surge import M3DC1Dataset

# Load with fixed resolution (all cases same m-spectrum shape)
dataset = M3DC1Dataset.from_batch_dir(
    "${SURGE_SCRATCH}/mp288/jobs/batch_16",
    target_shape=(50, 50),  # 50 modes x 50 psi = 2500 outputs, same for every case
)

# Or use mode_step/psi_step for uniform downsampling
dataset = M3DC1Dataset.from_batch_dir(
    "${SURGE_SCRATCH}/mp288/jobs/batch_16",
    mode_step=4,
    psi_step=4,
)

# With eigenmode features (requires m3dc1/fpy in M3DC1 env)
dataset = M3DC1Dataset.from_batch_dir(
    "${SURGE_SCRATCH}/mp288/jobs/batch_16",
    target_shape=(50, 50),
    include_eigenmodes=True,
)
```

For CFS SPARC batches (batch_0–batch_7, etc.), you need postprocessed `sdata_pertfields_grid_complex_v2.h5` files. If they are produced elsewhere, document that path here.

## Per-mode format (n, m → profile)

For learning **δp_n,m(ψ_N)** per mode: one row per (case, m). Inputs include `n` (from input_ntor) and `m` (poloidal mode, -100..99). Outputs = 200 profile points. Native resolution.

```bash
python scripts/m3dc1/build_delta_p_per_mode.py /pscratch/.../batch_16 --out data/datasets/SPARC/delta_p_per_mode.pkl
```

```python
from surge import M3DC1Dataset
dataset = M3DC1Dataset.from_batch_dir_per_mode("/pscratch/.../batch_16")
# Each row: (eq_*, input_*, n, m) → 200 profile values
```

Plot: `python scripts/m3dc1/plot_profile.py data/.../delta_p_per_mode.pkl --n 9 --m -7`

## Eigenmode Integration

Optional `include_eigenmodes=True` extracts eigenmode_amp_0..eigenmode_amp_19 from C1.h5 (max amplitude per mode). Requires m3dc1 and fpy (M3DC1 environment). See `scripts/m3dc1/eigenmode_features.py`.

## Reference

- `scripts/m3dc1/dataset_complex_v2.py` – `find_complex_v2_files`, `load_complex_v2_for_surge`
- `scripts/m3dc1/eigenmode_features.py` – `get_eigenmode_amplitudes_safe`
- `surge/datasets/m3dc1.py` – `M3DC1Dataset.from_batch_dir()`
- `docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md` – Training guide
