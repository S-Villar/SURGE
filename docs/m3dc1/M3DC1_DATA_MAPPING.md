# M3DC1 Delta p Data Mapping

Mapping of postprocessed delta p spectra (`sdata_pertfields_grid_complex_v2.h5`) availability across pscratch batch directories.

## Root

- **Path:** `${SURGE_SCRATCH}/mp288/jobs`
- **Structure:** `batch_N/run*/sparc_*/`

## Key Finding

**Only `run1` in `batch_16` has been postprocessed** to produce `sdata_pertfields_grid_complex_v2.h5`.

| Location | complex_v2 count | Notes |
|----------|------------------|-------|
| batch_16/run1 | 31 | Has sdata_pertfields_grid_complex_v2.h5 in each sparc_* |
| batch_16/run2..run120 | 0 | C1.h5 exists; complex_v2 not yet generated |
| batch_1 | 0 | No postprocessed files |
| batch_2 | 0 | No postprocessed files |
| batch_10.. | 0 | No postprocessed files |

## Most Repeated Datasets

| Dataset | Count | Used for |
|---------|-------|----------|
| `sdata_pertfields_grid_complex_v2.h5` | 31 | Delta p per-mode surrogate (current) |
| `C1.h5` | ~9800+ | Raw M3DC1 output (all runs) |
| `sdata_pertfields_grid.h5` | varies | Non-complex variant |
| `sdata_pertfields_grid_eq.h5` | varies | Equilibrium variant |

## Why Only 31 Cases?

The postprocessing step that produces `sdata_pertfields_grid_complex_v2.h5` from `C1.h5` (or pertfields) has been run only on **batch_16/run1** (31 sparc dirs). The other ~117 runs in batch_16 and all runs in other batches have `C1.h5` but have not been postprocessed.

## To Get ~9800 Delta p Cases

1. **Run postprocessing** on all sparc dirs that have `C1.h5` but lack `sdata_pertfields_grid_complex_v2.h5`.
2. The postprocessing script (e.g. `sdata_write.py` or equivalent) should be run per `run*/sparc_*/` to produce `sdata_pertfields_grid_complex_v2.h5`.
3. Document the postprocessing command/location in `scripts/m3dc1/` or `docs/m3dc1/`.

## Discovery Script

```bash
python scripts/m3dc1/discover_m3dc1_data.py ${SURGE_SCRATCH}/mp288/jobs
python scripts/m3dc1/discover_m3dc1_data.py --sample-runs 5  # faster on large batches
```

## Reference

- `scripts/m3dc1/dataset_complex_v2.py` – `find_complex_v2_files`, `load_per_mode_for_surge`
- `scripts/m3dc1/build_delta_p_per_mode.py` – builds dataset from batch dir
- `docs/m3dc1/BATCH_DIRS_AND_DATA.md` – batch locations
