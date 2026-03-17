# SURGE Development Plan

Living document tracking two main development branches. Update this file as tasks progress.

---

## Branch 1: M3DC1 Dataset + Eigenmodes (mp288)

**Project:** mp288 | **Goal:** Create dataset from runs, integrate eigenmodes/spectra, improve M3DC1 surrogates

### Data Locations

| Location | Description |
|----------|-------------|
| `/global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC` | Batch config out_root (see `examples/batch_setup_m3dc1.yml`) |
| `${SURGE_SCRATCH}/mp288/jobs/batch_*` | Scratch batch runs |
| `batch_N/run*/sparc_*/sdata_pertfields_grid_complex_v2.h5` | Per-run delta p spectra HDF5 |
| `runs/m3dc1_aug_r75/` | Reference SURGE run (growth-rate surrogates, R² > 0.88) |

### Existing SURGE Components

- **M3DC1Dataset:** [surge/datasets/m3dc1.py](../../surge/datasets/m3dc1.py) – `from_batch_dir()`, `from_sdata_complex_v2()` for SurrogateDataset-compatible loading
- **Loader:** [scripts/m3dc1/loader.py](../../scripts/m3dc1/loader.py) – `convert_sdata_complex_v2_to_dataframe`, `read_sdata_complex_v2_structure`
- **Dataset builder:** [scripts/m3dc1/dataset_complex_v2.py](../../scripts/m3dc1/dataset_complex_v2.py) – `build_dataframe_from_batch`, `load_complex_v2_for_surge`
- **Training guide:** [docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md](docs/m3dc1/DELTA_P_SPECTRA_TRAINING.md)
- **Metadata:** [data/datasets/SPARC/delta_p_spectra_metadata.yaml](data/datasets/SPARC/delta_p_spectra_metadata.yaml)
- **Eigenmode scripts:** [scripts/m3dc1/interfaces/](scripts/m3dc1/interfaces/) – `test_eigenmode_extraction.py`, `reconstruct_field_from_eigenmodes.py`, etc.

### Recipe Steps

1. Identify data sources – Document batch locations, which have `sdata_pertfields_grid_complex_v2.h5`
2. Extend dataset from runs – Use `build_dataframe_from_batch` / `load_complex_v2_for_surge` with batch dirs
3. Surge dataset wrapper – **Done:** `M3DC1Dataset.from_batch_dir()` and `from_sdata_complex_v2()`
4. Include eigenmodes – Integrate eigenmode fields into dataset pipeline
5. Train from spectra – Improve models with eigenmode features per DELTA_P_SPECTRA_TRAINING.md

### Task Tracking

- [ ] 1.1 Document batch dir locations and sdata_complex_v2 availability
- [ ] 1.2 Build DataFrame from batch dirs via dataset_complex_v2
- [x] 1.3 Add SurrogateDataset loader for M3DC1 batch/spectra (`M3DC1Dataset`)
- [ ] 1.4 Integrate eigenmode extraction into dataset pipeline
- [ ] 1.5 Train/improve surrogates with eigenmode features

---

## Branch 2: XGC A_parallel Surrogate (m499)

**Project:** m499 | **Goal:** Relocate prior work, build SURGE loader, train A_parallel surrogates for XGC

### Data Locations

| Location | Description |
|----------|-------------|
| `/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025/` | OLCF hackathon data (lower beta: set1; beta0.5: set2) |
| `/global/u2/a/asvillar/AI_OCLF/` | **Prior work:** notebooks, models (not in m499 project dir) |

### OLCF Hackathon Data Layout

| File | Shape | Description |
|------|-------|-------------|
| `data_nprev5_set1_data.npy` | (28994400, 201) | 201 inputs |
| `data_nprev5_set1_target.npy` | (28994400, 2) | 2 outputs |
| `data_nprev5_set1_tags.npy` | (28994400, 3) | Tags |
| `data_nprev5_set1_var_all.npy` | 14 vars | aparh, apars, dBphi, dBpsi, dBtheta, ejpar, ijpar, dpot, epara, epara2, epsi, etheta, eden, iden |
| `data_nprev5_set2_beta0p5_*` | Similar | Beta 0.5 case |

### Prior Work (AI_OCLF)

| File | Description |
|------|-------------|
| `Ah_prediction_NN1_ASV_Geometrical.ipynb` | Main prediction notebook |
| `Ah_prediction_NN1_ASV_Geometrical_inference.ipynb` | Inference notebook |
| `Ah_prediction_NN1_ASV_Geometrical_firstmodel.ipynb` | First model |
| `Ah_ML_NN_ASV_train.ipynb` | Training |
| `Ah_ML_NN_ASV_load_infer.ipynb` | Load and infer |
| `model.onnx` | ONNX export |
| `model_epoch_500.pth` | PyTorch checkpoint (epoch 500) |
| `model_epoch_100.pth`, `model_epoch_200.pth`, etc. | Other checkpoints |

### Recipe Steps

1. Relocate prior work – **Done:** Found in `/global/u2/a/asvillar/AI_OCLF/`
2. Inspect data format – Document 201 inputs, 2 outputs, variable mapping
3. Build SURGE loader – **Done:** `XGCDataset.from_olcf_hackathon()` and `from_npy()` in [surge/datasets/xgc.py](../../surge/datasets/xgc.py)
4. Train with SURGE – Config and workflow for XGC A_parallel surrogate
5. Retrain/compare – Compare prior models with SURGE-trained models

### Task Tracking

- [ ] 2.1 Document AI_OCLF notebook architecture and model format
- [ ] 2.2 Inspect and document data_nprev5_set1/set2 variable mapping
- [x] 2.3 Create XGC loader (scripts/xgc or surge/datagen) for SURGE (`XGCDataset`)
- [ ] 2.4 Add config template for XGC A_parallel surrogate
- [ ] 2.5 Train with SURGE and compare with prior model_epoch_500.pth

---

## Application Dataset Classes

SURGE provides `M3DC1Dataset` and `XGCDataset` in `surge.datasets` (and `surge` package exports):

```python
from surge import M3DC1Dataset, XGCDataset

# M3DC1: from batch dir or single sdata_complex_v2.h5
dataset = M3DC1Dataset.from_batch_dir("/path/to/batch_16")
# or
dataset = M3DC1Dataset.from_sdata_complex_v2("path/to/sdata_complex_v2.h5")

# XGC: from OLCF hackathon or generic .npy files
dataset = XGCDataset.from_olcf_hackathon(
    "/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025",
    set_name="set1",
    sample=10000,  # optional subsample
)
# or
dataset = XGCDataset.from_npy("data.npy", "target.npy")
```

Both return SurrogateDataset-compatible instances (`df`, `input_columns`, `output_columns`) for use with `SurrogateEngine` and `run_surrogate_workflow`.

---

## Cross-References

- [docs/README.md](../README.md) – Documentation navigation
- [docs/dev/README.md](README.md) – Dev folder purpose
- [SURGE_OVERVIEW.md](../SURGE_OVERVIEW.md) – Framework overview
