# Training Surrogates on Delta p Spectra from sdata_complex_v2.h5

This guide explains how to train SURGE surrogate models on **delta p (pressure perturbation) spectra** from M3DC1 `sdata_complex_v2.h5` files.

## Overview

`sdata_complex_v2.h5` contains eigenmode spectra for δp (delta p) as a function of flux surface. Each run has:
- **Inputs**: Equilibrium parameters (R0, a, kappa, delta, q0, q95, qmin, p0, etc.) and run inputs (ntor, pscale, batemanscale)
- **Output**: Delta p spectrum — a vector of amplitudes (or complex values) at each flux surface / mode

## Quick Start

### 1. Inspect your file structure

If your file has a non-standard structure, inspect it first:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts/m3dc1")))
from loader import read_sdata_complex_v2_structure

# Inspect structure and output keys
read_sdata_complex_v2_structure("path/to/sdata_complex_v2.h5")
```

This prints the root groups, input keys, and **output keys with shapes**. Use the output key name for `delta_p_key` if auto-detect fails.

### 2. Convert to DataFrame

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("scripts/m3dc1")))
from loader import convert_sdata_complex_v2_to_dataframe

df = convert_sdata_complex_v2_to_dataframe("path/to/sdata_complex_v2.h5")
# Or with explicit key if auto-detect fails:
# df = convert_sdata_complex_v2_to_dataframe("...", delta_p_key="p")
```

The DataFrame has:
- `input_*`, `eq_*`, `q0`, `q95`, `qmin`, `p0` as inputs
- `output_p_0`, `output_p_1`, ... as the delta p spectrum (magnitude by default for complex data)

### 3. Train with SURGE

```python
from surge.data.dataset import SurrogateDataset
from surge.workflow.spec import SurrogateWorkflowSpec
from surge.workflow.run import run_surrogate_workflow

# Option A: Save DataFrame and use metadata
df.to_pickle("data/datasets/SPARC/delta_p_spectra.pkl")

spec = SurrogateWorkflowSpec(
    dataset_path="data/datasets/SPARC/delta_p_spectra.pkl",
    metadata_path="data/datasets/SPARC/delta_p_spectra_metadata.yaml",
    output_dir="runs/delta_p_spectra",
    models=[{"key": "random_forest"}, {"key": "torch_mlp"}],
)
run_surrogate_workflow(spec)
```

### 4. Metadata for delta p spectra

Create `data/datasets/SPARC/delta_p_spectra_metadata.yaml` (or use the provided template):

```yaml
inputs:
  - eq_R0
  - eq_a
  - eq_kappa
  - eq_delta
  - q0
  - q95
  - qmin
  - p0
  - input_ntor
  - input_pscale
  - input_batemanscale
metadata:
  campaign: M3DC1_delta_p_spectra
  description: Delta p eigenmode spectrum surrogate
```

Output columns (`output_p_0`, `output_p_1`, ...) are auto-detected by SURGE as a profile group. If your column names differ, adjust `inputs` to match the DataFrame.

## Supported output key names

The loader auto-detects delta p spectra from these keys (in order):
- `p`
- `delta_p`
- `p_spectrum`
- `eigenmode_p`
- `eignemode`

If your key is different, pass it explicitly:

```python
df = convert_sdata_complex_v2_to_dataframe(
    "sdata_complex_v2.h5",
    delta_p_key="your_custom_key",
)
```

## Complex vs magnitude

By default, complex spectra are converted to **magnitude** (`|z|`) for each component. To keep real and imaginary parts as separate columns, use:

```python
df = convert_sdata_complex_v2_to_dataframe(
    "sdata_complex_v2.h5",
    use_magnitude=False,  # Not yet implemented for real/imag columns
)
```

(Currently only magnitude is implemented; real/imag expansion can be added if needed.)

## File format detection

`convert_to_dataframe` auto-detects `sdata_complex_v2` when the filename contains both `"complex"` and `"v2"`. You can force the format:

```python
df = convert_to_dataframe("myfile.h5", format="sdata_complex_v2")
```

## See also

- [scripts/m3dc1/README.md](../../scripts/m3dc1/README.md) — M3DC1 loader overview
- [M3DC1_CAPABILITIES_RESULTS.md](M3DC1_CAPABILITIES_RESULTS.md) — SURGE integration results
