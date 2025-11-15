# SPARC M3DC1 Dataset (sparc-m3dc1-D1)

- **Source**: SPARC linear MHD (M3DC1-LRMHD) batch processed with `sdata_write_fast.py`.
- **File**: `sparc-m3dc1-D1.pkl`
- **Content**: 9,891 curated cases with completeness mask and equilibrium/profile parameters.

## Fields
- Equilibrium geometry: `R0`, `a`, `kappa`, `delta`
- Plasma/magnetic properties: `q0`, `q95`, `qmin`, `p0`
- Input parameters: `ntor`, `pscale`, `batemanscale`
- Output target: `gamma`
- Metadata: `run_id`, `eqid`, `completeness`

## Usage
```python
from pathlib import Path
import pandas as pd
from surge import SurrogateDataset

pkl_path = Path('data/datasets/SPARC/sparc-m3dc1-D1.pkl')
df = pd.read_pickle(pkl_path)

engine = SurrogateDataset()
engine.load_dataframe(df, input_cols=['R0', 'a', 'kappa', 'delta', 'q0', 'q95', 'qmin', 'p0'], output_cols=['gamma'])
```
