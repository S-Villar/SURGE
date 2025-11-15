# Sample Dataset Files

These are small representative samples of the full datasets for testing and examples.

## Purpose

Full datasets are excluded from version control due to size constraints (95MB+ pickle files, large CSV files). 
These samples allow:
- Testing SURGE functionality without large data files
- Examples to run quickly
- CI/CD pipelines to test code without data dependencies
- Developers to understand dataset structure

## Sample Files

### M3DC1
- **File**: `M3DC1/m3dc1_synthetic_profiles_sample.csv`
- **Size**: First 100 rows of the synthetic profiles dataset
- **Format**: CSV with same columns as full dataset
- **Full dataset**: `m3dc1_synthetic_profiles.csv` (271KB, ~9,981 rows)

### SPARC
- **File**: `SPARC/sparc-m3dc1-D1_sample.pkl` (to be created)
- **Size**: First 100 rows of the SPARC M3DC1 dataset
- **Format**: Pandas DataFrame pickle
- **Full dataset**: `sparc-m3dc1-D1.pkl` (95MB, ~9,891 rows)

**Creating SPARC Sample:**
```python
import pandas as pd
df = pd.read_pickle('data/datasets/SPARC/sparc-m3dc1-D1.pkl')
df.head(100).to_pickle('data/datasets/SPARC/sparc-m3dc1-D1_sample.pkl')
```

## Dataset Characteristics

Samples maintain:
- Same column structure and names
- Same data types
- Same metadata format
- Representative data distribution (first N rows or stratified sample)

## Usage

Samples can be used directly in examples and tests:
```python
from surge import SurrogateDataset

# Works with sample or full dataset
dataset = SurrogateDataset.from_path('data/datasets/SPARC/sparc-m3dc1-D1_sample.pkl')
```

**Note**: For production model training, use the full datasets stored in external storage or project directories.
