# SURGE Data and Resources Access Guide

## Overview

SURGE has two ways to access data files, templates, and resources from the repository:

1. **Automatic detection** (works after `pip install -e .`)
2. **Environment variable** (`$SURGE`) for explicit control

## Method 1: Automatic Detection (Recommended)

After installing SURGE with `pip install -e .`, the code automatically finds its location:

```python
from surge import get_data_path

# Access dataset
data_path = get_data_path("HHFW-NSTX")
print(f"Data located at: {data_path}")

# Load your data
import pandas as pd
df = pd.read_csv(data_path / "training_data.csv")
```

**How it works:**
- SURGE detects its location from the installed package
- No manual configuration needed
- Works anywhere on your system

## Method 2: Environment Variable (Explicit Control)

For more control or when working with data/templates, set the `$SURGE` environment variable:

### Quick Setup

```bash
cd /path/to/SURGE
./scripts/setup_surge_env.sh
source ~/.bashrc  # or ~/.zshrc
```

### Manual Setup

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export SURGE="/path/to/your/SURGE"
export PYTHONPATH="$SURGE:$PYTHONPATH"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Using $SURGE in Scripts

```python
import os
from pathlib import Path

# Get SURGE root directory
surge_root = Path(os.environ.get('SURGE', Path(__file__).parent))

# Access data files
data_file = surge_root / "data" / "datasets" / "my_dataset.h5"

# Access templates
template_dir = surge_root / "examples" / "datagen" / "templates"
template_file = template_dir / "C1input"

# Access any resource
resource = surge_root / "resources" / "config.json"
```

## Real-World Examples

### Example 1: Loading Training Data

```python
from surge import get_data_path
import pandas as pd

# Method 1: Using get_data_path (automatic)
data_path = get_data_path("HHFW-NSTX")
df = pd.read_csv(data_path / "training_data.csv")

# Method 2: Using $SURGE environment variable
import os
from pathlib import Path
surge_root = Path(os.environ['SURGE'])
df = pd.read_csv(surge_root / "data" / "datasets" / "HHFW-NSTX" / "training_data.csv")
```

### Example 2: DataGenerator with Templates

```python
from surge.datagen import DataGenerator
import os
from pathlib import Path

# Templates are in: SURGE/examples/datagen/templates/
surge_root = Path(os.environ.get('SURGE', Path(__file__).parent))
template_dir = surge_root / "examples" / "datagen" / "templates"

# Create DataGenerator
dg = DataGenerator(template_file=template_dir / "C1input")
```

### Example 3: Accessing Models/Results

```python
import os
from pathlib import Path

# Get SURGE root
surge_root = Path(os.environ.get('SURGE'))

# Save model
model_dir = surge_root / "models" / "my_experiment"
model_dir.mkdir(parents=True, exist_ok=True)
model.save(model_dir / "best_model.pkl")

# Load results
results_file = surge_root / "results" / "optimization_results.json"
```

## Directory Structure Reference

```
$SURGE/
├── surge/                  # Python package (importable after pip install -e .)
├── data/                   # Datasets
│   └── datasets/
│       ├── HHFW-NSTX/
│       └── your_dataset/
├── examples/               # Example scripts
│   └── datagen/
│       ├── templates/      # DataGenerator templates
│       │   ├── C1input
│       │   └── global_ns.py
│       └── batch*/         # Generated batches
├── models/                 # Saved models
├── results/                # Experiment results
├── notebooks/              # Jupyter notebooks
└── scripts/                # Utility scripts
    └── setup_surge_env.sh  # Environment setup script
```

## When to Use Each Method

| Use Case | Recommended Method |
|----------|-------------------|
| Import Python modules | `pip install -e .` |
| Access datasets in code | `get_data_path()` |
| Use DataGenerator templates | `$SURGE` env variable |
| Save/load models | `$SURGE` env variable |
| Scripts that need file paths | `$SURGE` env variable |
| CI/CD pipelines | `$SURGE` env variable |

## Best Practices

### ✅ DO

```python
# Use SURGE utilities when available
from surge import get_data_path
data_path = get_data_path("my_dataset")

# Use environment variable with fallback
import os
from pathlib import Path
surge_root = Path(os.environ.get('SURGE', Path(__file__).parent.parent))

# Make paths relative to SURGE root
data_file = surge_root / "data" / "my_data.csv"
```

### ❌ DON'T

```python
# DON'T hardcode absolute paths
data_file = "/u/username/src/SURGE/data/my_data.csv"  # ❌

# DON'T assume current directory
data_file = "./data/my_data.csv"  # ❌ May not work

# DON'T use sys.path.append with hardcoded paths
sys.path.append('/u/username/src/SURGE')  # ❌
```

## Troubleshooting

### "SURGE environment variable not set"

**Solution:**
```bash
cd /path/to/SURGE
./scripts/setup_surge_env.sh
source ~/.bashrc
```

### "Cannot find dataset"

**Solution:**
```python
from surge import get_data_path
try:
    data_path = get_data_path("my_dataset")
except FileNotFoundError as e:
    print(e)  # Shows available datasets
```

### "Module 'surge' not found"

**Solution:**
```bash
cd /path/to/SURGE
pip install -e .
```

## Verification

Check your setup:

```bash
# Check environment variable
echo $SURGE

# Test Python import
python -c "import surge; print(surge.__file__)"

# Test data access
python -c "from surge import get_data_path; print(get_data_path('HHFW-NSTX'))"

# Run full diagnostic
python check_installation.py
```

## Summary

- **`pip install -e .`**: Makes SURGE importable everywhere ✅
- **`$SURGE` env variable**: Provides explicit path to SURGE root ✅
- **`get_data_path()`**: Smart function for accessing datasets ✅
- **Never hardcode paths**: Always use relative paths from `$SURGE` ✅

---

**Ready to access your data? 📊**
