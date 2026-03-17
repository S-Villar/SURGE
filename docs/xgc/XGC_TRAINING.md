# XGC A_parallel Surrogate Training Guide

## Quick Start

### 1. Run SURGE workflow (subsampled)

```bash
cd ${SURGE_HOME}
pip install -e .   # install in editable mode to get the surge CLI
surge run configs/xgc_aparallel_set1.yaml
```

Without installing, use the Python example:

```bash
python examples/m3dc1_workflow.py --spec configs/xgc_aparallel_set1.yaml
```

Or from Python:

```python
from pathlib import Path
from surge import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec
import yaml

with open("configs/xgc_aparallel_set1.yaml") as f:
    spec = SurrogateWorkflowSpec.from_dict(yaml.safe_load(f))
summary = run_surrogate_workflow(spec)
```

### 2. Config options

| Option | Description |
|--------|-------------|
| `dataset_path` | OLCF hackathon dir: `/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025` |
| `dataset_format` | `xgc` |
| `metadata_overrides.set_name` | `set1` or `set2_beta0p5` |
| `sample_rows` | Subsample size (e.g. 50000); set to `null` for full dataset |

### 3. Full training (no subsample)

Edit `configs/xgc_aparallel_set1.yaml`:

```yaml
sample_rows: null  # Use full ~29M samples (set1) or ~20M (set2)
```

## Compare with Prior Model (model_epoch_500.pth)

The prior AI_OCLF model uses **GeometricalMeanNN** (201→14→3→1→1). SURGE's `torch.mlp` uses a different architecture. To compare:

1. **Train SURGE model** on the same subsample used for the prior model (e.g. 100k rows).
2. **Load prior checkpoint** and run inference on the test split.
3. **Compare metrics**: R², RMSE, MAE on the same test set.

### Prior model inference

```python
import torch
import numpy as np
from surge import XGCDataset

# Load data
dataset = XGCDataset.from_olcf_hackathon(
    "/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025",
    set_name="set1",
    sample=100000,
)
X = dataset.df[dataset.input_columns].values.astype(np.float32)
y_true = dataset.df["output_1"].values.astype(np.float32)  # A_parallel

# Define GeometricalMeanNN (match prior architecture)
class GeometricalMeanNN(torch.nn.Module):
    def __init__(self, input_size=201, output_size=1, num_hidden_layers=3, dropout_rate=0.0):
        super().__init__()
        layers = []
        current_size = input_size
        for _ in range(num_hidden_layers):
            next_size = int((current_size * output_size) ** 0.5)
            layers.extend([
                torch.nn.Linear(current_size, next_size),
                torch.nn.BatchNorm1d(next_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            ])
            current_size = next_size
        layers.append(torch.nn.Linear(current_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Load prior checkpoint
model = GeometricalMeanNN(201, 1, 3, 0.0)
model.load_state_dict(torch.load("/global/u2/a/asvillar/AI_OCLF/model_epoch_500.pth"))
model.eval()

# Predict (apply same scaling as training if used)
with torch.no_grad():
    X_t = torch.from_numpy(X)
    y_pred = model(X_t).numpy().ravel()
```

### SURGE model comparison

After training, load SURGE artifacts from `runs/xgc_aparallel_set1/` and compare predictions on the same test indices.
