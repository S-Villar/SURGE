# AI_OCLF Notebook Architecture and Model Format

This document describes the prior AI_OCLF work for XGC A_parallel surrogate modeling, including notebook structure, model architecture, and checkpoint format.

## Prior Work Location

| Path | Description |
|------|-------------|
| `/global/u2/a/asvillar/AI_OCLF/` | Prior work (not in m499 project dir) |
| `/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025/` | OLCF hackathon data |

## Notebook Roles

| Notebook | Purpose |
|----------|---------|
| `Ah_prediction_NN1_ASV_Geometrical.ipynb` | Main prediction notebook – data load, train, infer |
| `Ah_prediction_NN1_ASV_Geometrical_inference.ipynb` | Inference-only notebook |
| `Ah_prediction_NN1_ASV_Geometrical_firstmodel.ipynb` | First model variant |
| `Ah_ML_NN_ASV_train.ipynb` | Training workflow |
| `Ah_ML_NN_ASV_load_infer.ipynb` | Load checkpoint and run inference |
| `read_analyze_data.ipynb` | Data exploration (set2_beta0p5) |

## Model Architecture: GeometricalMeanNN

The primary model is **GeometricalMeanNN**, a feedforward MLP with hidden layer sizes computed via geometric mean.

### Layer Structure

```
Input (201) → Linear(201, 14) → BatchNorm1d(14) → ReLU → Dropout
           → Linear(14, 3)   → BatchNorm1d(3)  → ReLU → Dropout
           → Linear(3, 1)   → BatchNorm1d(1)  → ReLU → Dropout
           → Linear(1, 1)   → Output (1)
```

Hidden sizes: `next_size = int((current_size * output_size) ** 0.5)` (geometric mean).

### Hyperparameters (from training_summary.txt)

| Parameter | Value |
|-----------|-------|
| num_hidden_layers | 3 |
| dropout_rate | 0.0 |
| test_size | 0.2 |
| num_epochs | 500 |
| batch_size | 128 |
| learning_rate | 0.001 |

### PyTorch Code

```python
class GeometricalMeanNN(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, dropout_rate):
        layers = []
        current_size = input_size
        for i in range(num_hidden_layers):
            next_size = int((current_size * output_size) ** 0.5)  # Geometrical mean
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = next_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
```

## Model Checkpoint Format

### PyTorch (.pth)

- **Path:** `model_epoch_500.pth` (and `model_epoch_100.pth`, `model_epoch_200.pth`, etc.)
- **Content:** `state_dict` only (weights and biases)
- **Load:** `model.load_state_dict(torch.load(model_path))`
- **Input shape:** `(batch, 201)`
- **Output shape:** `(batch, 1)` (A_parallel)

### ONNX Export

- **Path:** `model.onnx`
- **Input shape:** `(1, 201)` or `(batch, 201)` depending on export
- **Usage:** `onnxruntime.InferenceSession` for deployment

### Full Checkpoint (Optional)

Some notebooks save a full checkpoint with metadata:

```python
{
    'state_dict': model.state_dict(),
    'num_inputs': 201,
    'output_size': 1,
    'num_hidden_layers': 3,
    'dropout_rate': 0.0,
}
```

## Data Preprocessing

- **Input:** `StandardScaler` on X (201 features)
- **Output:** `StandardScaler` on y (target column 1)
- **Target:** `target[:, 1]` (A_parallel)

## Inference

Targets are predicted in scaled space; apply `scaler_y.inverse_transform()` for physical values.
