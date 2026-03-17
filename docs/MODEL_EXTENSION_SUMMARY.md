# Summary: How to Add a New Model Type to SURGE

## Answer to: "How would a new user use SURGE to expand to a new type of model?"

To add a new model type (e.g., CNN) to SURGE, you need to:

1. **Create an adapter class** that inherits from `BaseModelAdapter`
2. **Implement required methods**: `fit()` and `predict()`
3. **Optionally implement**: `predict_with_uncertainty()`, `save()`, `load()`
4. **Register the adapter** using `MODEL_REGISTRY.register()`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    SURGE Workflow                        │
│  (SurrogateEngine, SurrogateWorkflowSpec, etc.)         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Model Registry (MODEL_REGISTRY)             │
│  - Stores adapter classes                               │
│  - Creates instances via keys/aliases                   │
│  - Manages metadata (tags, descriptions, etc.)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            BaseModelAdapter (Abstract Base)              │
│  - Defines interface: fit(), predict()                  │
│  - Optional: predict_with_uncertainty(), save(), load() │
│  - Manages fitted state                                 │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬────────────┐
        ▼            ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Random   │  │ Torch    │  │ GPflow   │  │ Your     │
│ Forest   │  │ MLP      │  │ GPR      │  │ CNN      │
│ Adapter  │  │ Adapter  │  │ Adapter  │  │ Adapter  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## Required Methods

### 1. `fit(X, y) -> self`
**Purpose**: Train the model

**Input**:
- `X`: numpy array of shape `(n_samples, n_features)` - already standardized by SURGE
- `y`: numpy array of shape `(n_samples,)` or `(n_samples, n_outputs)` - already standardized if configured

**Requirements**:
- Must call `self.mark_fitted()` when training completes
- Should return `self` for method chaining
- Store any necessary state in `self.params` for save/load

**Example**:
```python
def fit(self, X, y):
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    
    # Your training code here
    self.model = YourModel(...)
    self.model.train(X_np, y_np)
    
    self.mark_fitted()  # REQUIRED!
    return self
```

### 2. `predict(X) -> np.ndarray`
**Purpose**: Make predictions

**Input**:
- `X`: numpy array of shape `(n_samples, n_features)` - already standardized

**Output**:
- numpy array of shape `(n_samples, n_outputs)`

**Requirements**:
- Must call `self.ensure_fitted()` first (validates model is trained)
- Return numpy array (not tensor, not list)

**Example**:
```python
def predict(self, X):
    self.ensure_fitted()  # REQUIRED!
    
    X_np = np.asarray(X, dtype=np.float32)
    predictions = self.model.predict(X_np)
    
    return predictions  # Must be numpy array
```

---

## Optional Methods

### 3. `predict_with_uncertainty(X) -> dict`
**Purpose**: Provide uncertainty estimates

**When to implement**: If your model can provide uncertainty (e.g., ensemble, MC-dropout, Bayesian)

**Output format**:
```python
return {
    "mean": np.ndarray,      # (n_samples, n_outputs)
    "variance": np.ndarray,  # (n_samples, n_outputs)
}
```

**Set class attribute**: `supports_uq = True`

### 4. `save(path: Path) -> None`
**Purpose**: Save trained model to disk

**When to implement**: If you want model persistence

**Set class attribute**: `supports_serialization = True`

### 5. `load(path: Path) -> self`
**Purpose**: Load trained model from disk

**When to implement**: If you want model persistence

**Requirements**:
- Must call `self.mark_fitted()` after loading
- Return `self` for chaining

---

## Registration

After creating your adapter class, register it:

```python
MODEL_REGISTRY.register(
    YourAdapterClass,
    key="backend.model_name",      # Primary key (required)
    name="Human Readable Name",    # Optional
    backend="backend_name",        # Optional (auto-detected from class)
    description="Model description", # Optional
    tags=["tag1", "tag2"],         # Optional
    aliases=["alias1", "alias2"],  # Optional alternative keys
    default_params={...},          # Optional default parameters
)
```

**Registration happens automatically** when the module is imported. Make sure to:
1. Add your adapter to `surge/models/adapters/__init__.py`
2. Import the module somewhere (usually in `surge/__init__.py`)

---

## Integration Points

### 1. In Code
```python
from surge import SurrogateEngine, ModelSpec

engine = SurrogateEngine(...)
engine.configure_from_dataset(dataset)
engine.prepare()

spec = ModelSpec(
    key="torch.cnn",  # Your registered key
    params={"epochs": 100, ...}
)
results = engine.run([spec])
```

### 2. In YAML Workflow
```yaml
models:
  - key: torch.cnn
    name: my_cnn
    params:
      epochs: 100
      batch_size: 256
    hpo:
      enabled: true
      n_trials: 20
      search_space:
        learning_rate:
          type: loguniform
          low: 1e-4
          high: 1e-2
```

### 3. With HPO (Hyperparameter Optimization)
Your adapter automatically works with SURGE's Optuna-based HPO. Just define search spaces in YAML or `HPOConfig`.

---

## Key Design Principles

1. **Separation of Concerns**:
   - SURGE handles: data loading, splitting, standardization, metrics, artifacts
   - Your adapter handles: model architecture, training, prediction

2. **State Management**:
   - SURGE tracks whether model is fitted
   - You must call `mark_fitted()` after training
   - SURGE validates with `ensure_fitted()` before prediction

3. **Data Preprocessing**:
   - SURGE provides standardized/scaled data to your adapter
   - You don't need to handle scaling yourself
   - Input/output shapes are consistent

4. **Flexibility**:
   - Support single or multi-output
   - Handle any number of features
   - Store training history if desired (in `self.training_history`)

---

## Files Created

1. **`docs/ADDING_NEW_MODEL_ADAPTER.md`** - Complete guide with CNN example
2. **`docs/QUICK_START_NEW_MODEL.md`** - Quick reference
3. **`examples/custom_cnn_adapter_template.py`** - Working template you can copy

---

## Example: CNN Adapter Structure

```
surge/models/adapters/cnn_adapter.py
├── CNNModel (PyTorch nn.Module)
│   └── Architecture definition
│
├── CNNAdapter (BaseModelAdapter)
│   ├── __init__() - Initialize with parameters
│   ├── fit() - Train model (REQUIRED)
│   ├── predict() - Make predictions (REQUIRED)
│   ├── save() - Save model (optional)
│   └── load() - Load model (optional)
│
└── MODEL_REGISTRY.register() - Register adapter
```

---

## Testing Your Adapter

```python
# Quick test
from surge import SurrogateEngine, SurrogateDataset, ModelSpec

# Load/create data
dataset = SurrogateDataset.from_path("data.pkl")

# Create engine
engine = SurrogateEngine()
engine.configure_from_dataset(dataset)
engine.prepare()

# Test your adapter
spec = ModelSpec(key="torch.cnn", params={...})
results = engine.run([spec])

print(f"R²: {results[0].val_metrics['r2']:.4f}")
```

---

## Common Patterns

### Pattern 1: Simple Wrapper (e.g., sklearn models)
```python
class SimpleAdapter(BaseModelAdapter):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = SomeModel(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.mark_fitted()
        return self
    
    def predict(self, X):
        self.ensure_fitted()
        return self.model.predict(X)
```

### Pattern 2: PyTorch Model
```python
class TorchAdapter(BaseModelAdapter):
    def __init__(self, **params):
        super().__init__(**params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def fit(self, X, y):
        # Build model
        self.model = YourTorchModel(...).to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            # ... training code ...
        
        self.mark_fitted()
        return self
    
    def predict(self, X):
        self.ensure_fitted()
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X)).numpy()
```

---

## Next Steps

1. **Read**: `docs/ADDING_NEW_MODEL_ADAPTER.md` for detailed guide
2. **Copy**: `examples/custom_cnn_adapter_template.py` as starting point
3. **Modify**: Adapt the template to your model architecture
4. **Test**: Verify it works with SURGE workflows
5. **Register**: Add to `surge/models/adapters/__init__.py`

---

## Support

- **Reference implementations**: See existing adapters in `surge/models/adapters/`
- **Base class**: `surge/registry.py` - `BaseModelAdapter`
- **Registry**: `surge/registry.py` - `MODEL_REGISTRY`
- **Examples**: `examples/` directory

---

*The SURGE adapter architecture is designed to be simple, flexible, and extensible. Most models can be integrated with just a few methods!*

