# Multi-Modal Diffusion Surrogates: Quick Summary

## What's Needed

To build a **multi-modal surrogate based on joint diffusion** in SURGE, you need to extend the architecture in 5 key areas:

---

## 1. Multi-Modal Data Layer

**Current**: SURGE only handles tabular data (DataFrames → numpy arrays)

**Needed**:
- `MultiModalDataset` class to handle:
  - Tabular data (parameters)
  - Images (plasma profiles, visualizations)
  - Sequences (time series, trajectories)
  - Point clouds (3D spatial data)
- Per-modality preprocessing pipelines
- Multi-modal data loaders

**Key Classes**:
```python
class MultiModalDataset:
    def add_modality(name, data, modality_type, is_output=False)
    def get_batch(indices) -> Dict[str, torch.Tensor]
```

---

## 2. Diffusion Model Core

**Needed**:
- Base `DiffusionModel` implementing DDPM (Denoising Diffusion Probabilistic Model)
- Noise schedules (forward diffusion: add noise)
- Sampling methods (reverse diffusion: generate from noise)
- UNet architecture for denoising network

**Key Components**:
- `q_sample()`: Forward diffusion (add noise)
- `p_sample()`: Reverse diffusion (denoise one step)
- `p_sample_loop()`: Full generation from noise

---

## 3. Multi-Modal Architecture

**Needed**:
- **Encoders** (one per input modality):
  - Tabular encoder (MLP)
  - Image encoder (CNN/ViT)
  - Sequence encoder (RNN/Transformer)
- **Fusion layer**: Combines modalities into shared latent space
  - Options: Concatenation, Cross-attention, Mixture-of-Experts
- **Denoising network**: UNet operating on latent space
- **Decoders** (one per output modality):
  - Tabular decoder (MLP)
  - Image decoder (CNN/ViT)

**Architecture Flow**:
```
Input Modalities → Encoders → Fusion → Latent Space
                                              ↓
                                    Diffusion Process
                                              ↓
Output Modalities ← Decoders ← Fusion ← Latent Space
```

---

## 4. SURGE Adapter Extension

**Current**: `BaseModelAdapter.fit(X: ndarray, y: ndarray)`

**Needed**: Extended adapter for multi-modal inputs

```python
class MultiModalDiffusionAdapter(BaseModelAdapter):
    def fit(self, dataset: MultiModalDataset)  # Not (X, y)
    def predict(self, inputs: Dict[str, Tensor]) -> Dict[str, ndarray]
    def predict_with_uncertainty(self, inputs, n_samples=100)
```

**Key Differences**:
- Accepts `MultiModalDataset` instead of numpy arrays
- Returns dict of outputs per modality
- Natural uncertainty via multiple diffusion samples

---

## 5. Engine & Workflow Extensions

**Needed**:
- `MultiModalSurrogateEngine` for multi-modal workflows
- Extended `SurrogateWorkflowSpec` for YAML configs
- Multi-modal artifact handling

**YAML Example**:
```yaml
dataset:
  type: multimodal
  modalities:
    tabular_params: {type: tabular, path: ...}
    plasma_images: {type: image, path: ...}
    output_profiles: {type: tabular, path: ..., is_output: true}
```

---

## Implementation Complexity

| Component | Complexity | Dependencies |
|-----------|-----------|--------------|
| Multi-Modal Dataset | Medium | torch, PIL |
| Diffusion Core | High | torch, math |
| Multi-Modal Architecture | Very High | torch, transformers |
| SURGE Integration | Medium | Existing SURGE |
| Workflow Extensions | Low | YAML parsing |

---

## Key Technical Challenges

1. **Modality Alignment**: Different modalities → common latent space
2. **Conditional Generation**: Generate outputs conditioned on inputs
3. **Training Stability**: Diffusion models can be unstable
4. **Computational Cost**: Diffusion requires many timesteps (100-1000)
5. **Evaluation**: Metrics for multi-modal generative models

---

## Minimal Viable Implementation

**Phase 1** (Tabular + Images):
- Multi-modal dataset loader
- Simple fusion (concatenation)
- Basic diffusion model
- SURGE adapter wrapper

**Phase 2** (Full Features):
- Advanced fusion (attention)
- Multiple modalities
- Efficient sampling (DDIM)
- Comprehensive evaluation

---

## Dependencies

```python
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0  # Optional: use their diffusion implementations
pillow>=9.0.0
einops>=0.6.0  # Tensor reshaping
```

---

## Example Usage (Target API)

```python
# Create multi-modal dataset
dataset = MultiModalDataset()
dataset.add_modality("params", tabular_data, "tabular")
dataset.add_modality("images", image_paths, "image")
dataset.add_modality("profiles", output_data, "tabular", is_output=True)

# Train diffusion model
adapter = MultiModalDiffusionAdapter(
    input_modalities={"params": {...}, "images": {...}},
    output_modalities={"profiles": {...}},
)
adapter.fit(dataset)

# Generate with uncertainty
predictions = adapter.predict(
    inputs={"params": ..., "images": ...},
    n_samples=10  # Multiple samples for uncertainty
)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              MultiModalDataset                          │
│  - Tabular: DataFrame columns                          │
│  - Images: File paths or arrays                        │
│  - Sequences: Time series data                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         MultiModalPreprocessor                          │
│  - Per-modality scaling/normalization                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│      MultiModalDiffusionAdapter                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │ Tabular      │  │ Image        │                    │
│  │ Encoder      │  │ Encoder      │                    │
│  └──────┬───────┘  └──────┬───────┘                    │
│         │                 │                             │
│         └────────┬────────┘                            │
│                  ▼                                      │
│         ┌─────────────────┐                            │
│         │  Fusion Layer   │                            │
│         └────────┬─────────┘                            │
│                  ▼                                      │
│         ┌─────────────────┐                            │
│         │  Latent Space   │                            │
│         └────────┬─────────┘                            │
│                  ▼                                      │
│         ┌─────────────────┐                            │
│         │ Diffusion UNet   │                            │
│         │ (Denoising)     │                            │
│         └────────┬─────────┘                            │
│                  ▼                                      │
│         ┌─────────────────┐                            │
│         │  Tabular        │                            │
│         │  Decoder        │                            │
│         └─────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Start Simple**: Tabular + one other modality (e.g., images)
2. **Use Existing Libraries**: Leverage `diffusers` for diffusion core
3. **Incremental Development**: Build one component at a time
4. **Test with Real Data**: Validate on actual multi-modal physics data

---

*See `MULTIMODAL_DIFFUSION_GUIDE.md` for detailed implementation guide.*




