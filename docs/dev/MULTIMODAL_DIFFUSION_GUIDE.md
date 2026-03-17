# Building Multi-Modal Surrogates with Joint Diffusion Models in SURGE

## Overview

This guide outlines what would be necessary to extend SURGE to support **multi-modal surrogates based on joint diffusion models**. This is a significant architectural extension that would enable:

- **Multi-modal inputs**: Tabular data, images, sequences, point clouds, etc.
- **Diffusion-based generation**: Probabilistic generation of outputs using diffusion processes
- **Joint modeling**: Learning shared representations across modalities
- **Uncertainty quantification**: Natural uncertainty from diffusion process

---

## Current SURGE Architecture (Tabular-Only)

```
┌─────────────────────────────────────────┐
│         SurrogateDataset                 │
│  - Loads CSV/Parquet/Pickle/HDF5        │
│  - Auto-detects input/output columns     │
│  - Returns pandas DataFrame              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         SurrogateEngine                 │
│  - Splits data (train/val/test)        │
│  - Standardizes inputs/outputs          │
│  - Passes numpy arrays to adapters     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         BaseModelAdapter                │
│  - fit(X: ndarray, y: ndarray)          │
│  - predict(X: ndarray) -> ndarray       │
└─────────────────────────────────────────┘
```

**Current Limitations**:
- Assumes tabular data (DataFrame → numpy arrays)
- Single modality (all inputs are numeric features)
- Deterministic or simple UQ (not generative)

---

## Required Extensions for Multi-Modal Diffusion

### 1. Multi-Modal Dataset Layer

#### 1.1 New Dataset Class: `MultiModalDataset`

```python
# surge/dataset/multimodal.py

from typing import Dict, List, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

class MultiModalDataset:
    """
    Dataset loader for multi-modal data.
    
    Supports:
    - Tabular: pandas DataFrame columns
    - Images: file paths or arrays
    - Sequences: time series, text, etc.
    - Point clouds: 3D coordinates
    """
    
    def __init__(self):
        self.modalities: Dict[str, Any] = {}
        self.modality_types: Dict[str, str] = {}  # 'tabular', 'image', 'sequence', etc.
        self.output_modalities: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_modality(
        self,
        name: str,
        data: Union[pd.DataFrame, np.ndarray, List[Path], Dict],
        modality_type: str,
        is_output: bool = False,
    ):
        """
        Add a data modality.
        
        Parameters
        ----------
        name : str
            Modality name (e.g., 'tabular_params', 'plasma_images', 'time_series')
        data : DataFrame, ndarray, List[Path], or Dict
            The actual data
        modality_type : str
            Type: 'tabular', 'image', 'sequence', 'point_cloud', etc.
        is_output : bool
            Whether this is an output (target) modality
        """
        self.modalities[name] = data
        self.modality_types[name] = modality_type
        if is_output:
            self.output_modalities.append(name)
    
    def get_modality(self, name: str):
        """Retrieve a specific modality."""
        return self.modalities.get(name)
    
    def get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Get a batch of multi-modal data.
        
        Returns
        -------
        dict mapping modality names to tensors
        """
        batch = {}
        for name, data in self.modalities.items():
            mod_type = self.modality_types[name]
            
            if mod_type == 'tabular':
                batch[name] = torch.from_numpy(data.iloc[indices].values).float()
            elif mod_type == 'image':
                # Load images if paths, or use arrays
                images = [self._load_image(data[i]) for i in indices]
                batch[name] = torch.stack(images)
            elif mod_type == 'sequence':
                batch[name] = torch.from_numpy(data[indices]).float()
            # ... handle other types
        
        return batch
```

#### 1.2 Multi-Modal Data Preprocessing

```python
# surge/preprocessing/multimodal.py

class MultiModalPreprocessor:
    """
    Preprocessing pipeline for multi-modal data.
    
    Each modality may need different preprocessing:
    - Tabular: StandardScaler, normalization
    - Images: Resize, normalize, augment
    - Sequences: Padding, normalization
    """
    
    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.transforms: Dict[str, Any] = {}
    
    def fit(self, dataset: MultiModalDataset):
        """Fit preprocessors for each modality."""
        for name, mod_type in dataset.modality_types.items():
            if mod_type == 'tabular':
                from sklearn.preprocessing import StandardScaler
                self.scalers[name] = StandardScaler()
                self.scalers[name].fit(dataset.modalities[name])
            elif mod_type == 'image':
                # Compute mean/std for normalization
                self.transforms[name] = self._compute_image_stats(dataset.modalities[name])
            # ... other modalities
    
    def transform(self, dataset: MultiModalDataset) -> MultiModalDataset:
        """Apply preprocessing transforms."""
        # Create transformed dataset
        # ...
        return transformed_dataset
```

---

### 2. Diffusion Model Architecture

#### 2.1 Base Diffusion Model

```python
# surge/models/diffusion/base.py

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class DiffusionModel(nn.Module):
    """
    Base class for diffusion models.
    
    Implements DDPM (Denoising Diffusion Probabilistic Model) or
    score-based diffusion.
    """
    
    def __init__(
        self,
        data_dim: int,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        **kwargs
    ):
        super().__init__()
        self.timesteps = timesteps
        self.data_dim = data_dim
        
        # Define noise schedule
        self.betas = self._get_beta_schedule(beta_schedule, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _get_beta_schedule(self, schedule: str, timesteps: int):
        """Generate noise schedule."""
        if schedule == "linear":
            return torch.linspace(0.0001, 0.02, timesteps)
        # ... other schedules
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion: add noise to data.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = torch.gather(
            torch.sqrt(self.alphas_cumprod), 0, t
        ).reshape(-1, *([1] * (len(x_start.shape) - 1)))
        
        sqrt_one_minus_alphas_cumprod_t = torch.gather(
            torch.sqrt(1.0 - self.alphas_cumprod), 0, t
        ).reshape(-1, *([1] * (len(x_start.shape) - 1)))
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        """
        Reverse diffusion: denoise one step.
        
        p(x_{t-1} | x_t) = denoising step
        """
        # Predict noise
        predicted_noise = model(x, t)
        
        # Compute x_{t-1}
        alpha_t = torch.gather(self.alphas, 0, t).reshape(-1, *([1] * (len(x.shape) - 1)))
        alpha_cumprod_t = torch.gather(self.alphas_cumprod, 0, t).reshape(-1, *([1] * (len(x.shape) - 1)))
        beta_t = torch.gather(self.betas, 0, t).reshape(-1, *([1] * (len(x.shape) - 1)))
        
        pred_x_start = (x - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        posterior_mean = (
            torch.sqrt(alpha_t) * beta_t / (1.0 - alpha_cumprod_t) * x +
            torch.sqrt(1.0 - beta_t) * (1.0 - alpha_cumprod_t) / (1.0 - alpha_cumprod_t) * pred_x_start
        )
        posterior_variance = beta_t * (1.0 - alpha_cumprod_t) / (1.0 - alpha_cumprod_t)
        
        if t[0] == 0:
            return pred_x_start
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...], device: torch.device):
        """
        Full reverse diffusion: generate sample from noise.
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Denoise step by step
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        
        return x
```

#### 2.2 Multi-Modal Diffusion Model

```python
# surge/models/diffusion/multimodal.py

class MultiModalDiffusionModel(nn.Module):
    """
    Joint diffusion model for multi-modal data.
    
    Architecture:
    1. Multi-modal encoders (one per input modality)
    2. Fusion layer (combines modalities)
    3. Diffusion denoising network
    4. Multi-modal decoders (one per output modality)
    """
    
    def __init__(
        self,
        input_modalities: Dict[str, Dict],  # {name: {type, dim, ...}}
        output_modalities: Dict[str, Dict],
        latent_dim: int = 512,
        timesteps: int = 1000,
        **kwargs
    ):
        super().__init__()
        
        # Encoders for each input modality
        self.input_encoders = nn.ModuleDict()
        for name, config in input_modalities.items():
            if config['type'] == 'tabular':
                self.input_encoders[name] = TabularEncoder(
                    input_dim=config['dim'],
                    hidden_dims=[256, 128],
                    output_dim=latent_dim
                )
            elif config['type'] == 'image':
                self.input_encoders[name] = ImageEncoder(
                    channels=config.get('channels', 3),
                    output_dim=latent_dim
                )
            # ... other modalities
        
        # Fusion layer
        self.fusion = FusionLayer(
            input_dims=[latent_dim] * len(input_modalities),
            output_dim=latent_dim
        )
        
        # Diffusion denoising network
        self.denoising_net = UNet(
            in_channels=latent_dim,
            out_channels=latent_dim,
            time_embed_dim=128
        )
        
        # Decoders for each output modality
        self.output_decoders = nn.ModuleDict()
        for name, config in output_modalities.items():
            if config['type'] == 'tabular':
                self.output_decoders[name] = TabularDecoder(
                    input_dim=latent_dim,
                    hidden_dims=[128, 256],
                    output_dim=config['dim']
                )
            elif config['type'] == 'image':
                self.output_decoders[name] = ImageDecoder(
                    input_dim=latent_dim,
                    channels=config.get('channels', 3),
                    output_size=config.get('size', (64, 64))
                )
            # ... other modalities
    
    def encode_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode all input modalities into shared latent space."""
        encoded = []
        for name, encoder in self.input_encoders.items():
            encoded.append(encoder(inputs[name]))
        return self.fusion(torch.stack(encoded))
    
    def decode_outputs(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent representation to output modalities."""
        outputs = {}
        for name, decoder in self.output_decoders.items():
            outputs[name] = decoder(latent)
        return outputs
    
    def forward(self, inputs: Dict[str, torch.Tensor], t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward pass: predict noise at timestep t.
        
        Parameters
        ----------
        inputs : dict
            Multi-modal input data
        t : torch.Tensor
            Diffusion timestep
        noise : torch.Tensor, optional
            Noise to predict (for training)
        """
        # Encode inputs
        latent = self.encode_inputs(inputs)
        
        # Add noise if training
        if noise is not None:
            latent_noisy = self.diffusion.q_sample(latent, t, noise)
        else:
            latent_noisy = latent
        
        # Predict noise
        predicted_noise = self.denoising_net(latent_noisy, t)
        
        return predicted_noise
```

---

### 3. Multi-Modal Adapter for SURGE

#### 3.1 Extended Adapter Interface

```python
# surge/models/adapters/multimodal_diffusion_adapter.py

from ...registry import BaseModelAdapter, MODEL_REGISTRY
from ...dataset.multimodal import MultiModalDataset
from ..diffusion.multimodal import MultiModalDiffusionModel
from ..diffusion.base import DiffusionModel

class MultiModalDiffusionAdapter(BaseModelAdapter):
    """
    SURGE adapter for multi-modal diffusion models.
    
    This adapter extends BaseModelAdapter to handle:
    - Multi-modal inputs (not just numpy arrays)
    - Diffusion-based generation
    - Probabilistic outputs
    """
    
    name = "MultiModalDiffusion"
    backend = "torch"
    supports_uq = True  # Diffusion naturally provides uncertainty
    supports_serialization = True
    
    def __init__(
        self,
        *,
        input_modalities: Dict[str, Dict],
        output_modalities: Dict[str, Dict],
        timesteps: int = 1000,
        latent_dim: int = 512,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        device: Optional[str] = None,
        **params: Any,
    ):
        super().__init__(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            timesteps=timesteps,
            latent_dim=latent_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            device=device,
            **params,
        )
        
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Initialize diffusion process
        self.diffusion = DiffusionModel(
            data_dim=latent_dim,
            timesteps=timesteps,
        )
        
        # Model will be created in fit()
        self.model: Optional[MultiModalDiffusionModel] = None
        self.training_history: List[Dict[str, float]] = []
    
    def fit(self, dataset: MultiModalDataset, **kwargs) -> "MultiModalDiffusionAdapter":
        """
        Train the multi-modal diffusion model.
        
        Note: This signature differs from BaseModelAdapter.fit(X, y)
        because we need multi-modal data, not just numpy arrays.
        """
        # Create model
        self.model = MultiModalDiffusionModel(
            input_modalities=self.params["input_modalities"],
            output_modalities=self.params["output_modalities"],
            latent_dim=self.params["latent_dim"],
            timesteps=self.params["timesteps"],
        ).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
        )
        
        # Training loop
        self.training_history.clear()
        n_samples = len(dataset)
        
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            
            # Create data loader
            dataloader = self._create_dataloader(dataset)
            
            for batch in dataloader:
                # Get inputs and outputs
                inputs = {k: v.to(self.device) for k, v in batch.items() 
                         if k not in dataset.output_modalities}
                outputs = {k: v.to(self.device) for k, v in batch.items() 
                          if k in dataset.output_modalities}
                
                # Encode outputs to latent space
                output_latent = self.model.encode_outputs(outputs)
                
                # Sample random timestep
                t = torch.randint(
                    0, self.params["timesteps"],
                    (batch_size,), device=self.device
                )
                
                # Sample noise
                noise = torch.randn_like(output_latent)
                
                # Add noise
                noisy_latent = self.diffusion.q_sample(output_latent, t, noise)
                
                # Predict noise
                predicted_noise = self.model(inputs, t, noise=None)
                
                # Loss: MSE between predicted and actual noise
                loss = nn.functional.mse_loss(predicted_noise, noise)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch)
            
            avg_loss = epoch_loss / max(1, n_samples)
            self.training_history.append({
                "epoch": epoch,
                "loss": float(avg_loss),
            })
        
        self.mark_fitted()
        return self
    
    def predict(self, inputs: Dict[str, torch.Tensor], n_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Generate predictions using diffusion sampling.
        
        Parameters
        ----------
        inputs : dict
            Multi-modal input data
        n_samples : int
            Number of samples to generate (for uncertainty)
        
        Returns
        -------
        dict mapping output modality names to numpy arrays
        """
        self.ensure_fitted()
        assert self.model is not None
        
        self.model.eval()
        
        # Move inputs to device
        inputs_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) 
            else torch.from_numpy(v).to(self.device)
            for k, v in inputs.items()
        }
        
        # Encode inputs
        input_latent = self.model.encode_inputs(inputs_device)
        
        # Generate samples using diffusion
        all_samples = []
        for _ in range(n_samples):
            # Start from noise
            latent_shape = (1, self.params["latent_dim"])
            latent = torch.randn(latent_shape, device=self.device)
            
            # Add input conditioning (concatenate or use cross-attention)
            conditioned_latent = torch.cat([input_latent, latent], dim=1)
            
            # Reverse diffusion
            generated_latent = self.diffusion.p_sample_loop(
                self.model.denoising_net,
                latent_shape,
                self.device
            )
            
            # Decode to output modalities
            outputs = self.model.decode_outputs(generated_latent)
            all_samples.append(outputs)
        
        # Average over samples (or return distribution)
        if n_samples > 1:
            # Stack and compute mean/variance
            stacked = {k: torch.stack([s[k] for s in all_samples]) 
                      for k in all_samples[0].keys()}
            result = {
                k: {
                    "mean": v.mean(dim=0).cpu().numpy(),
                    "variance": v.var(dim=0).cpu().numpy(),
                }
                for k, v in stacked.items()
            }
        else:
            result = {k: v.cpu().numpy() for k, v in all_samples[0].items()}
        
        return result
    
    def predict_with_uncertainty(self, inputs: Dict[str, torch.Tensor], n_samples: int = 100):
        """
        Generate predictions with uncertainty estimates.
        
        Uses multiple diffusion samples to estimate uncertainty.
        """
        return self.predict(inputs, n_samples=n_samples)
    
    def _create_dataloader(self, dataset: MultiModalDataset):
        """Create PyTorch DataLoader for multi-modal dataset."""
        from torch.utils.data import DataLoader, Dataset
        
        class MultiModalDatasetWrapper(Dataset):
            def __init__(self, mm_dataset):
                self.mm_dataset = mm_dataset
                self.length = len(mm_dataset.modalities[list(mm_dataset.modalities.keys())[0]])
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                return self.mm_dataset.get_batch(np.array([idx]))
        
        return DataLoader(
            MultiModalDatasetWrapper(dataset),
            batch_size=self.params["batch_size"],
            shuffle=True,
        )
```

---

### 4. Engine Extensions

#### 4.1 Multi-Modal Engine

```python
# surge/engine/multimodal.py

class MultiModalSurrogateEngine:
    """
    Extended engine for multi-modal surrogates.
    
    Handles:
    - Multi-modal dataset loading
    - Per-modality preprocessing
    - Multi-modal model training
    - Multi-modal prediction
    """
    
    def __init__(self, **kwargs):
        self.dataset: Optional[MultiModalDataset] = None
        self.preprocessor: Optional[MultiModalPreprocessor] = None
        self.models: List[BaseModelAdapter] = []
    
    def configure_multimodal_dataset(self, dataset: MultiModalDataset):
        """Configure multi-modal dataset."""
        self.dataset = dataset
        
        # Fit preprocessor
        self.preprocessor = MultiModalPreprocessor()
        self.preprocessor.fit(dataset)
        self.dataset = self.preprocessor.transform(dataset)
    
    def train_model(self, adapter: BaseModelAdapter):
        """Train a multi-modal model."""
        if isinstance(adapter, MultiModalDiffusionAdapter):
            adapter.fit(self.dataset)
        else:
            # Fallback to standard adapter (if it supports multi-modal)
            # ...
            pass
```

---

### 5. Workflow Integration

#### 5.1 Extended Workflow Spec

```yaml
# configs/multimodal_diffusion.yaml

dataset:
  type: multimodal
  modalities:
    tabular_params:
      type: tabular
      path: data/inputs/params.csv
      columns: [param1, param2, param3]
      is_input: true
    plasma_images:
      type: image
      path: data/inputs/images/
      size: [64, 64]
      channels: 3
      is_input: true
    output_profiles:
      type: tabular
      path: data/outputs/profiles.csv
      columns: [profile_1, profile_2, ...]
      is_output: true

models:
  - key: torch.multimodal_diffusion
    name: joint_diffusion_surrogate
    params:
      input_modalities:
        tabular_params:
          type: tabular
          dim: 10
        plasma_images:
          type: image
          channels: 3
          size: [64, 64]
      output_modalities:
        output_profiles:
          type: tabular
          dim: 100
      timesteps: 1000
      latent_dim: 512
      epochs: 2000
      batch_size: 32
    hpo:
      enabled: true
      n_trials: 50
      search_space:
        latent_dim:
          type: categorical
          choices: [256, 512, 1024]
        learning_rate:
          type: loguniform
          low: 1e-5
          high: 1e-3
```

---

## Implementation Roadmap

### Phase 1: Multi-Modal Data Layer
- [ ] Implement `MultiModalDataset` class
- [ ] Support for tabular, image, sequence modalities
- [ ] Multi-modal data loaders
- [ ] Per-modality preprocessing pipelines

### Phase 2: Diffusion Model Core
- [ ] Base `DiffusionModel` class (DDPM)
- [ ] Noise schedules (linear, cosine, etc.)
- [ ] Sampling methods (DDPM, DDIM)
- [ ] UNet architecture for denoising

### Phase 3: Multi-Modal Architecture
- [ ] Modality-specific encoders
- [ ] Fusion layers (concatenation, attention, etc.)
- [ ] Modality-specific decoders
- [ ] Conditional diffusion (conditioned on inputs)

### Phase 4: SURGE Integration
- [ ] `MultiModalDiffusionAdapter` implementation
- [ ] Extended `SurrogateEngine` for multi-modal
- [ ] Workflow spec extensions
- [ ] Artifact handling for multi-modal outputs

### Phase 5: Advanced Features
- [ ] Cross-modal attention mechanisms
- [ ] Hierarchical diffusion (coarse-to-fine)
- [ ] Latent space regularization
- [ ] Multi-scale generation

---

## Key Challenges & Solutions

### Challenge 1: Different Data Types
**Solution**: Modality-specific encoders/decoders that map to common latent space

### Challenge 2: Variable-Sized Inputs
**Solution**: Padding, masking, or attention mechanisms

### Challenge 3: Training Stability
**Solution**: Gradient clipping, learning rate scheduling, EMA of model weights

### Challenge 4: Computational Cost
**Solution**: 
- Efficient attention (Flash Attention)
- Reduced timesteps (DDIM sampling)
- Mixed precision training

### Challenge 5: Evaluation Metrics
**Solution**: 
- FID for images
- Profile similarity metrics for physics outputs
- Conditional likelihood for probabilistic evaluation

---

## Example Usage

```python
from surge import MultiModalDataset, MultiModalSurrogateEngine
from surge.models.adapters import MultiModalDiffusionAdapter

# Create multi-modal dataset
dataset = MultiModalDataset()
dataset.add_modality(
    "tabular_params",
    pd.read_csv("params.csv"),
    modality_type="tabular",
    is_output=False
)
dataset.add_modality(
    "plasma_images",
    image_paths,  # List of image paths
    modality_type="image",
    is_output=False
)
dataset.add_modality(
    "output_profiles",
    pd.read_csv("profiles.csv"),
    modality_type="tabular",
    is_output=True
)

# Create engine
engine = MultiModalSurrogateEngine()
engine.configure_multimodal_dataset(dataset)

# Create and train model
adapter = MultiModalDiffusionAdapter(
    input_modalities={
        "tabular_params": {"type": "tabular", "dim": 10},
        "plasma_images": {"type": "image", "channels": 3, "size": (64, 64)},
    },
    output_modalities={
        "output_profiles": {"type": "tabular", "dim": 100},
    },
    timesteps=1000,
    latent_dim=512,
)

engine.train_model(adapter)

# Generate predictions
inputs = {
    "tabular_params": torch.tensor([[1.0, 2.0, ...]]),
    "plasma_images": load_image("test_image.png"),
}
predictions = adapter.predict(inputs, n_samples=10)  # With uncertainty
```

---

## Dependencies

New dependencies would be needed:

```python
# requirements-multimodal.txt
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0  # HuggingFace diffusion library
transformers>=4.30.0  # For attention mechanisms
pillow>=9.0.0  # Image handling
einops>=0.6.0  # Tensor operations
```

---

## Summary

Building multi-modal diffusion surrogates in SURGE requires:

1. **Data Layer**: Multi-modal dataset classes and preprocessing
2. **Model Architecture**: Diffusion models with multi-modal encoders/decoders
3. **Adapter Interface**: Extended adapters for multi-modal inputs
4. **Engine Extensions**: Support for multi-modal workflows
5. **Workflow Integration**: YAML specs and artifact handling

This is a significant extension but follows SURGE's modular architecture, making it feasible to implement incrementally.

---

*This is a design document. Implementation would require careful consideration of specific use cases and performance requirements.*




