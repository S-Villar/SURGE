#!/usr/bin/env python3
"""
Example: train SURGE `pytorch.cnn` on an open-source image dataset surrogate task.

This script loads MNIST via sklearn's OpenML interface (falling back to the built-in
sklearn digits dataset if needed), compresses each image into a low-dimensional
parameter vector via PCA, and then trains SURGE's CNN adapter to reconstruct the
image from those parameters.

This demonstrates the parameter->image surrogate use case with a real image dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml, load_digits

# Ensure the checkout root is importable when running from examples/.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Import only the PyTorch CNN model directly to avoid GPflow/TensorFlow issues
from surge.model.pytorch_impl import PyTorchCNNModel
from surge.model.pytorch import _TORCH_CNN_PROFILE
from surge.model.base import BaseModelAdapter


class SimplePyTorchCNNAdapter(BaseModelAdapter):
    """Simple PyTorch CNN adapter for the example."""

    name = "pytorch.cnn"
    backend = "pytorch"
    uses_internal_preprocessing = True
    handles_output_scaling = True
    resource_profile = _TORCH_CNN_PROFILE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = None

    def _build_model(self, **kwargs: Any) -> PyTorchCNNModel:
        return PyTorchCNNModel(**kwargs)

    def fit(self, X: Any, y: Any, *, X_val: Any = None, y_val: Any = None, finetune: bool = False, **kwargs: Any) -> Any:
        if self._model is None and not finetune:
            self._model = self._build_model(**self.params)
        return self._model.fit(X, y, X_val=X_val, y_val=y_val, finetune=finetune)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available for SSIM computation. Install with: pip install scikit-image")


def load_mnist_like_images(n_samples: int = 2000, random_state: int = 42) -> np.ndarray:
    """Load 28x28 image data from MNIST or fall back to sklearn digits."""
    try:
        print("Loading MNIST from OpenML...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        images = mnist.data.astype(np.float32) / 255.0
        images = images.reshape(-1, 28, 28)
    except Exception as exc:  # pragma: no cover
        print("Failed to load MNIST from OpenML, falling back to sklearn digits:", exc)
        digits = load_digits()
        images = digits.images.astype(np.float32) / 16.0
        images = np.repeat(np.repeat(images, 4, axis=1), 4, axis=2)[:, :28, :28]

    rng = np.random.default_rng(random_state)
    if images.shape[0] > n_samples:
        indices = rng.choice(images.shape[0], size=n_samples, replace=False)
        images = images[indices]

    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}.")
    return images


def build_parameter_features(images: np.ndarray, n_components: int = 32) -> np.ndarray:
    """Create a low-dimensional parameter vector from images using PCA."""
    flattened = images.reshape(images.shape[0], -1)
    scaler = StandardScaler()
    flattened_scaled = scaler.fit_transform(flattened)

    pca = PCA(n_components=n_components, random_state=42)
    params = pca.fit_transform(flattened_scaled)
    print(f"Built {params.shape[1]}-dimensional parameter features from images.")
    return params


def main() -> None:
    images = load_mnist_like_images(n_samples=2000)
    X = build_parameter_features(images, n_components=32)
    y = images.reshape(images.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = SimplePyTorchCNNAdapter(
        hidden_layers=[128, 64],
        latent_dim=32,
        output_shape=(28, 28),
        learning_rate=1e-3,
        n_epochs=25,
        batch_size=64,
        patience=5,
        log_progress=True,
    )

    print("Training SURGE pytorch.cnn on the surrogate MNIST task...")
    model.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    print(f"Prediction shape: {y_pred.shape}")

    mse = np.mean((y_pred.reshape(y_pred.shape[0], -1) - y_test) ** 2)
    print(f"Test MSE: {mse:.6f}")

    # Demonstrate the output format and value range.
    print(f"Output dtype: {y_pred.dtype}")
    print(f"Output range: min={y_pred.min():.4f}, max={y_pred.max():.4f}")

    # Compute SSIM scores if scikit-image is available
    if HAS_SKIMAGE:
        print("\nComputing SSIM scores...")
        ssim_scores = []
        y_test_images = y_test.reshape(-1, 28, 28)
        y_pred_images = y_pred.reshape(-1, 28, 28)

        for i in range(len(y_test_images)):
            # SSIM expects values in [0, 1] range, which we already have
            score = ssim(y_test_images[i], y_pred_images[i], data_range=1.0)
            ssim_scores.append(score)

        ssim_scores = np.array(ssim_scores)
        best_idx = np.argmax(ssim_scores)
        worst_idx = np.argmin(ssim_scores)

        print(f"SSIM Statistics:")
        print(f"  Mean SSIM: {ssim_scores.mean():.4f}")
        print(f"  Std SSIM: {ssim_scores.std():.4f}")
        print(f"  Min SSIM: {ssim_scores.min():.4f}")
        print(f"  Max SSIM: {ssim_scores.max():.4f}")
        print(f"  Median SSIM: {np.median(ssim_scores):.4f}")

        print(f"\nBest SSIM: {ssim_scores[best_idx]:.4f} (sample {best_idx})")
        print(f"Worst SSIM: {ssim_scores[worst_idx]:.4f} (sample {worst_idx})")

        # Show parameter vectors for best/worst cases
        print(f"\nBest case parameters: {X_test[best_idx][:5]}...")  # Show first 5 params
        print(f"Worst case parameters: {X_test[worst_idx][:5]}...")
    else:
        print("\nSkipping SSIM computation (scikit-image not available)")

    assert y_pred.shape == (X_test.shape[0], 1, 28, 28)
    assert np.all(y_pred >= 0.0) and np.all(y_pred <= 1.0)

    print("\nExample finished successfully. The SURGE pytorch.cnn model reconstructed images from parameter vectors.")


if __name__ == "__main__":
    main()
