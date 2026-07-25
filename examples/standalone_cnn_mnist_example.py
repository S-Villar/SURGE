#!/usr/bin/env python3
"""
Standalone CNN example for parameter-to-image prediction using MNIST.

This script demonstrates a CNN that reconstructs images from compressed parameter vectors.
It uses PyTorch directly without SURGE dependencies to avoid import issues.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available for SSIM computation. Install with: pip install scikit-image")


class PyTorchCNN(nn.Module):
    """CNN with MLP encoder and transposed CNN decoder for parameter-to-image prediction."""

    def __init__(self, input_dim: int, hidden_layers: list[int], latent_dim: int, output_shape: tuple[int, int]):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Encoder: MLP that maps parameter vector to latent space
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: Transposed CNN that maps latent vector to 2D image
        # latent_dim -> 7x7 feature map -> upsample to 28x28
        self.decoder = nn.Sequential(
            # Reshape latent to 7x7 feature map (latent_dim must be 7*7*channels)
            nn.Unflatten(1, (latent_dim // 49, 7, 7)),  # Assuming latent_dim divisible by 49

            # Transposed conv layers to upsample
            nn.ConvTranspose2d(latent_dim // 49, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


class CNNTrainer:
    """Simple trainer for the PyTorch CNN."""

    def __init__(self, model: PyTorchCNN, learning_rate: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 25,
            batch_size: int = 64, patience: int = 5, log_progress: bool = True):

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)

            if log_progress and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if log_progress:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy()


def load_mnist_like_images(n_samples: int = 2000, random_state: int = 42) -> np.ndarray:
    """Load MNIST-like images, falling back to sklearn digits if needed."""
    try:
        # Try to fetch MNIST from OpenML
        print("Fetching MNIST from OpenML...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        images = mnist.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        print(f"Loaded {len(images)} MNIST images")

    except Exception as e:
        print(f"Failed to load MNIST from OpenML: {e}")
        print("Falling back to sklearn digits dataset...")
        digits = load_digits()
        images = digits.data.astype(np.float32) / 16.0  # Normalize to [0, 1]
        print(f"Loaded {len(images)} digits images")

    # Select subset and reshape to 28x28
    if len(images) > n_samples:
        np.random.seed(random_state)
        indices = np.random.choice(len(images), n_samples, replace=False)
        images = images[indices]

    # Ensure 28x28 shape
    if images.shape[1] == 784:  # MNIST flattened
        images = images.reshape(-1, 28, 28)
    elif images.shape[1] == 64:  # digits dataset
        # Pad digits to 28x28 (they're 8x8)
        padded = np.zeros((len(images), 28, 28), dtype=np.float32)
        padded[:, 10:18, 10:18] = images.reshape(-1, 8, 8)
        images = padded

    return images


def build_parameter_features(images: np.ndarray, n_components: int = 10) -> tuple[np.ndarray, PCA]:
    """Compress images into parameter vectors using PCA."""
    # Flatten images for PCA
    X_flat = images.reshape(len(images), -1)

    # Fit PCA to compress to lower dimension
    pca = PCA(n_components=n_components, random_state=42)
    X_params = pca.fit_transform(X_flat)

    print(f"Compressed {X_flat.shape[1]}D images to {n_components}D parameter vectors")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:3]}...")

    return X_params, pca


def main():
    """Main training and evaluation function."""
    print("Loading MNIST-like images...")
    images = load_mnist_like_images(n_samples=2000)

    print("Building parameter features via PCA...")
    X, pca = build_parameter_features(images, n_components=10)

    # Normalize parameters
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Target is the original images (normalized)
    y = images.reshape(len(images), 1, 28, 28)  # Add channel dimension

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} parameters -> {y.shape[1:]} images")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PyTorchCNN(
        input_dim=X.shape[1],
        hidden_layers=[128, 64],
        latent_dim=49,  # 7*7 for decoder
        output_shape=(28, 28)
    )

    trainer = CNNTrainer(model, learning_rate=1e-3, device=device)

    print("Training CNN surrogate on the parameter-to-image task...")
    trainer.fit(X_train, y_train, n_epochs=25, batch_size=64, patience=5, log_progress=True)

    print("Evaluating on test set...")
    y_pred = trainer.predict(X_test)
    print(f"Prediction shape: {y_pred.shape}")

    mse = np.mean((y_pred.reshape(y_pred.shape[0], -1) - y_test.reshape(y_test.shape[0], -1)) ** 2)
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

    print("\nExample finished successfully. The CNN reconstructed images from parameter vectors.")


if __name__ == "__main__":
    main()