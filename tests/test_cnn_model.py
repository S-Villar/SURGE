#!/usr/bin/env python3
"""
Test script for PyTorch CNN model in SURGE.

Tests parameter-to-image prediction using synthetic data.
"""

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from surge.models import create_model


def create_synthetic_image_data(n_samples=1000, input_dim=10, image_size=(28, 28)):
    """
    Create synthetic data for parameter-to-image prediction.

    Parameters:
    - n_samples: Number of samples
    - input_dim: Dimension of input parameters
    - image_size: Size of output images (H, W)

    Returns:
    - X: Input parameters (n_samples, input_dim)
    - y: Output images (n_samples, H*W)
    """
    # Generate random input parameters
    X, _ = make_regression(n_samples=n_samples, n_features=input_dim, noise=0.1, random_state=42)

    # Create synthetic images based on input parameters
    # Simple pattern: each parameter influences different parts of the image
    H, W = image_size
    y = np.zeros((n_samples, H * W))

    for i in range(n_samples):
        # Create a pattern where parameters control intensity in different regions
        img = np.zeros((H, W))

        # Use first few parameters to create patterns
        for j in range(min(input_dim, 4)):
            # Create circular patterns at different positions
            center_x = int((j % 2) * H * 0.75 + H * 0.25)
            center_y = int((j // 2) * W * 0.75 + W * 0.25)
            radius = max(3, int(abs(X[i, j]) * 5))

            y_coords, x_coords = np.ogrid[:H, :W]
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            mask = dist_from_center <= radius
            intensity = (X[i, j] + 3) / 6  # Normalize to [0, 1]
            img[mask] += intensity

        # Clip to [0, 1] range
        img = np.clip(img, 0, 1)
        y[i] = img.flatten()

    return X, y


def test_cnn_model():
    """Test the PyTorch CNN model with synthetic data."""
    print("Testing PyTorch CNN model...")

    # Create synthetic data
    print("Creating synthetic data...")
    X, y = create_synthetic_image_data(n_samples=500, input_dim=5, image_size=(28, 28))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Create CNN model
    print("Creating CNN model...")
    model = create_model('pytorch.cnn',
                        hidden_layers=[64, 32],
                        latent_dim=16,
                        output_shape=(28, 28),
                        learning_rate=1e-3,
                        n_epochs=10,  # Small number for quick test
                        batch_size=32,
                        patience=5)

    # Fit the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    print(f"Predictions shape: {y_pred.shape}")

    # Basic validation
    assert y_pred.shape[0] == X_test.shape[0], f"Wrong prediction batch size: {y_pred.shape[0]} != {X_test.shape[0]}"
    assert y_pred.shape[1] == 1, f"Wrong number of channels: {y_pred.shape[1]} != 1"
    assert y_pred.shape[2] == 28 and y_pred.shape[3] == 28, f"Wrong image size: {y_pred.shape[2:]} != (28, 28)"

    # Check predictions are in reasonable range (should be close to [0, 1] after sigmoid)
    assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "Predictions should be in [0, 1] range"

    print("✓ CNN model test passed!")

    # Calculate basic metrics
    y_test_reshaped = y_test.reshape(-1, 1, 28, 28)
    mse = np.mean((y_pred - y_test_reshaped) ** 2)
    mae = np.mean(np.abs(y_pred - y_test_reshaped))

    print(".6f")
    print(".6f")

    return model


def test_cnn_model_registry():
    """Test that CNN model is properly registered."""
    print("\nTesting CNN model registry...")

    from surge.models import list_models, MODEL_REGISTRY

    models = list_models()
    print(f"Available models: {list(models.keys())}")

    # Check if CNN is registered
    cnn_keys = [k for k in models.keys() if 'cnn' in k.lower()]
    assert len(cnn_keys) > 0, f"No CNN models found in registry: {list(models.keys())}"

    print(f"✓ CNN model registered with keys: {cnn_keys}")

    # Test creation
    model = MODEL_REGISTRY.create('pytorch.cnn')
    assert model.name == 'pytorch.cnn', f"Wrong model name: {model.name}"

    print("✓ CNN model creation test passed!")


if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch
        print("PyTorch version:", torch.__version__)
    except ImportError:
        print("PyTorch not available, skipping CNN tests")
        exit(0)

    test_cnn_model_registry()
    test_cnn_model()
    print("\n🎉 All CNN tests passed!")