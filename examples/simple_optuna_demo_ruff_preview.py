"""Simple Hyperparameter Optimization Demo.

This module demonstrates hyperparameter optimization for RandomForestRegressor
using Optuna with basic convergence visualization.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Test basic imports first
print("Testing imports...")
try:
    import optuna

    print("✅ Optuna imported successfully")
except ImportError as e:
    print(f"❌ Optuna import failed: {e}")
    sys.exit(1)

try:
    from optuna.integration import BoTorchSampler

    print("✅ BoTorchSampler imported successfully")
    BOTORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ BoTorchSampler not available: {e}")
    BOTORCH_AVAILABLE = False

print(f"BoTorch available: {BOTORCH_AVAILABLE}")

# Global variables for the objective function
X_train: NDArray[np.floating[Any]] | None = None
y_train: NDArray[np.floating[Any]] | None = None


def generate_test_data() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate synthetic regression data for testing.
    
    Returns:
        tuple: Feature matrix X and target vector y
    """
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 2 * X[:, 0] + X[:, 1] ** 2 + 0.5 * X[:, 2] + 0.1 * np.random.randn(200)
    return X, y


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object for hyperparameter suggestions
        
    Returns:
        Mean cross-validation R² score
    """
    if X_train is None or y_train is None:
        msg = "Training data not initialized"
        raise ValueError(msg)
        
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": 42,
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
    return float(scores.mean())


def plot_convergence(trial_values: list[float]) -> None:
    """Create and save convergence plots.
    
    Args:
        trial_values: List of trial objective values
    """
    plt.figure(figsize=(10, 4))

    # Plot all trial scores
    plt.subplot(1, 2, 1)
    plt.plot(trial_values, "b-o", alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("R² Score")
    plt.title("Trial Scores")
    plt.grid(True, alpha=0.3)

    # Calculate and plot running maximum
    running_max = []
    current_max = -np.inf
    for val in trial_values:
        if val > current_max:
            current_max = val
        running_max.append(current_max)

    plt.subplot(1, 2, 2)
    plt.plot(running_max, "r-o", alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("Best R² Score")
    plt.title("Best Score Progress")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simple_optimization_demo.png", dpi=150)
    print("\nPlot saved as 'simple_optimization_demo.png'")
    plt.show()


def main() -> None:
    """Run the hyperparameter optimization demo."""
    global X_train, y_train
    
    print("\n🚀 Running Simple Hyperparameter Optimization Demo")

    # Generate data
    X, y = generate_test_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Run optimization
    print("\nRunning optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print(f"\nBest R² score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Create convergence plots
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    plot_convergence(trial_values)

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
