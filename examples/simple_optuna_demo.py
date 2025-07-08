"""
Simple Hyperparameter Optimization Demo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score

# Test basic imports first
print("Testing imports...")
try:
    import optuna
    print("✅ Optuna imported successfully")
except ImportError as e:
    print(f"❌ Optuna import failed: {e}")
    exit(1)

try:
    from optuna.integration import BoTorchSampler
    print("✅ BoTorchSampler imported successfully")
    BOTORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ BoTorchSampler not available: {e}")
    BOTORCH_AVAILABLE = False

print(f"BoTorch available: {BOTORCH_AVAILABLE}")

# Generate simple test data
def generate_test_data():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 2*X[:, 0] + X[:, 1]**2 + 0.5*X[:, 2] + 0.1*np.random.randn(200)
    return X, y

# Simple objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    return scores.mean()

if __name__ == "__main__":
    print("\n🚀 Running Simple Hyperparameter Optimization Demo")
    
    # Generate data
    X, y = generate_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Run optimization
    print("\nRunning optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print(f"\nBest R² score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Simple convergence plot
    trial_values = [trial.value for trial in study.trials]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trial_values, 'b-o', alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel('R² Score')
    plt.title('Trial Scores')
    plt.grid(True, alpha=0.3)
    
    # Running maximum
    running_max = []
    current_max = -np.inf
    for val in trial_values:
        if val > current_max:
            current_max = val
        running_max.append(current_max)
    
    plt.subplot(1, 2, 2)
    plt.plot(running_max, 'r-o', alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel('Best R² Score')
    plt.title('Best Score Progress')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_optimization_demo.png', dpi=150)
    print("\nPlot saved as 'simple_optimization_demo.png'")
    plt.show()
    
    print("\n✅ Demo completed successfully!")
