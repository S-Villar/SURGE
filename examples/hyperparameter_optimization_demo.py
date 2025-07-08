"""
Hyperparameter Optimization Demo using Optuna with BoTorchSampler
Demonstrates Bayesian optimization for RandomForestRegressor with convergence analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import BoTorchSampler
try:
    from optuna.integration import BoTorchSampler
    BOTORCH_AVAILABLE = True
    print("BoTorchSampler is available")
except ImportError:
    BOTORCH_AVAILABLE = False
    print("BoTorchSampler not available, will use TPESampler instead")
    print("To install: pip install botorch")


def generate_sample_data(n_samples=1000, n_features=10, noise_level=0.1, random_state=42):
    """
    Generate sample regression data for testing.
    """
    np.random.seed(random_state)
    
    # Create feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Create a non-linear target function
    y = (2 * X[:, 0] + 
         1.5 * X[:, 1]**2 + 
         0.8 * X[:, 2] * X[:, 3] + 
         0.5 * np.sin(X[:, 4] * 3) +
         0.3 * X[:, 5] * X[:, 6] * X[:, 7] +
         noise_level * np.random.randn(n_samples))
    
    return X, y


def objective_function(trial, X_train, y_train, X_val, y_val, cv_folds=3):
    """
    Objective function for Optuna optimization.
    """
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    # Create and train model
    model = RandomForestRegressor(**params)
    
    if cv_folds > 1:
        # Use cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv_folds, 
            scoring='r2',
            n_jobs=-1
        )
        return cv_scores.mean()
    else:
        # Use simple train/validation split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return r2_score(y_val, y_pred)


def run_optimization_study(X_train, y_train, X_val, y_val, 
                          sampler_type="botorch", n_trials=100, cv_folds=3):
    """
    Run Optuna optimization study with specified sampler.
    """
    # Create sampler
    if sampler_type == "botorch" and BOTORCH_AVAILABLE:
        sampler = BoTorchSampler()
        print(f"Using BoTorchSampler for Bayesian optimization")
    else:
        sampler = TPESampler(seed=42)
        print(f"Using TPESampler")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize R²
        sampler=sampler
    )
    
    # Define objective with data
    def objective(trial):
        return objective_function(trial, X_train, y_train, X_val, y_val, cv_folds)
    
    # Run optimization
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    return study


def analyze_convergence(study, save_plots=True):
    """
    Analyze and visualize the convergence of the optimization.
    """
    # Extract optimization history
    trials = study.trials
    trial_numbers = [trial.number for trial in trials]
    values = [trial.value for trial in trials if trial.value is not None]
    
    # Calculate running maximum
    running_max = []
    current_max = -np.inf
    for value in values:
        if value > current_max:
            current_max = value
        running_max.append(current_max)
    
    # Create convergence plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Trial values over time
    axes[0, 0].plot(trial_numbers[:len(values)], values, 'b-', alpha=0.6, label='Trial R²')
    axes[0, 0].plot(trial_numbers[:len(values)], running_max, 'r-', linewidth=2, label='Best R²')
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Optimization Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Running maximum improvement
    improvement = np.diff([0] + running_max)
    axes[0, 1].bar(range(len(improvement)), improvement, alpha=0.7)
    axes[0, 1].set_xlabel('Trial Number')
    axes[0, 1].set_ylabel('R² Improvement')
    axes[0, 1].set_title('Per-Trial Improvement')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Convergence rate
    convergence_rate = []
    for i in range(1, len(running_max)):
        if i <= 10:
            rate = (running_max[i] - running_max[0]) / i
        else:
            rate = (running_max[i] - running_max[i-10]) / 10
        convergence_rate.append(rate)
    
    axes[1, 0].plot(range(1, len(convergence_rate) + 1), convergence_rate, 'g-')
    axes[1, 0].set_xlabel('Trial Number')
    axes[1, 0].set_ylabel('Convergence Rate')
    axes[1, 0].set_title('Convergence Rate (10-trial window)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of trial scores
    axes[1, 1].hist(values, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(running_max[-1], color='red', linestyle='--', 
                       label=f'Best R²: {running_max[-1]:.4f}')
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Trial Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('hyperparameter_optimization_convergence.png', dpi=300, bbox_inches='tight')
        print("Convergence plots saved as 'hyperparameter_optimization_convergence.png'")
    
    plt.show()
    
    # Print convergence statistics
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    print(f"Total trials: {len(values)}")
    print(f"Best R² score: {running_max[-1]:.6f}")
    print(f"Initial R² score: {values[0]:.6f}")
    print(f"Total improvement: {running_max[-1] - values[0]:.6f}")
    print(f"Final 10-trial improvement: {running_max[-1] - running_max[-11]:.6f}")
    print(f"Convergence achieved (< 0.001 improvement in last 10 trials): {improvement[-10:].sum() < 0.001}")
    
    return {
        'trial_numbers': trial_numbers[:len(values)],
        'values': values,
        'running_max': running_max,
        'improvement': improvement,
        'convergence_rate': convergence_rate
    }


def compare_samplers(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Compare BoTorchSampler vs TPESampler performance.
    """
    results = {}
    
    # Test TPE sampler
    print("\n" + "="*50)
    print("RUNNING TPE SAMPLER")
    print("="*50)
    study_tpe = run_optimization_study(
        X_train, y_train, X_val, y_val, 
        sampler_type="tpe", n_trials=n_trials, cv_folds=3
    )
    results['tpe'] = study_tpe
    
    # Test BoTorch sampler if available
    if BOTORCH_AVAILABLE:
        print("\n" + "="*50)
        print("RUNNING BOTORCH SAMPLER")
        print("="*50)
        study_botorch = run_optimization_study(
            X_train, y_train, X_val, y_val, 
            sampler_type="botorch", n_trials=n_trials, cv_folds=3
        )
        results['botorch'] = study_botorch
        
        # Compare results
        print("\n" + "="*60)
        print("SAMPLER COMPARISON")
        print("="*60)
        print(f"TPE Best R²: {study_tpe.best_value:.6f}")
        print(f"BoTorch Best R²: {study_botorch.best_value:.6f}")
        print(f"BoTorch improvement: {study_botorch.best_value - study_tpe.best_value:.6f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 5))
        
        # Extract values
        tpe_values = [t.value for t in study_tpe.trials if t.value is not None]
        botorch_values = [t.value for t in study_botorch.trials if t.value is not None]
        
        # Running maximums
        tpe_max = []
        botorch_max = []
        tpe_current = -np.inf
        botorch_current = -np.inf
        
        for val in tpe_values:
            if val > tpe_current:
                tpe_current = val
            tpe_max.append(tpe_current)
            
        for val in botorch_values:
            if val > botorch_current:
                botorch_current = val
            botorch_max.append(botorch_current)
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(tpe_max)), tpe_max, 'b-', linewidth=2, label='TPE Sampler')
        plt.plot(range(len(botorch_max)), botorch_max, 'r-', linewidth=2, label='BoTorch Sampler')
        plt.xlabel('Trial Number')
        plt.ylabel('Best R² Score')
        plt.title('Sampler Comparison: Best Score Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(tpe_values)), tpe_values, 'b.', alpha=0.6, label='TPE Trials')
        plt.plot(range(len(botorch_values)), botorch_values, 'r.', alpha=0.6, label='BoTorch Trials')
        plt.xlabel('Trial Number')
        plt.ylabel('R² Score')
        plt.title('Sampler Comparison: All Trial Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sampler_comparison.png', dpi=300, bbox_inches='tight')
        print("Sampler comparison saved as 'sampler_comparison.png'")
        plt.show()
    
    return results


def evaluate_best_model(study, X_train, y_train, X_test, y_test):
    """
    Evaluate the best model found during optimization.
    """
    print("\n" + "="*60)
    print("BEST MODEL EVALUATION")
    print("="*60)
    
    # Get best parameters
    best_params = study.best_params
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Train best model
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"  Training R²: {train_r2:.6f}")
    print(f"  Test R²: {test_r2:.6f}")
    print(f"  Training MSE: {train_mse:.6f}")
    print(f"  Test MSE: {test_mse:.6f}")
    
    # Feature importance
    feature_importance = best_model.feature_importances_
    print(f"\nFeature Importance (top 5):")
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i in range(min(5, len(feature_importance))):
        idx = sorted_idx[i]
        print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    return best_model, {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'feature_importance': feature_importance
    }


def main():
    """
    Main function to run the complete hyperparameter optimization demo.
    """
    print("🚀 HYPERPARAMETER OPTIMIZATION DEMO")
    print("Using RandomForestRegressor with Optuna + BoTorchSampler")
    print("="*60)
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, n_features=10, noise_level=0.1)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    print(f"Data shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Run optimization
    sampler_type = "botorch" if BOTORCH_AVAILABLE else "tpe"
    study = run_optimization_study(
        X_train, y_train, X_val, y_val, 
        sampler_type=sampler_type, n_trials=100, cv_folds=3
    )
    
    # Analyze convergence
    print("\nAnalyzing convergence...")
    convergence_data = analyze_convergence(study, save_plots=True)
    
    # Evaluate best model
    best_model, performance = evaluate_best_model(study, X_train, y_train, X_test, y_test)
    
    # Compare samplers if BoTorch is available
    if BOTORCH_AVAILABLE:
        print("\nComparing samplers...")
        comparison_results = compare_samplers(X_train, y_train, X_val, y_val, n_trials=50)
    
    print("\n🎉 Demo completed successfully!")
    return study, convergence_data, best_model, performance


if __name__ == "__main__":
    # Install required packages if needed
    try:
        study, convergence_data, best_model, performance = main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nTo install required packages:")
        print("  pip install optuna")
        print("  pip install botorch  # For BoTorchSampler")
        print("  pip install matplotlib  # For plots")
