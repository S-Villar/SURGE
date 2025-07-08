"""
Comprehensive Hyperparameter Optimization Demo
RandomForestRegressor with Optuna: TPE vs BoTorchSampler comparison
Shows convergence analysis with 100 trials for Bayesian optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.integration import BoTorchSampler
import warnings
import time

warnings.filterwarnings('ignore')

def generate_sample_data(n_samples=1000, n_features=10, noise_level=0.1, random_state=42):
    """Generate sample regression data for testing."""
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

def objective_function(trial, X_train, y_train, cv_folds=3):
    """Objective function for Optuna optimization."""
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
    
    # Create and evaluate model
    model = RandomForestRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2', n_jobs=-1)
    return cv_scores.mean()

def run_optimization_study(X_train, y_train, sampler_type="tpe", n_trials=100):
    """Run Optuna optimization study with specified sampler."""
    
    # Create sampler
    if sampler_type == "botorch":
        sampler = BoTorchSampler()
        print(f"🔬 Using BoTorchSampler (Bayesian Optimization)")
    else:
        sampler = TPESampler(seed=42)
        print(f"🔬 Using TPESampler")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"RandomForest_{sampler_type}"
    )
    
    # Define objective with data
    def objective(trial):
        return objective_function(trial, X_train, y_train, cv_folds=5)
    
    # Run optimization with timing
    print(f"🚀 Starting optimization with {n_trials} trials...")
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    end_time = time.time()
    
    print(f"⏱️ Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"🎯 Best R² score: {study.best_value:.6f}")
    
    return study

def analyze_convergence(studies, save_plots=True):
    """Analyze and visualize convergence for multiple studies."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange']
    study_names = list(studies.keys())
    
    for idx, (name, study) in enumerate(studies.items()):
        color = colors[idx % len(colors)]
        
        # Extract values
        trial_values = [trial.value for trial in study.trials if trial.value is not None]
        trial_numbers = list(range(len(trial_values)))
        
        # Calculate running maximum
        running_max = []
        current_max = -np.inf
        for value in trial_values:
            if value > current_max:
                current_max = value
            running_max.append(current_max)
        
        # Plot 1: Running maximum comparison
        axes[0, 0].plot(trial_numbers, running_max, color=color, linewidth=2, 
                       label=f'{name} (Best: {running_max[-1]:.4f})')
        
        # Plot 2: All trial scores
        axes[0, 1].scatter(trial_numbers, trial_values, color=color, alpha=0.6, 
                          s=20, label=f'{name}')
    
    # Configure Plot 1
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('Best R² Score')
    axes[0, 0].set_title('Convergence Comparison: Best Score Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Configure Plot 2
    axes[0, 1].set_xlabel('Trial Number')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('All Trial Scores by Sampler')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Improvement rate analysis
    for idx, (name, study) in enumerate(studies.items()):
        color = colors[idx % len(colors)]
        trial_values = [trial.value for trial in study.trials if trial.value is not None]
        
        # Calculate improvements
        improvements = []
        for i in range(1, len(trial_values)):
            window_size = min(10, i)
            recent_max = max(trial_values[max(0, i-window_size):i])
            current_max = max(trial_values[:i+1])
            improvement = current_max - recent_max
            improvements.append(improvement)
        
        axes[1, 0].plot(range(1, len(improvements) + 1), improvements, 
                       color=color, alpha=0.7, label=f'{name}')
    
    axes[1, 0].set_xlabel('Trial Number')
    axes[1, 0].set_ylabel('Improvement Rate')
    axes[1, 0].set_title('Improvement Rate Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final performance comparison
    final_scores = []
    sampler_names = []
    for name, study in studies.items():
        trial_values = [trial.value for trial in study.trials if trial.value is not None]
        final_scores.append(max(trial_values))
        sampler_names.append(name)
    
    bars = axes[1, 1].bar(sampler_names, final_scores, color=colors[:len(sampler_names)], alpha=0.7)
    axes[1, 1].set_ylabel('Best R² Score')
    axes[1, 1].set_title('Final Best Scores Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('comprehensive_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Analysis plots saved as 'comprehensive_optimization_analysis.png'")
    
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("📈 DETAILED CONVERGENCE ANALYSIS")
    print("="*80)
    
    for name, study in studies.items():
        trial_values = [trial.value for trial in study.trials if trial.value is not None]
        running_max = []
        current_max = -np.inf
        for value in trial_values:
            if value > current_max:
                current_max = value
            running_max.append(current_max)
        
        # Calculate convergence metrics
        final_best = running_max[-1]
        initial_score = trial_values[0]
        total_improvement = final_best - initial_score
        
        # Find when 90% of final improvement was achieved
        target_score = initial_score + 0.9 * total_improvement
        convergence_trial = next((i for i, score in enumerate(running_max) if score >= target_score), len(running_max))
        
        print(f"\n🔬 {name.upper()} Sampler:")
        print(f"   Best R² score: {final_best:.6f}")
        print(f"   Initial score: {initial_score:.6f}")
        print(f"   Total improvement: {total_improvement:.6f}")
        print(f"   90% convergence at trial: {convergence_trial}")
        print(f"   Final 10-trial stability: {np.std(running_max[-10:]):.6f}")

def evaluate_best_model(study, X_train, y_train, X_test, y_test):
    """Evaluate the best model found during optimization."""
    print("\n" + "="*80)
    print("🏆 BEST MODEL EVALUATION")
    print("="*80)
    
    # Get best parameters
    best_params = study.best_params
    print("📋 Best parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Train best model
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    print(f"\n📊 Model Performance:")
    print(f"   Training R²: {train_r2:.6f}")
    print(f"   Test R²: {test_r2:.6f}")
    print(f"   Training MSE: {train_mse:.6f}")
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Overfitting check: {train_r2 - test_r2:.6f}")
    
    # Feature importance
    feature_importance = best_model.feature_importances_
    print(f"\n🎯 Feature Importance (top 5):")
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i in range(min(5, len(feature_importance))):
        idx = sorted_idx[i]
        print(f"   Feature {idx}: {feature_importance[idx]:.4f}")
    
    return best_model

def main():
    """Main function to run the comprehensive optimization demo."""
    print("🚀 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION DEMO")
    print("RandomForestRegressor with Optuna: TPE vs BoTorchSampler")
    print("="*80)
    
    # Generate sample data
    print("📊 Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, n_features=10, noise_level=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Training data shape: {X_train.shape}")
    print(f"   Test data shape: {X_test.shape}")
    
    # Run optimization studies
    studies = {}
    n_trials = 100
    
    print(f"\n🔬 Running optimization studies with {n_trials} trials each...")
    
    # TPE Sampler
    print("\n1️⃣ TPE Sampler Study:")
    studies['TPE'] = run_optimization_study(X_train, y_train, 'tpe', n_trials)
    
    # BoTorch Sampler
    print("\n2️⃣ BoTorch Sampler Study:")
    studies['BoTorch'] = run_optimization_study(X_train, y_train, 'botorch', n_trials)
    
    # Analyze convergence
    print("\n📈 Analyzing convergence...")
    analyze_convergence(studies, save_plots=True)
    
    # Evaluate best models
    print("\n🏆 Evaluating best models...")
    for name, study in studies.items():
        print(f"\n{'='*40} {name} BEST MODEL {'='*40}")
        evaluate_best_model(study, X_train, y_train, X_test, y_test)
    
    # Summary comparison
    print("\n" + "="*80)
    print("📋 FINAL COMPARISON SUMMARY")
    print("="*80)
    
    for name, study in studies.items():
        best_score = study.best_value
        n_completed = len([t for t in study.trials if t.value is not None])
        print(f"{name:>10}: Best R² = {best_score:.6f} ({n_completed} trials)")
    
    print("\n🎉 Comprehensive demo completed successfully!")
    print("📊 Check the generated plots for detailed convergence analysis.")
    
    return studies

if __name__ == "__main__":
    studies = main()
