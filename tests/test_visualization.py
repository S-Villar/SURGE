"""
Minimal test for visualization functions.

Demonstrates:
1. plot_regression_results (single output, train/test/both)
2. plot_all_outputs (multi-output comparison)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Legacy pre-refactor API tests. The fixture builds a SurrogateEngine
# through the old constructor signature; skipped at module level until
# ported. Tracked in docs/REFACTORING_PLAN.md §1.9.
pytestmark = pytest.mark.skip(
    reason="legacy pre-refactor API; pending migration (docs/REFACTORING_PLAN.md §1.9)"
)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from surge import SurrogateEngine


@pytest.fixture
def trained_model():
    """Create a trained model for visualization testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 5
    n_outputs = 2
    
    X = np.random.random((n_samples, n_features))
    y = (
        2.0 * X[:, 0] ** 2 +
        1.5 * X[:, 1] * X[:, 2] +
        0.1 * np.random.randn(n_samples)
    )
    
    y_multi = np.column_stack([
        y,
        y * 1.2 + 0.1 * np.random.randn(n_samples)
    ])
    
    input_names = [f'input_{i+1}' for i in range(n_features)]
    output_names = [f'output_{i+1}' for i in range(n_outputs)]
    
    df = pd.DataFrame(X, columns=input_names)
    for i, output_name in enumerate(output_names):
        df[output_name] = y_multi[:, i]
    
    engine = SurrogateEngine(n_features=len(input_names), n_outputs=len(output_names))
    engine.load_df_dataset(df, input_names, output_names)
    engine.train_test_split(test_split=0.2)
    engine.standardize_data()
    engine.init_model(0, n_estimators=50, max_depth=8, random_state=42)
    engine.train(0)
    engine.predict_output(0)
    
    return engine


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_plot_regression_results_both(trained_model, tmp_path):
    """Test plot_regression_results with dataset='both'."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    fig, axes, results = trained_model.plot_regression_results(
        model_index=0,
        output_index=0,
        dataset='both',
        save_path=str(plots_dir / 'comparison.png'),
    )
    
    assert 'r2_train' in results
    assert 'r2_test' in results
    assert (plots_dir / 'comparison.png').exists()


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_plot_regression_results_train(trained_model, tmp_path):
    """Test plot_regression_results with dataset='train'."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    fig, ax, results = trained_model.plot_regression_results(
        model_index=0,
        output_index=0,
        dataset='train',
        save_path=str(plots_dir / 'train.png'),
    )
    
    assert 'r2_train' in results
    assert (plots_dir / 'train.png').exists()


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_plot_regression_results_test(trained_model, tmp_path):
    """Test plot_regression_results with dataset='test'."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    fig, ax, results = trained_model.plot_regression_results(
        model_index=0,
        output_index=0,
        dataset='test',
        save_path=str(plots_dir / 'test.png'),
    )
    
    assert 'r2_test' in results
    assert (plots_dir / 'test.png').exists()


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_plot_all_outputs(trained_model, tmp_path):
    """Test plot_all_outputs for multi-output models."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    fig, axes, results = trained_model.plot_all_outputs(
        model_index=0,
        max_outputs=2,
        save_path=str(plots_dir / 'all_outputs.png'),
    )
    
    assert len(results) > 0
    assert (plots_dir / 'all_outputs.png').exists()
    
    # Close figures
    plt.close('all')

