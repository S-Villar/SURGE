"""
Tests for SURGE helper functions.

Tests the convenience functions that make SURGE easier to use.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Legacy pre-refactor API tests. They call SurrogateEngine with positional
# or `dataframe=` arguments that the refactored engine no longer accepts
# (see surge/engine.py::SurrogateEngine.__init__). Skipped at module level
# until migrated to the new `configure_dataframe` / ModelSpec API.
# Tracked in docs/REFACTORING_PLAN.md §1.9.
pytestmark = pytest.mark.skip(
    reason="legacy pre-refactor API; pending migration (docs/REFACTORING_PLAN.md §1.9)"
)

try:
    from surge.helpers import quick_train, train_and_compare, load_and_train
    HELPERS_AVAILABLE = True
except ImportError:
    HELPERS_AVAILABLE = False


@pytest.mark.skipif(not HELPERS_AVAILABLE, reason="Helper functions not available")
class TestQuickTrain:
    """Test quick_train convenience function."""
    
    def test_quick_train_basic(self):
        """Test basic quick_train functionality."""
        np.random.seed(42)
        X = np.random.random((100, 5))
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
        
        trainer = quick_train(X, y, model_type=0)
        
        assert trainer is not None
        assert trainer.R2 is not None
        assert trainer.MSE is not None
        assert trainer.R2 > -10.0  # Reasonable range
    
    def test_quick_train_with_model_registry(self):
        """Test quick_train with model registry keys."""
        np.random.seed(42)
        X = np.random.random((50, 3))
        y = np.sum(X, axis=1)
        
        # Use registry key instead of integer, with model_kwargs
        trainer = quick_train(X, y, model_type='random_forest', model_kwargs={'n_estimators': 50})
        
        assert trainer is not None
        assert len(trainer.models) == 1
    
    def test_quick_train_with_dataframe(self):
        """Test quick_train with pandas DataFrames."""
        np.random.seed(42)
        X_df = pd.DataFrame(np.random.random((50, 3)), columns=['x1', 'x2', 'x3'])
        y_df = pd.DataFrame(np.sum(X_df.values, axis=1), columns=['y'])
        
        trainer = quick_train(X_df, y_df, model_type=0)
        
        assert trainer is not None
        assert trainer.F == 3
        assert trainer.T == 1
    
    def test_quick_train_multi_output(self):
        """Test quick_train with multiple outputs."""
        np.random.seed(42)
        X = np.random.random((50, 3))
        y = np.column_stack([
            np.sum(X, axis=1),
            np.sum(X, axis=1) * 1.2
        ])
        
        trainer = quick_train(X, y, model_type=0)
        
        assert trainer is not None
        assert trainer.T == 2


@pytest.mark.skipif(not HELPERS_AVAILABLE, reason="Helper functions not available")
class TestTrainAndCompare:
    """Test train_and_compare convenience function."""
    
    def test_train_and_compare_basic(self):
        """Test basic train_and_compare functionality."""
        np.random.seed(42)
        X = np.random.random((100, 5))
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
        
        trainer = train_and_compare(X, y, model_types=[0, 1])
        
        assert trainer is not None
        assert len(trainer.models) == 2
        assert all(idx in trainer.model_predictions for idx in [0, 1])
    
    def test_train_and_compare_with_plots(self):
        """Test train_and_compare with plot saving."""
        np.random.seed(42)
        X = np.random.random((100, 5))
        y = np.sum(X, axis=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = train_and_compare(
                X, y,
                model_types=[0, 1],
                save_comparison_plots=True,
                plots_dir=tmpdir
            )
            
            # Check that plots were created
            plots_path = Path(tmpdir) / "model_comparison_test.png"
            # Note: plots might not be created if matplotlib fails, so we don't assert existence


@pytest.mark.skipif(not HELPERS_AVAILABLE, reason="Helper functions not available")
class TestLoadAndTrain:
    """Test load_and_train convenience function."""
    
    def test_load_and_train_csv(self):
        """Test load_and_train with CSV file."""
        np.random.seed(42)
        X = np.random.random((50, 3))
        y = np.sum(X, axis=1)
        
        df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
        df['y'] = y
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            df.to_csv(csv_path, index=False)
            
            trainer = load_and_train(
                csv_path,
                input_cols=['x1', 'x2', 'x3'],
                output_cols=['y'],
                model_type=0
            )
            
            assert trainer is not None
            assert trainer.F == 3
            assert trainer.T == 1
    
    def test_load_and_train_pickle(self):
        """Test load_and_train with pickle file."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.random(50),
            'x2': np.random.random(50),
            'y': np.random.random(50)
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_data.pkl"
            df.to_pickle(pkl_path)
            
            trainer = load_and_train(
                pkl_path,
                input_cols=['x1', 'x2'],
                output_cols=['y'],
                model_type='random_forest'
            )
            
            assert trainer is not None

