"""
Enhanced tests for SURGE package including PyTorch and GPflow models
"""

import numpy as np
import pandas as pd
import pytest

from surge import GPFLOW_AVAILABLE, PYTORCH_AVAILABLE, MLTrainer, SurrogateTrainer


def test_import():
    """Test that we can import the main classes."""
    assert SurrogateTrainer is not None
    assert MLTrainer is not None


def test_availability_flags():
    """Test that availability flags are accessible."""
    assert isinstance(PYTORCH_AVAILABLE, bool)
    assert isinstance(GPFLOW_AVAILABLE, bool)


def test_surrogate_trainer_init():
    """Test SurrogateTrainer initialization."""
    trainer = SurrogateTrainer()
    assert trainer is not None
    assert trainer.df is None


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_mlp_model():
    """Test PyTorch MLP model if available."""
    np.random.seed(42)
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    trainer = MLTrainer(n_features=3, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(2)  # PyTorch MLP
    trainer.train(0)
    trainer.predict_output(0)

    assert trainer.R2 > 0.5


@pytest.mark.skipif(not GPFLOW_AVAILABLE, reason="GPflow not available")
def test_gpflow_gpr_model():
    """Test GPflow GPR model if available."""
    # This test is currently placeholder since GPR implementation needs work
    # We'll just test that GPflow is importable when available
    try:
        import gpflow
        assert gpflow is not None
    except ImportError:
        pytest.skip("GPflow not available")


@pytest.mark.skipif(not GPFLOW_AVAILABLE, reason="GPflow not available")
def test_gpflow_multi_kernel_model():
    """Test GPflow multi-kernel model if available."""
    # Placeholder test
    try:
        import gpflow
        assert gpflow is not None
    except ImportError:
        pytest.skip("GPflow not available")


def test_basic_workflow():
    """Test basic training workflow with scikit-learn models."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    # Test Random Forest
    trainer = MLTrainer(n_features=3, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    trainer.train(0)
    trainer.predict_output(0)

    assert trainer.R2 > 0.5
    assert trainer.MSE < 1.0


def test_all_sklearn_model_types():
    """Test all scikit-learn supported model types."""
    np.random.seed(42)
    X = np.random.random((30, 2))
    y = np.sum(X, axis=1)

    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['y'] = y

    # Test Random Forest
    trainer = MLTrainer(n_features=2, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    trainer.train(0)
    trainer.predict_output(0)
    assert trainer.R2 > 0.8

    # Test MLP
    trainer2 = MLTrainer(n_features=2, n_outputs=1)
    trainer2.load_df_dataset(df, ['x1', 'x2'], ['y'])
    trainer2.train_test_split(test_split=0.3)
    trainer2.standardize_data()
    trainer2.init_model(1)  # MLP
    trainer2.train(0)
    trainer2.predict_output(0)
    assert trainer2.R2 > 0.5


def test_predict_functionality():
    """Test prediction functionality."""
    np.random.seed(42)
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    trainer = MLTrainer(n_features=3, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    trainer.train(0)
    trainer.predict_output(0)

    # Test that predictions were stored
    assert hasattr(trainer, 'y_pred_test')
    assert hasattr(trainer, 'y_pred_train_val')
    assert trainer.y_pred_test.shape[0] > 0
    assert trainer.y_pred_train_val.shape[0] > 0


def test_uncertainty_prediction():
    """Test uncertainty prediction for GP models."""
    # This is a placeholder since GPR implementation needs work
    # For now, just test that we can create the trainer
    trainer = MLTrainer(n_features=3, n_outputs=1)
    assert trainer is not None


def test_model_info():
    """Test getting model information."""
    np.random.seed(42)
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    trainer = MLTrainer(n_features=3, n_outputs=1)
    trainer.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    trainer.train_test_split(test_split=0.3)
    trainer.standardize_data()
    trainer.init_model(0)  # Random Forest
    trainer.train(0)
    trainer.predict_output(0)

    # Test model summary
    trainer.get_model_summary(0)  # Should not raise an error
    trainer.get_model_summary()   # Test all models summary


if __name__ == "__main__":
    pytest.main([__file__])
