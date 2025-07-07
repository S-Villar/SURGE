"""
Enhanced tests for SURGE package including PyTorch and GPflow models
"""

import pytest
import numpy as np
from surge import SurrogateTrainer, PYTORCH_AVAILABLE, GPFLOW_AVAILABLE


def test_import():
    """Test that we can import the main class."""
    assert SurrogateTrainer is not None


def test_availability_flags():
    """Test that availability flags are accessible."""
    assert isinstance(PYTORCH_AVAILABLE, bool)
    assert isinstance(GPFLOW_AVAILABLE, bool)


def test_surrogate_trainer_init():
    """Test SurrogateTrainer initialization with different model types."""
    # Test basic models
    trainer = SurrogateTrainer(model_type="rfr")
    assert trainer is not None
    
    trainer = SurrogateTrainer(model_type="mlp")
    assert trainer is not None
    
    trainer = SurrogateTrainer(model_type="gpr")
    assert trainer is not None
    
    # Test invalid model type
    with pytest.raises(ValueError):
        SurrogateTrainer(model_type="invalid")


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_mlp_model():
    """Test PyTorch MLP model if available."""
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)
    
    trainer = SurrogateTrainer(
        model_type="pytorch_mlp",
        hidden_layers=[16, 8],
        n_epochs=50,
        learning_rate=1e-2
    )
    trainer.fit(X, y)
    results = trainer.cross_validate(n_splits=3)
    
    assert "r2_mean" in results
    assert "mse_mean" in results
    assert isinstance(results["r2_mean"], float)


@pytest.mark.skipif(not GPFLOW_AVAILABLE, reason="GPflow not available")
def test_gpflow_gpr_model():
    """Test GPflow GPR model if available."""
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)
    
    trainer = SurrogateTrainer(
        model_type="gpflow_gpr",
        kernel_type="matern32",
        optimize=True,
        maxiter=100
    )
    trainer.fit(X, y)
    results = trainer.cross_validate(n_splits=3)
    
    assert "r2_mean" in results
    assert "mse_mean" in results
    assert isinstance(results["r2_mean"], float)


@pytest.mark.skipif(not GPFLOW_AVAILABLE, reason="GPflow not available")
def test_gpflow_multi_kernel_model():
    """Test GPflow multi-kernel model if available."""
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)
    
    trainer = SurrogateTrainer(
        model_type="gpflow_multi",
        kernel_types=["matern32", "rbf"],
        optimize=True,
        maxiter=50
    )
    trainer.fit(X, y)
    
    # Test that we can get model info
    info = trainer.get_model_info()
    assert "model_type" in info
    assert info["model_type"] == "gpflow_multi"


def test_basic_workflow():
    """Test basic training workflow with scikit-learn models."""
    # Generate sample data
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)
    
    # Test with Random Forest
    trainer = SurrogateTrainer(model_type="rfr", n_estimators=10)
    trainer.fit(X, y)
    results = trainer.cross_validate(n_splits=3)
    
    assert "r2_mean" in results
    assert "mse_mean" in results
    assert isinstance(results["r2_mean"], float)


def test_all_sklearn_model_types():
    """Test all scikit-learn supported model types."""
    X = np.random.random((30, 2))
    y = np.sum(X, axis=1)
    
    for model_type in ["rfr", "mlp", "gpr"]:
        if model_type == "mlp":
            trainer = SurrogateTrainer(
                model_type=model_type, max_iter=50, hidden_layer_sizes=(10,)
            )
        elif model_type == "gpr":
            trainer = SurrogateTrainer(model_type=model_type)
        else:
            trainer = SurrogateTrainer(model_type=model_type, n_estimators=10)
        
        trainer.fit(X, y)
        results = trainer.cross_validate(n_splits=2)
        assert "r2_mean" in results


def test_predict_functionality():
    """Test prediction functionality."""
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)
    
    trainer = SurrogateTrainer(model_type="rfr", n_estimators=10)
    trainer.fit(X, y)
    
    # Fit the model on training data
    trainer.model.fit(trainer.X_tv, trainer.y_tv)
    
    # Test prediction
    predictions = trainer.predict(trainer.X_test)
    assert predictions is not None
    assert len(predictions) == len(trainer.X_test)


def test_uncertainty_prediction():
    """Test uncertainty prediction for GP models."""
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)
    
    # Test with sklearn GP
    trainer = SurrogateTrainer(model_type="gpr")
    trainer.fit(X, y)
    trainer.model.fit(trainer.X_tv, trainer.y_tv)
    
    predictions, uncertainty = trainer.predict_with_uncertainty(trainer.X_test)
    assert predictions is not None
    # For sklearn GP, uncertainty will be None since it's not implemented
    # in the wrapper yet


def test_model_info():
    """Test getting model information."""
    X = np.random.random((30, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)
    
    trainer = SurrogateTrainer(model_type="rfr", n_estimators=10)
    trainer.fit(X, y)
    trainer.cross_validate(n_splits=3)
    
    info = trainer.get_model_info()
    assert "model_type" in info
    assert info["model_type"] == "rfr"
    assert "cv_results" in info


if __name__ == "__main__":
    pytest.main([__file__])
