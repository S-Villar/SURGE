"""
Basic tests for SURGE package
"""
import pytest
import numpy as np
from surge import SurrogateTrainer


def test_import():
    """Test that we can import the main class."""
    assert SurrogateTrainer is not None


def test_surrogate_trainer_init():
    """Test SurrogateTrainer initialization."""
    trainer = SurrogateTrainer(model_type='rfr')
    assert trainer is not None
    
    # Test invalid model type
    with pytest.raises(ValueError):
        SurrogateTrainer(model_type='invalid')


def test_basic_workflow():
    """Test basic training workflow."""
    # Generate sample data
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)
    
    # Test with Random Forest
    trainer = SurrogateTrainer(model_type='rfr', n_estimators=10)
    trainer.fit(X, y)
    results = trainer.cross_validate(n_splits=3)
    
    assert 'r2_mean' in results
    assert 'mse_mean' in results
    assert isinstance(results['r2_mean'], float)


def test_all_model_types():
    """Test all supported model types."""
    X = np.random.random((30, 2))
    y = np.sum(X, axis=1)
    
    for model_type in ['rfr', 'mlp', 'gpr']:
        if model_type == 'mlp':
            trainer = SurrogateTrainer(model_type=model_type, max_iter=50, hidden_layer_sizes=(10,))
        elif model_type == 'gpr':
            trainer = SurrogateTrainer(model_type=model_type)
        else:
            trainer = SurrogateTrainer(model_type=model_type, n_estimators=10)
        
        trainer.fit(X, y)
        results = trainer.cross_validate(n_splits=2)
        assert 'r2_mean' in results


if __name__ == "__main__":
    pytest.main([__file__])
