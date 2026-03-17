"""Tests for SHAP feature importance (TreeExplainer and KernelExplainer)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from surge.viz.importance import (
    SHAP_AVAILABLE,
    compute_shap_kernel,
    compute_shap_tree,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_summary,
    plot_shap_waterfall,
    run_shap_analysis,
    shap_importance_mean_abs,
)


@pytest.fixture
def synthetic_data():
    """Synthetic data for SHAP tests."""
    np.random.seed(42)
    n_samples, n_features = 80, 4
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    return X, y


@pytest.fixture
def rf_model(synthetic_data):
    """Trained Random Forest for TreeExplainer."""
    X, y = synthetic_data
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def mlp_model(synthetic_data):
    """Trained MLP for KernelExplainer (non-tree)."""
    X, y = synthetic_data
    model = MLPRegressor(hidden_layer_sizes=(20,), max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_compute_shap_tree(rf_model, synthetic_data):
    """TreeExplainer returns correct shapes."""
    X, _ = synthetic_data
    X_explain = X[:30]
    X_background = X[:20]

    explainer, shap_values = compute_shap_tree(
        rf_model, X_explain, X_background=X_background
    )

    assert shap_values.shape == (30, 4)
    assert explainer is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_compute_shap_kernel(mlp_model, synthetic_data):
    """KernelExplainer returns correct shapes."""
    X, _ = synthetic_data
    X_explain = X[:15]
    X_background = X[:20]

    explainer, shap_values = compute_shap_kernel(
        mlp_model, X_explain, X_background, nsamples=10
    )

    assert shap_values.shape == (15, 4)
    assert explainer is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_shap_importance_mean_abs(rf_model, synthetic_data):
    """Mean absolute SHAP produces one value per feature."""
    X, _ = synthetic_data
    _, shap_values = compute_shap_tree(rf_model, X[:20], X_background=X[:10])

    importance = shap_importance_mean_abs(shap_values)
    assert importance.shape == (4,)
    assert np.all(importance >= 0)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_plot_shap_summary(rf_model, synthetic_data):
    """plot_shap_summary saves bar plot without error."""
    X, _ = synthetic_data
    _, shap_values = compute_shap_tree(rf_model, X[:20], X_background=X[:10])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shap_bar.png"
        fig = plot_shap_summary(
            shap_values, X[:20],
            feature_names=["f0", "f1", "f2", "f3"],
            plot_type="bar",
            save_path=path,
        )
        assert path.exists()
        assert fig is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_run_shap_analysis_tree(rf_model, synthetic_data):
    """run_shap_analysis with tree model produces tree results."""
    X, _ = synthetic_data
    with tempfile.TemporaryDirectory() as tmp:
        results = run_shap_analysis(
            rf_model, X[:40],
            X_background=X[:20],
            use_tree=True,
            use_kernel=False,
            save_dir=tmp,
            feature_names=["a", "b", "c", "d"],
        )

    assert "tree" in results
    assert "importance_tree" in results
    assert results["importance_tree"].shape == (4,)
    assert len(results["saved_paths"]) >= 1


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_run_shap_analysis_kernel(mlp_model, synthetic_data):
    """run_shap_analysis with KernelExplainer produces kernel results."""
    X, _ = synthetic_data
    with tempfile.TemporaryDirectory() as tmp:
        results = run_shap_analysis(
            mlp_model, X[:30],
            X_background=X[:20],
            use_tree=False,
            use_kernel=True,
            kernel_nsamples=10,
            save_dir=tmp,
        )

    assert "kernel" in results
    assert "importance_kernel" in results
    assert results["importance_kernel"].shape == (4,)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_tree_explainer_rejects_non_tree(mlp_model, synthetic_data):
    """TreeExplainer raises ValueError for non-tree models."""
    X, _ = synthetic_data
    with pytest.raises(ValueError, match="TreeExplainer requires"):
        compute_shap_tree(mlp_model, X[:10], X_background=X[:5])


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_plot_shap_dependence(rf_model, synthetic_data):
    """plot_shap_dependence saves figure without error."""
    X, _ = synthetic_data
    _, shap_values = compute_shap_tree(rf_model, X[:30], X_background=X[:15])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shap_dep.png"
        fig = plot_shap_dependence(
            shap_values, X[:30], 0,
            feature_names=["f0", "f1", "f2", "f3"],
            save_path=path,
        )
        assert path.exists()
        assert fig is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_plot_shap_waterfall(rf_model, synthetic_data):
    """plot_shap_waterfall saves figure without error."""
    X, _ = synthetic_data
    explainer, shap_values = compute_shap_tree(rf_model, X[:30], X_background=X[:15])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shap_waterfall.png"
        fig = plot_shap_waterfall(
            explainer, shap_values, X[:30],
            sample_index=0,
            feature_names=["f0", "f1", "f2", "f3"],
            save_path=path,
        )
        assert path.exists()
        assert fig is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_plot_shap_force(rf_model, synthetic_data):
    """plot_shap_force saves figure without error."""
    X, _ = synthetic_data
    explainer, shap_values = compute_shap_tree(rf_model, X[:30], X_background=X[:15])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shap_force.png"
        fig = plot_shap_force(
            explainer, shap_values, X[:30],
            sample_index=0,
            feature_names=["f0", "f1", "f2", "f3"],
            save_path=path,
        )
        assert path.exists()
        assert fig is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_plot_shap_summary_auto_feature_names_from_dataframe(rf_model, synthetic_data):
    """plot_shap_summary infers feature names from DataFrame columns."""
    X, _ = synthetic_data
    df = pd.DataFrame(X, columns=["βN", "κ", "Δne,ped", "ΔTe,ped"])
    _, shap_values = compute_shap_tree(rf_model, X[:20], X_background=X[:10])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shap_bar.png"
        plot_shap_summary(shap_values, df[:20], plot_type="bar", save_path=path)
        assert path.exists()
        # Labels should come from df.columns, not feature_0, feature_1, etc.
        # We verify the plot was created; exact label check would need figure inspection


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_run_shap_analysis_with_dataframe(rf_model, synthetic_data):
    """run_shap_analysis infers feature names from DataFrame."""
    X, _ = synthetic_data
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    with tempfile.TemporaryDirectory() as tmp:
        results = run_shap_analysis(
            rf_model, df[:40],
            X_background=df[:20],
            use_tree=True,
            use_kernel=False,
            save_dir=tmp,
        )
    assert "tree" in results
    assert results["importance_tree"].shape == (4,)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_viz_shap_missing_run_dir():
    """viz_shap returns [] when run_dir has no spec."""
    from surge.viz import viz_shap

    with tempfile.TemporaryDirectory() as tmp:
        result = viz_shap(Path(tmp))
    assert result == []


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_run_shap_analysis_with_all_plots(rf_model, synthetic_data):
    """run_shap_analysis with plot_type=dot and extra plots."""
    X, _ = synthetic_data
    with tempfile.TemporaryDirectory() as tmp:
        results = run_shap_analysis(
            rf_model, X[:40],
            X_background=X[:20],
            use_tree=True,
            use_kernel=False,
            save_dir=tmp,
            plot_type="dot",
            save_dependence=True,
            save_waterfall=True,
            save_force=True,
        )
    assert "tree" in results
    assert any("dependence" in p for p in results["saved_paths"])
    assert any("waterfall" in p for p in results["saved_paths"])
    assert any("force" in p for p in results["saved_paths"])
