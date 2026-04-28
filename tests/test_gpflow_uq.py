from __future__ import annotations

import numpy as np
import pytest

from surge.engine import SurrogateEngine


def test_uncertainty_tuple_normalizes_to_mapping() -> None:
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])

    payload = SurrogateEngine._normalize_uncertainty_payload((mean, std))

    assert set(payload) == {"mean", "std", "variance"}
    np.testing.assert_allclose(payload["mean"], mean)
    np.testing.assert_allclose(payload["std"], std)
    np.testing.assert_allclose(payload["variance"], std**2)


def test_gpflow_multi_kernel_exposes_uncertainty() -> None:
    pytest.importorskip("gpflow")

    from surge.model.gpflow import GPflowMultiKernelAdapter

    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(24, 2))
    y = (np.sin(X[:, 0]) + 0.2 * X[:, 1]).reshape(-1, 1)

    model = GPflowMultiKernelAdapter(
        kernel_types=["rbf"],
        lengthscales_range=(0.25, 0.25),
        variance_range=(0.5, 0.5),
        optimize=False,
        maxiter=2,
    )
    model.fit(X, y)
    mean, std = model.predict_with_uncertainty(X[:5])

    assert np.asarray(mean).shape == (5,)
    assert np.asarray(std).shape == (5,)
    assert np.all(np.asarray(std) >= 0.0)
