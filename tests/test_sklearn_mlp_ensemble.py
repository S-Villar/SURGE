from __future__ import annotations

import numpy as np

from surge.model.sklearn import MLPModel


def test_sklearn_mlp_ensemble_predicts_mean_and_uncertainty() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 3))
    y = (0.5 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]).reshape(-1, 1)

    model = MLPModel(
        hidden_layer_sizes=(8,),
        max_iter=80,
        random_state=7,
        ensemble_n=3,
    )
    model.fit(X, y.ravel())

    pred = np.asarray(model.predict(X[:5]))
    uq = model.predict_with_uncertainty(X[:5])

    assert pred.shape == (5,)
    assert int(uq["n_members"]) == 3
    assert np.asarray(uq["mean"]).shape == (5,)
    assert np.asarray(uq["std"]).shape == (5,)
    assert np.all(np.asarray(uq["variance"]) >= 0.0)
