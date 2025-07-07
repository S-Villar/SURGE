from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

class RFRModel:
    def __init__(self, **kwargs):
        self._rf = RandomForestRegressor(**kwargs)
    def fit(self, X, y):    return self._rf.fit(X, y)
    def predict(self, X):   return self._rf.predict(X)

class MLPModel:
    def __init__(self, **kwargs):
        self._mlp = MLPRegressor(**kwargs)
    def fit(self, X, y):    return self._mlp.fit(X, y)
    def predict(self, X):   return self._mlp.predict(X)

class GPRModel:
    def __init__(self, **kwargs):
        self._gpr = GaussianProcessRegressor(**kwargs)
    def fit(self, X, y):    return self._gpr.fit(X, y)
    def predict(self, X):   return self._gpr.predict(X)
