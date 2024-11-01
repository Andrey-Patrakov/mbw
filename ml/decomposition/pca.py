import numpy as np


class PCA:

    def __init__(self, n_components=None) -> None:
        self._n_components = n_components
        self._x_components = None
        self._cumsum = None

    @property
    def cumsum(self):
        return self._cumsum

    @property
    def n_components(self):
        if self._n_components:
            return min(self._n_components, self._x_components)

        return self._x_components

    @property
    def W(self):
        return self._V[:, :self.n_components]

    @n_components.setter
    def n_components(self, value):
        self._n_components = value

    def fit(self, X, y=None):
        X = np.array(X)
        self._x_components = X.shape[0]

        U, D, Vt = np.linalg.svd(X)
        self._V = Vt.T
        self._cumsum = np.cumsum(D**2 / sum(D**2) * 100)

        return self.cumsum

    def transform(self, X, y=None):
        return np.array(X) @ self.W

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
