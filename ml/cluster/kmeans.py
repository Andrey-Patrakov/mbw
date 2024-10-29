import numpy as np
from mbw.ml.metrics.distance import euclidean


class KMeans:

    def __init__(self,
                 n_clusters=8,
                 max_iter=300,
                 tol=0.0001,
                 random_state=None) -> None:

        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._centroids = None
        self._histoty = []

    @property
    def history(self):
        return np.array(self._histoty)

    def fit(self, X, y=None):
        self._histoty = []
        X = np.array(X)
        if self._random_state is not None:
            np.random.seed(int(self._random_state))

        ids = np.arange(X.shape[0])
        self._centroids = X[np.random.choice(ids, self._n_clusters)]
        self._histoty.append(self._centroids.copy())

        for _ in range(self._max_iter):
            print(_)
            distances = np.apply_along_axis(self._get_distances, 1, X)
            closest = distances.argmin(axis=1)

            old_centroids = self._centroids.copy()
            for centroid in range(self._n_clusters):
                if any(closest == centroid):
                    self._centroids[centroid] = np.mean(X[closest == centroid])

            self._histoty.append(self._centroids.copy())
            tol_distances = euclidean(old_centroids, self._centroids)
            if max(tol_distances) < self._tol:
                break

        return self.history

    def _get_distances(self, X):
        return euclidean(X, self._centroids)

    def predict(self, X):
        distances = np.apply_along_axis(self._get_distances, 1, np.array(X))
        return distances.argmin(axis=1)
