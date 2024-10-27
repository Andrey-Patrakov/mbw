import numpy as np
from abc import ABC, abstractmethod

from mbw.ml.metrics.distance import minkowski
from mbw.ml.metrics.distance import manhattan
from mbw.ml.metrics.distance import euclidean


class KNeighbors(ABC):

    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='minkowski',
                 metric_params=None,
                 p=2) -> None:

        self._X = None
        self._y = None

        self._n_neighbors = n_neighbors
        self._weights = weights.lower()
        if self._weights not in ['uniform', 'distance']:
            raise ValueError(f'Unknown value "{weights} for weights param!"')

        if isinstance(metric_params, dict):
            self._metric_params = metric_params
        elif metric_params is None:
            self._metric_params = {}
        else:
            raise TypeError('Unknown param type for metric_params')

        self._distance = None
        if callable(metric):
            self._distance = metric
        elif metric.lower() == 'minkowski':
            self._distance = minkowski
            self._metric_params['p'] = p
        elif metric.lower() == 'manhattan':
            self._distance = manhattan
        elif metric.lower() == 'euclidean':
            self._distance = euclidean
        else:
            raise ValueError(f'Unknown metric {metric} for KNeighbors class!')

    @abstractmethod
    def _predict(self, y, weights):
        return None

    def fit(self, X, y):
        self._X, self._y = np.array(X), np.array(y)
        if self._X.shape[0] != self._y.shape[0]:
            error = 'Found input variables with inconsistent numbers of samples: '  # noqa
            error += f'[{self._X.shape[0]}, {self._y.shape[0]}]'
            raise ValueError(error)

    def predict(self, X):
        return np.apply_along_axis(self._brute_force, 1, np.array(X))

    def _brute_force(self, X):
        distances = self._distance(X, self._X, **self._metric_params)
        prediction = distances.argsort()[:self._n_neighbors]
        weights = self._calc_weights(distances[prediction])
        return self._predict(self._y[prediction], weights)

    def _calc_weights(self, distance):
        if self._weights == 'distance':
            return max(distance) - distance + 0.1

        return np.ones(len(distance))


class KNeighborsClassifier(KNeighbors):

    def _predict(self, y, weights):
        results = []
        predictions = np.unique(y)
        for pred in predictions:
            results.append(sum(weights[y == pred]))

        return predictions[np.array(results).argmax()]


class KNeighborsRegressor(KNeighbors):

    def _predict(self, y, weights):
        return sum(y * weights) / sum(weights)
