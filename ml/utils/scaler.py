import numpy as np
from abc import ABC, abstractmethod


class Scaler(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class MinMaxScaler(Scaler):

    def __init__(self, feature_range=(0, 1)) -> None:
        self._x_min = None
        self._x_max = None
        self._min_range = min(feature_range)
        self._max_range = max(feature_range)

    def fit(self, X, y=None):
        X = np.array(X)
        self._x_min = X.min(axis=0)
        self._x_max = X.max(axis=0)

    def transform(self, X, y=None):
        if self._x_min is None or self._x_max is None:
            raise RuntimeError(
                "MinMaxScaler is not fited (fit method was never called)!"
            )

        X = np.array(X)
        X_std = (X - self._x_min) / (self._x_max - self._x_min)
        return X_std * (self._max_range - self._min_range) + self._min_range


class StandardScaler(Scaler):

    def __init__(self) -> None:
        self._x_mean = None
        self._x_std = None

    def fit(self, X, y=None):
        X = np.array(X)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)

    def transform(self, X, y=None):
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError(
                "StandardScaler is not fited (fit method was never called)!"
            )

        X = np.array(X)
        return (X - self._x_mean) / self._x_std
