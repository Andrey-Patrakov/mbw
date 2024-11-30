import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, error, lr):
        pass


class Linear(Layer):

    def __init__(self, in_features: int, out_features: int):
        self._in_features = in_features
        self._out_features = out_features

        self._w = 2 * np.random.random((in_features, out_features)) - 1
        self._b = 2 * np.random.random((1, out_features)) - 1

    def forward(self, x):
        self._cache = x
        return x @ self._w + self._b

    def backward(self, error, lr):
        self._w += (self._cache.T @ error) * lr
        self._b += np.sum(error, axis=0, keepdims=True) * lr
        return error @ self._w.T
