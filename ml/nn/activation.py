import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):

    def __init__(self) -> None:
        self._cache = None

    @abstractmethod
    def call(self, x):
        pass

    @abstractmethod
    def deriv(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self._cache = self.call(x)
        return self._cache

    def backward(self, error, lr=None):
        return error * self.deriv(self._cache)


class ReLU(Activation):

    def call(self, x):
        return np.maximum(0, x)

    def deriv(self, x):
        return x > 0


class Sigmoid(Activation):

    def call(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv(self, x):
        return self.call(x) * (1 - self.call(x))
