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


class LeakyReLU(Activation):

    def __init__(self, alpha=0.03) -> None:
        super().__init__()
        self._alpha = alpha

    def call(self, x):
        return np.maximum(self._alpha * x, x)

    def deriv(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = self._alpha
        return dx


class Sigmoid(Activation):

    def call(self, x):
        return 1 / (1 + np.exp(np.minimum(-x, 50.0)))

    def deriv(self, x):
        return self.call(x) * (1 - self.call(x))


class Tanh(Activation):

    def call(self, x):
        return (2 / (1 + np.exp(np.minimum(-2 * x, 50.0)))) - 1

    def deriv(self, x):
        return 1 - self.call(x)**2
