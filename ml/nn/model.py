import numpy as np
from mbw.ml.nn.layers import Layer


class NeuralNetwork:
    def __init__(self,
                 learning_rate=1e-3,
                 random_seed=None,
                 verbose_callback=None) -> None:

        self._layers = []
        self._lr = learning_rate
        self._random_seed = random_seed
        self._v_callback = verbose_callback

    def add(self, layer: Layer):
        self._layers.append(layer)

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)

        return x

    def backward(self, error, lr):
        for layer in self._layers[::-1]:
            error = layer.backward(error, lr)

        return error

    def fit(self, x, y, epochs, learning_rate=None):
        lr = learning_rate if learning_rate else self._lr
        if self._random_seed is not None:
            np.random.seed(self._random_seed)

        history = {
            'epoch': [],
            'error': [],
        }

        for epoch in range(epochs):
            y_pred = self.forward(x)
            error = y - y_pred
            self.backward(error, lr)

            history['epoch'].append(epoch + 1)
            history['error'].append(error)

            if self._v_callback is not None:
                self._v_callback(self)

        return history

    def predict(self, x):
        return self.forward(x)
