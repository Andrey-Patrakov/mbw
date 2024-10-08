import numpy as np
from mbw.ml.linear_model.base import BaseGD, SGD, MiniBatchGD
from mbw.ml.metrics.loss import log_loss
from mbw.ml.metrics.score import f_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(BaseGD):

    def __init__(self,
                 threshold=0.5,
                 **kwargs):

        super().__init__(**kwargs)
        self._threshold = threshold

    def _gradient_step(self, X, y):
        n = len(y)
        p_pred = self.predict_proba(X)
        grad = (1 / n) * ((p_pred - y) @ X)
        penalty = self._l1_reg()
        penalty += self._l2_reg()
        self._w -= self._eta * grad + penalty

    def predict(self, X):
        return np.where(self.predict_proba(X) >= self._threshold, 1, 0)

    def predict_proba(self, X):
        return sigmoid(self._prepare_data(X) @ self._w)

    def set_threshold(self, threshold):
        self._threshold = threshold

    def _loss(self, X, y_true):
        p_pred = self.predict_proba(X)
        return log_loss(y_true, p_pred)

    def _score(self, X, y_true):
        y_pred = self.predict(X)
        return f_score(y_true, y_pred)


class StochasticLogisticRegression(SGD, LogisticRegression):
    pass


class MiniBatchLogisticRegression(MiniBatchGD, LogisticRegression):
    pass
