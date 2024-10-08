from mbw.ml.linear_model.base import BaseGD, SGD, MiniBatchGD
from mbw.ml.metrics.loss import mse
from mbw.ml.metrics.score import r2_score


class LinearRegression(BaseGD):

    def _gradient_step(self, X, y):
        n = len(y)
        y_pred = self.predict(X)

        grad = 2/n * (X.T @ (y_pred - y))
        penalty = self._l1_reg()
        penalty += self._l2_reg()
        self._w -= self._eta * grad + penalty
        return y_pred

    def predict(self, X):
        return self._prepare_data(X) @ self._w

    def _loss(self, X, y):
        y_pred = self.predict(X)
        return mse(y, y_pred)

    def _score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class StochasticLinearRegression(SGD, LinearRegression):
    pass


class MiniBatchLinearRegression(MiniBatchGD, LinearRegression):
    pass
