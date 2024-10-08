import numpy as np
import pandas as pd
from mbw.ml.utils.regularization import lasso, ridge
from mbw.ml.utils.batch_loader import BatchLoader

from abc import ABC, abstractmethod


class LinearModel(ABC):

    def __init__(self,
                 eta=1e-3,
                 max_iter=1000,
                 class_weight=None,
                 random_state=None,
                 fit_intercept=True,
                 l1_ratio=None,
                 l2_ratio=None,
                 tol=None,  # Tolerance for stopping criteria
                 history_step=100):

        self._eta = eta
        self._max_iter = max_iter

        self._w = class_weight
        self._seed = random_state
        self._fit_intercept = fit_intercept
        self._data_prepared = False

        self._l1_ratio = l1_ratio
        self._l2_ratio = l2_ratio

        self._prev_error = float("inf")
        self._tolerance = tol

        self._history_step = history_step
        self.clear_history()

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def _gradient_step(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _loss(self, X, y):
        pass

    @abstractmethod
    def _score(self, X, y):
        pass

    @property
    def history(self):
        return pd.DataFrame(self._history)

    @property
    def coef_(self):
        return self._w.copy()

    @property
    def coef_history(self):
        return self._coef_history.copy()

    def _prepare_data(self, X, y=None):
        X = np.array(X)
        if not self._data_prepared:
            if len(X.shape) != 2:
                raise ValueError("Wrong X shape (len(X.shape) != 2)!")

            if self._fit_intercept:
                intercept = np.ones((X.shape[0], 1))
                X = np.hstack((intercept, X))

        if y is None:
            return X

        y = np.array(y)
        return X, y

    def _init_weights(self, size):
        if self._w is None:
            np.random.seed(self._seed)
            self._w = np.random.rand(size)
            self.clear_history()

    def _l1_reg(self):
        if self._l1_ratio is None:
            return 0

        return lasso(self._w, lambda_=self._l1_ratio)

    def _l2_reg(self):
        if self._l2_ratio is None:
            return 0

        return ridge(self._w, lambda_=self._l2_ratio)

    def _early_stop(self, X, y):
        if self._tolerance is not None:
            loss = self._loss(X, y)
            if abs(loss - self._prev_error) < self._tolerance:
                return True

            self._prev_error = loss

        return False

    def clear_history(self):
        self._history = {"step": [], "loss": [], "score": []}
        self._coef_history = None

    def _save_history(self, X, y, iteration=None):
        if iteration is None or iteration % self._history_step == 0:
            self._history["step"].append(len(self._history["step"]))
            self._history["loss"].append(self._loss(X, y))
            self._history["score"].append(self._score(X, y))

            if self._coef_history is None:
                self._coef_history = self.coef_

            self._coef_history = np.vstack((self._coef_history, self.coef_))


class BaseGD(LinearModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        X, y = self._prepare_data(X, y)
        self._init_weights(X.shape[1])
        self._data_prepared = True

        for i in range(self._max_iter):
            self._save_history(X, y, i)
            self._gradient_step(X, y)
            if self._early_stop(X, y):
                break

        self._save_history(X, y)
        self._data_prepared = False
        return self.history


class SGD(LinearModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        X, y = self._prepare_data(X, y)
        self._init_weights(X.shape[1])
        self._data_prepared = True
        rand_indexes = np.random.randint(len(y), size=(self._max_iter,))

        for i, idx in enumerate(rand_indexes):
            self._save_history(X, y, i)
            self._gradient_step(X[idx:idx+1], y[idx:idx+1])
            if self._early_stop(X, y):
                break

        self._save_history(X, y)
        self._data_prepared = False
        return self.history


class MiniBatchGD(LinearModel):

    def __init__(self, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size

    def fit(self, X, y):
        X, y = self._prepare_data(X, y)
        self._init_weights(X.shape[1])
        self._data_prepared = True

        loader = BatchLoader(X, y, batch_size=self._batch_size)
        for i in range(self._max_iter):
            self._save_history(X, y, i)
            for batch_X, batch_y in loader:
                self._gradient_step(batch_X, batch_y)

            if self._early_stop(X, y):
                break

        self._save_history(X, y)
        self._data_prepared = False
        return self.history
