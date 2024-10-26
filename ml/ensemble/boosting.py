import numpy as np
from mbw.ml.utils.train_test_split import train_test_split


class BoostingRegressor:

    def __init__(self,
                 estimator=None,
                 eta=0.1,
                 n_estimators=100,
                 random_state=None,
                 estimator_coefs=None,
                 subsample=1.0,
                 **kwargs) -> None:

        self._estimators = []
        self._base_prediction = None

        self._eta = eta
        self._estimator = estimator
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._estimator_coefs = estimator_coefs
        self._subsample = subsample
        self._estimator_args = kwargs

    def _bias(self, X, y):
        z = self.predict(X)
        return -2 * (z - y)

    def fit(self, X, y):
        if self._random_state is not None:
            np.random.seed(int(self._random_state))

        X, y = np.array(X), np.array(y)

        if self._estimator_coefs is None:
            self._estimator_coefs = np.ones(self._n_estimators)

        for _ in range(self._n_estimators):
            estimator = self._estimator(**self._estimator_args)

            data = train_test_split(X, y, train_size=self._subsample)
            X_train, _, y_train, _ = data

            if len(self._estimators) == 0:
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, self._bias(X_train, y_train))

            self._estimators.append(estimator)

    def predict(self, X):
        y_pred = []
        for estimator, coef in zip(self._estimators, self._estimator_coefs):
            y_pred.append(self._eta * coef * estimator.predict(X))

        y_pred = np.array(y_pred).T
        y_pred = y_pred.sum(axis=1)

        return y_pred
