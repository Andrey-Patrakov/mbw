import numpy as np


def vote(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[counts.argmax()]


class Bagging:

    def __init__(self,
                 estimator=None,
                 n_estimators=100,
                 random_state=None,
                 prediction='voting',
                 **kwargs) -> None:

        self._estimators = []

        self._estimator = estimator
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._estimator_args = kwargs
        self._prediction = prediction.lower()

        if self._prediction not in ['voting', 'mean']:
            error = f'Unknown value for {prediction} for prediction param!'
            error += ' (value not in ("voting", "mean"))'
            raise ValueError(error)

    @property
    def oob(self):
        return self._oob

    @property
    def oob_true(self):
        return self._oob[0]

    @property
    def oob_pred(self):
        return self._oob[1]

    def fit(self, X, y):
        oob_values = {}
        self._estimators = []

        if self._random_state is not None:
            np.random.seed(int(self._random_state))

        X, y = np.array(X), np.array(y)
        for _ in range(self._n_estimators):
            data, labels, oob = self._get_bootstrap(X, y)
            estimator = self._estimator(**self._estimator_args)
            estimator.fit(data, labels)
            self._estimators.append(estimator)

            oob_pred = estimator.predict(oob[1])
            for idx in range(len(oob[0])):
                index = oob[0][idx]
                if index not in oob_values:
                    oob_values[index] = {'real': oob[2][idx], 'pred': []}

                oob_values[index]['pred'].append(oob_pred[idx])

        oob = []
        for row in oob_values.values():
            real = row['real']
            pred = self._get_prediction(row['pred'])
            oob.append(np.array([real, pred]))

        self._oob = np.array(oob).T

    def predict(self, X):
        y_pred = []
        for estimator in self._estimators:
            y_pred.append(estimator.predict(X))

        y_pred = np.array(y_pred).T
        results = []
        for row in y_pred:
            results.append(self._get_prediction(row))

        return np.array(results)

    def _get_prediction(self, y_pred):
        y_pred = np.array(y_pred)
        if self._prediction == 'voting':
            return vote(y_pred)

        elif self._prediction == 'mean':
            return np.mean(y_pred)

    def _get_bootstrap(self, X, y):
        n_samples = X.shape[0]
        all_indexes = np.arange(n_samples)
        rand_indexes = np.random.randint(n_samples, size=(n_samples,))
        oob_indexes = all_indexes[~np.isin(all_indexes, rand_indexes)]

        oob = (oob_indexes, X[oob_indexes], y[oob_indexes])
        return X[rand_indexes], y[rand_indexes], oob
