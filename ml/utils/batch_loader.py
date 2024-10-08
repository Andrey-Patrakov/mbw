import numpy as np


class BatchLoader:

    def __init__(self,
                 dataset,
                 y=None,
                 batch_size=1,
                 shuffle=False,
                 random_state=None):

        self._dataset = np.array(dataset)
        self._y = y
        self._indexes = np.arange(dataset.shape[0])
        self._batch_size = batch_size

        if self._y is not None:
            self._y = np.array(self._y)
            if self._y.shape[0] != self._dataset.shape[0]:
                error = 'X and y has different row count'
                error += f' ({self._dataset.shape[0]} != {self._y.shape[0]})'
                raise ValueError(error)

        np.random.seed(random_state)
        if shuffle:
            np.random.shuffle(self._indexes)

        self._get_batch_count()

    def __len__(self):
        return self._batch_count

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                f'Index {index} is out of bounds [0, {len(self)-1}]'
            )

        start = self._batch_size * (index)
        end = self._batch_size * (index + 1)
        idx = self._indexes[start:end]

        if self._y is not None:
            return self._dataset[idx, ...], self._y[idx, ...]

        return self._dataset[idx, ...]

    def _get_batch_count(self):
        n = self._dataset.shape[0]
        self._batch_count = max(1, n // self._batch_size)
        if n % self._batch_count != 0:
            self._batch_count += 1
