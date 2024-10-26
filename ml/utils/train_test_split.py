import numpy as np
from math import ceil


def train_test_split(*arrays,
                     random_state=None,
                     train_size=None,
                     test_size=None,
                     shuffle=True):
    results = []

    if train_size is None:
        if test_size is None:
            test_size = 0.25

        train_size = 1 - test_size

    train_size = max(train_size, 0)
    train_size = min(train_size, 1)
    indexes = np.arange(len(arrays[0]))

    if random_state is not None:
        np.random.seed(int(random_state))

    if shuffle:
        np.random.shuffle(indexes)

    idx = ceil(len(indexes) * train_size)
    train_idx = indexes[:idx]
    test_idx = indexes[idx:]
    for arr in arrays:
        array = np.array(arr)
        results.append(array[train_idx, ...])
        results.append(array[test_idx, ...])

    return results
