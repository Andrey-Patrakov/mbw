import numpy as np


def manhattan(x, y):
    x, y = np.array(x), np.array(y)
    return np.sum(abs(x - y), axis=1)


def euclidean(x, y):
    x, y = np.array(x), np.array(y)
    return np.sum((x - y)**2, axis=1) ** (1/2)


def minkowski(x, y, p=2):
    x, y = np.array(x), np.array(y)
    return np.sum(abs(x - y) ** p, axis=1) ** (1/p)
