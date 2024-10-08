import numpy as np


# L1 regularization
def lasso(w, lambda_):
    return lambda_ * np.where(w != 0, w/abs(w), 0)


# L2 regularization
def ridge(w, lambda_):
    return 2 * lambda_ * w
