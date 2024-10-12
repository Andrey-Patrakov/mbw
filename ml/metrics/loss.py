import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.mean(- y * np.log(p) - (1.0 - y) * np.log(1.0 - p))
