import numpy as np


def r2_score(y_true, y_pred):
    SSres = np.sum((y_true - y_pred)**2)
    SStot = np.sum((y_true - np.mean(y_pred))**2)
    return 1 - (SSres / SStot)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    positive = y_pred.astype(int) == 1
    negative = y_pred.astype(int) == 0

    tp = sum(y_true[positive] == y_pred[positive])
    fp = sum(y_true[positive] != y_pred[positive])
    tn = sum(y_true[negative] == y_pred[negative])
    fn = sum(y_true[negative] != y_pred[negative])

    return np.array([[tn, fn], [fp, tp]])


def precision(y_true, y_pred):
    (tn, fn), (fp, tp) = confusion_matrix(y_true, y_pred)
    if tp + fp == 0:
        return 1

    return tp / (tp + fp)


def recall(y_true, y_pred):
    (tn, fn), (fp, tp) = confusion_matrix(y_true, y_pred)
    if tp + fp == 0:
        return 1

    return tp / (tp + fn)


def f_score(y_true, y_pred, B=1):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return (1 + B) * (precision_ * recall_) / (B**2 * precision_ + recall_)
