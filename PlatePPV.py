import numpy as np
from sklearn.metrics import make_scorer


def ppv_top_n(y_true, y_pred, top_n: int, threshold: float = 0.5) -> float:
    """
    calculate the ppv (precision) taking into account only the top N prediction with the highest score/probability
    :param y_true: real classes
    :param y_pred: predicted probability of positive class
    :param top_n: number of top prediction to choose (default = 128)
    :param threshold: the probability threshold to consider a compound active or not
    :return: float; the metric value (0-1)
    """
    y_pred = np.atleast_1d(y_pred)
    y_true = np.atleast_1d(y_true)
    _tmp = np.vstack((y_true, y_pred)).T[y_pred.argsort()[::-1]][:top_n, :]
    _tmp = _tmp[np.where(_tmp[:, 1] > threshold)[0]].copy()
    return np.sum(_tmp[:, 0]) / len(_tmp)


# this function is can be used anywhere sklearn expects a scorer (like in the `cross_validate` function)
plate_ppv = make_scorer(ppv_top_n, needs_proba=True, top_n=128)  # change 'top_n' to whatever N you want to use
