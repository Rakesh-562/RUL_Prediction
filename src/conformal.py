"""Prediction intervals with split conformal prediction.

Idea: measure how wrong the model is on calibration engines it never trained on,
then widen every new prediction by that amount. The width is computed separately
for three ranges of predicted RUL, so the interval can be narrower near failure.
"""
import numpy as np

BINS = [50, 100]   # predicted RUL < 50 | 50-100 | > 100


def _bin(pred):
    return np.digitize(pred, BINS)


def fit(pred_cal, y_cal, alpha=0.1):
    """Half-width q for each bin: the (1 - alpha) quantile of |error| on calibration data."""
    q = []
    for b in range(len(BINS) + 1):
        err = np.abs(y_cal - pred_cal)[_bin(pred_cal) == b]
        n = len(err)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)   # small-sample correction
        q.append(np.quantile(err, level, method="higher"))
    return np.array(q)


def interval(pred, q):
    """Lower and upper bound for each prediction."""
    half = q[_bin(pred)]
    return np.clip(pred - half, 0, None), pred + half


def coverage(y, lower, upper):
    """Fraction of true values that fall inside their interval."""
    return np.mean((y >= lower) & (y <= upper))
