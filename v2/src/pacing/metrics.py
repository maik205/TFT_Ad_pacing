"""Forecast and pacing metrics (PROTOCOL 7). Each function works on one series or episode."""

import numpy as np


def pinball(y, yq, q):
    u = np.asarray(y) - np.asarray(yq)
    return float(np.mean(np.maximum(q * u, (q - 1) * u)))


def seasonal_naive_scale(train_series, season=24):
    """In-sample MAE of the lag-`season` naive forecast on one campaign's training series."""
    s = np.asarray(train_series, dtype=float)
    return float(np.mean(np.abs(s[season:] - s[:-season])))


def mase(y, y_med, scale):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_med))) / scale)


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def crossing_rate(qpreds):
    """Share of (origin, step) points whose quantiles are not non-decreasing.
    qpreds: array (..., n_quantiles) ordered by increasing quantile level."""
    d = np.diff(qpreds, axis=-1)
    return float(np.mean(np.any(d < 0, axis=-1)))


def pacing_volatility(alphas):
    """RMS of hour-to-hour changes of the participation rate within a day."""
    a = np.asarray(alphas, dtype=float)
    return float(np.sqrt(np.mean(np.diff(a) ** 2)))


def depletion_hour(cum_spend_end_of_hour, budget, tol):
    """Index h such that the budget is exhausted by the end of hour h; 24 if never."""
    hit = np.nonzero(np.asarray(cum_spend_end_of_hour) >= tol * budget)[0]
    return int(hit[0]) if len(hit) else 24


def profile_deviation(hourly_spend, reference_hourly_spend, budget):
    """Total-variation distance between the spend profile and a reference, as a share of budget."""
    return float(np.sum(np.abs(np.asarray(hourly_spend) - np.asarray(reference_hourly_spend))) / (2 * budget))
