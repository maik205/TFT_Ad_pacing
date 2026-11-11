"""Metrics on observed clicks: log loss, AUC, expected calibration error, calibration in the large."""
import numpy as np
from sklearn.metrics import roc_auc_score

from . import contract as K


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=20):
    """Expected calibration error over equal-mass bins of the prediction."""
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return float(sum(abs(p[i].mean() - y[i].mean()) * i.size for i in parts if i.size) / p.size)


def reliability(y, p, bins=10):
    order = np.argsort(p, kind="stable")
    return [{"pred": float(p[i].mean()), "observed": float(y[i].mean()), "n": int(i.size)}
            for i in np.array_split(order, bins) if i.size]


def summary(y, p):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    out = {"n": int(y.size), "clicks": int(y.sum()), "logloss": logloss(y, p),
           "mean_pred": float(p.mean()), "ctr": float(y.mean()), "ece": ece(y, p)}
    # a model that knows only the base rate: the reference log loss
    base = np.full_like(p, y.mean())
    out["logloss_base_rate"] = logloss(y, base)
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else None
    return out


def report(y, p, fmt_tokens):
    out = {"all": summary(y, p), "reliability": reliability(y, p)}
    for i, f in enumerate(K.FORMATS):
        sel = fmt_tokens == i + 1
        if sel.any():
            out[f] = summary(y[sel], p[sel])
    return out
