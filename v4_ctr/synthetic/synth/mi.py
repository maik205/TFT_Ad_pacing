"""Within-user MI estimator of Section 14 (same definition as the KuaiRand measurement)."""
import numpy as np

from .opportunities import local_time
from .world import BLOCK_STARTS, K


def block_of(local_hour):
    return np.searchsorted(BLOCK_STARTS, local_hour, side="right") - 1


def _mi(user, block, cat, n_users, min_events):
    T = np.zeros((n_users, 5, K))
    np.add.at(T, (user, block, cat), 1.0)
    tot = T.sum(axis=(1, 2))
    keep = tot >= min_events
    T, tot = T[keep], tot[keep]
    p = T / tot[:, None, None]
    pr = p.sum(axis=2, keepdims=True)
    pc = p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p / (pr * pc)), 0.0)
    mi_u = terms.sum(axis=(1, 2))
    return float((mi_u * tot).sum() / tot.sum())


def mi_excess(user, block, cat, n_users, seed=0, n_shuffles=5, min_events=50):
    """Observed MI minus the mean MI after shuffling blocks within each user."""
    rng = np.random.default_rng(seed)
    obs = _mi(user, block, cat, n_users, min_events)
    order = np.argsort(user, kind="stable")
    u_s, b_s, c_s = user[order], block[order], cat[order]
    nulls = []
    for _ in range(n_shuffles):
        perm = np.lexsort((rng.random(u_s.size), u_s))  # random order within each user
        nulls.append(_mi(u_s, b_s[perm], c_s, n_users, min_events))
    return obs - float(np.mean(nulls)), obs, float(np.mean(nulls))


def synthetic_mi_excess(cfg, users, opp, seed=0):
    _, _, _, lh = local_time(opp["ts"], users["tz"][opp["user"]])
    return mi_excess(opp["user"], block_of(lh), opp["context"], cfg["n_users"], seed=seed)
