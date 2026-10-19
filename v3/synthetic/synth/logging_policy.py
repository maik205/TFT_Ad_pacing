"""Logging policy (Section 10): epsilon-mixture of a user-blind heuristic softmax and uniform, with exact
propensities, paced by a per-campaign throttle against C1 budgets."""
import numpy as np

DAY = 86400


def campaign_local_day(ts, tz):
    return np.floor((ts + 3600.0 * tz) / DAY).astype(np.int64)


def run_logging(cfg, camp, opp, rng, budget_c1=None):
    """Returns choice (-1 = no ad), propensity, eligible count, and the C3 internal second bid.

    budget_c1: per-campaign daily budget in C1 units, or None for the unbudgeted pass. The throttle
    lets campaign c bid in a step only while its spend today is below B_c * (fraction of its local day
    elapsed by the end of the step). Spend is checked once per step, so a campaign can overshoot B_c
    within a step; measured mean utilisation under C1 is about 1.17 (spec Section 17).
    """
    C = camp["v"].size
    match = (camp["m"].T > 0)  # (K, C)
    ts = opp["ts"]
    S = ts.size
    step = 3600 / cfg["pacing_steps_per_hour"]
    choice = np.full(S, -1, np.int64)
    prop = np.zeros(S)
    n_elig = np.zeros(S, np.int64)
    bid2 = np.full(S, np.nan)
    days = cfg["days"]
    spend = np.zeros((C, days + 3))  # by campaign-local day, offset by 1
    bounds = np.searchsorted(ts, np.arange(0, days * DAY + step, step))
    eps, tau = cfg["epsilon"], cfg["tau"]
    for b in range(bounds.size - 1):
        lo, hi = bounds[b], bounds[b + 1]
        if lo == hi:
            continue
        t0 = b * step
        if budget_c1 is None:
            elig = np.ones(C, bool)
        else:
            d = campaign_local_day(np.full(C, t0), camp["tz"])
            frac = np.minimum(1.0, ((t0 + 3600.0 * camp["tz"]) - d * DAY + step) / DAY)
            sp = spend[np.arange(C), d + 1]
            elig = (sp < budget_c1 * frac) & (sp < budget_c1)
        ctx = opp["context"][lo:hi]
        n = hi - lo
        score = camp["v"][None, :] * (1 + cfg["delta_ctx"] * match[ctx])  # (n, C)
        z = np.where(elig[None, :], score / tau, -np.inf)
        z = np.concatenate([z, np.full((n, 1), cfg["none_score"] / tau)], axis=1)
        z -= z.max(axis=1, keepdims=True)
        soft = np.exp(z)
        soft /= soft.sum(axis=1, keepdims=True)
        ne = int(elig.sum())
        uni = np.concatenate([elig, [True]]).astype(float) / (ne + 1)
        p = (1 - eps) * soft + eps * uni[None, :]
        cdf = np.cumsum(p, axis=1)
        a = np.minimum((cdf < rng.random(n)[:, None]).sum(axis=1), C)
        prop[lo:hi] = p[np.arange(n), a]
        a = np.where(a == C, -1, a)
        choice[lo:hi] = a
        n_elig[lo:hi] = ne
        # C3: highest other eligible internal bid, bid = m_c * (1 + delta_ctx * context match)
        bids = np.where(elig[None, :], camp["m_cpm"][None, :] * (1 + cfg["delta_ctx"] * match[ctx]), -np.inf)
        shown = a >= 0
        if shown.any():
            r = np.flatnonzero(shown)
            bids_o = bids[r].copy()
            bids_o[np.arange(r.size), a[r]] = -np.inf
            best = bids_o.max(axis=1)
            bid2[lo + r] = np.where(np.isfinite(best), best, np.nan)
            if budget_c1 is not None:
                cc = a[r]
                dd = campaign_local_day(ts[lo + r], camp["tz"][cc])
                np.add.at(spend, (cc, dd + 1), camp["m_cpm"][cc])
    return {"choice": choice, "propensity": prop, "n_eligible": n_elig, "c3_bid2": bid2}


def training_daily_mean(cfg, camp, ts, campaign, amount):
    """Mean spend per complete campaign-local day inside the training days, per campaign."""
    C = camp["v"].size
    tz = camp["tz"][campaign]
    d = campaign_local_day(ts, tz)
    tot = np.zeros((C, cfg["days"] + 3))
    np.add.at(tot, (campaign, d + 1), amount)
    out = np.zeros(C)
    end = cfg["train_days"] * DAY
    for c in range(C):
        days = [dd for dd in range(-1, cfg["days"] + 1)
                if dd * DAY - 3600 * camp["tz"][c] >= 0 and (dd + 1) * DAY - 3600 * camp["tz"][c] <= end]
        out[c] = tot[c, [dd + 1 for dd in days]].mean()
    return out
