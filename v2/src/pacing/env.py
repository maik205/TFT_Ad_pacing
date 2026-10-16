"""Impression-replay pacing environment with probabilistic throttling (PROTOCOL 4).

An episode is one campaign-day. At the start of each hour the policy sets a participation
rate alpha in [0, 1]. Each logged impression of that hour is kept when its uniform draw u < alpha,
and kept impressions are served in log order until the next one would exceed the budget.
Draws u are fixed per (replay seed, campaign, hour), so all policies face the same
randomness (common random numbers).
"""

from dataclasses import dataclass

import numpy as np

from . import metrics as M
from . import config as C


@dataclass
class Episode:
    c: int               # campaign index
    day: int             # day index; hours 24*day .. 24*day+23
    budget: float
    cost: list           # 24 arrays of impression costs (log order)
    click: list
    conv: list

    @property
    def demand(self):
        return np.array([x.sum() for x in self.cost])


class ImpressionStore:
    def __init__(self, imp, campaigns):
        cidx = {c: i for i, c in enumerate(campaigns)}
        self.data = {}
        for (camp, hour), g in imp.groupby(["campaign", "hour"], sort=False):
            self.data[(cidx[camp], int(hour))] = (
                g.cost.to_numpy(np.float64), g.click.to_numpy(np.int64), g.attribution.to_numpy(np.int64))
        self.empty = (np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64))

    def episode(self, c, day, budget):
        hours = [self.data.get((c, 24 * day + h), self.empty) for h in range(24)]
        return Episode(c, day, budget, [x[0] for x in hours], [x[1] for x in hours], [x[2] for x in hours])


def uniforms(ep, seed):
    return [np.random.default_rng([seed, ep.c, 24 * ep.day + h]).random(len(ep.cost[h])) for h in range(24)]


def run_episode(ep, policy, u):
    policy.reset(ep)
    B, S = ep.budget, 0.0
    alphas, spends, cums = [], [], []
    clicks = convs = 0
    for h in range(24):
        a = float(np.clip(policy.act(h, S, spends), 0.0, 1.0))
        cost = ep.cost[h]
        spent = 0.0
        if S < B and len(cost):
            kept = np.nonzero(u[h] < a)[0]
            cs = np.cumsum(cost[kept])
            n_ok = int(np.searchsorted(cs, B - S, side="right"))
            if n_ok:
                served = kept[:n_ok]
                spent = float(cs[n_ok - 1])
                clicks += int(ep.click[h][served].sum())
                convs += int(ep.conv[h][served].sum())
        S += spent
        alphas.append(a)
        spends.append(spent)
        cums.append(S)
    return {
        "spend": S, "clicks": clicks, "convs": convs,
        "depletion_hour": M.depletion_hour(cums, B, C.DEPLETION_TOL),
        "pvi": M.pacing_volatility(alphas),
        "hourly_spend": np.array(spends), "alphas": np.array(alphas),
    }
