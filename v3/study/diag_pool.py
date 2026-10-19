"""Diagnostic: does the O-P4 pool, with and without the true presence weights, project the right amount of traffic?

    python diag_pool.py WORLD_DIR [WORLD_DIR ...]

For each test day D (24..27), compares the opportunity count of day D with the pool projection from days D-1..D-7:
    unweighted = sum over lags d of count(D-d) / 7
    weighted   = sum over pool items of the O-P4 weight / 7   (full pool, no sampling)
A ratio of 1.0 means the projection is unbiased in total traffic.
"""
import sys

import numpy as np

from pacing.harness import World
from pacing.observed import Observed
from pacing.oracle_policies import DAY, LAGS, OracleForecast


def main(wdirs):
    for wdir in wdirs:
        obs, world = Observed(wdir), World(wdir)
        pol = OracleForecast(None, world)
        pol.obs = obs
        cnt = np.bincount((obs.ts // DAY).astype(int), minlength=28)
        us = world.users
        rows = []
        for D in range(24, 28):
            idx = np.flatnonzero((obs.ts >= (D - LAGS) * DAY) & (obs.ts < D * DAY))
            lag = D - (obs.ts[idx] // DAY).astype(int)  # 1..LAGS
            ar = np.arange(idx.size)
            tot = {"none": idx.size / LAGS, "user": pol._presence(idx)[ar, lag - 1].sum() / LAGS}
            for mode in ("agg", "user_norm"):
                pm = OracleForecast(None, world, weights=mode)
                pm.obs = obs
                tot[mode] = pm._expected_weights(idx)[ar, lag - 1].sum() / LAGS
            rows.append((D, cnt[D], tot))
        alive = [float(((us["join"] * 24 <= (d + 0.5) * 24) & ((d + 0.5) * 24 < us["leave"] * 24)).mean()) for d in (17, 24)]
        print(wdir.split("/")[-1], "alive share day17 %.2f day24 %.2f" % tuple(alive))
        for D, c, tot in rows:
            print(f"  day {D}: actual {c}  " + "  ".join(f"{k} {v / c:.3f}" for k, v in tot.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
