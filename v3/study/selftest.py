"""Development checks of the oracle policies on a (small) world.

    python selftest.py WORLD_DIR [COST] [DAY0 DAY1]
"""
import sys
import time

import numpy as np

from pacing.harness import World, run
from pacing.observed import Observed
from pacing.oracle_policies import OracleForecast, OracleReactive, Truth, score_run, upper_bound
from pacing.policies import EvenThrottle


def main(world_dir, cost="C1", days=(21, 24)):
    obs, world = Observed(world_dir), World(world_dir)
    # 1. matrices equal the oracle's expected values
    for k in ("C1", "C2", "C4", "C5", "C6"):
        tr = Truth(world_dir, k)
        rng = np.random.default_rng(1)
        opp = np.sort(rng.choice(obs.n, 40, replace=False))
        n_uc = rng.integers(0, 3, (opp.size, obs.C)).astype(float)
        P, K = tr.pk(opp, n_uc)
        ex = tr.o.expected(np.repeat(opp, obs.C), np.tile(np.arange(obs.C), opp.size), n_uc.ravel())
        ep, ek = ex["pctr"].to_numpy().reshape(P.shape), ex[f"cost_{k}"].to_numpy().reshape(K.shape)
        print(k, "max |dP|", float(np.abs(P - ep).max()), "max |dK|", float(np.abs(K - ek).max()))
        assert np.abs(P - ep).max() < 1e-9 and np.abs(K - ek).max() < 1e-9
    # 2. policies and the bound
    tr = Truth(world_dir, cost)
    t = time.time()
    ub, diag = upper_bound(tr, obs, cost, 0.5, days)
    print("UB", {k: round(v, 4) for k, v in diag.items()}, f"{time.time() - t:.0f} s")
    pols = [EvenThrottle(seed=1), OracleReactive(tr), OracleReactive(tr, eta=300.0),
            OracleForecast(tr, world, user_weights=False), OracleForecast(tr, world)]
    for p in pols:
        t = time.time()
        res = run(world, obs, p, cost=cost, rho=0.5, days=days)
        cd = score_run(res, obs, ub)
        print(f"{res['policy']:6s} sum-ratio {cd['value'].sum() / cd['ub_value'].sum():.4f} (UB {cd['ub_value'].sum():.0f})  mean-ratio {cd['ratio'].mean():.3f}  value {cd['value'].sum():8.1f}  "
              f"util {cd['utilisation'].mean():.3f}  early-depl {cd['early_depletion'].mean():.2f}  "
              f"max ratio {cd['ratio'].max():.3f}  {time.time() - t:.0f} s")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2] if len(a) > 2 else "C1", (int(a[3]), int(a[4])) if len(a) > 4 else (21, 24))
