"""Equal-budget tuning of the oracle policies (study protocol Section 7): 20 random-search configurations on the
validation days (21-23) of one world, scored on the primary metric (value relative to UB, ratio of totals over
complete campaign-days; protocol deviation 2). The chosen configuration is then used unchanged on every world.

    python tune.py WORLD_DIR METHOD [--n 20] [--seed 0] [--cost C1] [--rho 0.5] [--out tuned_METHOD.json]

METHOD is O-P1 or O-P4. Search spaces (fixed here, before any test run):
    O-P1: eta log-uniform [1, 500], lam0 uniform [0, 1]
    O-P4: damp uniform [0.3, 1.0], kappa uniform [0.8, 1.2]
"""
import argparse
import json
import time

import numpy as np

from pacing.harness import World, run
from pacing.observed import Observed
from pacing.oracle_policies import OracleForecast, OracleReactive, Truth, score_run, upper_bound

VALID_DAYS = (21, 24)


def sample(method, rng):
    if method == "O-P1":
        return {"eta": float(np.exp(rng.uniform(np.log(1), np.log(500)))), "lam0": float(rng.uniform(0, 1))}
    return {"damp": float(rng.uniform(0.3, 1.0)), "kappa": float(rng.uniform(0.8, 1.2))}


def build(method, truth, world, params):
    if method == "O-P1":
        return OracleReactive(truth, **params)
    return OracleForecast(truth, world, **params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("method", choices=["O-P1", "O-P4"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost", default="C1")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    obs, world, truth = Observed(a.world), World(a.world), Truth(a.world, a.cost)
    t = time.time()
    ub, diag = upper_bound(truth, obs, a.cost, a.rho, VALID_DAYS)
    print(f"UB {json.dumps(diag)} {time.time() - t:.0f} s", flush=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for i in range(a.n):
        params = sample(a.method, rng)
        t = time.time()
        res = run(world, obs, build(a.method, truth, world, params), cost=a.cost, rho=a.rho, days=VALID_DAYS)
        cd = score_run(res, obs, ub)
        score = float(cd["value"].sum() / cd["ub_value"].sum())
        rows.append({"params": params, "score": score, "utilisation": float(cd["utilisation"].mean()),
                     "clicks_expected": float(cd["exp_clicks"].sum()), "seconds": round(time.time() - t)})
        print(i, json.dumps(rows[-1]), flush=True)
    best = max(rows, key=lambda r: r["score"])
    out = {"method": a.method, "world": a.world, "cost": a.cost, "rho": a.rho, "days": VALID_DAYS, "n": a.n,
           "seed": a.seed, "best": best, "all": rows, "ub": diag}
    path = a.out or f"tuned_{a.method}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
