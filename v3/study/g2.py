"""Gate G2 (study protocol Section 8.1): O-P4 - O-P1 on the test days, oracle predictions, with tuned settings.

    python g2.py OUT_PREFIX TUNED_O-P1.json TUNED_O-P4.json WORLD_DIR [WORLD_DIR ...] [--costs C1 C2 C4 C5 C6] [--rho 0.5]
    python g2.py --combine OUT.json CELLS.npz [CELLS.npz ...]

First form: for every world and cost model, runs P0, O-P1 and O-P4 on test days 24-27, computes UB, and saves the
per-campaign-day cells to OUT_PREFIX.npz and the per-world summaries to OUT_PREFIX.json. One world is held in
memory at a time, so runs can be split over processes by cost model.
Second form: pools the saved cells of several runs and reports, per cost model, the primary metric (summed
value over complete campaign-days relative to summed UB value; deviation 2) and the paired gain O-P4 - O-P1
with a hierarchical bootstrap 95% CI (world, then campaign-day; 2,000 resamples).
Verdict: gain >= 1% of UB (the threshold of G2) or not; the author decides what follows.
"""
import argparse
import json
import sys
import time

import numpy as np

TEST_DAYS = (24, 28)
KEYS = ("P0", "O-P1", "O-P4")


def bootstrap_gain(cells, n=2000, seed=0):
    """cells: per world, arrays (v1, v4, ub) over campaign-days. Gain = (sum v4 - sum v1) / sum ub."""
    rng = np.random.default_rng(seed)
    W = len(cells)
    out = np.empty(n)
    for b in range(n):
        s4 = s1 = su = 0.0
        for w in rng.integers(0, W, W):
            v1, v4, ub = cells[w]
            i = rng.integers(0, ub.size, ub.size)
            s1, s4, su = s1 + v1[i].sum(), s4 + v4[i].sum(), su + ub[i].sum()
        out[b] = (s4 - s1) / su
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def compute(a):
    from pacing.harness import World, run
    from pacing.observed import Observed
    from pacing.oracle_policies import OracleForecast, OracleReactive, Truth, score_run, upper_bound
    from pacing.policies import EvenThrottle

    p1 = json.load(open(a.tuned_p1))["best"]["params"]
    p4 = json.load(open(a.tuned_p4))["best"]["params"]
    summaries, cells = [], {}
    for wdir in a.worlds:
        obs, world = Observed(wdir), World(wdir)
        for cost in a.costs:
            truth = Truth(wdir, cost)
            t = time.time()
            ub, diag = upper_bound(truth, obs, cost, a.rho, TEST_DAYS)
            rec = {"world": wdir, "cost": cost, "ub": diag}
            cd = {}
            for name, pol in (("P0", EvenThrottle(seed=1)), ("O-P1", OracleReactive(truth, **p1)),
                              ("O-P4", OracleForecast(truth, world, **p4))):
                res = run(world, obs, pol, cost=cost, rho=a.rho, days=TEST_DAYS)
                x = cd[name] = score_run(res, obs, ub)
                rec[name] = {"value_over_ub": float(x["value"].sum() / x["ub_value"].sum()),
                             "mean_ratio_per_campaign_day": float(x["ratio"].mean()),
                             "expected_clicks": float(x["exp_clicks"].sum()),
                             "utilisation": float(x["utilisation"].mean()),
                             "early_depletion_share": float(x["early_depletion"].mean()),
                             "wasted_spend_share": float(x["absent_spend"].sum() / max(x["spend"].sum(), 1e-300)),
                             "campaign_days": int(len(x))}
            rec["gain_O-P4_minus_O-P1"] = rec["O-P4"]["value_over_ub"] - rec["O-P1"]["value_over_ub"]
            rec["seconds"] = round(time.time() - t)
            summaries.append(rec)
            tag = f"{cost}|{wdir}"
            for k in KEYS:
                cells[f"{tag}|{k}"] = cd[k]["value"].to_numpy()
            cells[f"{tag}|ub"] = cd["O-P1"]["ub_value"].to_numpy()
            print(cost, wdir, json.dumps({k: rec[k] for k in ("gain_O-P4_minus_O-P1", "seconds")}), flush=True)
            np.savez(a.out + ".npz", **cells)
            json.dump({"tuned": {"O-P1": p1, "O-P4": p4}, "rho": a.rho, "days": TEST_DAYS, "runs": summaries},
                      open(a.out + ".json", "w"), indent=1)


def combine(out, files):
    cells = {}
    for f in files:
        z = np.load(f)
        cells.update({k: z[k] for k in z.files})
    tags = sorted({k.rsplit("|", 1)[0] for k in cells})
    costs = sorted({t.split("|")[0] for t in tags})
    res = {"worlds_per_cost": {}, "costs": {}}
    for cost in costs:
        worlds = sorted(t.split("|", 1)[1] for t in tags if t.split("|")[0] == cost)
        per, bs = [], []
        for w in worlds:
            g = lambda k: cells[f"{cost}|{w}|{k}"]
            ub = g("ub")
            keep = ub > 0
            tot = {k: float(g(k).sum() / ub.sum()) for k in KEYS}
            per.append({"world": w, **tot, "gain": tot["O-P4"] - tot["O-P1"]})
            bs.append((g("O-P1")[keep], g("O-P4")[keep], ub[keep]))
        gain = float(np.mean([r["gain"] for r in per]))
        ci = bootstrap_gain(bs)
        res["costs"][cost] = {"worlds": len(worlds), "per_world": per, "gain": gain, "ci95": ci,
                              "mean_value_over_ub": {k: float(np.mean([r[k] for r in per])) for k in KEYS},
                              "meets_threshold": bool(gain >= 0.01), "ci_excludes_zero": bool(ci[0] > 0)}
        print(cost, f"worlds {len(worlds)}  gain {gain:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"O-P1 {res['costs'][cost]['mean_value_over_ub']['O-P1']:.3f}  "
              f"O-P4 {res['costs'][cost]['mean_value_over_ub']['O-P4']:.3f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--combine":
        return combine(sys.argv[2], sys.argv[3:])
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("tuned_p1")
    ap.add_argument("tuned_p4")
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--costs", nargs="+", default=["C1", "C2", "C4", "C5", "C6"])
    ap.add_argument("--rho", type=float, default=0.5)
    compute(ap.parse_args())


if __name__ == "__main__":
    main()
