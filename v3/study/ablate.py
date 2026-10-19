"""Ablation of O-P4: what do the user-level presence weights add over plain traffic forecasting?

    python ablate.py OUT_PREFIX TUNED_O-P4.json CELLS.npz[,CELLS.npz ...] WORLD_DIR [WORLD_DIR ...] --costs C1 C4
    python ablate.py --combine OUT.json CELLS.npz [CELLS.npz ...] ABLATION.npz [ABLATION.npz ...]

First form: runs O-P4a (the O-P4 scenario pool with every opportunity weighted 1, i.e. aggregate traffic
forecast; same tuned settings) on the test days and saves its per-campaign-day value. UB, O-P1 and O-P4 come
from the cells saved by g2.py (same worlds, same campaign-day order, checked by length).
Second form: per cost model, the shares of UB for O-P1, O-P4a and O-P4, and the paired gains
O-P4a - O-P1 (traffic forecast alone) and O-P4 - O-P4a (user-level presence weights), with the g2.py bootstrap.
"""
import argparse
import json
import sys
import time

import numpy as np

from g2 import TEST_DAYS, bootstrap_gain


def compute(a):
    from pacing.harness import World, run
    from pacing.metrics import campaign_days
    from pacing.observed import Observed
    from pacing.oracle_policies import OracleForecast, Truth

    p4 = json.load(open(a.tuned_p4))["best"]["params"]
    saved = {}
    for f in a.cells.split(","):
        z = np.load(f)
        saved.update({k: z[k] for k in z.files})
    out = {}
    keymap = OracleForecast.MODES
    for wdir in a.worlds:
        obs, world = Observed(wdir), World(wdir)
        for cost in a.costs:
            for mode in a.modes:
                t = time.time()
                res = run(world, obs, OracleForecast(Truth(wdir, cost), world, weights=mode, **p4),
                          cost=cost, rho=a.rho, days=TEST_DAYS)
                v = campaign_days(res, obs)["value"].to_numpy()
                tag, key = f"{cost}|{wdir}", keymap[mode]
                ub = saved[f"{tag}|ub"]
                assert ub.size == v.size, (tag, ub.size, v.size)
                out[f"{tag}|{key}"] = v
                g = (v.sum() - saved[f"{tag}|O-P1"].sum()) / ub.sum()
                print(cost, wdir, key, f"gain over O-P1 {g:+.4f}  {time.time() - t:.0f} s", flush=True)
                np.savez(a.out + ".npz", **out)


def combine(out, files):
    cells = {}
    for f in files:
        z = np.load(f)
        cells.update({k: z[k] for k in z.files})
    keys = [k for k in ("O-P1", "O-P4a", "O-P4g", "O-P4n", "O-P4") if any(c.endswith("|" + k) for c in cells)]
    tags = sorted({c.rsplit("|", 1)[0] for c in cells if c.endswith("|ub")})
    res = {}
    for cost in sorted({t.split("|")[0] for t in tags}):
        ws = [t for t in tags if t.split("|")[0] == cost and all(f"{t}|{k}" in cells for k in keys)]
        if not ws:
            continue
        g = lambda t, k: cells[f"{t}|{k}"]
        keep = [g(t, "ub") > 0 for t in ws]
        sh = {k: float(np.mean([g(t, k).sum() / g(t, "ub").sum() for t in ws])) for k in keys}
        r = {"worlds": len(ws), "share_of_ub": sh, "gain_over_O-P1": {}, "gain_over_O-P4a": {}}
        for base, name in (("O-P1", "gain_over_O-P1"), ("O-P4a", "gain_over_O-P4a")):
            for k in keys:
                if k == base or (base == "O-P4a" and k == "O-P1"):
                    continue
                bs = [(g(t, base)[m], g(t, k)[m], g(t, "ub")[m]) for t, m in zip(ws, keep)]
                r[name][k] = {"gain": sh[k] - sh[base], "ci95": bootstrap_gain(bs)}
        res[cost] = r
        print(cost, f"worlds {len(ws)}  " + "  ".join(f"{k} {v:.3f}" for k, v in sh.items()), flush=True)
        for name, d in (("vs O-P1", r["gain_over_O-P1"]), ("vs O-P4a", r["gain_over_O-P4a"])):
            print("   " + name + ": " + "  ".join(f"{k} {v['gain']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for k, v in d.items()), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--combine":
        files = sys.argv[3:]
        return combine(sys.argv[2], files)
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("tuned_p4")
    ap.add_argument("cells")
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--costs", nargs="+", default=["C1", "C4", "C5", "C6"])
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=["none"], choices=list(("none", "user", "agg", "user_norm")))
    compute(ap.parse_args())


if __name__ == "__main__":
    main()
