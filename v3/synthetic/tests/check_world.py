"""Spec Section 15 test 9 on a full-size generated world: fits hit their targets.

    python tests/check_world.py OUT_DIR [--gamma-real]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# KuaiRand-Pure AD views: watch-fraction quantiles at p10..p90 (spec Section 7a.2)
TARGET_P = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
TARGET_Q = np.array([0.00, 0.01, 0.02, 0.13, 0.70])
TOL = 0.02


def infeed_watch_fraction(world_dir):
    obs = os.path.join(world_dir, "observed")
    opp = pd.read_parquet(os.path.join(obs, "synth_opportunities.parquet"),
                          columns=["opp_id", "format", "video_duration_s"])
    eng = pd.read_parquet(os.path.join(obs, "synth_engagement.parquet"), columns=["opp_id", "watch_time_s"])
    m = eng.merge(opp[opp["format"] == "infeed"], on="opp_id")
    return (m["watch_time_s"] / m["video_duration_s"]).to_numpy()


def levy_points(f):
    """Per target point: quantile distance, share distance, and whether either is within TOL (deviation 3)."""
    q = np.quantile(f, TARGET_P)
    share = np.array([(f <= t).mean() for t in TARGET_Q])
    dq, dp = np.abs(q - TARGET_Q), np.abs(share - TARGET_P)
    return q, share, (dq <= TOL) | (dp <= TOL)


def main(world_dir, gamma_real=False):
    meta = json.load(open(os.path.join(world_dir, "observed", "synth_meta.json")))
    ch, fails = meta["checks"], []
    f = ch.get("infeed")
    if f:
        q, share, ok = levy_points(infeed_watch_fraction(world_dir))
        for i in np.flatnonzero(~ok):
            fails.append(f"in-feed point p{int(TARGET_P[i] * 100)}: quantile {q[i]:.4f} vs {TARGET_Q[i]}, "
                         f"share {share[i]:.4f} vs {TARGET_P[i]}")
        if abs(f["long_view"] - 0.24) > 0.02:
            fails.append(f"long view {f['long_view']:.3f} vs 0.24")
    imp = pd.read_parquet(os.path.join(world_dir, "observed", "synth_impressions.parquet"),
                          columns=["campaign_id", "cost_C5"]).dropna()
    lc = np.log(imp["cost_C5"].to_numpy())
    sd = np.std(lc - imp.assign(l=lc).groupby("campaign_id")["l"].transform("mean").to_numpy())
    if abs(sd / 0.865 - 1) > 0.02:
        fails.append(f"C5 within-campaign log-cost sd {sd:.4f} vs 0.865")
    if gamma_real:
        ex = ch["mi_excess_bits"]["excess"]
        if abs(ex / 0.0255 - 1) > 0.10:
            fails.append(f"MI excess {ex:.4f} vs 0.0255")
    print(json.dumps({"world": world_dir, "config_hash": meta["config_hash"], "fails": fails}, indent=1))
    return not fails


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], "--gamma-real" in sys.argv) else 1)
