"""One-off calibration before generation (spec Sections 7a.2, 13, 14). Writes calibration.json.

    python -m synth.calibrate [--n-users 50000]

1. gamma_real: bisection on gamma so the synthetic within-user MI excess equals KuaiRand's 0.0255 bits.
2. In-feed watch fraction: fit (e0, c0, aL, bL, aS, bS) on the default world (gamma 1.0, seed 0) so the
   logged in-feed watch-fraction quantiles and long-view rate match KuaiRand-Pure AD views.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from .config import make_config
from .generate import streams
from .logging_policy import run_logging, training_daily_mean
from .mi import mi_excess, block_of
from .opportunities import draw_context, local_time, make_opportunities
from .outcomes import impression_frame, infeed_watch_fraction, sig
from .world import make_campaigns, make_users

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MI_TARGET = 0.0255
Q_TARGET = np.array([0.0, 0.01, 0.02, 0.13, 0.70])
LV_TARGET = 0.24
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=50_000)
    ap.add_argument("--out", default=os.path.join(HERE, "calibration.json"))
    ap.add_argument("--set", action="append", default=[], help="key=json_value")
    a = ap.parse_args(argv)
    t0 = time.time()
    over = {"n_users": a.n_users}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = json.loads(v)
    cfg = make_config(over)
    R = streams(0)
    users = make_users(cfg, R["users"])
    camp = make_campaigns(cfg, R["campaigns"])
    opp = make_opportunities(cfg, users, R["opportunities"])
    _, _, _, lh = local_time(opp["ts"], users["tz"][opp["user"]])
    blk = block_of(lh)
    it = opp["visit_interest"][opp["visit"]]

    def excess(g):
        ctx = draw_context(it, g, opp["u_context"], cfg["context_uniform_share"])
        return mi_excess(opp["user"], blk, ctx, cfg["n_users"])[0]

    trace = []
    lo, hi = 0.0, 1.0
    while excess(hi) < MI_TARGET and hi < 8:
        hi *= 2
    for _ in range(14):
        mid = (lo + hi) / 2
        e = excess(mid)
        trace.append((mid, e))
        lo, hi = (mid, hi) if e < MI_TARGET else (lo, mid)
    gamma_real = (lo + hi) / 2
    e_real, e_def, e_zero = excess(gamma_real), excess(1.0), excess(0.0)
    print(f"gamma_real {gamma_real:.4f}: MI excess {e_real:.4f} (gamma 1.0: {e_def:.4f}; gamma 0: {e_zero:.4f})")

    # in-feed fit on the default world's logged impressions
    log_u = run_logging(cfg, camp, opp, R["log_unbudgeted"])
    sh = log_u["choice"] >= 0
    S1 = training_daily_mean(cfg, camp, opp["ts"][sh], log_u["choice"][sh], camp["m_cpm"][log_u["choice"][sh]])
    log = run_logging(cfg, camp, opp, R["log"], budget_c1=cfg["logging_rho"] * S1)
    imp = impression_frame(cfg, camp, users, opp, log)
    feed = np.flatnonzero(imp["format"] == 2)
    rng = np.random.default_rng(123)
    feed = rng.permutation(feed)[:300_000]
    st, D = imp["state"][feed], imp["duration"][feed]
    x = sig(cfg["gamma"] * imp["aff"][feed])
    u1, u2, u3 = rng.random(feed.size), rng.random(feed.size), rng.random(feed.size)
    absent_share = float(np.mean(st == 2))
    print(f"in-feed impressions used {feed.size}; absent share {absent_share:.3f}")

    def unpack(th):
        return {"e0": th[0], "c0": th[1], "aL": np.exp(th[2]), "bL": np.exp(th[3]),
                "aS": np.exp(th[4]), "bS": np.exp(th[5])}

    def stats(prm):
        f = infeed_watch_fraction(prm, st, x, u1, u2, u3, cfg["absent_loop_mean"])
        return np.quantile(f, QS), float(np.mean(f * D >= np.minimum(D, 18.0)))

    def loss(th):
        q, lv = stats(unpack(th))
        return float(((q - Q_TARGET) ** 2).sum() + 4 * (lv - LV_TARGET) ** 2)

    # the loss is multimodal: several starts, keep the best
    starts = [np.array([0.0, -1.5, np.log(2.0), np.log(1.0), np.log(0.5), np.log(15.0)]),
              np.array([0.05, -3.7, np.log(0.02), np.log(1.0), np.log(0.75), np.log(6.6)]),
              np.array([0.0, -3.0, np.log(0.1), np.log(1.0), np.log(0.7), np.log(30.0)])]
    srng = np.random.default_rng(7)
    starts += [starts[1] + srng.normal(0, 0.5, 6) for _ in range(5)]
    best = None
    for th0 in starts:
        r = minimize(loss, th0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    prm = {k: float(v) for k, v in unpack(best.x).items()}
    q, lv = stats(prm)
    print("in-feed fit:", {k: round(v, 4) for k, v in prm.items()}, "quantiles", q.round(4).tolist(),
          "long_view", round(lv, 4))
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "spec_sha256": open(os.path.join(HERE, "GENERATOR_SPEC.sha256")).read().split()[0],
        "world": {"n_users": a.n_users, "seed": 0, "gamma": cfg["gamma"]},
        "gamma_real": gamma_real,
        "gamma_real_fit": {"target_bits": MI_TARGET, "achieved_bits": e_real, "gamma_1_bits": e_def,
                           "gamma_0_bits": e_zero, "bisection": trace},
        "infeed": prm,
        "infeed_fit": {"target_quantiles": Q_TARGET.tolist(), "achieved_quantiles": q.tolist(),
                       "target_long_view": LV_TARGET, "achieved_long_view": lv, "loss": float(best.fun),
                       "absent_share": absent_share, "n_impressions": int(feed.size)},
        "sources": {"mi": "KuaiRand-1K standard log 2022-04-08..21, first-level category, 5 local blocks",
                    "infeed": "KuaiRand-Pure AD views, all three logs"},
        "seconds": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
