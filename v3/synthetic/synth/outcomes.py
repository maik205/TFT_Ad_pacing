"""Engagement signals (7a.2), clicks (8) and the six cost models (9) for shown impressions."""
import numpy as np
from scipy.optimize import brentq

from .opportunities import local_time
from .world import rhythm_table

sig = lambda x: 1 / (1 + np.exp(-x))
DWELL = [(np.log(20), 1.0), (np.log(8), 1.0), (np.log(300), 0.8)]
P_INTERACT = np.array([0.6, 0.2, 0.03])
P_MUTED = np.array([0.05, 0.25, 0.5])
P_COMPLETE_PREROLL = np.array([0.85, 0.5, 1.0])


def fatigue(user, campaign, ts, C):
    """n_uc(t): impressions of campaign c shown to user u in the preceding 24 hours."""
    key = user.astype(np.int64) * C + campaign
    val = key * 10_000_000 + np.floor(ts).astype(np.int64)
    order = np.argsort(val, kind="stable")
    sv = val[order]
    left = np.searchsorted(sv, sv - 86400, side="right")
    n = np.empty(ts.size, np.int64)
    n[order] = np.arange(ts.size) - left
    return n


def population_curve(users):
    """g(h): mean user rhythm by local hour (weekday 5/7, weekend 2/7), normalised to mean 1."""
    R = rhythm_table()  # (A, 2, G)
    G = R.shape[2] // 24
    hourly = R.reshape(R.shape[0], 2, 24, G).mean(axis=3)
    r = np.einsum("na,adh->dh", users["w"], hourly) / users["w"].shape[0]
    g = (5 * r[0] + 2 * r[1]) / 7
    return g / g.mean()


def impression_frame(cfg, camp, users, opp, log):
    """Index arrays and planted quantities for shown impressions (no randomness)."""
    shown = np.flatnonzero(log["choice"] >= 0)
    c = log["choice"][shown]
    u = opp["user"][shown]
    ts = opp["ts"][shown]
    interest = opp["visit_interest"][opp["visit"][shown]]
    aff = np.einsum("nk,nk->n", camp["m"][c], interest)
    match = camp["m"][c, opp["context"][shown]] > 0
    n_uc = fatigue(u, c, ts, camp["v"].size)
    st = opp["state"][shown]
    z = (cfg["gamma"] * aff + camp["q"][c] + cfg["delta"] * match - cfg["phi"] * np.log1p(n_uc)
         + np.array(cfg["s_state"])[st])
    return {"opp": shown, "campaign": c, "user": u, "ts": ts, "aff": aff, "match": match, "n_uc": n_uc,
            "state": st, "format": opp["format"][shown], "duration": opp["duration"][shown],
            "c3_bid2": log["c3_bid2"][shown], "logit_no_beta0": z}


def solve_beta0(z, target):
    return brentq(lambda b: sig(b + z).mean() - target, -30, 30, xtol=1e-10)


def infeed_watch_fraction(prm, state, x, u_long, u_beta, u_loop, loop_mean):
    """Watch fraction f for in-feed video (Section 7a.2); uniforms passed in for common random numbers."""
    from scipy.stats import beta as B
    p_long = np.where(state == 0, sig(prm["e0"] + 3 * x), sig(prm["c0"] + 2 * x))
    long_ = u_long < p_long
    f = np.where(long_, B.ppf(u_beta, prm["aL"], prm["bL"]), B.ppf(u_beta, prm["aS"], prm["bS"]))
    return np.where(state == 2, 1.0 - loop_mean * np.log1p(-u_loop), f)


def draw_outcomes(cfg, camp, users, imp, beta0, rng, r_bar=None):
    n = imp["opp"].size
    st, fmt, D = imp["state"], imp["format"], imp["duration"]
    x = sig(cfg["gamma"] * imp["aff"])
    # pre-roll
    skipped = np.full(n, np.nan)
    skip_time = np.full(n, np.nan)
    watch = np.full(n, np.nan)
    completed = np.full(n, np.nan)
    pre = fmt == 1
    skippable = pre & (D >= 15)
    p_skip = np.select([st == 0, st == 1], [sig(0.5 - 3 * x), sig(1.5 - 2 * x)], 0.0)
    sk = skippable & (rng.random(n) < p_skip)
    skipped[pre] = sk[pre]
    skip_time[sk] = 5.0
    comp_pre = pre & ~sk & (rng.random(n) < P_COMPLETE_PREROLL[st])
    stop = 5.0 + (D - 5.0) * rng.random(n)
    watch[pre] = np.where(sk, 5.0, np.where(comp_pre, D, stop))[pre]
    completed[pre] = comp_pre[pre]
    # in-feed
    feed = fmt == 2
    if feed.any():
        f = infeed_watch_fraction(cfg["infeed"], st[feed], x[feed], rng.random(feed.sum()),
                                  rng.random(feed.sum()), rng.random(feed.sum()), cfg["absent_loop_mean"])
        wt = f * D[feed]
        watch[feed] = wt
        completed[feed] = wt >= np.minimum(D[feed], 18.0)  # KuaiRand long_view rule
    # all formats
    amb = (st == 0) & (rng.random(n) < cfg["a_amb"])
    dist_state = np.where(amb, 2, st)
    mu = np.array([d[0] for d in DWELL])[dist_state]
    sd = np.array([d[1] for d in DWELL])[dist_state]
    dwell = np.exp(mu + sd * rng.standard_normal(n))
    interacted = rng.random(n) < P_INTERACT[st]
    muted = rng.random(n) < P_MUTED[st]
    # clicks
    pctr = sig(beta0 + imp["logit_no_beta0"])
    clicked = rng.random(n) < pctr
    # costs
    c = imp["campaign"]
    m = camp["m_cpm"][c]
    g = population_curve(users)
    _, _, _, lh = local_time(imp["ts"], users["tz"][imp["user"]])
    gh = g[np.minimum(lh.astype(np.int64), 23)]
    m_bar = camp["m_cpm"].mean()
    E = np.exp(np.log(m_bar * gh) + cfg["c3_ext_sigma"] * rng.standard_normal(n))
    s5 = cfg["c5_sigma"]
    video = fmt > 0
    if r_bar is None:
        r_bar = float(completed[video].mean()) if video.any() else 1.0
    costs = {
        "cost_C1": m,
        "cost_C2": camp["v"][c] * clicked,
        "cost_C3": np.fmax(E, imp["c3_bid2"]),
        "cost_C4": m * gh,
        "cost_C5": m * np.exp(s5 * rng.standard_normal(n) - s5 ** 2 / 2),
        "cost_C6": np.where(video, np.where(completed == 1, m / r_bar, 0.0), m),
    }
    eng = {"skipped": skipped, "skip_time_s": skip_time, "watch_time_s": watch, "completed": completed,
           "post_dwell_s": dwell, "interacted": interacted, "muted_or_backgrounded": muted}
    return {"pctr": pctr, "clicked": clicked, "ambiguous": amb, "r_bar": r_bar, **costs, **eng}
