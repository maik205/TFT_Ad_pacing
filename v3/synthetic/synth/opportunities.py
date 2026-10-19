"""Visits, slots, attention states, formats and page context (Sections 7 and 7a.1).

Nothing here depends on which ad is shown, so opportunities are generated once per world and the
logging policy runs on top of them.
"""
import numpy as np

from .world import GRID, K, interest_table, rhythm_table

DAY = 86400


def local_time(ts, tz):
    """Local day index, day type (0 weekday, 1 weekend), grid index and local hour (float)."""
    t = ts + 3600.0 * tz
    day = np.floor(t / DAY).astype(np.int64)
    sec = t - day * DAY
    dt = (np.mod(day, 7) >= 5).astype(np.int64)  # world start is a Monday
    g = np.minimum((sec / (3600 / GRID)).astype(np.int64), 24 * GRID - 1)
    return day, dt, g, sec / 3600.0


def visit_interest(users, uid, ts, chunk=400_000):
    """interest_u(k, t) for each visit: archetype mix of planted tables + personal noise + drift."""
    I = interest_table().astype(np.float32)  # (A, 2, G, K)
    A_, _, G, _ = I.shape
    I = I.reshape(A_, 2 * G, K)
    out = np.empty((uid.size, K), np.float32)
    days = users["drift"].shape[1]
    for s in range(0, uid.size, chunk):
        u, t = uid[s:s + chunk], ts[s:s + chunk]
        _, dt, g, _ = local_time(t, users["tz"][u])
        Ig = I[:, dt * G + g, :]  # (A, n, K)
        mix = np.einsum("na,ank->nk", users["w"][u].astype(np.float32), Ig)
        d = np.minimum((t // DAY).astype(np.int64), days - 1)
        out[s:s + chunk] = mix + users["eps"][u] + users["drift"][u, d]
    return out


def make_visits(cfg, users, rng):
    R = rhythm_table()  # (A, 2, G)
    T = np.einsum("na,adg->ndg", users["w"], R).astype(np.float32)
    n = cfg["n_users"]
    uids, tss = [], []
    for H in range(cfg["days"] * 24):
        mid = H + 0.5
        alive = (users["join"] * 24 <= mid) & (mid < users["leave"] * 24)
        _, dt, g, _ = local_time(np.full(n, mid * 3600.0), users["tz"])
        rate = np.where(alive, users["lam"] * T[np.arange(n), dt, g] / 24.0, 0.0)
        k = rng.poisson(rate)
        u = np.repeat(np.arange(n), k)
        uids.append(u)
        tss.append(H * 3600.0 + rng.uniform(0, 3600, u.size))
    uid = np.concatenate(uids)
    ts = np.concatenate(tss)
    order = np.argsort(ts, kind="stable")
    return uid[order], ts[order]


def make_slots(cfg, users, uid, ts, interest, rng):
    """Expand visits into slots with the hidden attention state (Section 7a.1)."""
    V = uid.size
    sig = lambda x: 1 / (1 + np.exp(-x))
    p_eng = sig(cfg["eta0"] + cfg["eta1"] * interest.max(axis=1))
    state = np.where(rng.random(V) < p_eng, 0, 1)  # 0 engaged, 1 casual, 2 absent
    att_left = 1 + rng.poisson(cfg["extra_slots_mean"], V)
    auto_left = np.zeros(V, np.int64)
    active = np.ones(V, bool)
    tz = users["tz"][uid]
    rows = []
    j = 0
    while active.any():
        idx = np.flatnonzero(active)
        st = state[idx]
        t_slot = ts[idx] + j * cfg["slot_gap_s"]
        is_video = rng.random(idx.size) < cfg["p_video"]
        is_feed = rng.random(idx.size) < cfg["p_feed"]
        fmt = np.where(is_video, np.where(is_feed, 2, 1), 0)  # 0 display, 1 preroll, 2 infeed
        rows.append((idx, np.full(idx.size, j), t_slot, st.copy(), fmt))
        # transitions after this slot
        attentive = st < 2
        _, _, _, lh = local_time(t_slot, tz[idx])
        h = cfg["h0"] * np.where(lh < 6, cfg["night_mult"], 1.0)
        go_abs = attentive & (rng.random(idx.size) < h)
        cool = attentive & ~go_abs & (st == 0) & (rng.random(idx.size) < cfg["engaged_to_casual"])
        new_st = st.copy()
        new_st[go_abs] = 2
        new_st[cool] = 1
        state[idx] = new_st
        auto_left[idx[go_abs]] = rng.geometric(1 / cfg["autoplay_mean"], go_abs.sum())
        att_left[idx[attentive & ~go_abs]] -= 1
        was_abs = st == 2
        auto_left[idx[was_abs]] -= 1
        done = np.where(new_st == 2, auto_left[idx] <= 0, att_left[idx] <= 0)
        active[idx[done]] = False
        j += 1
    vis = np.concatenate([r[0] for r in rows])
    slot = np.concatenate([r[1] for r in rows])
    t = np.concatenate([r[2] for r in rows])
    st = np.concatenate([r[3] for r in rows])
    fmt = np.concatenate([r[4] for r in rows])
    order = np.lexsort((slot, t))
    vis, slot, t, st, fmt = vis[order], slot[order], t[order], st[order], fmt[order]
    S = vis.size
    dur = np.full(S, np.nan)
    pre = fmt == 1
    dur[pre] = rng.choice(cfg["preroll_durations"], pre.sum(), p=cfg["preroll_probs"])
    feed = fmt == 2
    dur[feed] = np.exp(cfg["infeed_dur_logmean"] + cfg["infeed_dur_logsd"] * rng.standard_normal(feed.sum()))
    return {"visit": vis, "slot_index": slot, "ts": t, "state": st, "format": fmt, "duration": dur,
            "u_context": rng.random(S)}


def draw_context(interest_slot, gamma, u, uniform_share):
    """Page-context category from (1 - s) * softmax(gamma * interest) + s * uniform, by inverse CDF."""
    z = gamma * interest_slot
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    p = (1 - uniform_share) * p + uniform_share / K
    cdf = np.cumsum(p, axis=1)
    return np.minimum((cdf < u[:, None]).sum(axis=1), K - 1)


def make_opportunities(cfg, users, rng):
    uid, ts = make_visits(cfg, users, rng)
    interest = visit_interest(users, uid, ts)
    slots = make_slots(cfg, users, uid, ts, interest, rng)
    slots["user"] = uid[slots["visit"]]
    slots["context"] = draw_context(interest[slots["visit"]], cfg["gamma"], slots["u_context"],
                                    cfg["context_uniform_share"])
    return {"visit_user": uid, "visit_ts": ts, "visit_interest": interest, **slots}
