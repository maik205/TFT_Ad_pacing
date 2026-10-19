"""Planted world: archetypes (Section 4), users (Section 5), campaigns (Section 6)."""
import numpy as np

from .config import TZ_MIXES

CATS = ["enterprise_software", "office_equipment", "video_games", "gaming_hardware", "streaming_media",
        "fashion", "beauty", "groceries", "home_family", "travel", "personal_finance", "education"]
K = len(CATS)
ARCHS = ["office_worker", "night_gamer", "student", "parent", "shift_worker"]
A = len(ARCHS)
BLOCK_STARTS = [0, 6, 9, 17, 20]  # night, morning, work, evening, late
GRID = 12  # time-of-day grid points per hour (5 minutes)

# Section 4.1: (mu, k, w) per bump; [weekday, weekend]
BUMPS = {
    "office_worker": [[(12.5, 3, 1.0), (21, 4, 0.8)], [(11, 2, 0.6), (21, 3, 1.0)]],
    "night_gamer": [[(23, 4, 1.2), (14, 2, 0.3)], [(22, 3, 1.2), (15, 2, 0.6)]],
    "student": [[(14, 2, 0.6), (22, 3, 1.0)], [(16, 1.5, 1.0), (23, 3, 0.8)]],
    "parent": [[(7, 5, 0.7), (20.5, 4, 1.0)], [(10, 2, 0.9), (20, 3, 0.7)]],
    "shift_worker": [[(4, 3, 0.8), (16, 3, 0.8)], [(13, 1.5, 0.8), (21, 2, 0.6)]],
}
BASE_RATE = 0.05

# Section 4.2: per block, {category: log-affinity}; a (weekday, weekend) pair where they differ
INTEREST = {
    "office_worker": [{}, {"personal_finance": 1.0},
                      ({"enterprise_software": 2.0, "office_equipment": 1.5}, {"travel": 1.0}),
                      {"groceries": 0.8}, {"video_games": 1.5, "streaming_media": 1.2}],
    "night_gamer": [{"video_games": 2.0, "gaming_hardware": 1.5}, {}, {"streaming_media": 0.8},
                    {"video_games": 1.0}, {"video_games": 2.0, "gaming_hardware": 1.8}],
    "student": [{"streaming_media": 1.0}, {"education": 1.2}, ({"education": 1.5}, {"fashion": 1.2}),
                {"fashion": 1.0}, {"video_games": 1.5, "beauty": 0.8}],
    "parent": [{}, {"groceries": 1.5}, {"home_family": 1.0}, {"groceries": 1.2, "home_family": 1.2},
               {"travel": 1.2, "personal_finance": 0.8}],
    "shift_worker": [{"groceries": 1.0, "streaming_media": 1.0}, {"streaming_media": 0.8}, {},
                     {"video_games": 0.8}, {"home_family": 0.8}],
}


def rhythm_table():
    """R[a, daytype, g]: activity at local time g/GRID hours, normalised to mean 1 per day type."""
    h = np.arange(24 * GRID) / GRID
    R = np.empty((A, 2, h.size))
    for ai, a in enumerate(ARCHS):
        for dt in range(2):
            r = np.full(h.size, BASE_RATE)
            for mu, k, w in BUMPS[a][dt]:
                r += w * np.exp(k * (np.cos(2 * np.pi * (h - mu) / 24) - 1))
            R[ai, dt] = r / r.mean()
    return R


def interest_table():
    """I[a, daytype, g, k] on the 5-minute grid, block edges blended linearly over one hour."""
    blocks = np.zeros((A, 2, 5, K))
    for ai, a in enumerate(ARCHS):
        for b, entry in enumerate(INTEREST[a]):
            for dt in range(2):
                d = entry[dt] if isinstance(entry, tuple) else entry
                for cat, v in d.items():
                    blocks[ai, dt, b, CATS.index(cat)] = v
    h = np.arange(24 * GRID) / GRID
    bidx = np.searchsorted(BLOCK_STARTS, h, side="right") - 1
    I = blocks[:, :, bidx, :].copy()
    for b, e in enumerate(BLOCK_STARTS):
        prev = (b - 1) % 5
        t = ((h - e + 0.5) % 24)  # position inside [e - 0.5, e + 0.5)
        sel = t < 1.0
        w = t[sel][None, None, :, None]
        I[:, :, sel, :] = (1 - w) * blocks[:, :, prev, None, :] + w * blocks[:, :, b, None, :]
    return I


def tz_offsets(cfg, rng, n):
    mix = TZ_MIXES[cfg["tz_mix"]]
    offs = np.array(sorted(mix))
    zone = rng.choice(offs.size, size=n, p=np.array([mix[o] for o in offs]))
    return offs, zone


def make_users(cfg, rng):
    n, days = cfg["n_users"], cfg["days"]
    pi = np.array(cfg["prevalence"])
    w = rng.dirichlet(cfg["alpha"] * pi, size=n)
    eps = rng.normal(0, cfg["eps_sd"], size=(n, K))
    steps = rng.normal(0, cfg["drift_sd"], size=(n, days, K))
    steps[:, 0, :] = 0.0
    drift = np.cumsum(steps, axis=1).astype(np.float32)
    lam = np.exp(np.log(cfg["lambda_median"]) + cfg["lambda_sigma"] * rng.standard_normal(n))
    late = rng.random(n) < cfg["late_join_share"]
    join = np.where(late, rng.uniform(0, days, n), 0.0)
    L = cfg["mean_lifetime_days"]
    leave = join + (rng.exponential(L, n) if L else np.full(n, np.inf))
    offs, zone = tz_offsets(cfg, rng, n)
    rpz = cfg["regions_per_zone"]
    region = zone * rpz + rng.integers(0, rpz, n)
    n_regions = offs.size * rpz
    u = rng.random(n)
    region_obs = region.copy()
    missing = u < cfg["p_missing"]
    wrong = (u >= cfg["p_missing"]) & (u < cfg["p_missing"] + cfg["p_wrong"])
    other = rng.integers(0, n_regions - 1, n)
    region_obs[wrong] = np.where(other[wrong] >= region[wrong], other[wrong] + 1, other[wrong])
    region_obs[missing] = -1
    return {
        "w": w, "eps": eps, "drift": drift, "lam": lam, "join": join, "leave": leave,
        "tz": offs[zone], "region": region, "region_obs": region_obs,
        "region_tz": np.repeat(offs, rpz),  # public mapping: region code -> zone offset
    }


def make_campaigns(cfg, rng):
    C = cfg["n_campaigns"]
    per = C // K
    cat1 = np.concatenate([np.tile(np.arange(K), per), rng.choice(K, C - per * K, replace=False)])
    rng.shuffle(cat1)
    m = np.zeros((C, K))
    m[np.arange(C), cat1] = 1.0
    cat2 = np.full(C, -1)
    two = rng.random(C) < cfg["second_cat_share"]
    for c in np.flatnonzero(two):
        cat2[c] = rng.choice([k for k in range(K) if k != cat1[c]])
        m[c, cat2[c]] = 0.5
    mu = np.zeros(K)
    mu[[CATS.index(c) for c in ("enterprise_software", "personal_finance", "travel")]] = 0.7
    mu[[CATS.index(c) for c in ("groceries", "beauty")]] = -0.3
    v = np.exp(mu[cat1] + cfg["v_sigma"] * rng.standard_normal(C))
    q = rng.normal(0, cfg["q_sd"], C)
    offs, zone = tz_offsets(cfg, rng, C)
    return {"cat1": cat1, "cat2": cat2, "m": m, "v": v, "q": q, "tz": offs[zone],
            "m_cpm": 0.01 * v * cfg["target_ctr"]}
