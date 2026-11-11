"""Model inputs for every shown impression of a world, built from `observed/` only (rule R2).

    python -m ctr.dataset WORLD_DIR OUT.npz

Each row is one served impression: the request context, the user's previous served impressions (newest
first, at most HISTORY_LEN, strictly earlier), the shown campaign as the single candidate, and the click.
The same rules are applied by the ad server at request time (contract.py).
"""
import os
import sys

import numpy as np
import pandas as pd

from . import contract as K

DAY = 86400
SPLITS = {"train": (0, 21), "val": (21, 24), "test": (24, 28)}  # UTC days, base spec Section 12


def load_world(world_dir):
    obs = os.path.join(world_dir, "observed")
    if not os.path.isdir(obs):
        raise FileNotFoundError(obs)
    opp = pd.read_parquet(os.path.join(obs, "synth_opportunities.parquet"),
                          columns=["opp_id", "ts_utc", "user_id", "region_obs", "context_category", "format"])
    imp = pd.read_parquet(os.path.join(obs, "synth_impressions.parquet"), columns=["opp_id", "campaign_id"])
    clk = pd.read_parquet(os.path.join(obs, "synth_clicks.parquet"))
    camp = pd.read_parquet(os.path.join(obs, "synth_campaigns.parquet"), columns=["campaign_id", "category", "category2"])
    reg = pd.read_parquet(os.path.join(obs, "synth_regions.parquet"))
    if list(opp["format"].cat.categories) != K.FORMATS:
        raise ValueError(f"world formats {list(opp['format'].cat.categories)} are not the fork's {K.FORMATS}")
    if list(opp["context_category"].cat.categories) != K.CATEGORIES:
        raise ValueError("world categories differ from the contract")
    if reg.sort_values("region_code")["tz_offset"].tolist() != K.REGION_OFFSETS:
        raise ValueError("world regions differ from the contract (only the default 4-zone mix is mapped)")
    return opp, imp, clk, camp


def campaign_tokens(camp):
    idx = {c: i for i, c in enumerate(K.CATEGORIES)}
    toks = np.zeros((int(camp["campaign_id"].max()) + 1, K.CANDIDATE_CATEGORIES), np.int8)
    for row in camp.itertuples():
        toks[row.campaign_id, 0] = K.category_token(idx[row.category])
        if row.category2 is not None and isinstance(row.category2, str):
            toks[row.campaign_id, 1] = K.category_token(idx[row.category2])
    return toks


def context(ts, fmt, ctx_cat, region_obs):
    """int8 context rows; region_obs = -1 when missing."""
    n = ts.size
    known = region_obs >= 0
    off = np.where(known, np.array(K.REGION_OFFSETS)[np.maximum(region_obs, 0)], 0)
    local = ts + 3600.0 * off
    lh = np.floor(np.mod(local / 3600.0, 24)).astype(np.int64)
    wd = np.mod(np.floor(local / DAY).astype(np.int64), 7)  # world start is a Monday
    ctx = np.zeros((n, len(K.CTX_FIELDS)), np.int8)
    ctx[:, 0] = fmt + 1
    ctx[:, 1] = K.CAT_BASE + ctx_cat
    ctx[:, 2] = np.where(known, region_obs + 1, 0)
    ctx[:, 3] = np.where(known, lh + 1, 0)
    ctx[:, 4] = np.where(known, wd + 1, 0)
    ctx[:, 5] = np.floor(np.mod(ts / 3600.0, 24)).astype(np.int64) + 1
    return ctx


def build(world_dir):
    opp, imp, clk, camp = load_world(world_dir)
    shown = imp.dropna(subset=["campaign_id"])
    d = (shown.merge(opp, on="opp_id").merge(clk, on="opp_id", how="left")
         .sort_values(["user_id", "ts_utc", "opp_id"], kind="stable").reset_index(drop=True))
    n = len(d)
    u = d["user_id"].to_numpy(np.int64)
    ts = d["ts_utc"].to_numpy(np.float64)
    c = d["campaign_id"].to_numpy(np.int64)
    fmt = d["format"].cat.codes.to_numpy(np.int64)
    clicked = d["clicked"].fillna(False).to_numpy(bool)
    region_obs = d["region_obs"].fillna(-1).to_numpy(np.int64)
    ctx_cat = d["context_category"].cat.codes.to_numpy(np.int64)
    ctoks = campaign_tokens(camp)

    L = K.HISTORY_LEN
    hist_cat = np.zeros((n, L, K.CANDIDATE_CATEGORIES), np.int8)
    hist_fmt = np.zeros((n, L), np.int8)
    hist_click = np.zeros((n, L), np.int8)
    hist_age = np.zeros((n, L), np.float32)
    fatigue = np.zeros(n, np.int16)
    i = np.arange(n)
    for k in range(1, L + 1):
        j = i - k
        ok = j >= 0
        jj = np.where(ok, j, 0)
        ok &= (u[jj] == u) & (ts[jj] < ts)
        age_s = ts - ts[jj]
        hist_cat[ok, k - 1] = ctoks[c[jj[ok]]]
        hist_fmt[ok, k - 1] = fmt[jj[ok]] + 1
        tok = np.where(age_s < K.OUTCOME_DELAY_S, K.CLICK_UNKNOWN, np.where(clicked[jj], K.CLICK_YES, K.CLICK_NO))
        hist_click[ok, k - 1] = tok[ok]
        hist_age[ok, k - 1] = (age_s[ok] / 3600.0).astype(np.float32)
        fatigue += (ok & (c[jj] == c) & (age_s < K.FATIGUE_WINDOW_S)).astype(np.int16)
    fatigue = np.minimum(fatigue, K.FATIGUE_CAP)

    day = np.floor(ts / DAY).astype(np.int64)
    split = np.full(n, -1, np.int8)
    for s, (name, (lo, hi)) in enumerate(SPLITS.items()):
        split[(day >= lo) & (day < hi)] = s
    return {
        "opp_id": d["opp_id"].to_numpy(np.int64), "user": u, "ts": ts, "campaign": c, "split": split,
        "ctx": context(ts, fmt, ctx_cat, region_obs),
        "hist_cat": hist_cat, "hist_fmt": hist_fmt, "hist_click": hist_click, "hist_age_h": hist_age,
        "cand_cat": ctoks[c], "cand_fatigue": fatigue.astype(np.float32), "clicked": clicked,
    }


def main(argv):
    world, out = argv
    data = build(world)
    np.savez(out, **data)
    n = data["clicked"].size
    print(f"{world}: {n} impressions, {int(data['clicked'].sum())} clicks, "
          + ", ".join(f"{s}={int((data['split'] == i).sum())}" for i, s in enumerate(SPLITS)))


if __name__ == "__main__":
    main(sys.argv[1:])
