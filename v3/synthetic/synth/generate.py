"""Generate one synthetic world.

    python -m synth.generate --out OUT_DIR --seed 0 [--set gamma=0.5 --set tz_mix=single]

Writes OUT_DIR/observed (models may read) and OUT_DIR/truth (evaluation only), per spec Section 11.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .config import FORMATS, STATES, config_hash, make_config
from .logging_policy import run_logging, training_daily_mean
from .mi import synthetic_mi_excess
from .opportunities import local_time, make_opportunities
from .outcomes import draw_outcomes, impression_frame, population_curve, solve_beta0
from .world import ARCHS, CATS, BUMPS, INTEREST, make_campaigns, make_users

HERE = os.path.dirname(os.path.abspath(__file__))
COSTS = [f"cost_C{i}" for i in range(1, 7)]


def code_version():
    """SHA-256 (first 16 hex) over the generator's source files, so it works outside a git checkout."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def nullable_int(a):
    """Integer column with null where a < 0."""
    return pd.Series(a, dtype="Int64").mask(a < 0)


def streams(seed):
    ss = np.random.SeedSequence(seed)
    names = ["users", "campaigns", "opportunities", "log_unbudgeted", "log", "out_unbudgeted", "out"]
    return {n: np.random.default_rng(s) for n, s in zip(names, ss.spawn(len(names)))}


def build(cfg, seed):
    """All in-memory parts of a world (no files)."""
    if cfg["infeed"] is None and cfg["p_video"] * cfg["p_feed"] > 0:
        raise ValueError("in-feed parameters missing: run synth.calibrate first")
    R = streams(seed)
    users = make_users(cfg, R["users"])
    camp = make_campaigns(cfg, R["campaigns"])
    opp = make_opportunities(cfg, users, R["opportunities"])
    # 1. unbudgeted logging pass -> C1 budgets for the logging throttle
    log_u = run_logging(cfg, camp, opp, R["log_unbudgeted"])
    sh_u = log_u["choice"] >= 0
    S1 = training_daily_mean(cfg, camp, opp["ts"][sh_u], log_u["choice"][sh_u], camp["m_cpm"][log_u["choice"][sh_u]])
    # 2. budgeted logging run (the logged data)
    log = run_logging(cfg, camp, opp, R["log"], budget_c1=cfg["logging_rho"] * S1)
    imp = impression_frame(cfg, camp, users, opp, log)
    beta0 = solve_beta0(imp["logit_no_beta0"], cfg["target_ctr"])
    out = draw_outcomes(cfg, camp, users, imp, beta0, R["out"])
    # 3. budgets for every cost model from the unbudgeted pass, same beta0 and r_bar
    imp_u = impression_frame(cfg, camp, users, opp, log_u)
    out_u = draw_outcomes(cfg, camp, users, imp_u, beta0, R["out_unbudgeted"], r_bar=out["r_bar"])
    S = {k: training_daily_mean(cfg, camp, imp_u["ts"], imp_u["campaign"], out_u[k]) for k in COSTS}
    return {"users": users, "camp": camp, "opp": opp, "log": log, "imp": imp, "out": out,
            "beta0": beta0, "S": S}


def checks(cfg, w):
    """Section 14 checks, written to metadata (never tuned)."""
    users, opp, imp, out = w["users"], w["opp"], w["imp"], w["out"]
    res = {}
    ex, obs, null = synthetic_mi_excess(cfg, users, opp)
    res["mi_excess_bits"] = {"excess": ex, "observed": obs, "shuffled": null, "kuairand_target": 0.0255}
    # rhythm stability, week 1 vs week 2, on visits
    vu, vt = opp["visit_user"], opp["visit_ts"]
    _, _, _, lh = local_time(vt, users["tz"][vu])
    wk = (vt // (7 * 86400)).astype(int)
    stab = {}
    for thr in (200, 20):
        H = np.zeros((cfg["n_users"], 2, 24))
        sel = wk < 2
        np.add.at(H, (vu[sel], wk[sel], lh[sel].astype(int)), 1)
        n = H.sum(axis=2)
        ok = np.flatnonzero((n >= thr).all(axis=1))
        if ok.size < 10:
            stab[f"min_{thr}"] = {"users": int(ok.size)}
            continue
        a, b = H[ok, 0], H[ok, 1]
        same = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]
        perm = np.random.default_rng(0).permutation(ok.size)
        rnd = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b[perm])]
        stab[f"min_{thr}"] = {"users": int(ok.size), "same_user_median": float(np.nanmedian(same)),
                              "random_user_median": float(np.nanmedian(rnd))}
    res["rhythm_stability"] = {**stab, "kuairand": {"same_user_median": 0.48, "random_user_median": 0.15}}
    # ad views per user over the first two weeks
    two = imp["ts"] < 14 * 86400
    apu = np.bincount(imp["user"][two], minlength=cfg["n_users"])
    apu = apu[apu > 0]
    res["ad_views_per_user_2w"] = {"median": float(np.median(apu)), "p90": float(np.quantile(apu, 0.9)),
                                   "kuairand": {"median": 12, "p90": 48}}
    # in-feed watch fraction (a fit; reported again here)
    feed = imp["format"] == 2
    if feed.any():
        f = out["watch_time_s"][feed] / imp["duration"][feed]
        res["infeed"] = {"quantiles_p10_p90": np.quantile(f, [0.1, 0.25, 0.5, 0.75, 0.9]).round(4).tolist(),
                         "long_view": float(np.mean(out["completed"][feed])),
                         "absent_share": float(np.mean(imp["state"][feed] == 2)),
                         "kuairand_pure": {"quantiles_p10_p90": [0.0, 0.01, 0.02, 0.13, 0.70], "long_view": 0.24}}
    # traffic shape vs iPinYou (UTC+8 users, local hour, hours 1-23)
    ipy = np.array([8.33, 2.47, 1.33, 1.25, 0.65, 0.71, 0.90, 2.04, 2.91, 3.23, 5.26, 5.40, 5.43, 5.94, 5.36,
                    5.88, 5.87, 5.60, 6.08, 4.48, 4.92, 4.08, 6.94, 4.94])
    ipy[0] = (ipy[23] + ipy[1]) / 2
    ipy /= ipy.sum()
    z8 = users["tz"][vu] == 8
    if z8.any():
        s = np.bincount(lh[z8].astype(int), minlength=24).astype(float)
        s /= s.sum()
        res["traffic_vs_ipinyou"] = {"total_variation": float(0.5 * np.abs(s - ipy).sum()),
                                     "correlation": float(np.corrcoef(s, ipy)[0, 1])}
    res["population_curve_g"] = population_curve(users).round(4).tolist()
    res["counts"] = {"visits": int(vu.size), "slots": int(opp["ts"].size), "impressions": int(imp["opp"].size),
                     "clicks": int(out["clicked"].sum()), "fill_rate": float(imp["opp"].size / opp["ts"].size)}
    res["state_share"] = {s: float(np.mean(opp["state"] == i)) for i, s in enumerate(STATES)}
    return res


def write(cfg, seed, w, out_dir):
    users, camp, opp, log, imp, out = w["users"], w["camp"], w["opp"], w["log"], w["imp"], w["out"]
    obs_dir, tru_dir = os.path.join(out_dir, "observed"), os.path.join(out_dir, "truth")
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(tru_dir, exist_ok=True)
    S_n = opp["ts"].size
    pq = dict(index=False, engine="pyarrow", compression="zstd")
    pd.DataFrame({
        "opp_id": np.arange(S_n), "ts_utc": opp["ts"], "user_id": opp["user"], "visit_id": opp["visit"],
        "region_obs": nullable_int(users["region_obs"][opp["user"]]),
        "context_category": pd.Categorical.from_codes(opp["context"], CATS),
        "slot_index": opp["slot_index"],
        "format": pd.Categorical.from_codes(opp["format"], FORMATS),
        "video_duration_s": opp["duration"],
    }).to_parquet(os.path.join(obs_dir, "synth_opportunities.parquet"), **pq)
    imp_df = pd.DataFrame({"opp_id": np.arange(S_n),
                           "campaign_id": nullable_int(log["choice"]),
                           "propensity": log["propensity"]})
    for k in COSTS:
        col = np.full(S_n, np.nan)
        col[imp["opp"]] = out[k]
        imp_df[k] = col
    imp_df.to_parquet(os.path.join(obs_dir, "synth_impressions.parquet"), **pq)
    eng = {"opp_id": imp["opp"]}
    for k in ["skipped", "skip_time_s", "watch_time_s", "completed", "post_dwell_s", "interacted",
              "muted_or_backgrounded"]:
        v = out[k]
        if k in ("skipped", "completed"):
            v = pd.Series(np.nan_to_num(v).astype(bool), dtype="boolean").mask(np.isnan(v))
        eng[k] = v
    pd.DataFrame(eng).to_parquet(os.path.join(obs_dir, "synth_engagement.parquet"), **pq)
    pd.DataFrame({"opp_id": imp["opp"], "clicked": out["clicked"]}).to_parquet(
        os.path.join(obs_dir, "synth_clicks.parquet"), **pq)
    cdf = pd.DataFrame({
        "campaign_id": np.arange(camp["v"].size),
        "category": [CATS[c] for c in camp["cat1"]],
        "category2": [CATS[c] if c >= 0 else None for c in camp["cat2"]],
        "advertiser_tz": camp["tz"], "value_per_click": camp["v"], "cpm_unit_cost": camp["m_cpm"],
    })
    for k in COSTS:
        for rho in cfg["rhos"]:
            cdf[f"budget_{k[5:]}_rho{rho}"] = rho * w["S"][k]
    cdf.to_parquet(os.path.join(obs_dir, "synth_campaigns.parquet"), **pq)
    pd.DataFrame({"region_code": np.arange(users["region_tz"].size), "tz_offset": users["region_tz"]}).to_parquet(
        os.path.join(obs_dir, "synth_regions.parquet"), **pq)
    # truth
    tu = pd.DataFrame({"user_id": np.arange(cfg["n_users"]), "tz_offset": users["tz"], "region": users["region"],
                       "lambda": users["lam"], "join_day": users["join"], "leave_day": users["leave"]})
    for i, a in enumerate(ARCHS):
        tu[f"w_{a}"] = users["w"][:, i]
    for i, c in enumerate(CATS):
        tu[f"eps_{c}"] = users["eps"][:, i]
    tu.to_parquet(os.path.join(tru_dir, "truth_users.parquet"), **pq)
    np.save(os.path.join(tru_dir, "truth_drift.npy"), users["drift"])
    np.save(os.path.join(tru_dir, "truth_visit_interest.npy"), opp["visit_interest"])
    amb = np.zeros(S_n, bool)
    amb[imp["opp"]] = out["ambiguous"]
    pd.DataFrame({"opp_id": np.arange(S_n), "state": pd.Categorical.from_codes(opp["state"], STATES),
                  "ambiguous": amb}).to_parquet(os.path.join(tru_dir, "truth_states.parquet"), **pq)
    pd.DataFrame({"opp_id": imp["opp"], "pctr": out["pctr"], "affinity": imp["aff"], "n_uc": imp["n_uc"],
                  "context_match": imp["match"]}).to_parquet(os.path.join(tru_dir, "truth_impressions.parquet"), **pq)
    world = {"archetypes": ARCHS, "categories": CATS, "bumps": BUMPS, "interest": {
        a: [list(e) if isinstance(e, tuple) else e for e in v] for a, v in INTEREST.items()},
        "beta0": w["beta0"], "gamma": cfg["gamma"], "delta": cfg["delta"], "phi": cfg["phi"],
        "s_state": cfg["s_state"], "campaign_q": camp["q"].tolist(), "campaign_m": camp["m"].tolist(),
        "r_bar": out["r_bar"], "infeed": cfg["infeed"], "c5_sigma": cfg["c5_sigma"],
        "unbudgeted_daily_spend": {k: w["S"][k].tolist() for k in COSTS}}
    with open(os.path.join(tru_dir, "truth_world.json"), "w") as f:
        json.dump(world, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="key=json_value")
    ap.add_argument("--calibration", default=os.path.join(os.path.dirname(HERE), "calibration.json"))
    a = ap.parse_args(argv)
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v
    calib = json.load(open(a.calibration)) if os.path.exists(a.calibration) else None
    cfg = make_config(over, calib)
    t0 = time.time()
    w = build(cfg, a.seed)
    write(cfg, a.seed, w, a.out)
    meta = {"synthetic": True, "config": cfg, "config_hash": config_hash(cfg), "world_seed": a.seed,
            "code_version": code_version(), "spec_sha256": open(os.path.join(os.path.dirname(HERE),
                                                                              "GENERATOR_SPEC.sha256")).read().split()[0],
            "beta0": w["beta0"], "checks": checks(cfg, w), "generation_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "observed", "synth_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print(json.dumps({k: meta[k] for k in ("config_hash", "world_seed", "beta0", "generation_seconds")}))
    print(json.dumps(meta["checks"]["counts"]), json.dumps(meta["checks"]["state_share"]))


if __name__ == "__main__":
    main()
