"""When does forecast-driven pacing help on Criteo? Click ratio by campaign group, from the v2 test episodes.

    python criteo_groups.py EPISODES.parquet WORK_DIR OUT_PREFIX

Groups use training hours only (0-503): size = mean daily spend, volatility = coefficient of variation of daily
spend over the 21 training days; each in terciles. Budget tightness is the episode setting rho (0.3, 0.5, 0.7).
Metric as in v2: summed clicks over summed same-day hindsight-constant-rate clicks. Seeds of one forecaster are
pooled. Differences are paired; 95% CIs resample campaigns (clusters), 2,000 resamples.
"""
import json
import sys

import numpy as np
import pandas as pd

TRAIN_END = 504
METHODS = {
    "TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Replay": "naive_seed",
}
PAIRS = [("TFT", "Replay"), ("TFT", "Even"), ("TFT", "NHITS"), ("LightGBM", "Replay"), ("Replay", "Even"),
         ("Replay", "PID")]


def family(pol):
    if pol == "Even pacing":
        return "Even"
    if pol == "PID (tuned)":
        return "PID"
    if pol.endswith("(q=0.5)"):
        for k, pref in METHODS.items():
            if pol.startswith(pref):
                return k
    return None


def main(ep_path, work, out):
    ep = pd.read_parquet(ep_path)
    ep["method"] = ep.policy.map(family)
    ep = ep[ep.method.notna()]
    ids = pd.read_parquet(f"{work}/campaigns.parquet")["campaign"].to_numpy()  # episode "campaign" = position in this list
    ep["campaign"] = ids[ep["campaign"].to_numpy()]
    panel = pd.read_parquet(f"{work}/panel.parquet")
    panel = panel[panel.campaign.isin(ids)]
    tr = panel[panel.hour < TRAIN_END].copy()
    tr["day"] = tr.hour // 24
    daily = tr.groupby(["campaign", "day"]).spend.sum().unstack()
    st = pd.DataFrame({"size": daily.mean(axis=1), "vol": daily.std(axis=1, ddof=1) / daily.mean(axis=1)})
    for c in ("size", "vol"):
        st[c + "_grp"] = pd.qcut(st[c], 3, labels=["low", "mid", "high"])
    print(st.groupby("size_grp", observed=True)["size"].agg(["min", "max", "count"]).round(4).to_string())
    print(st.groupby("vol_grp", observed=True)["vol"].agg(["min", "max", "count"]).round(3).to_string())
    print("correlation size vs volatility:", round(float(st[["size", "vol"]].corr(method="spearman").iloc[0, 1]), 3))

    # per (campaign, rho, method): clicks, plus the reference clicks of the same episodes
    pc = ep.groupby(["campaign", "rho", "method"]).agg(clicks=("clicks", "sum"), ref=("ref_clicks", "sum")).reset_index()
    nfam = ep.groupby("method").policy.nunique()  # seeds per forecaster family (3 for TFT and NHITS, else 1)
    pc["clicks"] = pc["clicks"] / pc["method"].map(nfam)  # average over seeds, so every method is on the same episodes
    clicks = pc.pivot_table(index=["campaign", "rho"], columns="method", values="clicks")
    ref = pc[pc.method == "Even"].set_index(["campaign", "rho"])["ref"].rename("ref")
    tab = clicks.join(ref).reset_index().merge(st[["size_grp", "vol_grp"]], left_on="campaign", right_index=True)
    rng = np.random.default_rng(0)
    rows = []

    names = list(METHODS) + ["Even", "PID"]

    def cell(g, label, value, rho):
        pcm = g.groupby("campaign")[names + ["ref"]].sum()  # one row per campaign
        X = pcm[names].to_numpy(float)
        R = pcm["ref"].to_numpy(float)
        n = len(pcm)
        res = {"group": label, "value": value, "rho": rho, "campaigns": n}
        for j, m in enumerate(names):
            res[f"ratio_{m}"] = float(X[:, j].sum() / R.sum())
        idx = rng.integers(0, n, (2000, n))
        bX, bR = X[idx].sum(axis=1), R[idx].sum(axis=1)
        for a_, b_ in PAIRS:
            ja, jb = names.index(a_), names.index(b_)
            d = (bX[:, ja] - bX[:, jb]) / bR
            lo, hi = np.quantile(d, [0.025, 0.975])
            res[f"diff_{a_}_minus_{b_}"] = float((X[:, ja].sum() - X[:, jb].sum()) / R.sum())
            res[f"ci_{a_}_minus_{b_}"] = [float(lo), float(hi)]
        rows.append(res)

    for rho in sorted(tab.rho.unique()):
        t = tab[tab.rho == rho]
        cell(t, "all", "all", rho)
        for gcol, name in (("size_grp", "size"), ("vol_grp", "volatility")):
            for v in ("low", "mid", "high"):
                cell(t[t[gcol] == v], name, v, rho)
    json.dump(rows, open(out + ".json", "w"), indent=1)
    flat = pd.DataFrame([{k: (v if not isinstance(v, list) else f"[{v[0]:+.4f},{v[1]:+.4f}]") for k, v in r.items()} for r in rows])
    flat.to_csv(out + ".csv", index=False)
    show = ["group", "value", "rho", "campaigns", "ratio_Replay", "ratio_TFT", "diff_TFT_minus_Replay", "ci_TFT_minus_Replay",
            "diff_TFT_minus_NHITS", "ci_TFT_minus_NHITS", "diff_TFT_minus_Even"]
    pd.set_option("display.width", 250)
    print(flat[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
