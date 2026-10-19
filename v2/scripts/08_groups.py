"""Click ratio by campaign group (exploratory; PROTOCOL 10.17). Reads the test episodes, writes pacing_groups.json.

Groups are fixed from training hours only (hours < TRAIN_END): size = mean daily spend, volatility = coefficient of
variation of daily spend over the training days; each in terciles of the evaluated campaigns. Budget tightness is the
episode setting rho. The metric is the one of the main analysis: summed clicks over summed hindsight-reference clicks.
The seeds of one forecaster are averaged. Differences are paired; the 95% intervals resample campaigns
(C.N_BOOTSTRAP resamples) and are not corrected for the number of groups.
"""

import json

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p
from pacing.provenance import write_meta

R = p("results")
FAMILY = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Seasonal naive": "naive_seed"}
METHODS = list(FAMILY) + ["Even pacing", "PID"]
PAIRS = [("TFT", "Seasonal naive"), ("NHITS", "Seasonal naive"), ("TFT", "NHITS"), ("LightGBM", "Seasonal naive"),
         ("TFT", "Even pacing"), ("TFT", "PID")]


def family(policy):
    if policy == "Even pacing":
        return "Even pacing"
    if policy == "PID (tuned)":
        return "PID"
    if policy.endswith("(q=0.5)"):
        for name, prefix in FAMILY.items():
            if policy.startswith(prefix):
                return name
    return None


panel, campaigns, stat = load_work()
ids = np.asarray(campaigns)
ep = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
ep["method"] = ep.policy.map(family)
ep = ep[ep.method.notna()].copy()
ep["campaign"] = ids[ep["campaign"].to_numpy()]  # the episodes label a campaign by its position in the list

tr = panel[(panel.campaign.isin(ids)) & (panel.hour < C.TRAIN_END)].copy()
tr["day"] = tr.hour // 24
daily = tr.groupby(["campaign", "day"]).spend.sum().unstack()
st = pd.DataFrame({"size": daily.mean(axis=1), "vol": daily.std(axis=1, ddof=1) / daily.mean(axis=1)})
bounds = {}
for c in ("size", "vol"):
    st[c + "_grp"], edges = pd.qcut(st[c], 3, labels=["low", "mid", "high"], retbins=True)
    bounds[c] = [float(x) for x in edges]
corr = float(st[["size", "vol"]].corr(method="spearman").iloc[0, 1])

pc = ep.groupby(["campaign", "rho", "method"]).agg(clicks=("clicks", "sum"), ref=("ref_clicks", "sum")).reset_index()
n_fam = ep.groupby("method").policy.nunique()  # seeds per family (neural models have several)
pc["clicks"] = pc["clicks"] / pc["method"].map(n_fam)
tab = pc.pivot_table(index=["campaign", "rho"], columns="method", values="clicks")
tab["ref"] = pc[pc.method == "Even pacing"].set_index(["campaign", "rho"])["ref"]
tab = tab.reset_index().merge(st[["size_grp", "vol_grp"]], left_on="campaign", right_index=True)

rng = np.random.default_rng(0)
rows = []


def cell(g, group, value, rho):
    pcm = g.groupby("campaign")[METHODS + ["ref"]].sum()
    X, Rf, n = pcm[METHODS].to_numpy(float), pcm["ref"].to_numpy(float), len(pcm)
    res = {"group": group, "value": value, "rho": float(rho), "campaigns": int(n)}
    for j, m in enumerate(METHODS):
        res[f"ratio_{m}"] = float(X[:, j].sum() / Rf.sum())
    idx = rng.integers(0, n, (C.N_BOOTSTRAP, n))
    bX, bR = X[idx].sum(axis=1), Rf[idx].sum(axis=1)
    for a, b in PAIRS:
        ja, jb = METHODS.index(a), METHODS.index(b)
        d = (bX[:, ja] - bX[:, jb]) / bR
        res[f"diff_{a}_minus_{b}"] = float((X[:, ja].sum() - X[:, jb].sum()) / Rf.sum())
        res[f"ci_{a}_minus_{b}"] = [float(x) for x in np.quantile(d, [0.025, 0.975])]
    rows.append(res)


for rho in sorted(tab.rho.unique()):
    t = tab[tab.rho == rho]
    cell(t, "all", "all", rho)
    for col, group in (("size_grp", "size"), ("vol_grp", "volatility")):
        for v in ("low", "mid", "high"):
            cell(t[t[col] == v], group, v, rho)

out = {"campaigns": int(len(st)), "tercile_bounds": bounds, "spearman_size_vol": corr, "rows": rows}
with open(f"{R}/pacing_groups.json", "w") as f:
    json.dump(out, f, indent=1)
write_meta("pacing_groups", n_rows=len(rows))
print(pd.DataFrame(rows)[["group", "value", "rho", "campaigns", "ratio_Seasonal naive", "ratio_TFT",
                          "diff_TFT_minus_Seasonal naive", "diff_TFT_minus_NHITS"]].round(4).to_string(index=False))
