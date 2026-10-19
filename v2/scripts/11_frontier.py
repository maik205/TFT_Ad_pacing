"""Spend-matched comparison (post-hoc, PROTOCOL 10.18). Scaling the seasonal-naive total traces a curve of clicks against
spend. Is a forecaster above that curve at its own spend? Reads pacing_test_episodes.parquet and
scaled_naive_episodes.parquet; writes frontier.json."""

import json

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import p

R = p("results")
old = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
new = pd.read_parquet(f"{R}/scaled_naive_episodes.parquet")
FAM = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Naive": "naive_seed"}


def fam(s):
    if s in ("Even pacing",):
        return s
    if s == "PID (tuned)":
        return "PID"
    if s.endswith("(q=0.5)"):
        for k, v in FAM.items():
            if s.startswith(v):
                return k
    return None


old["method"] = old.policy.map(fam)
old = old[old.method.notna()]
n_fam = old.groupby("method").policy.nunique()
new["method"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
cols = ["clicks", "spend", "ref_clicks"]
T = pd.concat([old.groupby(["campaign", "rho", "method"])[cols].sum().reset_index(),
               new.groupby(["campaign", "rho", "method"])[cols].sum().reset_index()])
T["w"] = T.method.map(n_fam).fillna(1)
for c in ("clicks", "spend"):
    T[c] = T[c] / T.w
SC = sorted({m for m in T.method if m.startswith("naive x")}, key=lambda m: float(m.split("x")[1]))
POINTS = ["TFT", "NHITS", "LightGBM", "Naive", "PID", "Even pacing"]

rng = np.random.default_rng(0)
out = []
for rho in C.RHOS:
    t = T[T.rho == rho]
    cl = t.pivot_table(index="campaign", columns="method", values="clicks")
    sp = t.pivot_table(index="campaign", columns="method", values="spend")
    rf = t[t.method == "Even pacing"].set_index("campaign").ref_clicks.loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = np.vstack([np.arange(n), rng.integers(0, n, (C.N_BOOTSTRAP, n))])
    Ck = np.stack([cl[m].to_numpy() for m in SC], 1)      # (n, scales)
    Sk = np.stack([sp[m].to_numpy() for m in SC], 1)
    ck, sk = Ck[idx].sum(1), Sk[idx].sum(1)                 # (B+1, scales)
    rb = rf[idx].sum(1)
    for m in POINTS:
        cm, sm = cl[m].to_numpy()[idx].sum(1), sp[m].to_numpy()[idx].sum(1)
        gap, eff = [], []
        for b in range(len(idx)):
            o = np.argsort(sk[b])
            inside = sk[b][o][0] <= sm[b] <= sk[b][o][-1]
            fc = np.interp(sm[b], sk[b][o], ck[b][o])
            gap.append(100 * (cm[b] - fc) / rb[b] if inside else np.nan)
        gap = np.array(gap)
        out.append({"rho": float(rho), "method": m, "within_scale_range": bool(~np.isnan(gap[0])),
                    "spend_share_vs_naive_x1": float(sm[0] / sk[0][SC.index("naive x1.0")]),
                    "points_above_curve": None if np.isnan(gap[0]) else float(gap[0]),
                    "ci": None if np.isnan(gap[1:]).all() else np.nanquantile(gap[1:], [0.025, 0.975]).tolist(),
                    "boot_valid": float(1 - np.isnan(gap[1:]).mean())})
    # curve itself, and the efficiency of the neural forecasters against the scaled naive nearest in spend
    out.append({"rho": float(rho), "curve_spend": (sk[0] / sk[0][SC.index("naive x1.0")]).tolist(),
                "curve_points_vs_hindsight": (100 * (ck[0] / rb[0] - ck[0][SC.index("naive x1.0")] / rb[0])).tolist(),
                "scales": [float(m.split("x")[1]) for m in SC]})
json.dump(out, open(f"{R}/frontier.json", "w"), indent=1)
d = pd.DataFrame([r for r in out if "method" in r])
d["ci"] = d.ci.map(lambda x: "-" if x is None else f"[{x[0]:.2f},{x[1]:.2f}]")
pd.set_option("display.width", 200)
print(d.round(3).to_string(index=False))
for r in out:
    if "scales" in r:
        print(r["rho"], "spend vs unscaled naive:", np.round(r["curve_spend"], 3).tolist())
