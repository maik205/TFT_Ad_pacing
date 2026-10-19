"""Effective quantile of the summed hedge plans (descriptive, PROTOCOL 10.18).

At origin hour h of a test day the throttle uses P_q = sum of the q-quantile forecasts of hours h..23. This script
measures how often the realised spend of those hours, T, is at or below P_q, by forecaster family and q, and the
factor on the median plan that would give coverage q (the q-quantile of T / P_0.5). Writes hedge_coverage.json."""

import glob
import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p, spend_matrix

R = p("results")
panel, campaigns, stat = load_work()
Y = spend_matrix(panel, campaigns)
DAYS = list(range(C.VAL_END // 24, C.TEST_END // 24))
QS = list(C.QUANTILES)
FAM = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Seasonal naive": "naive_seed"}

rows = {k: [] for k in FAM}
for path in sorted(glob.glob(f"{R}/forecasts/*.npz")):
    name = os.path.basename(path)[:-4]
    fam = next((k for k, v in FAM.items() if name.startswith(v)), None)
    if fam is None:
        continue
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    F, o0 = z["F"].astype(np.float64), int(z["origins"][0])
    for d in DAYS:
        for h in range(24):
            P = F[:, 24 * d + h - o0, : 24 - h, :].sum(1)            # (n, 4) summed plans
            T = Y[:, 24 * d + h: 24 * d + 24].sum(1)                  # realised rest-of-day spend
            hourly = Y[:, 24 * d + h: 24 * d + 24][:, :, None] <= F[:, 24 * d + h - o0, : 24 - h, :]   # per-hour hits
            rows[fam].append(pd.DataFrame({"h": h, "T": T, **{f"P{q}": P[:, k] for k, q in enumerate(QS)},
                                           **{f"hit{q}": hourly.mean(1)[:, k] for k, q in enumerate(QS)}}))
out = {}
for fam, parts in rows.items():
    df = pd.concat(parts)
    df = df[df["P0.5"] > 0]
    res = {"rows": int(len(df))}
    ratio = df["T"] / df["P0.5"]
    for q in (0.5, 0.7, 0.9, 0.1):
        P = df[f"P{q}"]
        res[f"q{q}"] = {
            "nominal": q,
            "hourly_coverage": float(df[f"hit{q}"].mean()),                       # share of single hours at or below the quantile
            "summed_plan_coverage": float((df["T"] <= P).mean()),                  # share of rest-of-day totals at or below the sum
            "median_plan_factor": float((P / df["P0.5"]).median()),                # implied scale of the plan on the median plan
            "calibrated_factor": float(ratio.quantile(q)),                          # scale on the median plan that gives coverage q
        }
    byh = {}
    for lo, hi in ((0, 5), (6, 11), (12, 17), (18, 23)):
        s = df[(df.h >= lo) & (df.h <= hi)]
        byh[f"{lo}-{hi}"] = {f"q{q}": float((s["T"] <= s[f"P{q}"]).mean()) for q in (0.5, 0.7, 0.9)}
    res["summed_plan_coverage_by_hours"] = byh
    out[fam] = res
json.dump(out, open(f"{R}/hedge_coverage.json", "w"), indent=1)
tab = pd.DataFrame([{"forecaster": f, "q": q, **{k: v for k, v in r[f"q{q}"].items() if k != "nominal"}}
                    for f, r in out.items() for q in (0.5, 0.7, 0.9)])
pd.set_option("display.width", 200)
print(tab.round(3).to_string(index=False))
print()
print(pd.DataFrame({f: {f"{h} q0.9": r["summed_plan_coverage_by_hours"][h]["q0.9"] for h in r["summed_plan_coverage_by_hours"]}
                    for f, r in out.items()}).round(3).to_string())
