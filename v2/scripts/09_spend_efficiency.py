"""Spend versus efficiency (post-hoc, PROTOCOL 10.18). usage: 09_spend_efficiency.py <episodes.parquet> <out.json>
Exploratory, descriptive: reads pacing_test_episodes.parquet only. Totals over the test days; seeds of a forecaster averaged;
95% intervals resample campaigns (paired)."""
import json
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
FAMILY = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Seasonal naive": "naive_seed"}
ep = pd.read_parquet(src)


def family(policy, q):
    if policy in ("Even pacing", "Hindsight constant"):
        return policy
    if policy == "PID (tuned)":
        return "PID"
    if policy.endswith(f"(q={q})"):
        for name, prefix in FAMILY.items():
            if policy.startswith(prefix):
                return name
    return None


PAIRS = [("TFT", "Seasonal naive"), ("NHITS", "Seasonal naive"), ("LightGBM", "Seasonal naive"),
         ("TFT", "Even pacing"), ("TFT", "PID"), ("Hindsight constant", "Even pacing")]
rng = np.random.default_rng(0)
rows = []
for q in ("0.5", "0.9"):
    e = ep.copy()
    e["method"] = e.policy.map(lambda s: family(s, q))
    e = e[e.method.notna()]
    nfam = e.groupby("method").policy.nunique()
    pc = e.groupby(["campaign", "rho", "method"])[["clicks", "spend", "budget"]].sum().reset_index()
    for c in ("clicks", "spend", "budget"):
        pc[c] = pc[c] / pc["method"].map(nfam)
    for rho in sorted(pc.rho.unique()):
        t = pc[pc.rho == rho]
        T = {c: t.pivot_table(index="campaign", columns="method", values=c) for c in ("clicks", "spend", "budget")}
        n = len(T["clicks"])
        idx = rng.integers(0, n, (2000, n))

        def stats(a, b, ii=None):
            cl, sp = T["clicks"], T["spend"]
            ca, cb = cl[a].to_numpy(), cl[b].to_numpy()
            sa, sb = sp[a].to_numpy(), sp[b].to_numpy()
            if ii is not None:
                ca, cb, sa, sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
            else:
                ca, cb, sa, sb = ca.sum(), cb.sum(), sa.sum(), sb.sum()
            return 100 * (ca / cb - 1), 100 * (sa / sb - 1), 100 * ((ca / sa) / (cb / sb) - 1)

        for a, b in PAIRS:
            if a not in T["clicks"] or b not in T["clicks"] or (q == "0.9" and b in ("PID", "Even pacing") and a == "Hindsight constant"):
                continue
            pt = stats(a, b)
            bs = [np.quantile(x, [0.025, 0.975]) for x in stats(a, b, idx)]
            rows.append({"q": q, "rho": float(rho), "a": a, "b": b, "campaigns": int(n),
                         "clicks_pct": pt[0], "clicks_ci": bs[0].tolist(),
                         "spend_pct": pt[1], "spend_ci": bs[1].tolist(),
                         "eff_pct": pt[2], "eff_ci": bs[2].tolist()})
        # level: share of budget spent, clicks per unit spend, for each method
        sp, bd, cl = T["spend"].sum(), T["budget"].sum(), T["clicks"].sum()
        for m in sp.index:
            rows.append({"q": q, "rho": float(rho), "method": m, "budget_spent": float(sp[m] / bd[m]),
                         "clicks_per_spend": float(cl[m] / sp[m])})
json.dump(rows, open(out, "w"), indent=1)
d = pd.DataFrame([r for r in rows if "a" in r])
pd.set_option("display.width", 200)
for q in ("0.5", "0.9"):
    print("\n=== q =", q)
    s = d[d.q == q].copy()
    for c in ("clicks", "spend", "eff"):
        s[c] = s[c + "_pct"].round(2).astype(str) + " " + s[c + "_ci"].map(lambda x: f"[{x[0]:.1f},{x[1]:.1f}]")
    print(s[["rho", "a", "b", "clicks", "spend", "eff"]].to_string(index=False))
lv = pd.DataFrame([r for r in rows if "method" in r])
print("\n=== levels")
print(lv.round(4).to_string(index=False))
