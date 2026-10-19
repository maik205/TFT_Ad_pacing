"""Scaled seasonal naive (post-hoc, PROTOCOL 10.18). Does one scale on the forecast total close the gap to TFT/NHITS?

The throttle of Section 5 is run with the seasonal-naive median forecast multiplied by s (13 values) and with the
7-day mean. s is chosen per budget level on the validation days, then everything is simulated once on the test days
with the same uniform draws as the main run. Writes scaled_naive.json.
"""

import json

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p, spend_matrix
from pacing.env import ImpressionStore
from pacing.forecasters import seasonal_naive, test_origins
from pacing.policies import ForecastThrottle
from pacing.provenance import write_meta
from pacing.simulate import simulate

R = p("results")
VAL_DAYS = list(range(C.TRAIN_END // 24, C.VAL_END // 24))
TEST_DAYS = list(range(C.VAL_END // 24, C.TEST_END // 24))
VAL_SEEDS, TEST_SEEDS = [0, 1, 2], list(range(C.N_REPLAY_SEEDS))
SCALES = [round(float(x), 4) for x in np.arange(0.80, 1.1001, 0.025)]
MEDIAN = list(C.QUANTILES).index(0.5)

panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
val_o, test_o = np.arange(C.TRAIN_END, C.VAL_END), test_origins()


def seasonal_mean(origins, n_days=7):
    k = np.arange(1, n_days + 1)
    idx = origins[:, None, None] + np.arange(C.HORIZON)[None, :, None] - 24 * k[None, None, :]
    m = Y[:, idx].mean(-1)
    return np.repeat(m[..., None], len(C.QUANTILES), -1)


def policies(origins):
    F = seasonal_naive(Y, origins)
    out = [ForecastThrottle(F * s, origins, 0.5, f"naive x{s}") for s in SCALES]
    out.append(ForecastThrottle(seasonal_mean(origins), origins, 0.5, "naive mean7"))
    return out


# 1. choose the scale per budget level on the validation days
PV = policies(val_o)
val = pd.DataFrame(simulate(store, daily, lambda: PV, range(C.MAX_CAMPAIGNS or len(campaigns)), VAL_DAYS, VAL_SEEDS))
val["policy"] = val.policy.str.replace(" (q=0.5)", "", regex=False)
vs = val.groupby(["rho", "policy"]).apply(lambda g: g.clicks.sum() / g.ref_clicks.sum(), include_groups=False)
chosen = {}
for rho, s in vs.groupby(level=0):
    s = s.droplevel(0)
    best = s[[f"naive x{x}" for x in SCALES]].idxmax()
    chosen[float(rho)] = {"scale": float(best.split("x")[1]), "val_ratio": float(s[best]), "val_ratio_unscaled": float(s["naive x1.0"]),
                          "val_ratio_mean7": float(s["naive mean7"]), "at_edge": bool(best in (f"naive x{SCALES[0]}", f"naive x{SCALES[-1]}"))}
print("chosen on validation:", json.dumps(chosen, indent=1))

# 2. test days, all scales, same draws as the main run
PT = policies(test_o)
new = pd.DataFrame(simulate(store, daily, lambda: PT, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, TEST_SEEDS))
new["policy"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
new.to_parquet(f"{R}/scaled_naive_episodes.parquet", index=False)

# 3. per-campaign totals for every method (forecast seeds averaged), then paired differences
old = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
FAM = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Naive": "naive_seed"}


def fam(s):
    if s == "Even pacing":
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


def totals(df, key):
    g = df.groupby(["campaign", "rho", key])[["clicks", "spend", "ref_clicks"]].sum().reset_index()
    return g.rename(columns={key: "method"})


T = pd.concat([totals(old, "method"), totals(new.assign(method=new.policy), "method")])
T["w"] = T.method.map(n_fam).fillna(1)
for c in ("clicks", "spend"):
    T[c] = T[c] / T.w
ref = T[T.method == "Even pacing"].set_index(["campaign", "rho"]).ref_clicks

rng = np.random.default_rng(0)
res = {"chosen": chosen, "scales": SCALES, "curve": {}, "pairs": []}
for rho in C.RHOS:
    t = T[T.rho == rho]
    s_star = chosen[float(rho)]["scale"]
    tune = f"naive x{s_star}"
    cl = t.pivot_table(index="campaign", columns="method", values="clicks")
    sp = t.pivot_table(index="campaign", columns="method", values="spend")
    rf = ref.xs(rho, level="rho").loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = rng.integers(0, n, (C.N_BOOTSTRAP, n))
    res["curve"][str(rho)] = {str(s): float(cl[f"naive x{s}"].sum() / rf.sum()) for s in SCALES}
    res["curve"][str(rho)].update({"mean7": float(cl["naive mean7"].sum() / rf.sum()), "TFT": float(cl["TFT"].sum() / rf.sum()),
                                   "NHITS": float(cl["NHITS"].sum() / rf.sum()), "Naive": float(cl["Naive"].sum() / rf.sum())})
    for a, b in (("naive mean7", "Naive"), (tune, "Naive"), ("TFT", "Naive"), ("NHITS", "Naive"), ("TFT", tune), ("NHITS", tune),
                 ("TFT", "PID"), (tune, "PID")):
        ca, cb, sa, sb = cl[a].to_numpy(), cl[b].to_numpy(), sp[a].to_numpy(), sp[b].to_numpy()

        def stats(ii):
            Ca, Cb, Sa, Sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
            return 100 * (Ca - Cb) / rf[ii].sum(-1), 100 * (Sa / Sb - 1), 100 * ((Ca / Sa) / (Cb / Sb) - 1)

        pt, bs = stats(np.arange(n)), stats(idx)
        res["pairs"].append({"rho": float(rho), "a": a.replace(tune, "Naive tuned") if a == tune else a,
                             "b": b.replace(tune, "Naive tuned") if b == tune else b, "scale": s_star, "campaigns": int(n),
                             "points_vs_hindsight": float(pt[0]), "ci": np.quantile(bs[0], [0.025, 0.975]).tolist(),
                             "spend_pct": float(pt[1]), "eff_pct": float(pt[2])})
json.dump(res, open(f"{R}/scaled_naive.json", "w"), indent=1)
write_meta("scaled_naive", scales=SCALES, val_days=VAL_DAYS, test_days=TEST_DAYS, seeds=TEST_SEEDS)

pd.set_option("display.width", 200)
print(pd.DataFrame(res["pairs"]).assign(ci=lambda d: d.ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]")).round(2).to_string(index=False))
print("\ncurve (click ratio vs hindsight):")
print(pd.DataFrame(res["curve"]).round(4).to_string())
