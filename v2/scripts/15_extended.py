"""Second batch of post-hoc experiments (PROTOCOL 10.19): extended scale grid, total/shape swap, oracle remaining demand.

1. Scaled seasonal-naive throttle with 25 scales (0.50 to 1.10); scale chosen per budget level on the validation days.
2. Static plans: planned remaining demand at hour h = day total x share of the day-start forecast in hours h..23, with the
   total and the shape each taken from the seasonal-naive or the TFT forecast issued at hour 0 (no re-planning).
3. Oracle: the true remaining demand of the day; and the true day total times a forecast shape.
Everything is simulated once on the test days (10 replay seeds, same draws as the main run). Writes extended.json."""

import glob
import json
import os
import re

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
SCALES = [round(float(x), 4) for x in np.arange(0.50, 1.1001, 0.025)]
K50 = list(C.QUANTILES).index(0.5)


class Scaled(ForecastThrottle):
    """The throttle of Section 5 with the median forecast sum multiplied by s."""

    def __init__(self, F, origins, label, s):
        super().__init__(F, origins, 0.5, label)
        self.s = s

    def act(self, h, S, spends):
        ep = self.ep
        R_ = ep.budget - S
        if R_ <= 0:
            return 0.0
        e = float(self.F[ep.c, 24 * ep.day + h - self.o0, : 24 - h, self.k].sum()) * self.s
        return 1.0 if e <= 0 else min(1.0, R_ / e)


class Static:
    """Planned remaining demand = day total x remaining share of the day-start shape; no re-planning."""

    def __init__(self, name, Ftot, Fshape, o0, oracle_total=False):
        self.name, self.Ftot, self.Fshape, self.o0, self.oracle_total = name, Ftot, Fshape, o0, oracle_total

    def reset(self, ep):
        self.ep = ep
        i0 = 24 * ep.day - self.o0
        sh = self.Fshape[ep.c, i0, :, K50].astype(np.float64)
        d = sh.sum()
        self.rem = np.array([sh[h:].sum() / d for h in range(24)]) if d > 0 else 1 - np.arange(24) / 24
        self.tot = float(ep.demand.sum()) if self.oracle_total else float(self.Ftot[ep.c, i0, :, K50].sum())

    def act(self, h, S, spends):
        R_ = self.ep.budget - S
        if R_ <= 0:
            return 0.0
        e = self.tot * self.rem[h]
        return 1.0 if e <= 0 else min(1.0, R_ / e)


class OracleRemaining:
    name = "OR remaining"

    def reset(self, ep):
        self.ep, self.d = ep, ep.demand

    def act(self, h, S, spends):
        R_ = self.ep.budget - S
        if R_ <= 0:
            return 0.0
        rem = float(self.d[h:].sum())
        return 1.0 if rem <= 0 else min(1.0, R_ / rem)


panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
val_o, test_o = np.arange(C.TRAIN_END, C.VAL_END), test_origins()
Fv, Ft = seasonal_naive(Y, val_o), seasonal_naive(Y, test_o)
o0 = int(test_o[0])

# 1. choose the scale per budget level on the validation days
PV = [Scaled(Fv, val_o, f"naive x{s}", s) for s in SCALES]
val = pd.DataFrame(simulate(store, daily, lambda: PV, range(C.MAX_CAMPAIGNS or len(campaigns)), VAL_DAYS, VAL_SEEDS))
val["policy"] = val.policy.str.replace(" (q=0.5)", "", regex=False)
vs = val.groupby(["rho", "policy"]).apply(lambda g: g.clicks.sum() / g.ref_clicks.sum(), include_groups=False)
chosen = {}
for rho, s in vs.groupby(level=0):
    s = s.droplevel(0)
    best = s[[f"naive x{x}" for x in SCALES]].idxmax()
    chosen[round(float(rho), 4)] = {"scale": float(best.split("x")[1]), "val_ratio": float(s[best]),
                                    "val_ratio_unscaled": float(s["naive x1.0"]),
                                    "at_edge": bool(best in (f"naive x{SCALES[0]}", f"naive x{SCALES[-1]}"))}
print("chosen on validation:", json.dumps({str(k): v for k, v in chosen.items()}, indent=1))

# 2. test days
tft = {}
for path in sorted(glob.glob(f"{R}/forecasts/TFT_seed*.npz")):
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    tft[os.path.basename(path)[:-4]] = z["F"]
PT = [Scaled(Ft, test_o, f"naive x{s}", s) for s in SCALES]
PT += [Static("ST naive-naive", Ft, Ft, o0), OracleRemaining(), Static("OR total-naive", None, Ft, o0, True)]
for name, F in tft.items():
    PT += [Static(f"ST TFT-TFT {name}", F, F, o0), Static(f"ST TFT-naive {name}", F, Ft, o0),
           Static(f"ST naive-TFT {name}", Ft, F, o0), Static(f"OR total-TFT {name}", None, F, o0, True)]
new = pd.DataFrame(simulate(store, daily, lambda: PT, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, TEST_SEEDS))
new = new[~new.policy.isin(["Hindsight constant"])].copy()
new["policy"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
new.to_parquet(f"{R}/extended_episodes.parquet", index=False)
new["method"] = new.policy.map(lambda s: re.sub(r" TFT_seed\d+", "", s))

# 3. paired comparisons (per-campaign totals, forecast seeds averaged)
old = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
FAM = {"TFT": "TFT_seed", "Naive": "naive_seed"}


def ofam(s):
    if s == "PID (tuned)":
        return "PID"
    if s == "Even pacing":
        return s
    if s.endswith("(q=0.5)"):
        for k, v in FAM.items():
            if s.startswith(v):
                return k
    return None


old["method"] = old.policy.map(ofam)
old = old[old.method.notna()]
cols = ["clicks", "spend", "ref_clicks"]
nf = {**old.groupby("method").policy.nunique().to_dict(), **new.groupby("method").policy.nunique().to_dict()}
T = pd.concat([d.groupby(["campaign", "rho", "method"])[cols].sum().reset_index() for d in (old, new)])
T["w"] = T.method.map(nf)
for c in ("clicks", "spend"):
    T[c] = T[c] / T.w

rng = np.random.default_rng(0)
res = {"chosen": {str(k): v for k, v in chosen.items()}, "scales": SCALES, "curve": {}, "pairs": [], "ratios": {}}
for rho in C.RHOS:
    t = T[T.rho == rho]
    tune = f"naive x{chosen[round(float(rho), 4)]['scale']}"
    cl = t.pivot_table(index="campaign", columns="method", values="clicks")
    sp = t.pivot_table(index="campaign", columns="method", values="spend")
    rf = t[t.method == "Even pacing"].set_index("campaign").ref_clicks.loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = rng.integers(0, n, (C.N_BOOTSTRAP, n))
    res["curve"][str(rho)] = {str(s): float(cl[f"naive x{s}"].sum() / rf.sum()) for s in SCALES}
    res["ratios"][str(rho)] = {m: float(cl[m].sum() / rf.sum()) for m in cl.columns if not m.startswith("naive x")}
    pairs = [(tune, "Naive"), ("TFT", "Naive"), ("TFT", tune),
             ("ST TFT-TFT", "ST naive-naive"), ("ST TFT-naive", "ST naive-naive"), ("ST naive-TFT", "ST naive-naive"),
             ("ST naive-naive", "Naive"), ("ST TFT-TFT", "TFT"),
             ("OR remaining", "TFT"), ("OR remaining", "Naive"), ("OR total-naive", "ST naive-naive"),
             ("OR total-TFT", "ST TFT-TFT"), ("OR remaining", "OR total-TFT")]
    for a, b in pairs:
        ca, cb, sa, sb = cl[a].to_numpy(), cl[b].to_numpy(), sp[a].to_numpy(), sp[b].to_numpy()

        def stats(ii):
            Ca, Cb, Sa, Sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
            return 100 * (Ca - Cb) / rf[ii].sum(-1), 100 * (Sa / Sb - 1), 100 * ((Ca / Sa) / (Cb / Sb) - 1)

        pt, bs = stats(np.arange(n)), stats(idx)
        res["pairs"].append({"rho": float(rho), "a": "Naive tuned" if a == tune else a, "b": "Naive tuned" if b == tune else b,
                             "scale": chosen[round(float(rho), 4)]["scale"], "points": float(pt[0]),
                             "ci": np.quantile(bs[0], [0.025, 0.975]).tolist(), "spend_pct": float(pt[1]), "eff_pct": float(pt[2])})
json.dump(res, open(f"{R}/extended.json", "w"), indent=1)
write_meta("extended", scales=SCALES, val_days=VAL_DAYS, test_days=TEST_DAYS, seeds=TEST_SEEDS)
pd.set_option("display.width", 220)
print(pd.DataFrame(res["pairs"]).assign(ci=lambda d: d.ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]")).round(2).to_string(index=False))
print("\nratios vs hindsight")
print(pd.DataFrame(res["ratios"]).round(4).to_string())
print("\ncurve")
print(pd.DataFrame(res["curve"]).round(4).to_string())
