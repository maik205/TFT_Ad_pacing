"""iPinYou season-2 replication (PROTOCOL 10.20). Small and descriptive: 5 advertisers, 7 days, 10 test episodes.

Campaign = advertiser. Hours 0-95 train, 96-119 validation (day 4), 120-167 test (days 5 and 6). Forecast = seasonal naive over
the previous 3 days (7 days are too few for more). Policies: no pacing, even pacing, PID (tuned on the validation day), dual
gradient flat/profile (eta tuned on the validation day), the naive throttle, the scaled naive throttle (scale chosen on the
validation day), and the oracle remaining demand. Writes ipinyou.json. Intervals resample the 10 test episodes."""

import itertools
import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.env import ImpressionStore
from pacing.forecasters import seasonal_naive
from pacing.policies import EvenPacing, ForecastThrottle, NoPacing, PID, TunedPID
from pacing.provenance import write_meta
from pacing.simulate import simulate
from pacing.data import p

R = p("results")
SRC = os.path.expanduser("~/datasets/ipinyou_work/impressions.parquet")
TRAIN_END, VAL_END, TEST_END = 96, 120, 168
VAL_DAYS, TEST_DAYS = [4], [5, 6]
VAL_SEEDS, TEST_SEEDS = [0, 1, 2], list(range(10))
SCALES = [round(float(x), 4) for x in np.arange(0.50, 1.1001, 0.025)]
ETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
KP = (0.03125, 0.125, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)
KI = (0, 0.05, 0.2, 0.5, 1)
KD = (0, 0.5, 2, 8, 32)
K50 = list(C.QUANTILES).index(0.5)

imp = pd.read_parquet(SRC)
campaigns = sorted(imp.campaign.unique())
cidx = {c: i for i, c in enumerate(campaigns)}
Y = np.zeros((len(campaigns), 168))
g = imp.groupby(["campaign", "hour"]).cost.sum()
for (c, h), v in g.items():
    Y[cidx[c], h] = v
daily = Y[:, :TRAIN_END].sum(1) / (TRAIN_END / 24)
store = ImpressionStore(imp[["campaign", "hour", "cost", "click", "attribution"]], campaigns)
prof = Y[:, :TRAIN_END].reshape(len(campaigns), -1, 24).mean(1)
prof = prof / prof.sum(1, keepdims=True)
val_o, test_o = np.arange(TRAIN_END, VAL_END), np.arange(VAL_END, TEST_END)
Fv, Ft = seasonal_naive(Y, val_o, n_days=3), seasonal_naive(Y, test_o, n_days=3)


def _by(v, rho):
    return v[rho] if isinstance(v, dict) else v


class DualGradient:
    def __init__(self, name, eta, profile=None):
        self.name, self.eta, self.profile = name, eta, profile

    def reset(self, ep):
        rho = round(ep.budget / daily[ep.c], 4)
        self.e = _by(self.eta, rho)
        a0 = min(1.0, ep.budget / daily[ep.c])
        self.mu = 1.0 / a0 - 1.0
        share = np.full(24, 1 / 24) if self.profile is None else self.profile[ep.c]
        self.tgt, self.B = ep.budget * share, ep.budget

    def act(self, h, S, spends):
        if h > 0 and self.tgt[h - 1] > 0:
            self.mu = max(0.0, self.mu + self.e * (spends[-1] - self.tgt[h - 1]) / self.tgt[h - 1])
        return 0.0 if self.B - S <= 0 else min(1.0, 1.0 / (1.0 + self.mu))


class Scaled(ForecastThrottle):
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


def run(policies, days, seeds):
    df = pd.DataFrame(simulate(store, daily, lambda: policies, range(len(campaigns)), days, seeds))
    return df[df.policy != "Hindsight constant"].copy()


# 1. tuning on the validation day
PID_GRID = list(itertools.product(KP, KI, KD))
PV = [PID(daily, *g_) for g_ in PID_GRID] + [DualGradient(f"DGflat {e}", e) for e in ETAS] + [DualGradient(f"DGprof {e}", e, prof) for e in ETAS]
PV += [Scaled(Fv, val_o, f"naive x{s}", s) for s in SCALES]
val = pd.DataFrame(simulate(store, daily, lambda: PV, range(len(campaigns)), VAL_DAYS, VAL_SEEDS))
val["policy"] = val.policy.str.replace(" (q=0.5)", "", regex=False)
vs = val.groupby(["rho", "policy"]).apply(lambda g_: g_.clicks.sum() / g_.ref_clicks.sum(), include_groups=False)
pid_g, eta_f, eta_p, scale = {}, {}, {}, {}
chosen = {}
for rho, s in vs.groupby(level=0):
    s = s.droplevel(0)
    rk = round(float(rho), 4)
    pn = s[[n for n in s.index if n.startswith("PID(")]].idxmax()
    pid_g[rk] = [float(x.split("=")[1]) for x in pn[4:-1].split(",")]
    eta_f[rk] = float(s[[f"DGflat {e}" for e in ETAS]].idxmax().split()[1])
    eta_p[rk] = float(s[[f"DGprof {e}" for e in ETAS]].idxmax().split()[1])
    scale[rk] = float(s[[f"naive x{x}" for x in SCALES]].idxmax().split("x")[1])
    chosen[str(rho)] = {"pid": pid_g[rk], "eta_flat": eta_f[rk], "eta_prof": eta_p[rk], "scale": scale[rk]}
print("chosen on the validation day:", json.dumps(chosen, indent=1))

# 2. test days
PT = [NoPacing(), EvenPacing(daily), TunedPID(daily, pid_g), DualGradient("DG flat", eta_f), DualGradient("DG prof", eta_p, prof),
      ForecastThrottle(Ft, test_o, 0.5, "naive"), OracleRemaining()]
PT += [Scaled(Ft, test_o, f"naive x{s}", s) for s in SCALES]
new = run(PT, TEST_DAYS, TEST_SEEDS)
new["policy"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
ref = new[new.policy == "No pacing"]

# 3. paired comparisons over the 10 test episodes (campaign x day)
cols = ["clicks", "spend", "ref_clicks", "budget"]
ep = new.groupby(["campaign", "day", "rho", "policy"])[cols].sum().reset_index()
rng = np.random.default_rng(0)
res = {"chosen": chosen, "levels": {}, "pairs": [], "episodes": int(ep.groupby(["rho", "policy"]).size().max())}
for rho in C.RHOS:
    t = ep[ep.rho == rho]
    tuned = f"naive x{scale[round(float(rho), 4)]}"
    cl = t.pivot_table(index=["campaign", "day"], columns="policy", values="clicks")
    sp = t.pivot_table(index=["campaign", "day"], columns="policy", values="spend")
    bu = t.pivot_table(index=["campaign", "day"], columns="policy", values="budget")
    rf = t[t.policy == "No pacing"].set_index(["campaign", "day"]).ref_clicks.loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = rng.integers(0, n, (C.N_BOOTSTRAP, n))
    names = {"No pacing": "No pacing", "Even pacing": "Even pacing", "PID (tuned)": "PID", "DG flat": "DG flat", "DG prof": "DG prof",
             "naive": "Naive", tuned: "Naive scaled", "OR remaining": "Oracle remaining"}
    res["levels"][str(rho)] = {v: {"click_ratio": float(cl[k].sum() / rf.sum()), "budget_spent": float(sp[k].sum() / bu[k].sum())}
                               for k, v in names.items()}
    pairs = [("naive", "Even pacing"), ("naive", "PID (tuned)"), ("naive", "DG prof"), ("naive", "DG flat"), (tuned, "naive"),
             (tuned, "DG prof"), (tuned, "PID (tuned)"), ("DG prof", "PID (tuned)"), ("DG flat", "PID (tuned)"),
             ("OR remaining", "naive"), ("OR remaining", tuned)]
    for a, b in pairs:
        ca, cb = cl[a].to_numpy(), cl[b].to_numpy()
        pt = 100 * (ca.sum() - cb.sum()) / rf.sum()
        bs = 100 * (ca[idx].sum(-1) - cb[idx].sum(-1)) / rf[idx].sum(-1)
        res["pairs"].append({"rho": float(rho), "a": names[a], "b": names[b], "points": float(pt), "ci": np.quantile(bs, [0.025, 0.975]).tolist()})
    curve = {str(s): float(cl[f"naive x{s}"].sum() / rf.sum()) for s in SCALES}
    res.setdefault("curve", {})[str(rho)] = curve

ok = []
for rho in C.RHOS:
    L = res["levels"][str(rho)]
    ok.append(L["Naive scaled"]["click_ratio"] >= L["Naive"]["click_ratio"] and L["Naive"]["click_ratio"] >= max(
        L["Even pacing"]["click_ratio"], L["PID"]["click_ratio"]))
res["consistent_levels"] = int(sum(ok))
res["verdict"] = "consistent" if sum(ok) >= 2 else "inconsistent"
json.dump(res, open(f"{R}/ipinyou.json", "w"), indent=1)
write_meta("ipinyou", train_hours=TRAIN_END, val_hours=[TRAIN_END, VAL_END], test_hours=[VAL_END, TEST_END], seeds=TEST_SEEDS,
           advertisers=[int(c) for c in campaigns])
pd.set_option("display.width", 200)
print(json.dumps(res["levels"], indent=1))
print(pd.DataFrame(res["pairs"]).assign(ci=lambda d: d.ci.map(lambda x: f"[{x[0]:.1f},{x[1]:.1f}]")).round(2).to_string(index=False))
print("consistent at", res["consistent_levels"], "of 3 budget levels ->", res["verdict"])
