"""Calibrated-total hedge (post-hoc, PROTOCOL 10.18). The hedged plan is the median plan times a factor, in place of the
sum of hourly q-quantile forecasts. Causal variant: factor for test day d from the earlier test days (all campaigns);
cross-fitted variant: factor from the other random half of the campaigns over all test days. Writes calibrated_hedge.json."""

import glob
import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p, spend_matrix
from pacing.env import ImpressionStore
from pacing.forecasters import test_origins
from pacing.policies import ForecastThrottle
from pacing.provenance import write_meta
from pacing.simulate import simulate

R = p("results")
D0 = C.VAL_END // 24
DAYS = list(range(D0, C.TEST_END // 24))
QS = (0.7, 0.9)
K50 = list(C.QUANTILES).index(0.5)
FAM = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "LightGBM": "lightgbm_seed", "Seasonal naive": "naive_seed"}

panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
n = len(campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
half = np.random.default_rng(0).permutation(n) % 2


def realised_and_plan(F, o0):
    T, P = np.zeros((n, len(DAYS), 24)), np.zeros((n, len(DAYS), 24))
    for j, d in enumerate(DAYS):
        for h in range(24):
            P[:, j, h] = F[:, 24 * d + h - o0, : 24 - h, K50].sum(1)
            T[:, j, h] = Y[:, 24 * d + h: 24 * d + 24].sum(1)
    return T, P


def factor_causal(ratio, q):
    f = np.ones((n, len(DAYS), 24))
    for j in range(1, len(DAYS)):
        f[:, j, :] = np.nanquantile(ratio[:, :j, :].reshape(-1, 24), q, axis=0)
    return f


def factor_crossfit(ratio, q):
    f = np.ones((n, len(DAYS), 24))
    for hv in (0, 1):
        fq = np.nanquantile(ratio[half != hv].reshape(-1, 24), q, axis=0)
        f[half == hv] = fq[None, None, :]
    return f


class CalibratedThrottle(ForecastThrottle):
    def __init__(self, F, origins, label, factor):
        super().__init__(F, origins, 0.5, label)
        self.factor = factor

    def act(self, h, S, spends):
        ep = self.ep
        R_ = ep.budget - S
        if R_ <= 0:
            return 0.0
        expected = float(self.F[ep.c, 24 * ep.day + h - self.o0, : 24 - h, self.k].sum()) * self.factor[ep.c, ep.day - D0, h]
        return 1.0 if expected <= 0 else min(1.0, R_ / expected)


forecasts = {}
for path in sorted(glob.glob(f"{R}/forecasts/*.npz")):
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    forecasts[os.path.basename(path)[:-4]] = z["F"]
test_o = test_origins()
pols, factors = [], {}
for name, F in forecasts.items():
    T, P = realised_and_plan(F.astype(np.float64), int(test_o[0]))
    ratio = np.where(P > 0, T / np.where(P > 0, P, 1), np.nan)
    for q in QS:
        for v, fn in (("b", factor_causal), ("a", factor_crossfit)):
            f = fn(ratio, q)
            factors[f"{v} {q} {name}"] = [float(x) for x in f[0, -1, ::6]]
            pols.append(CalibratedThrottle(F, test_o, f"CAL{v} {q} {name}", f))
print(len(pols), "policies")
EPI = f"{R}/calibrated_hedge_episodes.parquet"
if os.path.exists(EPI):  # the simulation is deterministic; reuse a saved run
    new = pd.read_parquet(EPI)
else:
    new = pd.DataFrame(simulate(store, daily, lambda: pols, range(C.MAX_CAMPAIGNS or n), DAYS, list(range(C.N_REPLAY_SEEDS))))
    new["policy"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
    new.to_parquet(EPI, index=False)
new = new[new.policy.str.startswith("CAL")].copy()  # drops the hindsight-reference rows


def family(name):
    return next(k for k, v in FAM.items() if v in name)


def label_new(s):
    v, q, name = s[3], s.split()[1], s.split()[2]
    return f"{family(name)} q{q} cal{v}"


def label_old(s):
    if not s.endswith(")") or "(q=" not in s:
        return None
    for k, v in FAM.items():
        if s.startswith(v):
            return f"{k} q{s.split('(q=')[1][:-1]} summed"
    return None


old = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
old["method"] = old.policy.map(label_old)
old = old[old.method.notna()]
new["method"] = new.policy.map(label_new)
nf = {**old.groupby("method").policy.nunique().to_dict(), **new.groupby("method").policy.nunique().to_dict()}
cols = ["clicks", "spend", "ref_clicks"]
rng = np.random.default_rng(0)
res = {"factors_last_day_every_6h": factors, "pairs": []}


def compare(min_day, tag, variants):
    o, nw = old[old.day >= min_day], new[new.day >= min_day]
    T = pd.concat([d.groupby(["campaign", "rho", "method"])[cols].sum().reset_index() for d in (o, nw)])
    T["w"] = T.method.map(nf)
    for c in ("clicks", "spend"):
        T[c] = T[c] / T.w
    for rho in C.RHOS:
        t = T[T.rho == rho]
        cl = t.pivot_table(index="campaign", columns="method", values="clicks")
        sp = t.pivot_table(index="campaign", columns="method", values="spend")
        eps = pd.read_parquet(f"{R}/pacing_test_episodes.parquet", columns=["policy", "rho", "campaign", "day", "ref_clicks"])
        ref = eps[(eps.policy == "Even pacing") & (eps.rho == rho) & (eps.day >= min_day)].groupby("campaign").ref_clicks.sum().loc[cl.index].to_numpy(float)
        m = len(cl)
        idx = rng.integers(0, m, (C.N_BOOTSTRAP, m))
        for fam in FAM:
            for q in QS:
                for v in variants:
                    cal, summ, med = f"{fam} q{q} cal{v}", f"{fam} q{q} summed", f"{fam} q0.5 summed"
                    for a, b, what in ((cal, summ, "calibrated - summed"), (cal, med, "calibrated - median plan"),
                                       (summ, med, "summed - median plan")):
                        ca, cb, sa, sb = cl[a].to_numpy(), cl[b].to_numpy(), sp[a].to_numpy(), sp[b].to_numpy()

                        def stats(ii):
                            Ca, Cb, Sa, Sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
                            return 100 * (Ca - Cb) / ref[ii].sum(-1), 100 * (Sa / Sb - 1), 100 * ((Ca / Sa) / (Cb / Sb) - 1)
                        pt, bs = stats(np.arange(m)), stats(idx)
                        res["pairs"].append({"set": tag, "variant": v, "rho": float(rho), "forecaster": fam, "q": q, "contrast": what,
                                             "points": float(pt[0]), "ci": np.quantile(bs[0], [0.025, 0.975]).tolist(),
                                             "spend_pct": float(pt[1]), "eff_pct": float(pt[2])})


compare(D0 + 1, "days 26-29", ["b", "a"])
compare(D0, "days 25-29", ["a"])
json.dump(res, open(f"{R}/calibrated_hedge.json", "w"), indent=1)
write_meta("calibrated_hedge", quantiles=QS, test_days=DAYS, seeds=C.N_REPLAY_SEEDS)
d = pd.DataFrame(res["pairs"])
d["ci"] = d.ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]")
pd.set_option("display.width", 220)
for (s, v), g in d.groupby(["set", "variant"]):
    print(f"\n=== {s}, variant {v}")
    print(g[g.forecaster.isin(["TFT", "Seasonal naive"])].drop(columns=["set", "variant"]).round(2).to_string(index=False))
