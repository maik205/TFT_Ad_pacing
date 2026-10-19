"""Forecast throttle plus PID correction (post-hoc, PROTOCOL 10.18). Gains are chosen per budget level on the
validation days for two seasonal-naive bases, then every forecaster is simulated once on the test days. Writes feedback.json."""

import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p, spend_matrix
from pacing.env import ImpressionStore, run_episode, uniforms
from pacing.forecasters import seasonal_naive, test_origins
from pacing.policies import ForecastThrottle
from pacing.provenance import write_meta
from pacing.simulate import simulate

R = p("results")
VAL_DAYS = list(range(C.TRAIN_END // 24, C.VAL_END // 24))
TEST_DAYS = list(range(C.VAL_END // 24, C.TEST_END // 24))
VAL_SEEDS, TEST_SEEDS = [0, 1, 2], list(range(C.N_REPLAY_SEEDS))
KP, KI, KD = (0, 2, 8, 32), (0, 0.5), (0, 4)
GRID = list(itertools.product(KP, KI, KD))
SCALE = {round(float(r), 4): v["scale"] for r, v in json.load(open(f"{R}/scaled_naive.json"))["chosen"].items()}


class Feedback(ForecastThrottle):
    """Forecast throttle plus PID correction on the gap to the day's planned spend path. gains / scale may be
    a value or a dict keyed by round(rho, 4)."""

    def __init__(self, F, origins, label, daily, gains, scale=1.0, q=0.5):
        super().__init__(F, origins, q, label)
        self.daily, self.gains, self.scale = daily, gains, scale

    @staticmethod
    def _by(v, rho):
        return v[rho] if isinstance(v, dict) else v

    def reset(self, ep):
        super().reset(ep)
        rho = round(ep.budget / self.daily[ep.c], 4)
        self.s = self._by(self.scale, rho)
        self.kp, self.ki, self.kd = self._by(self.gains, rho)
        d = self.F[ep.c, 24 * ep.day - self.o0, :, self.k] * self.s          # forecast issued at hour 0
        D = float(d.sum())
        before = np.concatenate([[0.0], np.cumsum(d)])[:24]                    # forecast demand before hour h
        self.plan = min(ep.budget, D) * before / D if D > 0 else np.zeros(24)
        self.I, self.prev = 0.0, 0.0

    def act(self, h, S, spends):
        ep = self.ep
        R_ = ep.budget - S
        if R_ <= 0:
            return 0.0
        expected = float(self.F[ep.c, 24 * ep.day + h - self.o0, : 24 - h, self.k].sum()) * self.s
        base = 1.0 if expected <= 0 else min(1.0, R_ / expected)
        e = (self.plan[h] - S) / ep.budget
        self.I += e
        de = e - self.prev if h > 0 else 0.0
        self.prev = e
        return float(np.clip(base + self.kp * e + self.ki * self.I + self.kd * de, 0.0, 1.0))


panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
val_o, test_o = np.arange(C.TRAIN_END, C.VAL_END), test_origins()
Fv, Ft = seasonal_naive(Y, val_o), seasonal_naive(Y, test_o)

# check: zero gains reproduce the existing throttle on one episode
ep = store.episode(0, VAL_DAYS[0], 0.5 * daily[0])
u = uniforms(ep, 0)
a = run_episode(ep, ForecastThrottle(Fv, val_o, 0.5, "x"), u)
b = run_episode(ep, Feedback(Fv, val_o, "x", daily, (0, 0, 0)), u)
assert a["spend"] == b["spend"] and a["clicks"] == b["clicks"] and np.allclose(a["alphas"], b["alphas"]), "zero-gain check failed"

# 1. gains per budget level on the validation days
PV = [Feedback(Fv, val_o, f"A {g}", daily, g) for g in GRID] + [Feedback(Fv, val_o, f"B {g}", daily, g, SCALE) for g in GRID]
val = pd.DataFrame(simulate(store, daily, lambda: PV, range(C.MAX_CAMPAIGNS or len(campaigns)), VAL_DAYS, VAL_SEEDS))
val["policy"] = val.policy.str.replace(" (q=0.5)", "", regex=False)
vs = val.groupby(["rho", "policy"]).apply(lambda g: g.clicks.sum() / g.ref_clicks.sum(), include_groups=False)
chosen = {"A": {}, "B": {}}
detail = {}
for rho, s in vs.groupby(level=0):
    s = s.droplevel(0)
    for base in "AB":
        names = [f"{base} {g}" for g in GRID]
        best = s[names].idxmax()
        g = GRID[names.index(best)]
        chosen[base][round(float(rho), 4)] = g
        detail[f"{base}@{rho}"] = {"gains": list(g), "val_ratio": float(s[best]), "val_ratio_no_feedback": float(s[f"{base} {(0, 0, 0)}"]),
                                   "at_edge": [bool(g[0] == KP[-1]), bool(g[1] == KI[-1]), bool(g[2] == KD[-1])]}
print("chosen on validation:", json.dumps(detail, indent=1))

# 2. test days
forecasts = {}
for path in sorted(glob.glob(f"{R}/forecasts/*.npz")):
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    forecasts[os.path.basename(path)[:-4]] = z["F"]
PT = [Feedback(Ft, test_o, "FB naive", daily, chosen["A"]), Feedback(Ft, test_o, "FB naive-scaled", daily, chosen["B"], SCALE)]
PT += [Feedback(F, test_o, f"FB {name}", daily, chosen["A"]) for name, F in forecasts.items() if name.startswith(("TFT", "NHITS"))]
new = pd.DataFrame(simulate(store, daily, lambda: PT, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, TEST_SEEDS))
new["policy"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
new.to_parquet(f"{R}/feedback_episodes.parquet", index=False)


def fbfam(s):
    for k in ("FB naive-scaled", "FB naive", "FB TFT", "FB NHITS"):
        if s.startswith(k):
            return k
    return None


# 3. paired comparisons (per campaign totals, forecast seeds averaged)
old = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
sn = pd.read_parquet(f"{R}/scaled_naive_episodes.parquet")
FAM = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "Naive": "naive_seed"}


def ofam(s):
    if s == "PID (tuned)":
        return "PID"
    if s.endswith("(q=0.5)"):
        for k, v in FAM.items():
            if s.startswith(v):
                return k
    return "Even pacing" if s == "Even pacing" else None


old["method"] = old.policy.map(ofam)
old = old[old.method.notna()]
new["method"] = new.policy.map(fbfam)
sn["method"] = sn.policy.str.replace(" (q=0.5)", "", regex=False)
sn["method"] = sn.method.where(sn.method.map(lambda m: m.startswith("naive x")))
sn = sn[sn.method.notna()]
cols = ["clicks", "spend", "ref_clicks"]
nf = {**old.groupby("method").policy.nunique().to_dict(), **new.groupby("method").policy.nunique().to_dict()}
T = pd.concat([d.groupby(["campaign", "rho", "method"])[cols].sum().reset_index() for d in (old, new, sn)])
T["w"] = T.method.map(nf).fillna(1)
for c in ("clicks", "spend"):
    T[c] = T[c] / T.w

rng = np.random.default_rng(0)
res = {"chosen": detail, "pairs": []}
for rho in C.RHOS:
    t = T[T.rho == rho]
    s_star = SCALE[round(float(rho), 4)]
    ns = f"naive x{s_star}"
    cl = t.pivot_table(index="campaign", columns="method", values="clicks")
    sp = t.pivot_table(index="campaign", columns="method", values="spend")
    rf = t[t.method == "Even pacing"].set_index("campaign").ref_clicks.loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = rng.integers(0, n, (C.N_BOOTSTRAP, n))
    res.setdefault("ratios", {})[str(rho)] = {m: float(cl[m].sum() / rf.sum()) for m in
                                              ["Naive", ns, "TFT", "NHITS", "PID", "FB naive", "FB naive-scaled", "FB TFT", "FB NHITS"]}
    for a, b in (("FB naive", "Naive"), ("FB naive-scaled", ns), ("FB TFT", "TFT"), ("FB NHITS", "NHITS"),
                 ("FB TFT", "FB naive-scaled"), ("FB NHITS", "FB naive-scaled"), ("FB TFT", ns), ("FB TFT", "PID"),
                 ("FB naive-scaled", "PID")):
        ca, cb, sa, sb = (x[m].to_numpy() for x, m in ((cl, a), (cl, b), (sp, a), (sp, b)))

        def stats(ii):
            Ca, Cb, Sa, Sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
            return 100 * (Ca - Cb) / rf[ii].sum(-1), 100 * (Sa / Sb - 1), 100 * ((Ca / Sa) / (Cb / Sb) - 1)

        pt, bs = stats(np.arange(n)), stats(idx)
        res["pairs"].append({"rho": float(rho), "a": a, "b": "naive scaled" if b == ns else b, "points_vs_hindsight": float(pt[0]),
                             "ci": np.quantile(bs[0], [0.025, 0.975]).tolist(), "spend_pct": float(pt[1]), "eff_pct": float(pt[2])})
json.dump(res, open(f"{R}/feedback.json", "w"), indent=1)
write_meta("feedback", grid={"kp": KP, "ki": KI, "kd": KD}, scales=SCALE, val_days=VAL_DAYS, test_days=TEST_DAYS, seeds=TEST_SEEDS)
pd.set_option("display.width", 200)
print(pd.DataFrame(res["pairs"]).assign(ci=lambda d: d.ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]")).round(2).to_string(index=False))
print(pd.DataFrame(res["ratios"]).round(4).to_string())
