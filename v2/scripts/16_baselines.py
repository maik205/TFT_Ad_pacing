"""Third batch (PROTOCOL 10.20): a tuned dual-gradient reactive pacer, and feedback on the hourly re-plan.

1. Dual gradient: multiplier mu, alpha = min(1, 1/(1+mu)); mu moves by eta times the relative gap between last hour's
   spend and its target (flat B/24, or B times the campaign's mean training-day hourly profile). eta chosen per budget
   level on the validation days.
2. Feedback on the hourly re-plan: a PID correction on the gap between each hour's realised spend and what the
   previous plan expected (alpha of that hour times the first-step forecast of that hour, times the scale). Gains chosen
   on the validation days for the plain and the scaled seasonal-naive base, then applied to TFT and NHITS.
Everything is simulated once on the test days (10 replay seeds, same draws as the main run). Writes baselines.json."""

import glob
import itertools
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
ETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
KP, KI, KD = (0, 2, 8, 32), (0, 0.5), (0, 4)
GRID = list(itertools.product(KP, KI, KD))
SCALE = {round(float(r), 4): v["scale"] for r, v in json.load(open(f"{R}/extended.json"))["chosen"].items()}


def _by(v, rho):
    return v[rho] if isinstance(v, dict) else v


class DualGradient:
    """Reactive dual-gradient throttle."""

    def __init__(self, name, daily, eta, profile=None):
        self.name, self.daily, self.eta, self.profile = name, daily, eta, profile

    def reset(self, ep):
        rho = round(ep.budget / self.daily[ep.c], 4)
        self.e = _by(self.eta, rho)
        a0 = min(1.0, ep.budget / self.daily[ep.c])
        self.mu = 1.0 / a0 - 1.0
        share = np.full(24, 1 / 24) if self.profile is None else self.profile[ep.c]
        self.tgt = ep.budget * share
        self.B = ep.budget

    def act(self, h, S, spends):
        if h > 0 and self.tgt[h - 1] > 0:
            self.mu = max(0.0, self.mu + self.e * (spends[-1] - self.tgt[h - 1]) / self.tgt[h - 1])
        if self.B - S <= 0:
            return 0.0
        return min(1.0, 1.0 / (1.0 + self.mu))


class FeedbackHour(ForecastThrottle):
    """Forecast throttle plus PID correction on the previous hour's gap between realised and expected spend."""

    def __init__(self, F, origins, label, daily, gains, scale=1.0, q=0.5):
        super().__init__(F, origins, q, label)
        self.daily, self.gains, self.scale = daily, gains, scale

    def reset(self, ep):
        super().reset(ep)
        rho = round(ep.budget / self.daily[ep.c], 4)
        self.s = _by(self.scale, rho)
        self.kp, self.ki, self.kd = _by(self.gains, rho)
        self.I, self.prev, self.a_prev = 0.0, 0.0, None

    def act(self, h, S, spends):
        ep = self.ep
        R_ = ep.budget - S
        if R_ <= 0:
            return 0.0
        i = 24 * ep.day + h - self.o0
        expected = float(self.F[ep.c, i, : 24 - h, self.k].sum()) * self.s
        base = 1.0 if expected <= 0 else min(1.0, R_ / expected)
        corr = 0.0
        if h > 0 and self.a_prev is not None:
            exp_prev = self.a_prev * float(self.F[ep.c, i - 1, 0, self.k]) * self.s
            e = (exp_prev - spends[-1]) / ep.budget
            self.I += e
            corr = self.kp * e + self.ki * self.I + self.kd * (e - self.prev)
            self.prev = e
        a = float(np.clip(base + corr, 0.0, 1.0))
        self.a_prev = a
        return a


panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
val_o, test_o = np.arange(C.TRAIN_END, C.VAL_END), test_origins()
Fv, Ft = seasonal_naive(Y, val_o), seasonal_naive(Y, test_o)
prof = Y[:, : C.TRAIN_END].reshape(len(campaigns), -1, 24).mean(axis=1)
prof = prof / np.where(prof.sum(1, keepdims=True) > 0, prof.sum(1, keepdims=True), 1.0)
prof = np.where(prof.sum(1, keepdims=True) > 0, prof, 1 / 24)

# 1. choose eta and the feedback gains on the validation days
PV = [DualGradient(f"DGflat {e}", daily, e) for e in ETAS] + [DualGradient(f"DGprof {e}", daily, e, prof) for e in ETAS]
PV += [FeedbackHour(Fv, val_o, f"A {g}", daily, g) for g in GRID]
PV += [FeedbackHour(Fv, val_o, f"B {g}", daily, g, SCALE) for g in GRID]
val = pd.DataFrame(simulate(store, daily, lambda: PV, range(C.MAX_CAMPAIGNS or len(campaigns)), VAL_DAYS, VAL_SEEDS))
val["policy"] = val.policy.str.replace(" (q=0.5)", "", regex=False)
vs = val.groupby(["rho", "policy"]).apply(lambda g: g.clicks.sum() / g.ref_clicks.sum(), include_groups=False)
eta_c = {"DGflat": {}, "DGprof": {}}
gains_c = {"A": {}, "B": {}}
chosen = {}
for rho, s in vs.groupby(level=0):
    s = s.droplevel(0)
    rk = round(float(rho), 4)
    for kind in eta_c:
        best = s[[f"{kind} {e}" for e in ETAS]].idxmax()
        eta_c[kind][rk] = float(best.split()[1])
        chosen[f"{kind}@{rho}"] = {"eta": eta_c[kind][rk], "val_ratio": float(s[best]), "at_edge": eta_c[kind][rk] in (ETAS[0], ETAS[-1])}
    for base in "AB":
        names = [f"{base} {g}" for g in GRID]
        best = s[names].idxmax()
        g = GRID[names.index(best)]
        gains_c[base][rk] = g
        chosen[f"FB2{base}@{rho}"] = {"gains": list(g), "val_ratio": float(s[best]), "val_ratio_no_feedback": float(s[f"{base} {(0, 0, 0)}"]),
                                      "at_edge": [bool(g[0] == KP[-1]), bool(g[1] == KI[-1]), bool(g[2] == KD[-1])]}
print("chosen on validation:", json.dumps(chosen, indent=1))

# 2. test days
fcs = {}
for path in sorted(glob.glob(f"{R}/forecasts/*.npz")):
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    n = os.path.basename(path)[:-4]
    if n.startswith(("TFT", "NHITS")):
        fcs[n] = z["F"]
PT = [DualGradient("DG flat", daily, eta_c["DGflat"]), DualGradient("DG prof", daily, eta_c["DGprof"], prof),
      FeedbackHour(Ft, test_o, "FB2 naive", daily, gains_c["A"]), FeedbackHour(Ft, test_o, "FB2 naive-scaled", daily, gains_c["B"], SCALE)]
PT += [FeedbackHour(F, test_o, f"FB2 {n}", daily, gains_c["A"]) for n, F in fcs.items()]
new = pd.DataFrame(simulate(store, daily, lambda: PT, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, TEST_SEEDS))
new = new[new.policy != "Hindsight constant"].copy()
new["policy"] = new.policy.str.replace(" (q=0.5)", "", regex=False)
new.to_parquet(f"{R}/baselines_episodes.parquet", index=False)
new["method"] = new.policy.map(lambda s: re.sub(r"_seed\d+", "", s))

# 3. paired comparisons
old = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
ext = pd.read_parquet(f"{R}/extended_episodes.parquet")
FAM = {"TFT": "TFT_seed", "NHITS": "NHITS_seed", "Naive": "naive_seed"}


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
ext = ext[ext.policy.str.startswith("naive x")].copy()
ext["method"] = ext.policy
cols = ["clicks", "spend", "ref_clicks"]
nf = {**old.groupby("method").policy.nunique().to_dict(), **new.groupby("method").policy.nunique().to_dict()}
T = pd.concat([d.groupby(["campaign", "rho", "method"])[cols].sum().reset_index() for d in (old, ext, new)])
T["w"] = T.method.map(nf).fillna(1)
for c in ("clicks", "spend"):
    T[c] = T[c] / T.w

rng = np.random.default_rng(0)
res = {"chosen": chosen, "pairs": [], "ratios": {}}
for rho in C.RHOS:
    t = T[T.rho == rho]
    tuned = f"naive x{SCALE[round(float(rho), 4)]}"
    cl = t.pivot_table(index="campaign", columns="method", values="clicks")
    sp = t.pivot_table(index="campaign", columns="method", values="spend")
    rf = t[t.method == "Even pacing"].set_index("campaign").ref_clicks.loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = rng.integers(0, n, (C.N_BOOTSTRAP, n))
    res["ratios"][str(rho)] = {("Naive tuned" if m == tuned else m): float(cl[m].sum() / rf.sum()) for m in cl.columns
                               if not m.startswith("naive x") or m == tuned}
    pairs = [("DG flat", "PID"), ("DG prof", "PID"), ("TFT", "DG flat"), ("TFT", "DG prof"), ("NHITS", "DG prof"),
             (tuned, "DG flat"), (tuned, "DG prof"), ("Naive", "DG prof"), ("DG prof", "DG flat"),
             ("FB2 naive", "Naive"), ("FB2 naive-scaled", tuned), ("FB2 TFT", "TFT"), ("FB2 NHITS", "NHITS"),
             ("FB2 TFT", "FB2 naive-scaled"), ("FB2 NHITS", "FB2 naive-scaled")]
    for a, b in pairs:
        ca, cb, sa, sb = cl[a].to_numpy(), cl[b].to_numpy(), sp[a].to_numpy(), sp[b].to_numpy()

        def stats(ii):
            Ca, Cb, Sa, Sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
            return 100 * (Ca - Cb) / rf[ii].sum(-1), 100 * (Sa / Sb - 1), 100 * ((Ca / Sa) / (Cb / Sb) - 1)

        pt, bs = stats(np.arange(n)), stats(idx)
        res["pairs"].append({"rho": float(rho), "a": "Naive tuned" if a == tuned else a, "b": "Naive tuned" if b == tuned else b,
                             "points": float(pt[0]), "ci": np.quantile(bs[0], [0.025, 0.975]).tolist(),
                             "spend_pct": float(pt[1]), "eff_pct": float(pt[2])})
json.dump(res, open(f"{R}/baselines.json", "w"), indent=1)
write_meta("baselines", etas=ETAS, grid={"kp": KP, "ki": KI, "kd": KD}, val_days=VAL_DAYS, test_days=TEST_DAYS, seeds=TEST_SEEDS)
pd.set_option("display.width", 220)
print(pd.DataFrame(res["pairs"]).assign(ci=lambda d: d.ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]")).round(2).to_string(index=False))
print(pd.DataFrame(res["ratios"]).round(4).to_string())
