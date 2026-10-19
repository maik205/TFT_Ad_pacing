"""Addition to PROTOCOL 10.20: dual-gradient pacer whose hourly gap is normalised by B/24 (a fixed scale) for both targets.

The pre-registered profile variant divides the gap by that hour's own target, which is tiny in low-traffic hours and then
makes mu jump and the pacer under-spend in the busy hours that follow. Here mu += eta * (spend - target) / (B / 24).
eta is chosen per budget level on the validation days from the same grid (3 replay seeds); everything is simulated once on
the test days with the same draws. Reads extended_episodes.parquet and pacing_test_episodes.parquet for the comparison
methods. Writes dual_gradient_fixed.json."""

import json

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p, spend_matrix
from pacing.env import ImpressionStore
from pacing.provenance import write_meta
from pacing.simulate import simulate

R = p("results")
VAL_DAYS = list(range(C.TRAIN_END // 24, C.VAL_END // 24))
TEST_DAYS = list(range(C.VAL_END // 24, C.TEST_END // 24))
VAL_SEEDS, TEST_SEEDS = [0, 1, 2], list(range(C.N_REPLAY_SEEDS))
ETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
SCALE = {round(float(r), 4): v["scale"] for r, v in json.load(open(f"{R}/extended.json"))["chosen"].items()}


class DG:
    def __init__(self, name, daily, eta, profile=None):
        self.name, self.daily, self.eta, self.profile = name, daily, eta, profile

    def reset(self, ep):
        rho = round(ep.budget / self.daily[ep.c], 4)
        self.e = self.eta[rho] if isinstance(self.eta, dict) else self.eta
        a0 = min(1.0, ep.budget / self.daily[ep.c])
        self.mu = 1.0 / a0 - 1.0
        share = np.full(24, 1 / 24) if self.profile is None else self.profile[ep.c]
        self.tgt, self.B, self.unit = ep.budget * share, ep.budget, ep.budget / 24

    def act(self, h, S, spends):
        if h > 0:
            self.mu = max(0.0, self.mu + self.e * (spends[-1] - self.tgt[h - 1]) / self.unit)
        return 0.0 if self.B - S <= 0 else min(1.0, 1.0 / (1.0 + self.mu))


panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
prof = Y[:, : C.TRAIN_END].reshape(len(campaigns), -1, 24).mean(axis=1)
prof = prof / np.where(prof.sum(1, keepdims=True) > 0, prof.sum(1, keepdims=True), 1.0)
prof = np.where(prof.sum(1, keepdims=True) > 0, prof, 1 / 24)

PV = [DG(f"DGprofN {e}", daily, e, prof) for e in ETAS] + [DG(f"DGflatN {e}", daily, e) for e in ETAS]
val = pd.DataFrame(simulate(store, daily, lambda: PV, range(C.MAX_CAMPAIGNS or len(campaigns)), VAL_DAYS, VAL_SEEDS))
vs = val.groupby(["rho", "policy"]).apply(lambda g: g.clicks.sum() / g.ref_clicks.sum(), include_groups=False)
eta_p, eta_f, chosen = {}, {}, {}
for rho, s in vs.groupby(level=0):
    s = s.droplevel(0)
    rk = round(float(rho), 4)
    eta_p[rk] = float(s[[f"DGprofN {e}" for e in ETAS]].idxmax().split()[1])
    eta_f[rk] = float(s[[f"DGflatN {e}" for e in ETAS]].idxmax().split()[1])
    chosen[str(rho)] = {"eta_prof": eta_p[rk], "eta_flat": eta_f[rk]}
print("chosen:", chosen)
PT = [DG("DG prof N", daily, eta_p, prof), DG("DG flat N", daily, eta_f)]
new = pd.DataFrame(simulate(store, daily, lambda: PT, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, TEST_SEEDS))
new = new[new.policy != "Hindsight constant"].copy()
new["method"] = new.policy

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
nf = {**old.groupby("method").policy.nunique().to_dict()}
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
    for a, b in (("DG prof N", "PID"), ("DG flat N", "PID"), ("DG prof N", "DG flat N"), ("TFT", "DG prof N"), ("TFT", "DG flat N"),
                 (tuned, "DG prof N"), (tuned, "DG flat N"), ("Naive", "DG prof N"), ("PID", "DG prof N")):
        ca, cb, sa, sb = cl[a].to_numpy(), cl[b].to_numpy(), sp[a].to_numpy(), sp[b].to_numpy()

        def stats(ii):
            Ca, Cb, Sa, Sb = ca[ii].sum(-1), cb[ii].sum(-1), sa[ii].sum(-1), sb[ii].sum(-1)
            return 100 * (Ca - Cb) / rf[ii].sum(-1), 100 * (Sa / Sb - 1), 100 * ((Ca / Sa) / (Cb / Sb) - 1)

        pt, bs = stats(np.arange(n)), stats(idx)
        res["pairs"].append({"rho": float(rho), "a": "Naive tuned" if a == tuned else a, "b": "Naive tuned" if b == tuned else b,
                             "points": float(pt[0]), "ci": np.quantile(bs[0], [0.025, 0.975]).tolist(),
                             "spend_pct": float(pt[1]), "eff_pct": float(pt[2])})
json.dump(res, open(f"{R}/dual_gradient_fixed.json", "w"), indent=1)
write_meta("dual_gradient_fixed", etas=ETAS, val_days=VAL_DAYS, test_days=TEST_DAYS, seeds=TEST_SEEDS)
pd.set_option("display.width", 200)
print(pd.DataFrame(res["pairs"]).assign(ci=lambda d: d.ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]")).round(2).to_string(index=False))
print(pd.DataFrame(res["ratios"]).round(4).to_string())
