"""Five neural predictors in pacing (PROTOCOL 10.21): TFT and NHITS (main study) plus PatchTST, LSTM and TiDE.

Reads the saved forecasts (results/forecasts/*.npz). Computes forecast accuracy (MASE as in Section 7), simulates the median-plan
throttle for the three new models once on the test days (10 replay seeds, same draws as the main run), and compares every neural model
with seasonal naive and with the scaling curve of the 25-scale run (results/extended_episodes.parquet), as 11_frontier.py does.
Writes neural_models.json."""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from pacing import config as C
from pacing import metrics as M
from pacing.data import load_work, p, spend_matrix
from pacing.env import ImpressionStore
from pacing.forecasters import STEPS, test_origins
from pacing.policies import ForecastThrottle
from pacing.provenance import write_meta
from pacing.simulate import simulate

R = p("results")
TEST_DAYS = list(range(C.VAL_END // 24, C.TEST_END // 24))
SEEDS = list(range(C.N_REPLAY_SEEDS))
NEURAL = ["TFT", "NHITS", "PatchTST", "LSTM", "TiDE"]
NEW = ["PatchTST", "LSTM", "TiDE"]
K = {q: list(C.QUANTILES).index(q) for q in C.QUANTILES}

panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
Y = spend_matrix(panel, campaigns)
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
test_o = test_origins()

fcs = {}
for path in sorted(glob.glob(f"{R}/forecasts/*.npz")):
    name = os.path.basename(path)[:-4]
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    fcs[name] = z["F"]
print("forecasts:", sorted(fcs))

# 1. forecast accuracy for every saved forecast
rows = []
for name, F in fcs.items():
    F = F.astype(np.float64)
    tgt = test_o[:, None] + STEPS[None, :]
    ok = tgt < C.N_HOURS
    ms = []
    for c in range(len(campaigns)):
        y = Y[c, np.minimum(tgt, C.N_HOURS - 1)][ok]
        f = F[c][ok]
        ms.append(M.mase(y, f[:, K[0.5]], M.seasonal_naive_scale(Y[c, :C.TRAIN_END])))
    rows.append({"forecast": name, "model": re.sub(r"_seed\d+$", "", name), "mase": float(np.mean(ms))})
acc = pd.DataFrame(rows).groupby("model").mase.agg(["mean", "std", "count"]).fillna(0)
mase = {m: {"mean": float(r["mean"]), "sd": float(r["std"]), "seeds": int(r["count"])} for m, r in acc.iterrows()}
print(json.dumps(mase, indent=1))

# 2. median-plan throttle for the new models
PT = [ForecastThrottle(F, test_o, 0.5, name) for name, F in fcs.items() if re.sub(r"_seed\d+$", "", name) in NEW]
new = pd.DataFrame(simulate(store, daily, lambda: PT, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, SEEDS))
new = new[new.policy != "Hindsight constant"].copy()
new["method"] = new.policy.str.replace(" (q=0.5)", "", regex=False).str.replace(r"_seed\d+$", "", regex=True)
new.to_parquet(f"{R}/neural_models_episodes.parquet", index=False)

# 3. comparisons
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
SC = sorted({m for m in T.method if m.startswith("naive x")}, key=lambda m: float(m.split("x")[1]))

rng = np.random.default_rng(0)
res = {"mase": mase, "ratios": {}, "pairs": []}
for rho in C.RHOS:
    t = T[T.rho == rho]
    cl = t.pivot_table(index="campaign", columns="method", values="clicks")
    sp = t.pivot_table(index="campaign", columns="method", values="spend")
    rf = t[t.method == "Even pacing"].set_index("campaign").ref_clicks.loc[cl.index].to_numpy(float)
    n = len(cl)
    idx = np.vstack([np.arange(n), rng.integers(0, n, (C.N_BOOTSTRAP, n))])
    rb = rf[idx].sum(1)
    Ck = np.stack([cl[m].to_numpy() for m in SC], 1)
    Sk = np.stack([sp[m].to_numpy() for m in SC], 1)
    ck, sk = Ck[idx].sum(1), Sk[idx].sum(1)
    res["ratios"][str(rho)] = {m: float(cl[m].sum() / rf.sum()) for m in NEURAL + ["Naive", "PID", "Even pacing"]}
    for m in NEURAL:
        cm, sm = cl[m].to_numpy()[idx].sum(1), sp[m].to_numpy()[idx].sum(1)
        cn, sn = cl["Naive"].to_numpy()[idx].sum(1), sp["Naive"].to_numpy()[idx].sum(1)
        gain = 100 * (cm - cn) / rb
        spend = 100 * (sm / sn - 1)
        eff = 100 * ((cm / sm) / (cn / sn) - 1)
        above = []
        for b in range(len(idx)):
            o = np.argsort(sk[b])
            inside = sk[b][o][0] <= sm[b] <= sk[b][o][-1]
            above.append(100 * (cm[b] - np.interp(sm[b], sk[b][o], ck[b][o])) / rb[b] if inside else np.nan)
        above = np.array(above)
        res["pairs"].append({
            "rho": float(rho), "model": m,
            "gain_vs_naive": float(gain[0]), "gain_ci": np.quantile(gain[1:], [0.025, 0.975]).tolist(),
            "spend_pct": float(spend[0]), "eff_pct": float(eff[0]),
            "above_curve": None if np.isnan(above[0]) else float(above[0]),
            "above_ci": None if np.isnan(above[1:]).all() else np.nanquantile(above[1:], [0.025, 0.975]).tolist(),
            "above_valid": float(1 - np.isnan(above[1:]).mean())})

d = pd.DataFrame(res["pairs"])
beats = {m: int(sum(r.gain_ci[0] > 0 for r in d[d.model == m].itertuples())) for m in NEURAL}
edge = {m: int(sum((r.above_ci is not None) and r.above_ci[0] > 0 for r in d[d.model == m].itertuples())) for m in NEURAL}
res["beats_naive_levels"], res["edge_levels"] = beats, edge
res["n_beat"] = int(sum(v >= 2 for v in beats.values()))
res["n_edge"] = int(sum(v >= 2 for v in edge.values()))
res["class_beats"] = res["n_beat"] >= 4
res["class_edge"] = res["n_edge"] >= 4
# descriptive: rank correlation of mean MASE with mean click gain over naive across all models with both
gains = d.groupby("model").gain_vs_naive.mean()
tab = pd.DataFrame({"mase": {m: mase[m]["mean"] for m in NEURAL}, "gain": gains.to_dict()}).dropna()
res["spearman_mase_gain"] = float(tab.mase.rank().corr(tab.gain.rank()))
json.dump(res, open(f"{R}/neural_models.json", "w"), indent=1)
write_meta("neural_models", models=NEURAL, seeds=SEEDS, test_days=TEST_DAYS)
pd.set_option("display.width", 220)
print(d.assign(gain_ci=d.gain_ci.map(lambda x: f"[{x[0]:.2f},{x[1]:.2f}]"),
               above_ci=d.above_ci.map(lambda x: "-" if x is None else f"[{x[0]:.2f},{x[1]:.2f}]")).round(2).to_string(index=False))
print("beats naive at >=2 levels:", beats, "-> n =", res["n_beat"], "class_beats", res["class_beats"])
print("edge at >=2 levels:", edge, "-> n =", res["n_edge"], "class_edge", res["class_edge"])
print("spearman(MASE, gain) over the neural models:", res["spearman_mase_gain"])
print(pd.DataFrame(res["ratios"]).round(4).to_string())
