"""Aggregate test-episode results, bootstrap CIs, paired tests, and LaTeX tables (PROTOCOL 7, 8).

Policies built from several seeds of the same forecaster (e.g. TFT_seed0..2) are pooled into one
row; the seed-to-seed spread is reported separately.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pacing import config as C
from pacing.data import p
from pacing.provenance import write_meta

R = p("results")
rows = pd.read_parquet(f"{R}/pacing_test_episodes.parquet")
rows["group"] = rows.policy.str.replace(r"_seed\d+", "", regex=True)
rows["seed_model"] = rows.policy.str.extract(r"_seed(\d+)", expand=False).fillna("-")
rows["early"] = rows.depletion_hour < C.EARLY_DEPLETION_HOUR
rows["bur"] = rows.spend / rows.budget
rows["binding"] = rows.demand > rows.budget
rng = np.random.default_rng(0)


def ratio(g, num, den):
    d = g[den].sum()
    return g[num].sum() / d if d > 0 else np.nan


def per_campaign(g):
    return g.groupby("campaign")[["clicks", "ref_clicks", "convs", "ref_convs"]].sum()


def boot_ci(pc, num="clicks", den="ref_clicks"):
    a, b = pc[num].to_numpy(), pc[den].to_numpy()
    idx = rng.integers(0, len(a), size=(C.N_BOOTSTRAP, len(a)))
    s = a[idx].sum(1) / b[idx].sum(1)
    return np.percentile(s, [2.5, 97.5])


summary = []
for (rho, group), g in rows.groupby(["rho", "group"]):
    pc = per_campaign(g)
    lo, hi = boot_ci(pc)
    seed_ratios = g.groupby("seed_model").apply(lambda x: ratio(x, "clicks", "ref_clicks"), include_groups=False)
    summary.append({
        "rho": rho, "policy": group,
        "click_ratio": ratio(g, "clicks", "ref_clicks"), "click_ci_lo": lo, "click_ci_hi": hi,
        "click_ratio_seed_sd": float(seed_ratios.std(ddof=1)) if len(seed_ratios) > 1 else 0.0,
        "conv_ratio": ratio(g, "convs", "ref_convs"),
        "click_ratio_binding": ratio(g[g.binding], "clicks", "ref_clicks"),
        "bur": g.bur.mean(), "early_depletion": g.early.mean(),
        "pvi": g.pvi.mean(), "profile_dev": g.profile_dev.mean(),
        "binding_share": g.binding.mean(), "n_episode_runs": len(g),
    })
S = pd.DataFrame(summary)
S.to_csv(f"{R}/pacing_summary.csv", index=False)

# ---------------------------------------------------------------- pre-specified paired comparisons
# every forecast-based policy vs. Even pacing and vs. tuned PID; and forecasters against each other
# at the same hedging quantile. Holm correction over all comparisons.
def pc_ratio(g):
    pc = per_campaign(g)
    pc = pc[pc.ref_clicks > 0]
    return pc.clicks / pc.ref_clicks


comps = []
fc_groups = sorted(g for g in S.policy.unique() if "(q=" in g)
for rho in C.RHOS:
    sub = rows[rows.rho == rho]
    by = {k: pc_ratio(v) for k, v in sub.groupby("group")}
    pairs = [(f, b) for f in fc_groups for b in ("Even pacing", "PID (tuned)")]
    for q in C.HEDGE_QUANTILES:
        fs = [f for f in fc_groups if f.endswith(f"(q={q})")]
        pairs += [(a, b) for i, a in enumerate(fs) for b in fs[i + 1:]]
    for a, b in pairs:
        x, y = by[a].align(by[b], join="inner")
        diff = x - y
        stat = wilcoxon(diff) if np.any(diff != 0) else None
        comps.append({"rho": rho, "a": a, "b": b, "median_diff": float(diff.median()),
                      "mean_diff": float(diff.mean()), "n_campaigns": int(len(diff)),
                      "p": float(stat.pvalue) if stat else 1.0})
K = pd.DataFrame(comps).sort_values("p").reset_index(drop=True)
m = len(K)
K["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * K.p))
K.to_csv(f"{R}/pacing_comparisons.csv", index=False)

# ---------------------------------------------------------------- LaTeX tables
os.makedirs(f"{R}/tables", exist_ok=True)


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def pacing_table(rho):
    t = S[S.rho == rho].sort_values("click_ratio", ascending=False)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Policy & Clicks/ref. [95\% CI] & Conv./ref. & BUR & Early depl. & PVI & Profile dev. \\",
             r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.policy} & {fmt(r.click_ratio)} [{fmt(r.click_ci_lo)}, {fmt(r.click_ci_hi)}] & "
                     f"{fmt(r.conv_ratio)} & {fmt(r.bur)} & {fmt(r.early_depletion)} & {fmt(r.pvi)} & "
                     f"{fmt(r.profile_dev)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


for rho in C.RHOS:
    with open(f"{R}/tables/pacing_rho{rho}.tex", "w") as f:
        f.write(pacing_table(rho))

fm = json.load(open(f"{R}/forecast_metrics.json"))
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Forecaster & MASE & sPL$_{0.5}$ & sPL$_{0.9}$ & Cov.\ 80\% & Crossing \\", r"\midrule"]
for model, v in fm.items():
    mu, sd = v["mean"], v["std_over_seeds"]
    pm = lambda k, d=3: fmt(mu[k], d) + (f" $\\pm$ {fmt(sd[k], d)}" if v["n_seeds"] > 1 else "")
    lines.append(f"{model} & {pm('mase')} & {pm('spl_0.5')} & {pm('spl_0.9')} & {pm('coverage80')} & {pm('crossing')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(f"{R}/tables/forecast.tex", "w") as f:
    f.write("\n".join(lines))

write_meta("pacing_summary", n_comparisons=m)
pd.set_option("display.width", 200)
print(S.round(3).to_string())
print(K.head(30).round(4).to_string())
