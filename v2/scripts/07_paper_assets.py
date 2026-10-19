"""Figures and LaTeX number macros for the paper, generated only from results files (PROTOCOL 8).

The paper cites numbers as \\res{key}; every key is defined in results/tables/numbers.tex.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pacing import config as C
from pacing import metrics as M
from pacing.data import load_work, p, spend_matrix
from pacing.forecasters import STEPS
from pacing.provenance import write_meta

R = p("results")
os.makedirs(f"{R}/tables", exist_ok=True)
os.makedirs(f"{R}/figures", exist_ok=True)

S = pd.read_csv(f"{R}/pacing_summary.csv")
K = pd.read_csv(f"{R}/pacing_comparisons.csv")
FM = json.load(open(f"{R}/forecast_metrics.json"))
DS = json.load(open(f"{R}/data_summary.json"))
PID = json.load(open(f"{R}/pid_tuned.json"))
G = json.load(open(f"{R}/pacing_groups.json"))  # exploratory group analysis (08_groups.py, PROTOCOL 10.17)

FORECASTERS = ["naive", "lightgbm", "NHITS", "TFT"]
LABEL = {"naive": "Seasonal naive", "lightgbm": "LightGBM", "NHITS": "NHITS", "TFT": "TFT"}


def slug(policy):
    fixed = {"Even pacing": "even", "PID (tuned)": "pid", "No pacing": "nopacing", "Hindsight constant": "hindsight"}
    if policy in fixed:
        return fixed[policy]
    m = re.fullmatch(r"(\w+) \(q=([\d.]+)\)", policy)
    return f"{m.group(1).lower()}-q{m.group(2)}"


S["slug"] = S.policy.map(slug)
V = {}  # key -> formatted string


def ratio(x):
    return f"{x:.3f}"


def pts(x):
    return f"{100 * x:.1f}"


def pct(x):
    return f"{100 * x:.1f}"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def get(sl, rho, col):
    return float(S[(S.slug == sl) & (S.rho == rho)][col].iloc[0])


# ---------------------------------------------------------------- data
V["data-impressions"] = thousands(DS["n_impressions"])
V["data-campaigns-total"] = str(DS["n_campaigns_total"])
V["data-campaigns"] = str(DS["n_campaigns_selected"])
V["data-spend-share"] = pct(DS["selected_share_of_total_spend"])
V["data-eval-impressions"] = thousands(DS["n_eval_impressions"])
V["data-test-episodes"] = thousands(DS["test_episodes"])
V["n-replay-seeds"] = str(C.N_REPLAY_SEEDS)
V["n-nn-seeds"] = str(C.N_NN_SEEDS)
V["n-bootstrap"] = thousands(C.N_BOOTSTRAP)

# ---------------------------------------------------------------- forecasting
for m, v in FM.items():
    k = m.lower()
    V[f"fc-{k}-mase"] = f"{v['mean']['mase']:.3f}"
    V[f"fc-{k}-mase-sd"] = f"{v['std_over_seeds']['mase']:.3f}"
    V[f"fc-{k}-medmase"] = f"{v['median_mase_over_campaigns']:.3f}"
    for q in ("0.1", "0.5", "0.9"):
        V[f"fc-{k}-spl{q}"] = f"{v['mean'][f'spl_{q}']:.3f}"
    V[f"fc-{k}-cov80"] = pct(v["mean"]["coverage80"])
    V[f"fc-{k}-cross"] = f"{100 * v['mean']['crossing']:.2f}"

# ---------------------------------------------------------------- pacing, per policy and budget level
for _, r in S.iterrows():
    base = f"{r.slug}-rho{r.rho}"
    V[f"{base}-click"] = ratio(r.click_ratio)
    V[f"{base}-click-lo"] = ratio(r.click_ci_lo)
    V[f"{base}-click-hi"] = ratio(r.click_ci_hi)
    V[f"{base}-conv"] = ratio(r.conv_ratio)
    V[f"{base}-bur"] = pct(r.bur)
    V[f"{base}-early"] = pct(r.early_depletion)
    V[f"{base}-pvi"] = f"{r.pvi:.3f}"
for rho in C.RHOS:
    V[f"binding-rho{rho}"] = pct(get("even", rho, "binding_share"))


def span(name, values, fmt):
    V[f"{name}-min"], V[f"{name}-max"] = fmt(min(values)), fmt(max(values))


nn = ("tft", "nhits")
span("gain-even", [get(f"{m}-q0.5", r, "click_ratio") - get("even", r, "click_ratio") for m in nn for r in C.RHOS], pts)
span("gain-pid", [get(f"{m}-q0.5", r, "click_ratio") - get("pid", r, "click_ratio") for m in nn for r in C.RHOS], pts)
span("hedge-cost", [get(f"{m.lower()}-q0.5", r, "click_ratio") - get(f"{m.lower()}-q0.9", r, "click_ratio")
                    for m in FORECASTERS for r in C.RHOS], pts)
span("hedge-bur", [get(f"{m.lower()}-q0.9", r, "bur") for m in FORECASTERS for r in C.RHOS], pct)
span("hedge-early", [get(f"{m.lower()}-q0.9", r, "early_depletion") for m in FORECASTERS for r in C.RHOS], pct)
span("tft-nhits-diff", [abs(get("tft-q0.5", r, "click_ratio") - get("nhits-q0.5", r, "click_ratio")) for r in C.RHOS], pts)
span("nn-lgb", [get(f"{m}-q0.5", r, "click_ratio") - get("lightgbm-q0.5", r, "click_ratio") for m in nn for r in C.RHOS], pts)
span("nn-naive", [get(f"{m}-q0.5", r, "click_ratio") - get("naive-q0.5", r, "click_ratio") for m in nn for r in C.RHOS], pts)
span("nopacing-conv", [get("nopacing", r, "conv_ratio") - 1 for r in C.RHOS], pts)
span("nopacing-early", [get("nopacing", r, "early_depletion") for r in C.RHOS], pct)
span("pid-pvi", [get("pid", r, "pvi") for r in C.RHOS], lambda x: f"{x:.2f}")
span("pid-early", [get("pid", r, "early_depletion") for r in C.RHOS], pct)
span("fc-pvi", [get(f"{m}-q0.5", r, "pvi") for m in nn for r in C.RHOS], lambda x: f"{x:.2f}")
span("fc-early", [get(f"{m}-q0.5", r, "early_depletion") for m in nn for r in C.RHOS], pct)
def sig_count(pairs, name):
    sub = K[[(a, b) in pairs for a, b in zip(K.a, K.b)]]
    V[f"{name}-sig"] = str(int((sub.p_holm < 0.05).sum()))
    V[f"{name}-neg"] = str(int((sub.mean_diff < 0).sum()))
    V[f"{name}-total"] = str(len(sub))
    V[f"{name}-pmax"] = f"{sub.p_holm.max():.2g}"


nn_med = [f"{m} (q=0.5)" for m in ("TFT", "NHITS")]
sig_count({(a, b) for a in nn_med for b in ("Even pacing", "PID (tuned)")}, "cmp-nn-baselines")
sig_count({(f"{m} (q=0.9)", b) for m in FORECASTERS for b in ("Even pacing", "PID (tuned)")}, "cmp-q09-baselines")
sig_count({("NHITS (q=0.5)", "TFT (q=0.5)"), ("TFT (q=0.5)", "NHITS (q=0.5)")}, "cmp-tft-nhits")
sig_count({(a, "lightgbm (q=0.5)") for a in nn_med}, "cmp-nn-lgb")
V["n-comparisons"] = str(len(K))
V["n-significant"] = str(int((K.p_holm < 0.05).sum()))
V["n-campaigns-paired"] = str(int(K.n_campaigns.min()))

# Revision round 1 (PROTOCOL 10.15): macros that only re-express existing results files or config
# constants. They add no analysis beyond descriptive aggregation of saved per-campaign metrics.
def pts2(x):
    return f"{100 * x:.2f}"


# design constants: read from config.py, never typed in the prose
V["cfg-n-hours"] = str(C.N_HOURS)
V["cfg-use-days"] = str(C.TEST_END // 24)
V["cfg-discard-hours"] = str(C.N_HOURS - C.TEST_END)
V["cfg-train-days"] = str(C.TRAIN_END // 24)
V["cfg-val-days"] = str((C.VAL_END - C.TRAIN_END) // 24)
V["cfg-test-days"] = str((C.TEST_END - C.VAL_END) // 24)
V["cfg-lookback"] = str(C.LOOKBACK)
V["cfg-min-active"] = str(int(round(100 * C.MIN_ACTIVE_FRAC)))
V["cfg-early-hour"] = str(C.EARLY_DEPLETION_HOUR)

# how much of the gain over even pacing a plain seasonal profile already delivers
span("gain-naive-even", [get("naive-q0.5", r, "click_ratio") - get("even", r, "click_ratio") for r in C.RHOS], pts)

# training-seed variability of the pacing outcome (clicks / reference, in points)
V["nhits-seed-sd-max"] = pts2(max(get("nhits-q0.5", r, "click_ratio_seed_sd") for r in C.RHOS))
V["tft-seed-sd-max"] = pts2(max(get("tft-q0.5", r, "click_ratio_seed_sd") for r in C.RHOS))
V["hedge-seed-sd-max"] = pts2(max(get(f"{m}-q0.9", r, "click_ratio_seed_sd") for m in nn for r in C.RHOS))
V["nhits-seed-sd-min"] = pts2(min(get("nhits-q0.5", r, "click_ratio_seed_sd") for r in C.RHOS))

# monotone decline of clicks with the hedging quantile
mono = sum(all(get(f"{m.lower()}-q{a}", r, "click_ratio") > get(f"{m.lower()}-q{b}", r, "click_ratio")
               for a, b in (("0.5", "0.7"), ("0.7", "0.9"))) for m in FORECASTERS for r in C.RHOS)
V["hedge-mono"] = str(mono)
V["hedge-mono-total"] = str(len(FORECASTERS) * len(C.RHOS))

# per-campaign view of forecast accuracy (descriptive; from forecast_metrics_per_campaign.parquet)
E = pd.read_parquet(f"{R}/forecast_metrics_per_campaign.parquet")
per_campaign_mase = E.groupby(["model", "campaign"]).mase.mean().unstack(0)
for a, b in (("TFT", "lightgbm"), ("NHITS", "lightgbm"), ("TFT", "naive"), ("NHITS", "naive"), ("TFT", "NHITS")):
    V[f"win-{a.lower()}-{b.lower()}"] = pct((per_campaign_mase[a] < per_campaign_mase[b]).mean())

for rho, v in PID.items():
    kp, ki, kd = v["gains"]
    V[f"pid-gains-rho{float(rho)}"] = f"$K_p{{=}}{kp:g}, K_i{{=}}{ki:g}, K_d{{=}}{kd:g}$"

# ---------------------------------------------------------------- exploratory groups (PROTOCOL 10.17)
def grp(group, value, rho):
    return next(r for r in G["rows"] if r["group"] == group and r["value"] == value and r["rho"] == rho)


def signed(x, digits=1):
    return f"{100 * x:.{digits}f}".replace("-", "$-$")


GAIN = "diff_TFT_minus_Seasonal naive"
CI = "ci_TFT_minus_Seasonal naive"
cells = [r for r in G["rows"] if r["group"] != "all"]
V["grp-per-group"] = str(G["campaigns"] // 3)
V["grp-spearman"] = f"{G['spearman_size_vol']:.2f}".replace("-", "$-$")
V["grp-n-cells"] = str(len(cells))
V["grp-gain-min"] = signed(min(r[GAIN] for r in cells))
V["grp-gain-max"] = signed(max(r[GAIN] for r in cells))
V["grp-ci-lo-min"] = signed(min(r[CI][0] for r in cells))
V["grp-ci-above-zero"] = str(sum(r[CI][0] > 0 for r in cells))
for rho in C.RHOS:
    V[f"grp-all-rho{rho}"] = signed(grp("all", "all", rho)[GAIN])
    V[f"grp-vol-high-rho{rho}"] = signed(grp("volatility", "high", rho)[GAIN])
    V[f"grp-vol-low-rho{rho}"] = signed(grp("volatility", "low", rho)[GAIN])
    sizes = [grp("size", v, rho)[GAIN] for v in ("low", "mid", "high")]
    V[f"grp-size-min-rho{rho}"], V[f"grp-size-max-rho{rho}"] = signed(min(sizes)), signed(max(sizes))
TN = "diff_TFT_minus_NHITS"
CTN = "ci_TFT_minus_NHITS"
V["grp-tft-nhits-max"] = signed(max(abs(r[TN]) for r in G["rows"]), 2)
V["grp-tft-nhits-excl"] = str(sum((r[CTN][0] > 0 or r[CTN][1] < 0) for r in G["rows"]))
V["grp-tft-nhits-total"] = str(len(G["rows"]))
V["grp-tft-nhits-pos"] = str(sum(r[CTN][0] > 0 for r in G["rows"]))
V["grp-tft-nhits-neg"] = str(sum(r[CTN][1] < 0 for r in G["rows"]))
V["grp-tft-nhits-hv-min"] = signed(min(grp("volatility", "high", rho)[TN] for rho in C.RHOS), 2)
V["grp-tft-nhits-hv-max"] = signed(max(grp("volatility", "high", rho)[TN] for rho in C.RHOS), 2)

labels = [(("all", "all"), "All campaigns"), (("size", "low"), "Small"), (("size", "mid"), "Medium-size"),
          (("size", "high"), "Large"), (("volatility", "low"), "Least volatile"),
          (("volatility", "mid"), "Middle volatility"), (("volatility", "high"), "Most volatile")]
rows = [r"\begin{tabular}{llccc}", r"\toprule",
        r"Group & Campaigns & $\rho{=}0.3$ & $\rho{=}0.5$ & $\rho{=}0.7$ \\", r"\midrule"]
for key, lab in labels:
    if key in (("size", "low"), ("volatility", "low")):
        rows.append(r"\addlinespace[2pt]")
    cs = []
    for rho in C.RHOS:
        r = grp(key[0], key[1], rho)
        cs.append(f"{signed(r[GAIN])} [{signed(r[CI][0])}, {signed(r[CI][1])}]")
    rows.append(f"{lab} & {grp(key[0], key[1], 0.3)['campaigns']} & " + " & ".join(cs) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
with open(f"{R}/tables/groups.tex", "w") as f:
    f.write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- post-hoc analyses (PROTOCOL 10.18)
SE = json.load(open(f"{R}/spend_efficiency.json"))   # 09_spend_efficiency.py
SN = json.load(open(f"{R}/scaled_naive.json"))       # 10_scaled_naive.py (13 scales; only the 7-day-mean pair is used)
EX = json.load(open(f"{R}/extended.json"))           # 15_extended.py (25 scales, swap, oracle; PROTOCOL 10.19)
BL = json.load(open(f"{R}/baselines.json"))          # 16_baselines.py (dual gradient, feedback on the hourly re-plan; PROTOCOL 10.20)
DF = json.load(open(f"{R}/dual_gradient_fixed.json"))  # 19_dual_gradient_fixed.py
IP = json.load(open(f"{R}/ipinyou.json"))            # 18_ipinyou.py
IPS = json.load(open(f"{R}/ipinyou_summary.json"))   # 17_ipinyou_build.py
FR = json.load(open(f"{R}/frontier.json"))           # 11_frontier.py
FB = json.load(open(f"{R}/feedback.json"))           # 12_feedback.py
HC = json.load(open(f"{R}/hedge_coverage.json"))     # 13_hedge_coverage.py
CH = json.load(open(f"{R}/calibrated_hedge.json"))   # 14_calibrated_hedge.py


def sg(x, d=1):
    """Format a value that is already in points or percent."""
    return f"{x:.{d}f}".replace("-", "$-$")


def ci2(c, d=1):
    return f"[{sg(c[0], d)}, {sg(c[1], d)}]"


def rng2(name, values, d=1):
    V[f"{name}-min"], V[f"{name}-max"] = sg(min(values), d), sg(max(values), d)


V["cmp-n-campaigns"] = str(int(K.n_campaigns.max()))
V["ph-test-days-causal"] = str((C.TEST_END - C.VAL_END) // 24 - 1)

# spend versus efficiency, median plans
def sep(rho, a, b, q="0.5"):
    return next(r for r in SE if "a" in r and r["q"] == q and r["rho"] == rho and r["a"] == a and r["b"] == b)


for a in ("TFT", "NHITS"):
    rows_ = [sep(r, a, "Seasonal naive") for r in C.RHOS]
    k = a.lower()
    rng2(f"se-{k}-naive-clicks", [r["clicks_pct"] for r in rows_])
    rng2(f"se-{k}-naive-spend", [r["spend_pct"] for r in rows_])
    rng2(f"se-{k}-naive-eff", [r["eff_pct"] for r in rows_], 2)
V["se-tft-naive-eff-ci0"] = str(sum(sep(r, "TFT", "Seasonal naive")["eff_ci"][0] <= 0 <= sep(r, "TFT", "Seasonal naive")["eff_ci"][1] for r in C.RHOS))
rng2("se-tft-pid-spend", [sep(r, "TFT", "PID")["spend_pct"] for r in C.RHOS])
rng2("se-tft-pid-eff", [sep(r, "TFT", "PID")["eff_pct"] for r in C.RHOS])
V["se-tft-pid-eff-ci-pos"] = str(sum(sep(r, "TFT", "PID")["eff_ci"][0] > 0 for r in C.RHOS))
rng2("se-tft-even-spend", [sep(r, "TFT", "Even pacing")["spend_pct"] for r in C.RHOS])
rng2("se-tft-even-eff", [sep(r, "TFT", "Even pacing")["eff_pct"] for r in C.RHOS])
rng2("se-q09-spend", [sep(r, "TFT", "Even pacing", "0.9")["spend_pct"] for r in C.RHOS])
rng2("se-q09-eff", [sep(r, "TFT", "Even pacing", "0.9")["eff_pct"] for r in C.RHOS])

# scaling the seasonal-naive total
def snp(rho, a, b):
    """Pair from the 25-scale run when it has it, else from the 13-scale run (7-day mean; scaled naive vs PID, whose scale is the same)."""
    r = next((r for r in EX["pairs"] if r["rho"] == rho and r["a"] == a and r["b"] == b), None)
    if r is not None:
        return {**r, "points_vs_hindsight": r["points"]}
    return next(r for r in SN["pairs"] if r["rho"] == rho and r["a"] == a and r["b"] == b)


for rho in C.RHOS:
    s = EX["chosen"][str(rho)]
    V[f"sn-scale-rho{rho}"] = f"{s['scale']:g}"
    tuned, tft = snp(rho, "Naive tuned", "Naive"), snp(rho, "TFT", "Naive")
    gap = snp(rho, "TFT", "Naive tuned")
    V[f"sn-gain-rho{rho}"] = sg(tuned["points_vs_hindsight"], 2)
    V[f"sn-gain-ci-rho{rho}"] = ci2(tuned["ci"], 2)
    V[f"sn-tft-rho{rho}"] = sg(tft["points_vs_hindsight"], 2)
    V[f"sn-share-rho{rho}"] = f"{100 * tuned['points_vs_hindsight'] / tft['points_vs_hindsight']:.0f}"
    V[f"sn-gap-rho{rho}"] = sg(gap["points_vs_hindsight"], 2)
    V[f"sn-gap-ci-rho{rho}"] = ci2(gap["ci"], 2)
    V[f"sn-spend-rho{rho}"] = sg(tuned["spend_pct"])
    V[f"sn-eff-rho{rho}"] = sg(tuned["eff_pct"], 2)
rng2("sn-mean7", [snp(r, "naive mean7", "Naive")["points_vs_hindsight"] for r in C.RHOS], 2)
V["sn-edge"] = ", ".join(f"{rho}" for rho in C.RHOS if EX["chosen"][str(rho)]["at_edge"]) or "none"
V["sn-pid-medium"] = sg(snp(0.5, "Naive tuned", "PID")["points_vs_hindsight"], 2)
V["sn-pid-medium-ci"] = ci2(snp(0.5, "Naive tuned", "PID")["ci"], 2)

# spend-matched comparison
def frr(rho, method):
    return next(r for r in FR if r.get("method") == method and r["rho"] == rho)


for m, k in (("TFT", "tft"), ("NHITS", "nhits"), ("LightGBM", "lgb"), ("PID", "pid")):
    for rho in C.RHOS:
        r = frr(rho, m)
        V[f"fr-{k}-rho{rho}"] = sg(r["points_above_curve"], 2)
        V[f"fr-{k}-ci-rho{rho}"] = ci2(r["ci"], 2)
rng2("fr-tft", [frr(r, "TFT")["points_above_curve"] for r in C.RHOS], 2)
rng2("fr-nhits", [frr(r, "NHITS")["points_above_curve"] for r in C.RHOS], 2)
V["np-gap-rho0.7"] = pts(abs(get("pid", 0.7, "click_ratio") - get("nopacing", 0.7, "click_ratio")))
V["fr-tft-spend-tight"] =sg(100 * (frr(0.3, "TFT")["spend_share_vs_naive_x1"] - 1))
V["fr-valid-tight"] = f"{100 * frr(0.3, 'TFT')['boot_valid']:.0f}"

# forecast plus feedback
def fbp(rho, a, b):
    return next(r for r in FB["pairs"] if r["rho"] == rho and r["a"] == a and r["b"] == b)


for rho in C.RHOS:
    for a, b, k in (("FB TFT", "TFT", "tft"), ("FB NHITS", "NHITS", "nhits"), ("FB naive", "Naive", "naive"),
                    ("FB naive-scaled", "naive scaled", "scaled"), ("FB TFT", "FB naive-scaled", "tft-vs-scaled")):
        r = fbp(rho, a, b)
        V[f"fb-{k}-rho{rho}"] = sg(r["points_vs_hindsight"], 2)
        V[f"fb-{k}-ci-rho{rho}"] = ci2(r["ci"], 2)
    V[f"fb-kp-rho{rho}"] = f"{FB['chosen'][f'A@{rho}']['gains'][0]:g}"
V["fb-val-gain-loose"] = sg(100 * (FB["chosen"]["A@0.7"]["val_ratio"] - FB["chosen"]["A@0.7"]["val_ratio_no_feedback"]), 1)
V["fb-edge-rho"] = ", ".join(str(rho) for rho in C.RHOS if FB["chosen"][f"A@{rho}"]["at_edge"][0]) or "none"

# coverage of the summed hedge plans
fams = ["TFT", "NHITS", "LightGBM", "Seasonal naive"]
rng2("hc-cov90", [100 * HC[f]["q0.9"]["summed_plan_coverage"] for f in fams])
rng2("hc-cov90-nn", [100 * HC[f]["q0.9"]["summed_plan_coverage"] for f in ("TFT", "NHITS", "LightGBM")])
rng2("hc-cov70", [100 * HC[f]["q0.7"]["summed_plan_coverage"] for f in fams])
rng2("hc-fac-sum", [HC[f]["q0.9"]["median_plan_factor"] for f in fams], 2)
rng2("hc-fac-cal", [HC[f]["q0.9"]["calibrated_factor"] for f in fams], 2)
V["hc-hourly90-tft"] = sg(100 * HC["TFT"]["q0.9"]["hourly_coverage"])
V["hc-cov90-naive"] = sg(100 * HC["Seasonal naive"]["q0.9"]["summed_plan_coverage"])
V["hc-cov90-nhits"] = sg(100 * HC["NHITS"]["q0.9"]["summed_plan_coverage"])

# calibrated-total hedge (causal variant, days 26-29)
def chp(rho, fam_, q, contrast, variant="b", set_="days 26-29"):
    return next(r for r in CH["pairs"] if r["set"] == set_ and r["variant"] == variant and r["rho"] == rho
                and r["forecaster"] == fam_ and r["q"] == q and r["contrast"] == contrast)


for rho in C.RHOS:
    for q, k in ((0.9, "ch"), (0.7, "ch70")):
        g = chp(rho, "TFT", q, "calibrated - summed")
        loss = chp(rho, "TFT", q, "summed - median plan")["points"]
        res_ = chp(rho, "TFT", q, "calibrated - median plan")
        V[f"{k}-gain-rho{rho}"] = sg(g["points"], 2)
        V[f"{k}-gain-ci-rho{rho}"] = ci2(g["ci"], 2)
        V[f"{k}-loss-rho{rho}"] = sg(-loss, 1)
        V[f"{k}-rec-rho{rho}"] = f"{100 * g['points'] / -loss:.0f}"
        V[f"{k}-resid-rho{rho}"] = sg(res_["points"], 2)
        V[f"{k}-resid-ci-rho{rho}"] = ci2(res_["ci"], 2)
rng2("ch-naive", [chp(r, "Seasonal naive", 0.9, "calibrated - summed")["points"] for r in C.RHOS])
rng2("ch-xfit", [chp(r, "TFT", 0.9, "calibrated - summed", "a")["points"] for r in C.RHOS], 2)

# total/shape swap and oracle headroom (15_extended.py, PROTOCOL 10.19)
def exp_(rho, a, b):
    return next(r for r in EX["pairs"] if r["rho"] == rho and r["a"] == a and r["b"] == b)


for rho in C.RHOS:
    g = exp_(rho, "ST TFT-TFT", "ST naive-naive")
    tt = exp_(rho, "ST TFT-naive", "ST naive-naive")
    ss = exp_(rho, "ST naive-TFT", "ST naive-naive")
    V[f"ex-g-rho{rho}"] = sg(g["points"], 2)
    V[f"ex-t-rho{rho}"], V[f"ex-t-ci-rho{rho}"] = sg(tt["points"], 2), ci2(tt["ci"], 2)
    V[f"ex-s-rho{rho}"], V[f"ex-s-ci-rho{rho}"] = sg(ss["points"], 2), ci2(ss["ci"], 2)
    V[f"ex-tshare-rho{rho}"] = f"{100 * tt['points'] / g['points']:.0f}"
    rp = exp_(rho, "ST TFT-TFT", "TFT")
    V[f"ex-replan-rho{rho}"] = sg(-rp["points"], 2)
    V[f"ex-replan-ci-rho{rho}"] = ci2([-rp["ci"][1], -rp["ci"][0]], 2)
    hd = exp_(rho, "OR remaining", "TFT")
    V[f"ex-head-rho{rho}"], V[f"ex-head-ci-rho{rho}"] = sg(hd["points"], 2), ci2(hd["ci"], 2)
    V[f"ex-head-naive-rho{rho}"] = sg(exp_(rho, "OR remaining", "Naive")["points"], 2)
    V[f"ex-tot-naive-rho{rho}"] = sg(exp_(rho, "OR total-naive", "ST naive-naive")["points"], 2)
    V[f"ex-tot-tft-rho{rho}"] = sg(exp_(rho, "OR total-TFT", "ST TFT-TFT")["points"], 2)
    V[f"ex-or-replan-rho{rho}"] = sg(exp_(rho, "OR remaining", "OR total-TFT")["points"], 2)
rng2("ex-head", [exp_(r, "OR remaining", "TFT")["points"] for r in C.RHOS], 2)
rng2("ex-tot-naive", [exp_(r, "OR total-naive", "ST naive-naive")["points"] for r in C.RHOS], 2)
rng2("ex-tot-tft", [exp_(r, "OR total-TFT", "ST TFT-TFT")["points"] for r in C.RHOS], 2)
rng2("ex-replan", [-exp_(r, "ST TFT-TFT", "TFT")["points"] for r in C.RHOS], 2)
rng2("ex-tshare", [100 * exp_(r, "ST TFT-naive", "ST naive-naive")["points"] / exp_(r, "ST TFT-TFT", "ST naive-naive")["points"] for r in C.RHOS], 0)

# dual-gradient reactive pacer and feedback on the hourly re-plan (16_baselines.py, 19_dual_gradient_fixed.py; PROTOCOL 10.20)
def blp(src, rho, a, b):
    return next(r for r in src["pairs"] if r["rho"] == rho and r["a"] == a and r["b"] == b)


for rho in C.RHOS:
    for a, b, k in (("DG flat", "PID", "flat-pid"), ("DG prof", "PID", "prof-pid"), ("TFT", "DG flat", "tft-flat"),
                    ("Naive tuned", "DG flat", "tuned-flat"), ("FB2 TFT", "TFT", "fb2-tft"), ("FB2 NHITS", "NHITS", "fb2-nhits"),
                    ("FB2 naive", "Naive", "fb2-naive"), ("FB2 naive-scaled", "Naive tuned", "fb2-scaled"),
                    ("FB2 TFT", "FB2 naive-scaled", "fb2-tft-vs-scaled")):
        r = blp(BL, rho, a, b)
        V[f"{'' if k.startswith('fb2') else 'dg-'}{k}-rho{rho}"] = sg(r["points"], 2)
        V[f"{'' if k.startswith('fb2') else 'dg-'}{k}-ci-rho{rho}"] = ci2(r["ci"], 2)
    for a, b, k in (("DG prof N", "PID", "profn-pid"), ("TFT", "DG prof N", "tft-profn"), ("Naive tuned", "DG prof N", "tuned-profn")):
        r = blp(DF, rho, a, b)
        V[f"dg-{k}-rho{rho}"], V[f"dg-{k}-ci-rho{rho}"] = sg(r["points"], 2), ci2(r["ci"], 2)
    V[f"dg-eta-flat-rho{rho}"] = f"{BL['chosen'][f'DGflat@{rho}']['eta']:g}"
    V[f"fb2-kp-rho{rho}"] = f"{BL['chosen'][f'FB2A@{rho}']['gains'][0]:g}"
rng2("dg-flat-pid", [blp(BL, r, "DG flat", "PID")["points"] for r in C.RHOS])
rng2("dg-prof-pid", [blp(BL, r, "DG prof", "PID")["points"] for r in C.RHOS])
rng2("dg-profn-pid", [blp(DF, r, "DG prof N", "PID")["points"] for r in C.RHOS])
rng2("dg-tft-flat", [blp(BL, r, "TFT", "DG flat")["points"] for r in C.RHOS])
rng2("dg-tft-profn", [blp(DF, r, "TFT", "DG prof N")["points"] for r in C.RHOS])

# iPinYou replication (17_ipinyou_build.py, 18_ipinyou.py; PROTOCOL 10.20)
V["ipy-impressions"] = f"{IPS['rows'] / 1e6:.1f}"
V["ipy-advertisers"] = str(len(IPS["advertisers"]))
V["ipy-episodes"] = str(IP["episodes"])
V["ipy-consistent"] = str(IP["consistent_levels"])
V["ipy-verdict"] = IP["verdict"]
for rho in C.RHOS:
    L = IP["levels"][str(rho)]
    for k, name in (("none", "No pacing"), ("even", "Even pacing"), ("pid", "PID"), ("dgflat", "DG flat"), ("naive", "Naive"),
                    ("scaled", "Naive scaled"), ("oracle", "Oracle remaining")):
        V[f"ipy-{k}-rho{rho}"] = ratio(L[name]["click_ratio"])
    V[f"ipy-scale-rho{rho}"] = f"{IP['chosen'][str(rho)]['scale']:g}"
    for a, b, k in (("Naive scaled", "Naive", "scaled-naive"), ("Naive", "Even pacing", "naive-even"), ("Naive", "PID", "naive-pid"),
                    ("Oracle remaining", "Naive", "oracle-naive")):
        r = next(r for r in IP["pairs"] if r["rho"] == rho and r["a"] == a and r["b"] == b)
        V[f"ipy-{k}-rho{rho}"], V[f"ipy-{k}-ci-rho{rho}"] = sg(r["points"], 1), ci2(r["ci"], 1)

# table: where the click gain comes from (median plans)
dec = [("TFT", "Naive", "TFT $-$ seasonal naive"), ("Naive tuned", "Naive", "Scaled naive $-$ seasonal naive"),
       ("TFT", "Naive tuned", "TFT $-$ scaled naive")]
rows = [r"\begin{tabular}{llccc}", r"\toprule",
        r"Comparison & $\rho$ & Clicks (points) & Spend (\%) & Clicks per spend (\%) \\", r"\midrule"]
for a, b, lab in dec:
    for j, rho in enumerate(C.RHOS):
        r = snp(rho, a, b)
        rows.append(f"{lab if j == 0 else ''} & {rho} & {sg(r['points_vs_hindsight'], 2)} {ci2(r['ci'], 2)} & "
                    f"{sg(r['spend_pct'], 1)} & {sg(r['eff_pct'], 2)} " + r"\\")
    if a != dec[-1][0] or b != dec[-1][1]:
        rows.append(r"\addlinespace[2pt]")
rows += [r"\bottomrule", r"\end{tabular}"]
with open(f"{R}/tables/decomp.tex", "w") as f:
    f.write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- compute cost
times = {}
for path in glob.glob(f"{R}/forecasts/*.meta.json"):
    meta = json.load(open(path))
    model = re.sub(r"_seed\d+$", "", meta["name"].split("/")[-1])
    t = meta.get("wall_seconds_fit_and_predict") or meta.get("fit_seconds") or meta.get("total_seconds")
    times.setdefault(model, []).append(t)
for model, ts in times.items():
    V[f"time-{model.lower()}"] = f"{np.mean(ts):.0f}"

# ---------------------------------------------------------------- figure 1: hedging quantile
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42})
colors = {"naive": "#7f7f7f", "lightgbm": "#d95f02", "NHITS": "#1b9e77", "TFT": "#7570b3"}
markers = {"naive": "s", "lightgbm": "^", "NHITS": "o", "TFT": "D"}
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
qs = list(C.HEDGE_QUANTILES)
for ax, rho in zip(axes, C.RHOS):
    for j, m in enumerate(FORECASTERS):
        ys = [get(f"{m.lower()}-q{q}", rho, "click_ratio") for q in qs]
        lo = [get(f"{m.lower()}-q{q}", rho, "click_ci_lo") for q in qs]
        hi = [get(f"{m.lower()}-q{q}", rho, "click_ci_hi") for q in qs]
        x = np.array(qs) + (j - 1.5) * 0.008
        ax.errorbar(x, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=colors[m], marker=markers[m],
                    ms=3, lw=1, capsize=1.5, label=LABEL[m])
    for sl, ls, name in (("even", "--", "Even pacing"), ("pid", ":", "PID (tuned)"), ("nopacing", "-.", "No pacing")):
        ax.axhline(get(sl, rho, "click_ratio"), color="black", ls=ls, lw=0.8, label=name)
    ax.axhline(1.0, color="#bbbbbb", lw=0.6)
    ax.set_title(f"$\\rho$ = {rho}")
    ax.set_xticks(qs)
    ax.set_xlabel("hedging quantile $q$")
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_ylabel("clicks / hindsight reference")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{R}/figures/fig_hedging.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 2: delivery trade-off at rho = 0.5
fig, ax = plt.subplots(figsize=(3.3, 2.4))
rho = 0.5
for m in FORECASTERS:
    xs = [100 * get(f"{m.lower()}-q{q}", rho, "bur") for q in qs]
    ys = [100 * get(f"{m.lower()}-q{q}", rho, "early_depletion") for q in qs]
    ax.plot(xs, ys, color=colors[m], marker=markers[m], ms=3.5, lw=1, label=LABEL[m])
    if m == "TFT":
        for q, x, y in zip(qs, xs, ys):
            ax.annotate(f"$q{{=}}{q}$", (x, y), fontsize=5.5, xytext=(4, 4), textcoords="offset points", color=colors[m])
offsets = {"even": (-42, -1), "pid": (-40, -2), "nopacing": (-40, -2), "hindsight": (-50, 5)}
names = {"even": "Even pacing", "pid": "PID (tuned)", "nopacing": "No pacing", "hindsight": "Hindsight ref."}
for sl, off in offsets.items():
    x, y = 100 * get(sl, rho, "bur"), 100 * get(sl, rho, "early_depletion")
    ax.scatter(x, y, color="black", marker="x", s=16, zorder=3)
    ax.annotate(names[sl], (x, y), fontsize=5.5, xytext=off, textcoords="offset points")
ax.set_yscale("log")
ax.set_xlabel("budget utilization (%)")
ax.set_ylabel("early depletion rate (%, log)")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.25, lw=0.4, which="both")
fig.tight_layout()
fig.savefig(f"{R}/figures/fig_tradeoff_rho0.5.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 3: accuracy by forecast step
panel, campaigns, stat = load_work()
Y = spend_matrix(panel, campaigns)
scale = np.array([M.seasonal_naive_scale(Y[c, :C.TRAIN_END]) for c in range(len(campaigns))])
k50 = list(C.QUANTILES).index(0.5)
fig, ax = plt.subplots(figsize=(3.3, 2.0))
step_mase = {}
for m in FORECASTERS:
    curves = []
    for path in sorted(glob.glob(f"{R}/forecasts/{m}_seed*.npz")):
        z = np.load(path)
        F, origins = z["F"].astype(float), z["origins"]
        tgt = origins[:, None] + STEPS[None, :]
        ok = tgt < C.N_HOURS
        err = np.abs(Y[:, np.minimum(tgt, C.N_HOURS - 1)] - F[..., k50])          # (n, no, H)
        err = np.where(ok[None], err, np.nan)
        curves.append(np.nanmean(np.nanmean(err, axis=1) / scale[:, None], axis=0))
    c = np.mean(curves, axis=0)
    step_mase[m] = c
    ax.plot(STEPS + 1, c, color=colors[m], marker=markers[m], ms=2, lw=1, label=LABEL[m])
ax.set_xlabel("forecast step (hours ahead)")
ax.set_ylabel("MASE")
ax.legend(frameon=False, ncol=2)
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
fig.savefig(f"{R}/figures/fig_horizon.pdf")
plt.close(fig)
for m, c in step_mase.items():
    V[f"fc-{m.lower()}-mase-step1"] = f"{c[0]:.3f}"
    V[f"fc-{m.lower()}-mase-step24"] = f"{c[-1]:.3f}"

# ---------------------------------------------------------------- compact tables for the paper
def display(policy):
    m = re.fullmatch(r"(\w+) \(q=([\d.]+)\)", policy)
    if m:
        return f"{LABEL.get(m.group(1), m.group(1))}, $q{{=}}{m.group(2)}$"
    return {"Hindsight constant": "Hindsight constant (ref.)"}.get(policy, policy)


order = (["Hindsight constant", "No pacing", "Even pacing", "PID (tuned)"]
         + [f"{m} (q={q})" for q in C.HEDGE_QUANTILES for m in FORECASTERS])
best = {rho: S[(S.rho == rho) & (S.policy != "Hindsight constant")].click_ratio.max() for rho in C.RHOS}
rows = [r"\begin{tabular}{l@{\hskip 6pt}ccc@{\hskip 8pt}cccc}", r"\toprule",
        r" & \multicolumn{3}{c}{Clicks / reference} & \multicolumn{4}{c}{At $\rho=0.5$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r"Policy & $\rho{=}0.3$ & $\rho{=}0.5$ & $\rho{=}0.7$ & Conv./ref. & BUR (\%) & Early depl. (\%) & PVI \\",
        r"\midrule"]
for i, pol in enumerate(order):
    if i in (1, 4, 8, 12):
        rows.append(r"\addlinespace[2pt]")
    cells = []
    for rho in C.RHOS:
        r = S[(S.policy == pol) & (S.rho == rho)].iloc[0]
        s = f"{r.click_ratio:.3f}"
        cells.append(rf"\textbf{{{s}}}" if np.isclose(r.click_ratio, best[rho]) else s)
    r5 = S[(S.policy == pol) & (S.rho == 0.5)].iloc[0]
    cells += [f"{r5.conv_ratio:.3f}", f"{100 * r5.bur:.1f}", f"{100 * r5.early_depletion:.1f}", f"{r5.pvi:.3f}"]
    rows.append(display(pol) + " & " + " & ".join(cells) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
with open(f"{R}/tables/pacing_main.tex", "w") as f:
    f.write("\n".join(rows) + "\n")

rows = [r"\begin{tabular}{lcccccr}", r"\toprule",
        r"Forecaster & MASE & MASE$_{h=1}$ & MASE$_{h=24}$ & sPL$_{0.9}$ & Cov.\,80\% & Time (s) \\", r"\midrule"]
for m in FORECASTERS:
    v = FM[m]
    sd = f" $\\pm$ {v['std_over_seeds']['mase']:.3f}" if v["n_seeds"] > 1 else ""
    rows.append(f"{LABEL[m]} & {v['mean']['mase']:.3f}{sd} & {step_mase[m][0]:.3f} & {step_mase[m][-1]:.3f} & "
                f"{v['mean']['spl_0.9']:.3f} & {100 * v['mean']['coverage80']:.1f} & {np.mean(times[m]):.0f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
with open(f"{R}/tables/forecast_main.tex", "w") as f:
    f.write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- write macros
lines = ["% Generated by scripts/07_paper_assets.py from results files. Do not edit by hand.",
         r"\makeatletter", r"\newcommand{\res}[1]{\@ifundefined{res@#1}{\textbf{??#1??}}{\csname res@#1\endcsname}}"]
for k in sorted(V):
    lines.append(rf"\expandafter\def\csname res@{k}\endcsname{{{V[k]}}}")
lines.append(r"\makeatother")
with open(f"{R}/tables/numbers.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
write_meta("paper_assets", n_macros=len(V))
print(f"{len(V)} macros; figures:", sorted(os.listdir(f"{R}/figures")))
