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

for rho, v in PID.items():
    kp, ki, kd = v["gains"]
    V[f"pid-gains-rho{float(rho)}"] = f"$K_p{{=}}{kp:g}, K_i{{=}}{ki:g}, K_d{{=}}{kd:g}$"

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
