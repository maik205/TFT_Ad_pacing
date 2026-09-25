"""Figures + summary for the S05 data report, computed from processed data only.

    python scripts/data_report.py --source criteo
Writes reports/data/<source>_*.png and reports/data/<source>_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.budget import RHOS, budgets  # noqa: E402
from src.data.timesplit import CRITEO_SPLIT  # noqa: E402


def acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[lag:] * x[:-lag]).sum() / d) if d > 0 else float("nan")


def criteo(out: Path) -> dict:
    sp = CRITEO_SPLIT
    hourly = pd.read_parquet(ROOT / "data/processed/criteo_hourly.parquet")
    sel = json.loads((ROOT / "data/processed_stats/criteo.json").read_text())["selected_campaigns"]
    h = hourly[hourly["campaign"].isin(sel)]
    wide = h.pivot(index="campaign", columns="hour", values="spend")
    train = wide.loc[:, wide.columns < sp.train_end].to_numpy()

    total = wide.sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(total.index, total.values, lw=0.8, color="#2b5c8f")
    for a, b, lab in [(0, sp.train_end, "train"), (sp.train_end, sp.val_end, "val"), (sp.val_end, sp.end, "test")]:
        ax.axvspan(a, b, alpha=0.08 if lab == "train" else 0.18, color={"train": "gray", "val": "orange", "test": "red"}[lab])
        ax.text((a + b) / 2, total.max() * 0.95, lab, ha="center", fontsize=9)
    ax.set_xlabel("hour since start of sample"); ax.set_ylabel("total spend (anon. units)")
    ax.set_title(f"D2 Criteo: total hourly spend of {len(sel)} selected campaigns")
    fig.tight_layout(); fig.savefig(out / "criteo_total_spend.png", dpi=150); plt.close(fig)

    norm = train / np.maximum(train.mean(axis=1, keepdims=True), 1e-12)
    prof = np.array([norm[:, k::24].mean() for k in range(24)])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(24), prof, marker="o", ms=3)
    ax.set_xlabel("hour index mod 24 (sample clock; true time-of-day unknown)")
    ax.set_ylabel("spend / series mean"); ax.set_title("Mean within-day profile (train)")
    fig.tight_layout(); fig.savefig(out / "criteo_daily_profile.png", dpi=150); plt.close(fig)

    b = {str(r): budgets(h, sp, r) for r in RHOS}
    test_spend = h[h["hour"] >= sp.val_end].groupby("campaign")["spend"].sum()
    ratio = {r: float(np.nanmedian(test_spend / b[r].reindex(test_spend.index))) for r in b}
    n_stopped_in_test = int((test_spend == 0).sum())
    summary = {
        "n_series": len(sel),
        "median_acf_train_lag24": float(np.median([acf(r, 24) for r in train])),
        "median_acf_train_lag168": float(np.median([acf(r, 168) for r in train])),
        "daily_profile_peak_to_trough": float(prof.max() / prof.min()),
        "median_test_spend_over_budget_by_rho": ratio,
        "n_series_zero_spend_in_test": n_stopped_in_test,
        "note": "test_spend/budget > 1 means the unthrottled campaign would overspend B (budget binds)",
    }
    (out / "criteo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True, choices=["criteo"])
    a = ap.parse_args()
    out = ROOT / "reports/data"; out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"criteo": criteo}[a.source](out), indent=2))
