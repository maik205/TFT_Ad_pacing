"""Hourly campaign panel and impression store built from the raw Criteo log."""

import os

import numpy as np
import pandas as pd

from . import config as C

RAW_COLS = ["timestamp", "campaign", "cost", "click", "attribution"]


def p(key):
    return os.path.expanduser(C.PATHS[key])


def load_impressions(path=None):
    path = os.path.expanduser(path or C.PATHS["raw"])
    df = pd.read_csv(path, sep="\t", usecols=RAW_COLS)
    df["hour"] = (df["timestamp"] // 3600).astype(np.int32)
    return df.sort_values(["campaign", "timestamp"], kind="stable").reset_index(drop=True)


def build_panel(imp):
    """Dense (campaign x hour) panel; hours with no impressions are explicit zeros."""
    g = imp.groupby(["campaign", "hour"]).agg(
        spend=("cost", "sum"), imps=("cost", "size"),
        clicks=("click", "sum"), convs=("attribution", "sum"),
    )
    campaigns = g.index.get_level_values(0).unique()
    full = pd.MultiIndex.from_product([campaigns, range(C.N_HOURS)], names=["campaign", "hour"])
    return g.reindex(full, fill_value=0).reset_index()


def select_campaigns(panel):
    """Inclusion rule uses training hours only (PROTOCOL 3)."""
    tr = panel[panel.hour < C.TRAIN_END]
    active = (tr.spend > 0).groupby(tr.campaign).mean()
    return sorted(active[active >= C.MIN_ACTIVE_FRAC].index)


def static_features(panel):
    """Per-campaign statistics from training hours only."""
    tr = panel[panel.hour < C.TRAIN_END]
    s = tr.groupby("campaign").agg(
        mean_hourly_spend=("spend", "mean"), std_hourly_spend=("spend", "std"),
        mean_imps=("imps", "mean"), clicks=("clicks", "sum"), imps=("imps", "sum"),
    )
    s["mean_daily_spend"] = 24 * s.mean_hourly_spend
    s["ctr"] = s.clicks / s.imps.clip(lower=1)
    s["cv"] = s.std_hourly_spend / s.mean_hourly_spend
    return s[["mean_hourly_spend", "mean_daily_spend", "mean_imps", "ctr", "cv"]]


def calendar(hours):
    """Phase features relative to dataset start (timestamp 0 is not local midnight)."""
    hours = np.asarray(hours)
    hod, dow = hours % 24, (hours // 24) % 7
    return pd.DataFrame({
        "hod_sin": np.sin(2 * np.pi * hod / 24), "hod_cos": np.cos(2 * np.pi * hod / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
    })


def spend_matrix(panel, campaigns):
    """(n_campaigns, N_HOURS) array of hourly spend, rows ordered as `campaigns`."""
    m = panel.pivot(index="campaign", columns="hour", values="spend")
    return m.loc[campaigns].to_numpy(dtype=np.float64)


def load_work():
    w = p("work")
    panel = pd.read_parquet(f"{w}/panel.parquet")
    campaigns = list(pd.read_parquet(f"{w}/campaigns.parquet").campaign)
    stat = pd.read_parquet(f"{w}/static.parquet").loc[campaigns]
    return panel, campaigns, stat
