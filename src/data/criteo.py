"""D2: Criteo Attribution Modeling for Bidding -> hourly spend per campaign.

Source: https://huggingface.co/datasets/criteo/criteo-attribution-dataset
Licence: CC BY-NC-SA 4.0. One row per displayed impression; `cost` is the price
paid for that impression (anonymised units), `timestamp` is seconds from the
start of the 30-day sample. Nothing is replicated or synthesised here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timesplit import TimeSplit, assert_no_future

USECOLS = ["timestamp", "campaign", "cost", "click", "conversion"]


def load_hourly(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Aggregate impressions to a complete (campaign x hour) grid.

    Returns long format with columns: campaign, hour, spend, impressions,
    clicks, conversions. Hours with no impression get zeros.
    """
    raw = pd.read_csv(path, sep="\t", usecols=USECOLS, nrows=nrows)
    raw["hour"] = (raw["timestamp"] // 3600).astype(np.int64)
    agg = raw.groupby(["campaign", "hour"], sort=True).agg(
        spend=("cost", "sum"),
        impressions=("cost", "size"),
        clicks=("click", "sum"),
        conversions=("conversion", "sum"),
    )
    hours = np.arange(int(raw["hour"].min()), int(raw["hour"].max()) + 1)
    grid = pd.MultiIndex.from_product([agg.index.levels[0], hours], names=["campaign", "hour"])
    out = agg.reindex(grid, fill_value=0).reset_index()
    out["impressions"] = out["impressions"].astype(np.int64)
    return out


def select_campaigns(hourly: pd.DataFrame, split: TimeSplit, min_active: float = 0.9,
                     min_active_val: float = 0.5) -> list:
    """Campaigns with spend > 0 in >= `min_active` of TRAIN hours and in
    >= `min_active_val` of VALIDATION hours (i.e. still running when the test
    period starts). Test hours are never read, so selection cannot depend on
    test behaviour; campaigns that stop during test are kept.
    """
    past = hourly[hourly["hour"] < split.val_end]
    assert_no_future(past, split.val_end)
    on = past.assign(on=past["spend"] > 0)
    tr = on[on["hour"] < split.train_end].groupby("campaign")["on"].mean()
    va = on[on["hour"] >= split.train_end].groupby("campaign")["on"].mean()
    ok = (tr >= min_active) & (va.reindex(tr.index).fillna(0) >= min_active_val)
    return sorted(tr[ok].index.tolist())
