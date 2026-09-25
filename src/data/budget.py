"""Exogenous campaign budgets for the pacing replay (fixes S01 finding F07).

B_i(rho) = rho * mean hourly spend of campaign i over the VALIDATION period
           * number of test hours.
Only rows with hour < split.val_end are read; test spend never enters B.
"""
from __future__ import annotations

import pandas as pd

from .timesplit import TimeSplit, assert_no_future

RHOS = (0.5, 0.7, 0.9, 1.1)


def budgets(hourly: pd.DataFrame, split: TimeSplit, rho: float,
            spend_col: str = "spend", key: str = "campaign") -> pd.Series:
    past = hourly[hourly["hour"] < split.val_end]
    assert_no_future(past, split.val_end)
    val = past[past["hour"] >= split.train_end]
    rate = val.groupby(key)[spend_col].mean()
    return (rho * rate * split.test_len).rename(f"budget_rho{rho}")
