"""Leakage guardrails for the chronological pipeline (S01 findings F07/F08).

    pytest tests/test_time_leakage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.budget import budgets  # noqa: E402
from src.data.criteo import select_campaigns  # noqa: E402
from src.data.timesplit import TimeSplit, assert_no_future, split_frame  # noqa: E402

SPLIT = TimeSplit(train_end=10, val_end=14, end=20)


def _hourly(seed=0):
    rng = np.random.default_rng(seed)
    rows = [(c, h, float(rng.uniform(1, 2))) for c in ("a", "b") for h in range(SPLIT.end)]
    return pd.DataFrame(rows, columns=["campaign", "hour", "spend"])


def test_split_is_chronological_and_disjoint():
    parts = split_frame(_hourly(), SPLIT)
    assert parts["train"]["hour"].max() < parts["val"]["hour"].min()
    assert parts["val"]["hour"].max() < parts["test"]["hour"].min()
    assert sum(len(p) for p in parts.values()) == len(_hourly())


def test_budget_ignores_test_period():
    df = _hourly()
    b0 = budgets(df, SPLIT, rho=0.7)
    df2 = df.copy()
    df2.loc[df2["hour"] >= SPLIT.val_end, "spend"] *= 1000.0  # change only the test period
    pd.testing.assert_series_equal(b0, budgets(df2, SPLIT, rho=0.7))


def test_campaign_selection_ignores_val_and_test():
    df = _hourly()
    df.loc[(df["campaign"] == "b") & (df["hour"] < SPLIT.train_end), "spend"] = 0.0
    sel0 = select_campaigns(df, SPLIT, min_active=0.9)
    df2 = df.copy()
    df2.loc[df2["hour"] >= SPLIT.train_end, "spend"] = 0.0  # wipe val+test
    assert sel0 == select_campaigns(df2, SPLIT, min_active=0.9) == ["a"]


def test_assert_no_future_detects_leak():
    with pytest.raises(AssertionError):
        assert_no_future(_hourly(), SPLIT.val_end)
