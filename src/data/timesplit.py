"""Chronological train/val/test splits on an integer hour index.

Intervals are half-open: train = [0, train_end), val = [train_end, val_end),
test = [val_end, end). Nothing fitted for the model or the policy may read rows
with hour >= val_end; see `assert_no_future` and tests/test_time_leakage.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: int
    val_end: int
    end: int

    def __post_init__(self):
        if not (0 < self.train_end < self.val_end < self.end):
            raise ValueError(f"invalid split {self}")

    @property
    def test_len(self) -> int:
        return self.end - self.val_end


# D2 Criteo attribution: 742 hours (S04 plan, approved at G3).
CRITEO_SPLIT = TimeSplit(train_end=480, val_end=600, end=742)


def split_frame(df: pd.DataFrame, split: TimeSplit, time_col: str = "hour") -> dict:
    t = df[time_col]
    parts = {
        "train": df[t < split.train_end],
        "val": df[(t >= split.train_end) & (t < split.val_end)],
        "test": df[(t >= split.val_end) & (t < split.end)],
    }
    assert_chronological(parts["train"], parts["val"], parts["test"], time_col)
    return parts


def assert_chronological(train, val, test, time_col: str = "hour") -> None:
    """Raise if any val/test timestamp is <= a train timestamp (or test <= val)."""
    if len(train) and len(val):
        assert train[time_col].max() < val[time_col].min(), "val overlaps train"
    if len(val) and len(test):
        assert val[time_col].max() < test[time_col].min(), "test overlaps val"
    if len(train) and len(test):
        assert train[time_col].max() < test[time_col].min(), "test overlaps train"


def assert_no_future(df: pd.DataFrame, limit: int, time_col: str = "hour") -> None:
    """Raise if `df` contains rows at or after `limit` (used before any fit)."""
    if len(df) and df[time_col].max() >= limit:
        raise AssertionError(
            f"frame reaches hour {df[time_col].max()} >= limit {limit}: future data leaked"
        )
