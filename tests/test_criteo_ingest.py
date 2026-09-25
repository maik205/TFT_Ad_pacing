"""Ingestion of D2 conserves spend and yields a complete campaign x hour grid."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.criteo import load_hourly  # noqa: E402


def test_conservation_and_grid(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    raw = pd.DataFrame({
        "timestamp": np.sort(rng.integers(0, 10 * 3600, n)),
        "uid": 0, "campaign": rng.choice([11, 22, 33], n),
        "conversion": rng.integers(0, 2, n), "click": rng.integers(0, 2, n),
        "cost": rng.uniform(1e-5, 1e-3, n),
    })
    p = tmp_path / "toy.tsv.gz"
    raw.to_csv(p, sep="\t", index=False)
    hourly = load_hourly(str(p))
    assert np.isclose(hourly["spend"].sum(), raw["cost"].sum())
    assert hourly["impressions"].sum() == n
    assert hourly["clicks"].sum() == raw["click"].sum()
    n_h = raw["timestamp"].max() // 3600 - raw["timestamp"].min() // 3600 + 1
    assert len(hourly) == 3 * n_h  # complete grid
