"""Replay simulator: identity at alpha = 1, monotone non-increasing spend in alpha."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sim.replay import auctionnet_spend, criteo_spend, ipinyou_spend  # noqa: E402

rng = np.random.default_rng(0)
PAYS = rng.uniform(10, 100, 1000)
BIDS = PAYS + rng.uniform(0, 200, 1000)          # won impressions: bid >= pay
LWC = rng.uniform(0.1, 0.5, 1000)
AN_BIDS = LWC + rng.uniform(-0.2, 0.3, 1000)
WON = AN_BIDS >= LWC
COST = np.where(WON, rng.uniform(0.05, 0.4, 1000), 0.0)


def test_identity_at_alpha_one():
    assert criteo_spend(123.4, 1.0) == 123.4
    assert ipinyou_spend(BIDS, PAYS, 1.0) == pytest.approx(PAYS.sum(), abs=1e-9)
    assert auctionnet_spend(AN_BIDS, LWC, COST, WON, 1.0) == pytest.approx(COST[WON].sum(), abs=1e-9)


@pytest.mark.parametrize("f", ["criteo", "ipinyou", "auctionnet"])
def test_monotone_in_alpha(f):
    alphas = np.linspace(0.05, 1.0, 20)
    if f == "criteo":
        s = [criteo_spend(100.0, a) for a in alphas]
    elif f == "ipinyou":
        s = [ipinyou_spend(BIDS, PAYS, a) for a in alphas]
    else:
        s = [auctionnet_spend(AN_BIDS, LWC, COST, WON, a) for a in alphas]
    assert all(x <= y + 1e-12 for x, y in zip(s, s[1:]))


def test_alpha_out_of_range():
    with pytest.raises(ValueError):
        criteo_spend(1.0, 1.5)
