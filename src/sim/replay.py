"""Counterfactual spend under a bid multiplier alpha in (0, 1].

Each dataset supports a different level of counterfactual fidelity; the paper
states which one is used (S04 plan, section 3):

* D2 Criteo  - only the paid cost is logged, so spend(alpha) = alpha * spend
               (linear-scaling ASSUMPTION, same as first-price pay-as-bid).
* D1 iPinYou - bid and paying price are logged for won impressions; with
               alpha <= 1 an impression is kept iff alpha * bid >= paying price,
               and it is still charged the paying price (second-price).
* D3 AuctionNet - bid, cost and leastWinningCost are logged for every
               opportunity; a won slot is kept iff alpha * bid >= leastWinningCost.
               The cost of a kept slot is the logged cost (rank changes among
               the winners are ignored - stated as an approximation).

Identity property (tested): alpha = 1 reproduces the logged spend exactly.
"""
from __future__ import annotations

import numpy as np


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")


def criteo_spend(spend: float, alpha: float) -> float:
    _check_alpha(alpha)
    return alpha * spend


def ipinyou_spend(bids: np.ndarray, pays: np.ndarray, alpha: float) -> float:
    _check_alpha(alpha)
    bids = np.asarray(bids, dtype=float)
    pays = np.asarray(pays, dtype=float)
    keep = alpha * bids >= pays
    return float(pays[keep].sum())


def auctionnet_spend(bids: np.ndarray, least_winning_cost: np.ndarray,
                     cost: np.ndarray, won: np.ndarray, alpha: float) -> float:
    _check_alpha(alpha)
    bids = np.asarray(bids, dtype=float)
    lwc = np.asarray(least_winning_cost, dtype=float)
    cost = np.asarray(cost, dtype=float)
    won = np.asarray(won).astype(bool)
    keep = won & (alpha * bids >= lwc)
    return float(cost[keep].sum())
