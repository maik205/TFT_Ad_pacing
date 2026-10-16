import numpy as np
import pytest

from pacing import metrics as M


def test_pinball_hand_values():
    # u = y - yq = [2, -2]; q=0.9 -> [1.8, 0.2] -> mean 1.0
    assert M.pinball([3, 1], [1, 3], 0.9) == pytest.approx(1.0)
    # q=0.5 is half the absolute error
    assert M.pinball([3, 1], [1, 3], 0.5) == pytest.approx(1.0)
    assert M.pinball([5], [5], 0.1) == 0.0


def test_seasonal_scale_uses_lag_within_one_series():
    s = np.r_[np.zeros(24), np.full(24, 2.0)]
    assert M.seasonal_naive_scale(s) == pytest.approx(2.0)
    assert M.seasonal_naive_scale(np.tile(np.arange(24.0), 3)) == 0.0


def test_mase():
    assert M.mase([1, 2, 3], [2, 2, 2], scale=0.5) == pytest.approx((2 / 3) / 0.5)


def test_coverage_and_crossing():
    assert M.coverage([1, 5, 10], lo=[0, 6, 9], hi=[2, 7, 11]) == pytest.approx(2 / 3)
    q = np.array([[1, 2, 3, 4], [1, 3, 2, 4], [0, 0, 0, 0]])
    assert M.crossing_rate(q) == pytest.approx(1 / 3)


def test_pacing_volatility():
    assert M.pacing_volatility([0.5] * 24) == 0.0
    assert M.pacing_volatility([0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_depletion_hour():
    cum = np.cumsum(np.full(24, 1.0))  # budget 10 reached at end of hour index 9
    assert M.depletion_hour(cum, 10.0, 0.99) == 9
    assert M.depletion_hour(cum, 100.0, 0.99) == 24


def test_profile_deviation():
    assert M.profile_deviation([1, 1], [1, 1], 2) == 0.0
    assert M.profile_deviation([2, 0], [0, 2], 2) == pytest.approx(1.0)
