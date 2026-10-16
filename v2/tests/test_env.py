import numpy as np
import pytest

from pacing import config as C
from pacing.env import Episode, run_episode, uniforms
from pacing.forecasters import seasonal_naive
from pacing.policies import EvenPacing, ForecastThrottle, Hindsight, NoPacing, PID


def make_episode(budget, per_hour=10, cost=1.0):
    return Episode(c=0, day=0, budget=budget,
                   cost=[np.full(per_hour, cost) for _ in range(24)],
                   click=[np.ones(per_hour, np.int64) for _ in range(24)],
                   conv=[np.zeros(per_hour, np.int64) for _ in range(24)])


def test_no_pacing_unbounded_budget_serves_all_demand():
    ep = make_episode(budget=1e9)
    r = run_episode(ep, NoPacing(), uniforms(ep, 0))
    assert r["spend"] == pytest.approx(240.0)
    assert r["clicks"] == 240 and r["depletion_hour"] == 24


def test_no_pacing_depletes_early_and_never_overspends():
    ep = make_episode(budget=55.0)
    r = run_episode(ep, NoPacing(), uniforms(ep, 0))
    assert r["spend"] == pytest.approx(55.0)      # 55 unit-cost impressions fit exactly
    assert r["depletion_hour"] == 5               # 10/h -> exhausted during hour index 5


def test_zero_rate_spends_nothing():
    class Zero(NoPacing):
        def act(self, h, S, spends):
            return 0.0
    ep = make_episode(budget=100.0)
    assert run_episode(ep, Zero(), uniforms(ep, 0))["spend"] == 0.0


def test_common_random_numbers_are_reproducible():
    ep = make_episode(budget=100.0, per_hour=50)
    u1, u2 = uniforms(ep, 3), uniforms(ep, 3)
    assert all(np.array_equal(a, b) for a, b in zip(u1, u2))
    r1 = run_episode(ep, Hindsight(), u1)
    r2 = run_episode(ep, Hindsight(), u2)
    assert r1["spend"] == r2["spend"]


def test_hindsight_spends_close_to_budget_across_day():
    ep = make_episode(budget=600.0, per_hour=100)      # demand 2400, alpha* = 0.25
    spends = [run_episode(ep, Hindsight(), uniforms(ep, s))["spend"] for s in range(20)]
    assert np.mean(spends) == pytest.approx(600.0, rel=0.05)


def test_even_and_pid_respect_budget():
    ep = make_episode(budget=100.0, per_hour=40)
    for pol in (EvenPacing(np.array([960.0])), PID(np.array([960.0]), 2.0, 0.1, 0.5)):
        r = run_episode(ep, pol, uniforms(ep, 1))
        assert r["spend"] <= 100.0 + 1e-9


def test_forecast_throttle_with_perfect_flat_forecast_matches_hindsight_rate():
    ep = make_episode(budget=60.0, per_hour=10)          # demand 240 -> alpha* = 0.25
    origins = np.array([0])
    F = np.zeros((1, 24, C.HORIZON, len(C.QUANTILES)))
    F[...] = 10.0
    pol = ForecastThrottle(F, origins, 0.5, "test")
    pol.reset(ep)
    assert pol.act(0, 0.0, []) == pytest.approx(0.25)
    assert pol.act(12, 60.0, []) == 0.0


def test_seasonal_naive_only_uses_past_hours():
    Y = np.arange(400, dtype=float)[None, :]
    F = seasonal_naive(Y, np.array([200]))
    # median of same hour on 7 previous days for step 0: hours 176, 152, ..., 32 -> median 104
    assert F[0, 0, 0, list(C.QUANTILES).index(0.5)] == pytest.approx(104.0)
    assert F.max() < 200 + 24 - 24
