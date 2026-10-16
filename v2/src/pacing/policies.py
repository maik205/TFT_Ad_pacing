"""Pacing policies (PROTOCOL 5). act(h, S, spends) returns the participation rate for hour h,
given cumulative spend S before hour h and the realized spend of earlier hours."""

import numpy as np

from . import config as C


class NoPacing:
    name = "No pacing"

    def reset(self, ep):
        pass

    def act(self, h, S, spends):
        return 1.0


class Hindsight:
    """Constant rate min(1, B/D) using the day's true demand D. A reference, not a bound."""
    name = "Hindsight constant"

    def reset(self, ep):
        D = ep.demand.sum()
        self.a = 1.0 if D <= 0 else min(1.0, ep.budget / D)

    def act(self, h, S, spends):
        return self.a


class EvenPacing:
    """Targets an even spend per remaining hour; multiplicative correction from last hour."""
    name = "Even pacing"
    a_min = 0.01

    def __init__(self, daily_train_spend):
        self.m = daily_train_spend          # per campaign, training hours only

    def reset(self, ep):
        self.B = ep.budget
        self.a = min(1.0, ep.budget / self.m[ep.c])

    def act(self, h, S, spends):
        R = self.B - S
        if R <= 0:
            return 0.0
        if h > 0 and spends[-1] > 0:
            target = R / (24 - h)
            self.a = float(np.clip(self.a * target / spends[-1], self.a_min, 1.0))
        return self.a


class PID:
    """Position-form PID on the normalized gap to the even cumulative target."""

    def __init__(self, daily_train_spend, kp, ki, kd):
        self.m, self.kp, self.ki, self.kd = daily_train_spend, kp, ki, kd
        self.name = f"PID(kp={kp},ki={ki},kd={kd})"

    def reset(self, ep):
        self.B = ep.budget
        self.a0 = min(1.0, ep.budget / self.m[ep.c])
        self.integral, self.prev = 0.0, 0.0

    def act(self, h, S, spends):
        e = (self.B * h / 24 - S) / self.B
        self.integral += e
        d = e - self.prev if h > 0 else 0.0
        self.prev = e
        return self.a0 + self.kp * e + self.ki * self.integral + self.kd * d


class TunedPID(PID):
    """PID whose gains were selected on validation days separately for each budget level rho."""
    name = "PID (tuned)"

    def __init__(self, daily_train_spend, gains_by_rho):
        super().__init__(daily_train_spend, 0, 0, 0)
        self.gains = {round(float(r), 4): g for r, g in gains_by_rho.items()}
        self.name = "PID (tuned)"

    def reset(self, ep):
        rho = round(ep.budget / self.m[ep.c], 4)
        self.kp, self.ki, self.kd = self.gains[rho]
        super().reset(ep)


class ForecastThrottle:
    """alpha_h = min(1, R_h / sum of q-quantile demand forecasts for the rest of the day),
    re-planned every hour from the forecast issued at that hour."""

    def __init__(self, F, origins, q, label):
        self.F, self.o0 = F, int(origins[0])
        self.k = list(C.QUANTILES).index(q)
        self.name = f"{label} (q={q})"

    def reset(self, ep):
        self.ep = ep

    def act(self, h, S, spends):
        ep = self.ep
        R = ep.budget - S
        if R <= 0:
            return 0.0
        i = 24 * ep.day + h - self.o0
        expected = float(self.F[ep.c, i, : 24 - h, self.k].sum())
        return 1.0 if expected <= 0 else min(1.0, R / expected)
