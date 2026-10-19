"""Policies. Policy code sees only `Observed` data and the harness's realised feedback (generator spec R2).

Interface, called by `pacing.harness.run`:
    start(obs, ctx)                    once; ctx has cost, rho, budget, step_s, t_start, t_end
    decide(t0, idx, spend, elig)       per pacing step with opportunities; returns a campaign per index (-1 = none)
    tick(t0, spend, elig)              per pacing step without opportunities
    observe(t0, opp, camp, feedback)   realised outcomes of the impressions just shown
"""
import numpy as np

from .observed import DAY, campaign_local_day


class Policy:
    name = "base"

    def start(self, obs, ctx):
        self.obs, self.ctx = obs, ctx

    def decide(self, t0, idx, spend, elig):
        raise NotImplementedError

    def tick(self, t0, spend, elig):
        pass

    def observe(self, t0, opp, camp, feedback):
        pass


class LoggedActions(Policy):
    """Replays the logged choices (harness validation only; run with hard_stop=False)."""
    name = "logged"

    def decide(self, t0, idx, spend, elig):
        return self.obs.log_choice[idx]


class EvenThrottle(Policy):
    """P0: the logging policy's rule (generator spec Section 10), user-blind: softmax over
    v_c * (1 + delta_ctx * context match) and a "no ad" score, epsilon-mixed with uniform, with a
    per-campaign throttle that lets a campaign bid only while its spend today is below its budget times
    the share of its local day elapsed by the end of the step."""
    name = "P0"

    def __init__(self, seed=0, epsilon=0.10, tau=1.0, delta_ctx=1.0, none_score=1.0):
        self.rng = np.random.default_rng(seed)
        self.eps, self.tau, self.dctx, self.none = epsilon, tau, delta_ctx, none_score

    def decide(self, t0, idx, spend, elig):
        o, B, step = self.obs, self.ctx["budget"], self.ctx["step_s"]
        C, n = o.C, idx.size
        d = campaign_local_day(np.full(C, t0), o.camp_tz)
        frac = np.minimum(1.0, ((t0 + 3600.0 * o.camp_tz) - d * DAY + step) / DAY)
        ok = elig & (spend < B * frac)
        match = o.camp_cats[:, o.context[idx]].T > 0  # (n, C)
        z = np.where(ok[None, :], o.v[None, :] * (1 + self.dctx * match) / self.tau, -np.inf)
        z = np.concatenate([z, np.full((n, 1), self.none / self.tau)], axis=1)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        uni = np.concatenate([ok, [True]]).astype(float) / (ok.sum() + 1)
        p = (1 - self.eps) * p + self.eps * uni[None, :]
        a = np.minimum((np.cumsum(p, axis=1) < self.rng.random(n)[:, None]).sum(axis=1), C)
        return np.where(a == C, -1, a)
