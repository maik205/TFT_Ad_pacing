"""Evaluation harness (study protocol Section 6). EVALUATION CODE: it rebuilds the world's ground truth in
memory and simulates outcomes; policies only ever receive observed arrays and realised feedback.

    res = run(world, obs, policy, cost="C1", rho=0.5, days=(24, 28))

Outcomes are drawn with the generator's own `draw_outcomes`, fed by random numbers keyed on
(world seed, opportunity, campaign), so two policies that make the same choice get the same outcome.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtri

SYNTH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "synthetic")
if SYNTH not in sys.path:
    sys.path.insert(0, SYNTH)

from synth.config import config_hash  # noqa: E402
from synth.generate import build  # noqa: E402
from synth.outcomes import draw_outcomes, fatigue  # noqa: E402

from .observed import DAY, campaign_local_day  # noqa: E402

FEEDBACK = ("clicked", "skipped", "watch_time_s", "completed", "post_dwell_s", "interacted",
            "muted_or_backgrounded")
OUTCOMES = FEEDBACK + ("pctr", "ambiguous") + tuple(f"cost_C{i}" for i in range(1, 7))


def _splitmix(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


class KeyedRNG:
    """Stand-in for a numpy Generator inside `draw_outcomes`: the i-th call returns, for each row, a
    uniform that depends only on (salt, key of the row, i). Every call must cover all rows."""

    def __init__(self, keys, salt):
        self.k = _splitmix(keys.astype(np.uint64) ^ np.uint64(salt))
        self.i = 0

    def random(self, n=None):
        assert n == self.k.size, "keyed draws must cover every row"
        self.i += 1
        with np.errstate(over="ignore"):
            x = _splitmix(self.k + np.uint64(self.i) * np.uint64(0xD1B54A32D192ED03))
        return ((x >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53

    def standard_normal(self, n=None):
        return ndtri(self.random(n))


class World:
    """Ground truth of a generated world, rebuilt from its config and seed (deterministic, spec R4)."""

    def __init__(self, world_dir):
        meta = json.load(open(os.path.join(world_dir, "observed", "synth_meta.json")))
        self.cfg, self.seed = meta["config"], meta["world_seed"]
        assert config_hash(self.cfg) == meta["config_hash"], "config does not reproduce the world"
        w = build(self.cfg, self.seed)
        ts = pd.read_parquet(os.path.join(world_dir, "observed", "synth_opportunities.parquet"),
                             columns=["ts_utc"])["ts_utc"].to_numpy()
        assert np.array_equal(ts, w["opp"]["ts"]), "rebuilt world differs from the files"
        assert abs(w["beta0"] - meta["beta0"]) < 1e-9
        self.users, self.camp, self.opp, self.beta0 = w["users"], w["camp"], w["opp"], w["beta0"]
        self.r_bar = w["out"]["r_bar"]
        self.logged = {"choice": w["log"]["choice"], "pctr": w["out"]["pctr"], "imp_opp": w["imp"]["opp"],
                       "n_uc": w["imp"]["n_uc"]}
        self.salt = 0x5EED0000 + self.seed

    def frame(self, opp, camp, n_uc):
        """Planted quantities for (opportunity, campaign) pairs, as in synth.outcomes.impression_frame."""
        cfg, o, cp = self.cfg, self.opp, self.camp
        interest = o["visit_interest"][o["visit"][opp]]
        aff = np.einsum("nk,nk->n", cp["m"][camp], interest)
        match = cp["m"][camp, o["context"][opp]] > 0
        st = o["state"][opp]
        z = (cfg["gamma"] * aff + cp["q"][camp] + cfg["delta"] * match - cfg["phi"] * np.log1p(n_uc)
             + np.array(cfg["s_state"])[st])
        return {"opp": opp, "campaign": camp, "user": o["user"][opp], "ts": o["ts"][opp], "aff": aff,
                "match": match, "n_uc": n_uc, "state": st, "format": o["format"][opp],
                "duration": o["duration"][opp], "c3_bid2": np.full(opp.size, np.nan), "logit_no_beta0": z}

    def outcomes(self, opp, camp, n_uc):
        """Realised outcomes of showing camp[i] in opp[i]; keyed draws, one call per format group so that
        every draw inside draw_outcomes covers all rows."""
        n = opp.size
        imp = self.frame(opp, camp, n_uc)
        res = {k: np.full(n, np.nan) for k in OUTCOMES}
        for f in np.unique(imp["format"]):
            g = np.flatnonzero(imp["format"] == f)
            sub = {k: v[g] for k, v in imp.items()}
            rng = KeyedRNG(opp[g].astype(np.uint64) * np.uint64(1024) + camp[g].astype(np.uint64), self.salt)
            out = draw_outcomes(self.cfg, self.camp, self.users, sub, self.beta0, rng, r_bar=self.r_bar)
            for k in res:
                res[k][g] = out[k]
        for k in ("clicked", "interacted", "muted_or_backgrounded"):
            res[k] = res[k] == 1
        res["absent"] = imp["state"] == 2
        res["value"] = res["pctr"] * self.camp["v"][camp]
        return res


def run(world, obs, policy, cost="C1", rho=0.5, days=(24, 28), hard_stop=True):
    """Replay the opportunities of UTC days [days[0], days[1]) through `policy`, online.

    Each pacing step the policy sees the step's opportunity indices, the campaigns' spend so far in their
    current local day, and which campaigns may still spend; it returns one campaign per opportunity
    (-1 = no ad). With `hard_stop`, an impression is dropped once its campaign's spend before it has
    reached the daily budget (so the overshoot is at most one impression).
    """
    C = obs.C
    budget = obs.budget(cost, rho)
    step = 3600.0 / obs.steps_per_hour
    t_start, t_end = days[0] * DAY, days[1] * DAY
    edges = np.arange(t_start, t_end + step / 2, step)
    bounds = np.searchsorted(obs.ts, edges)
    spend = np.zeros((C, obs.days + 3))
    hist_u = np.zeros(0, np.int64)
    hist_c = np.zeros(0, np.int64)
    hist_t = np.zeros(0)
    rec = []
    policy.start(obs, {"cost": cost, "rho": rho, "budget": budget, "step_s": step, "t_start": t_start,
                       "t_end": t_end})
    for b in range(edges.size - 1):
        t0 = edges[b]
        lo, hi = bounds[b], bounds[b + 1]
        d = campaign_local_day(np.full(C, t0), obs.camp_tz)
        sp = spend[np.arange(C), d + 1]
        elig = sp < budget if hard_stop else np.ones(C, bool)
        if lo == hi:
            policy.tick(t0, sp.copy(), elig.copy())
            continue
        idx = np.arange(lo, hi)
        choice = np.asarray(policy.decide(t0, idx, sp.copy(), elig.copy()), np.int64)
        assert choice.shape == idx.shape and choice.min() >= -1 and choice.max() < C
        s = choice >= 0
        assert elig[choice[s]].all(), "policy chose a campaign that has no budget left"
        opp, camp = idx[s], choice[s]
        # fatigue: the policy's own impressions of c to u in the preceding 24 h
        keep_h = hist_t >= t0 - DAY
        hist_u, hist_c, hist_t = hist_u[keep_h], hist_c[keep_h], hist_t[keep_h]
        u = obs.user[opp]
        n_uc = fatigue(np.concatenate([hist_u, u]), np.concatenate([hist_c, camp]),
                       np.concatenate([hist_t, obs.ts[opp]]), C)[hist_u.size:]
        out = world.outcomes(opp, camp, n_uc)
        cst = out[f"cost_{cost}"]
        if hard_stop and opp.size:
            # spend of each campaign before each of its impressions in this step (time order)
            order = np.lexsort((np.arange(opp.size), camp))
            cum = np.cumsum(cst[order])
            first = np.r_[True, camp[order][1:] != camp[order][:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(opp.size), 0))
            before = np.empty(opp.size)
            before[order] = cum - cst[order] - np.r_[0.0, cum][start]
            ok = sp[camp] + before < budget[camp]
            opp, camp, n_uc, cst = opp[ok], camp[ok], n_uc[ok], cst[ok]
            out = {k: v[ok] for k, v in out.items()}
        np.add.at(spend, (camp, d[camp] + 1), cst)
        hist_u = np.concatenate([hist_u, obs.user[opp]])
        hist_c = np.concatenate([hist_c, camp])
        hist_t = np.concatenate([hist_t, obs.ts[opp]])
        fb = {k: out[k] for k in FEEDBACK}
        fb["cost"] = cst
        policy.observe(t0, opp, camp, fb)
        rec.append(pd.DataFrame({"opp": opp, "campaign": camp, "ts": obs.ts[opp], "n_uc": n_uc,
                                 "pctr": out["pctr"], "value": out["value"], "clicked": out["clicked"],
                                 "cost": cst, "absent": out["absent"],
                                 "local_day": d[camp]}))
    imps = pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()
    return {"impressions": imps, "cost": cost, "rho": rho, "days": days, "budget": budget,
            "policy": getattr(policy, "name", type(policy).__name__)}
