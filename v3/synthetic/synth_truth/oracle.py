"""Ground-truth oracle (spec Section 11). EVALUATION CODE ONLY: model and policy code must not import this
module or read out/truth (spec R2; enforced by tests/test_integrity.py).

Given any allocation of logged opportunities to campaigns, it returns the exact expected clicks, value and
cost under the planted world, plus the true attention state for the wasted-spend metric.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta as B

sig = lambda x: 1 / (1 + np.exp(-x))
P_COMPLETE_PREROLL = np.array([0.85, 0.5, 1.0])


class Oracle:
    def __init__(self, world_dir):
        obs, tru = os.path.join(world_dir, "observed"), os.path.join(world_dir, "truth")
        o = pd.read_parquet(os.path.join(obs, "synth_opportunities.parquet"),
                            columns=["opp_id", "ts_utc", "user_id", "visit_id", "context_category", "format",
                                     "video_duration_s"])
        self.cats = list(o["context_category"].cat.categories)
        self.ts = o["ts_utc"].to_numpy()
        self.user = o["user_id"].to_numpy()
        self.visit = o["visit_id"].to_numpy()
        self.context = o["context_category"].cat.codes.to_numpy()
        self.format = o["format"].cat.codes.to_numpy()  # 0 display, 1 preroll, 2 infeed
        self.duration = o["video_duration_s"].to_numpy()
        self.state = pd.read_parquet(os.path.join(tru, "truth_states.parquet"),
                                     columns=["state"])["state"].cat.codes.to_numpy()
        self.interest = np.load(os.path.join(tru, "truth_visit_interest.npy"), mmap_mode="r")
        self.world = json.load(open(os.path.join(tru, "truth_world.json")))
        c = pd.read_parquet(os.path.join(obs, "synth_campaigns.parquet"))
        self.v = c["value_per_click"].to_numpy()
        self.m_cpm = c["cpm_unit_cost"].to_numpy()
        self.m = np.array(self.world["campaign_m"])
        self.q = np.array(self.world["campaign_q"])
        meta = json.load(open(os.path.join(obs, "synth_meta.json")))
        self.cfg = meta["config"]
        self.g = np.array(meta["checks"]["population_curve_g"])
        u = pd.read_parquet(os.path.join(tru, "truth_users.parquet"), columns=["tz_offset"])
        self.tz = u["tz_offset"].to_numpy()

    # ---- planted quantities for (opportunity, campaign) pairs
    def affinity(self, opp, camp):
        it = np.asarray(self.interest[self.visit[opp]])
        return np.einsum("nk,nk->n", self.m[camp], it)

    def pctr(self, opp, camp, n_uc):
        """Exact click probability; n_uc = the policy's own impressions of camp to that user in the prior 24 h."""
        w = self.world
        z = (w["beta0"] + w["gamma"] * self.affinity(opp, camp) + self.q[camp]
             + w["delta"] * (self.m[camp, self.context[opp]] > 0) - w["phi"] * np.log1p(n_uc)
             + np.array(w["s_state"])[self.state[opp]])
        return sig(z)

    def p_complete(self, opp, camp):
        """Exact probability that a video impression counts as completed (C6); NaN for display."""
        st, fmt, D = self.state[opp], self.format[opp], self.duration[opp]
        x = sig(self.world["gamma"] * self.affinity(opp, camp))
        out = np.full(opp.size, np.nan)
        pre = fmt == 1
        p_skip = np.select([st == 0, st == 1], [sig(0.5 - 3 * x), sig(1.5 - 2 * x)], 0.0) * (D >= 15)
        out[pre] = ((1 - p_skip) * P_COMPLETE_PREROLL[st])[pre]
        feed = fmt == 2
        if feed.any():
            prm = self.world["infeed"]
            thr = np.minimum(D, 18.0) / D
            p_long = np.where(st == 0, sig(prm["e0"] + 3 * x), sig(prm["c0"] + 2 * x))
            pc = p_long * B.sf(thr, prm["aL"], prm["bL"]) + (1 - p_long) * B.sf(thr, prm["aS"], prm["bS"])
            out[feed] = np.where(st == 2, 1.0, pc)[feed]
        return out

    def expected(self, opp, camp, n_uc=None):
        """Per-impression expected clicks, value and costs C1, C2, C4, C5, C6 (C3 depends on the policy's
        bids and is computed by the evaluation harness)."""
        opp, camp = np.asarray(opp), np.asarray(camp)
        if n_uc is None:
            n_uc = fatigue(self.user[opp], camp, self.ts[opp], self.v.size)
        p = self.pctr(opp, camp, n_uc)
        m = self.m_cpm[camp]
        lh = np.floor(np.mod(self.ts[opp] / 3600 + self.tz[self.user[opp]], 24)).astype(int)
        pc = self.p_complete(opp, camp)
        video = self.format[opp] > 0
        return pd.DataFrame({
            "opp_id": opp, "campaign_id": camp, "pctr": p, "value": p * self.v[camp],
            "cost_C1": m, "cost_C2": p * self.v[camp], "cost_C4": m * self.g[lh], "cost_C5": m,
            "cost_C6": np.where(video, pc * m / self.world["r_bar"], m),
            "absent": self.state[opp] == 2,
        })


def fatigue(user, campaign, ts, C):
    key = user.astype(np.int64) * C + campaign
    val = key * 10_000_000 + np.floor(ts).astype(np.int64)
    order = np.argsort(val, kind="stable")
    sv = val[order]
    left = np.searchsorted(sv, sv - 86400, side="right")
    n = np.empty(ts.size, np.int64)
    n[order] = np.arange(ts.size) - left
    return n


def wasted_spend(expected_df, cost="cost_C1"):
    """Share of each campaign's expected spend on impressions whose true state is absent."""
    g = expected_df.groupby("campaign_id")
    return (g.apply(lambda d: d.loc[d.absent, cost].sum() / d[cost].sum(), include_groups=False)
            .rename("wasted_share"))
