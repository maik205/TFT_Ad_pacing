"""Oracle policies O-P1 and O-P4 and the hindsight upper bound UB (study protocol Section 4).
EVALUATION CODE: these use the planted truth (true click probability, true presence). Learned methods
P1-P5 must not import this module (generator spec R2).

All three share the decision rule of Section 3 with `p_hat` = true pCTR and `a_hat` = 0 (true pCTR already
contains the attention term):  score_c = v_c * p - lambda_c * scale_c * cost_c;  show argmax if > 0.
`lambda_c` is dimensionless: `scale_c` = (value per click x logged CTR) / (logged mean cost per impression
of c), both observed, so lambda = 1 is roughly break-even for an average impression.
"""
import numpy as np
import pandas as pd

from . import harness  # noqa: F401  (puts the generator on sys.path)
from synth.opportunities import local_time  # noqa: E402
from synth.world import rhythm_table  # noqa: E402
from synth_truth.oracle import Oracle  # noqa: E402

from .metrics import planned_curve
from .observed import DAY, campaign_local_day
from .policies import Policy

sig = lambda x: 1.0 / (1.0 + np.exp(-x))
LAGS = 7  # scenario pool: the previous 7 days (Section 4.2)


class Truth:
    """Exact planted click probability and expected cost of every (opportunity, campaign) pair, as matrices.
    Same formulas as `synth_truth.oracle.Oracle.expected` (checked in selftest.py)."""

    def __init__(self, world_dir, cost):
        self.o, self.cost = Oracle(world_dir), cost
        o = self.o
        self.C = o.v.size
        self.s_state = np.array(o.world["s_state"])
        self.lh = np.floor(np.mod(o.ts / 3600 + o.tz[o.user], 24)).astype(np.int64)
        self.video = o.format > 0

    def pk(self, opp, n_uc=None, dtype=np.float64):
        """Returns P (n, C) true pCTR and K (n, C) expected cost; n_uc (n, C) = prior 24 h impressions of c to u."""
        o, w = self.o, self.o.world
        it = np.asarray(o.interest[o.visit[opp]], dtype=np.float64)
        aff = it @ o.m.T
        z = (w["beta0"] + w["gamma"] * aff + o.q[None, :] + w["delta"] * (o.m[:, o.context[opp]].T > 0)
             + self.s_state[o.state[opp]][:, None])
        if n_uc is not None:
            z = z - w["phi"] * np.log1p(n_uc)
        p = sig(z)
        m = o.m_cpm[None, :]
        if self.cost in ("C1", "C5"):
            K = np.broadcast_to(m, p.shape)
        elif self.cost == "C2":
            K = p * o.v[None, :]
        elif self.cost == "C4":
            K = m * o.g[self.lh[opp]][:, None]
        elif self.cost == "C6":
            n = opp.size
            pc = o.p_complete(np.repeat(opp, self.C), np.tile(np.arange(self.C), n)).reshape(n, self.C)
            K = np.where(self.video[opp][:, None], pc * m / w["r_bar"], m)
        else:
            raise ValueError(f"cost model {self.cost} is not supported by the oracle policies")
        return p.astype(dtype), np.ascontiguousarray(K, dtype=dtype)


def _scales(obs, cost):
    """scale_c from observed logged data only."""
    sh = obs.log_choice >= 0
    c = obs.log_choice[sh]
    p0 = float(obs.log_clicked[sh].mean())
    tot = np.bincount(c, weights=obs.log_cost[cost][sh], minlength=obs.C)
    cnt = np.bincount(c, minlength=obs.C)
    c0 = np.where(cnt > 0, tot / np.maximum(cnt, 1), obs.m_cpm)
    return obs.v * p0 / np.maximum(c0, 1e-12)


class _OraclePolicy(Policy):
    def __init__(self, truth):
        self.truth = truth

    def start(self, obs, ctx):
        super().start(obs, ctx)
        self.B = ctx["budget"]
        self.scale = _scales(obs, ctx["cost"])
        self.lam = np.zeros(obs.C)
        self.h_u, self.h_c, self.h_t = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)

    def _nuc(self, t0, idx):
        """n_uc (n, C) from this policy's own impressions in the preceding 24 h."""
        n, C = idx.size, self.obs.C
        M = np.zeros((n, C))
        if self.h_t.size == 0:
            return M
        keep = self.h_t >= t0 - DAY
        self.h_u, self.h_c, self.h_t = self.h_u[keep], self.h_c[keep], self.h_t[keep]
        users = self.obs.user[idx]
        sel = np.isin(self.h_u, users)
        if not sel.any():
            return M
        left = pd.DataFrame({"pos": np.arange(n), "user": users, "ts": self.obs.ts[idx]})
        right = pd.DataFrame({"user": self.h_u[sel], "camp": self.h_c[sel], "th": self.h_t[sel]})
        m = left.merge(right, on="user")
        m = m[(m["th"] >= m["ts"] - DAY) & (m["th"] < m["ts"])]
        np.add.at(M, (m["pos"].to_numpy(), m["camp"].to_numpy()), 1.0)
        return M

    def decide(self, t0, idx, spend, elig):
        self._pace(t0, spend)
        P, K = self.truth.pk(idx, self._nuc(t0, idx))
        S = self.obs.v[None, :] * P - (self.lam * self.scale)[None, :] * K
        S = np.where(elig[None, :], S, -np.inf)
        a = S.argmax(axis=1)
        return np.where(S[np.arange(idx.size), a] > 0, a, -1)

    def tick(self, t0, spend, elig):
        self._pace(t0, spend)

    def observe(self, t0, opp, camp, feedback):
        self.h_u = np.concatenate([self.h_u, self.obs.user[opp]])
        self.h_c = np.concatenate([self.h_c, camp])
        self.h_t = np.concatenate([self.h_t, self.obs.ts[opp]])

    def _pace(self, t0, spend):
        raise NotImplementedError


class OracleReactive(_OraclePolicy):
    """O-P1: true pCTR, reactive dual descent (Section 4.1). The plan is the cumulative share of the day's
    traffic by local hour on training days (`planned_curve`), interpolated within the hour; the error is the
    cumulative spend minus the cumulative plan, in units of the daily budget."""
    name = "O-P1"

    def __init__(self, truth, eta=50.0, lam0=0.0):
        super().__init__(truth)
        self.eta, self.lam0 = eta, lam0

    def start(self, obs, ctx):
        super().start(obs, ctx)
        self.lam[:] = self.lam0
        self.cum = np.concatenate([np.zeros((obs.C, 1)), planned_curve(obs)], axis=1)  # share at hour edges 0..24

    def _pace(self, t0, spend):
        C = self.obs.C
        lh = np.mod((t0 + 3600.0 * self.obs.camp_tz) / 3600.0, 24.0)
        i = np.floor(lh).astype(int)
        f = lh - i
        plan = self.cum[np.arange(C), i] * (1 - f) + self.cum[np.arange(C), i + 1] * f
        err = (spend - self.B * plan) / self.B
        self.lam = np.maximum(0.0, self.lam + self.eta * err)


class OracleForecast(_OraclePolicy):
    """O-P4 (user_weights=True) and O-P4a (False): true pCTR, forecast-driven multiplier (Section 4.2).

    Every `update_every` steps, lambda_c is set so that the projected spend of c over the rest of its local day
    equals kappa x its remaining budget. The projection replays the same clock window of each of the previous
    7 days (a sample of `q` of the opportunities), scores each opportunity with the true pCTR (fatigue
    ignored) and the current lambda, and sums the cost of those that campaign c wins. User-level weights are
    the true presence ratio rate_u(target time) / rate_u(historic time), where rate = alive x archetype-mixed
    rhythm. Per-campaign solves are exact (sort and cumulative sum), the joint solve is a few Jacobi rounds.
    """
    name = "O-P4"

    MODES = {"none": "O-P4a", "user": "O-P4", "user_norm": "O-P4n", "agg": "O-P4g"}

    def __init__(self, truth, world, user_weights=True, damp=0.7, kappa=1.0, q=0.1, update_every=10, rounds=3,
                 relax=0.6, seed=0, weights=None):
        """weights: "none" (O-P4a), "user" (O-P4, departures only), "agg" (O-P4g: pool scaled by the true expected
        traffic ratio, arrivals included, no user composition) or "user_norm" (O-P4n: user weights renormalised so
        the weighted pool carries that expected traffic). Default follows `user_weights`."""
        super().__init__(truth)
        self.mode = weights or ("user" if user_weights else "none")
        self.world, self.uw, self.damp, self.kappa = world, self.mode != "none", damp, kappa
        self.q, self.every, self.rounds, self.relax, self.seed = q, update_every, rounds, relax, seed
        self.name = self.MODES[self.mode]
        self.first, self.k = True, 0

    def start(self, obs, ctx):
        super().start(obs, ctx)
        lo, hi = ctx["t_start"] - LAGS * DAY, ctx["t_end"]
        cand = np.flatnonzero((obs.ts >= max(lo, 0)) & (obs.ts < hi))
        rng = np.random.default_rng(self.seed)
        samp = cand[rng.random(cand.size) < self.q]
        self.st = obs.ts[samp]
        n = samp.size
        self.V, self.K = np.empty((n, obs.C), np.float32), np.empty((n, obs.C), np.float32)
        for s in range(0, n, 20_000):
            p, k = self.truth.pk(samp[s:s + 20_000])
            self.V[s:s + 20_000] = obs.v[None, :] * p
            self.K[s:s + 20_000] = k
        self.W = np.full((n, LAGS), 1.0 / (LAGS * self.q))
        if self.mode == "user":
            self.W = self._presence(samp) / (LAGS * self.q)
        elif self.mode in ("agg", "user_norm"):
            self.W = self._expected_weights(samp) / (LAGS * self.q)

    def _traffic(self, hours):
        """Expected opportunity rate of the whole population in each integer UTC hour: sum over alive users of
        lambda_u x archetype-mixed rhythm (the generator's visit intensity; slots per visit assumed equal)."""
        us, R = self.world.users, rhythm_table()
        out = np.empty(hours.size)
        for i, h in enumerate(hours):
            tm = (h + 0.5) * 3600.0
            _, dt, g, _ = local_time(np.full(us["tz"].size, tm), us["tz"])
            alive = (us["join"] * 24 <= h + 0.5) & (h + 0.5 < us["leave"] * 24)
            out[i] = (alive * us["lam"] * np.einsum("na,an->n", us["w"], R[:, dt, g])).sum()
        return out

    def _expected_weights(self, samp):
        """Pool weights whose total equals the true expected traffic of the projected hour (arrivals included).
        agg: every item of an hour gets the ratio rho(H, d) = traffic(H + 24 d) / traffic(H).
        user_norm: the user-level presence weights, rescaled per (hour, lag) to carry rho(H, d) x (items in H)."""
        H = np.floor(self.obs.ts[samp] / 3600.0).astype(np.int64)
        h0 = int(H.min())
        N = self._traffic(np.arange(h0, int(H.max()) + 24 * LAGS + 1))
        b = H - h0
        rho = np.stack([N[b + 24 * d] / N[b] for d in range(1, LAGS + 1)], axis=1)
        if self.mode == "agg":
            return rho
        raw = self._presence(samp)
        n_b = np.bincount(b, minlength=N.size - 24 * LAGS).astype(float)
        out = np.empty_like(raw)
        for j in range(LAGS):
            Z = np.bincount(b, weights=raw[:, j], minlength=n_b.size)
            ok = Z[b] > 0
            out[:, j] = np.where(ok, raw[:, j] * rho[:, j] * n_b[b] / np.where(ok, Z[b], 1.0), rho[:, j])
        return out

    def _presence(self, samp):
        us = self.world.users
        R = rhythm_table()
        u = self.obs.user[samp]
        mid = (np.floor(self.obs.ts[samp] / 3600.0) + 0.5) * 3600.0
        _, dt, g, _ = local_time(mid, us["tz"][u])
        T_h = np.einsum("na,an->n", us["w"][u], R[:, dt, g])
        out = np.empty((samp.size, LAGS))
        for d in range(1, LAGS + 1):
            tm = mid + d * DAY
            _, dt2, g2, _ = local_time(tm, us["tz"][u])
            alive = (us["join"][u] * 24 <= tm / 3600.0) & (tm / 3600.0 < us["leave"][u] * 24)
            out[:, d - 1] = alive * np.einsum("na,an->n", us["w"][u], R[:, dt2, g2]) / T_h
        return out

    def _pace(self, t0, spend):
        due = self.k % self.every == 0
        self.k += 1
        if not due:
            return
        new = self._solve(t0, spend)
        self.lam = new if self.first else (1 - self.damp) * self.lam + self.damp * new
        self.first = False

    def _solve(self, t0, spend):
        obs = self.obs
        R = self.kappa * np.maximum(self.B - spend, 0.0)
        out = self.lam.copy()
        for tz in np.unique(obs.camp_tz):
            cs = np.flatnonzero(obs.camp_tz == tz)
            t_end = (np.floor((t0 + 3600.0 * tz) / DAY) + 1) * DAY - 3600.0 * tz
            idx, w = [], []
            for d in range(1, LAGS + 1):
                lo = np.searchsorted(self.st, t0 - d * DAY, "left")
                hi = np.searchsorted(self.st, t_end - d * DAY, "left")
                if hi > lo:
                    idx.append(np.arange(lo, hi))
                    w.append(self.W[lo:hi, d - 1])
            if not idx:
                continue
            idx, w = np.concatenate(idx), np.concatenate(w)
            Vp, Kp = self.V[idx], self.K[idx]
            lam = self.lam.copy()
            ar = np.arange(idx.size)
            for _ in range(self.rounds):
                S = Vp - (lam * self.scale)[None, :] * Kp
                a = S.argmax(axis=1)
                t1 = np.maximum(S[ar, a], 0.0)
                S[ar, a] = -np.inf
                t2 = np.maximum(S.max(axis=1), 0.0)
                step = lam.copy()
                for c in cs:
                    M = np.where(a == c, t2, t1)
                    step[c] = _dual_lambda(Vp[:, c] - M, Kp[:, c], w, R[c], self.scale[c])
                lam[cs] = self.relax * step[cs] + (1 - self.relax) * lam[cs]
            out[cs] = lam[cs]
        return out


def _dual_lambda(gain, K, w, budget, scale):
    """Smallest lambda >= 0 such that sum_{rows c wins} w*K <= budget, where c wins a row iff
    gain > lambda * scale * K. Exact: sort the win thresholds and take the cumulative cost."""
    pos = (gain > 0) & (K > 0)
    if not pos.any():
        return 0.0
    thr = gain[pos] / (scale * K[pos])
    cost = (w * K)[pos] if np.ndim(w) else w * K[pos]
    o = np.argsort(-thr)
    cum = np.cumsum(cost[o])
    if cum[-1] <= budget:
        return 0.0
    k = int(np.searchsorted(cum, budget, side="right"))  # rows 0..k-1 fit
    return float(thr[o][k])


def upper_bound(truth, obs, cost, rho, days, rounds=40, chunk=25_000):
    """UB: hindsight-optimal allocation with true pCTR (fatigue ignored, which only lowers p, so it is a bound)
    under a separate budget for every campaign-local day. Solved on the Lagrangian dual by coordinate descent
    over the (campaign, day) multipliers. Returns (per campaign-day table, diagnostics).

    Diagnostics include the dual bound L(lambda) = sum_o max(0, max_c score) + sum lambda B, a rigorous upper
    bound on the total value for any lambda >= 0, to compare with the primal total."""
    t0, t1 = days[0] * DAY, days[1] * DAY
    opp = np.flatnonzero((obs.ts >= t0) & (obs.ts < t1))
    n, C = opp.size, obs.C
    V, K = np.empty((n, C), np.float32), np.empty((n, C), np.float32)
    for s in range(0, n, chunk):
        p, k = truth.pk(opp[s:s + chunk])
        V[s:s + chunk] = obs.v[None, :] * p
        K[s:s + chunk] = k
    B = obs.budget(cost, rho)
    scale = _scales(obs, cost)
    dl = campaign_local_day(obs.ts[opp][:, None], obs.camp_tz[None, :])
    d0 = int(dl.min())
    dl = (dl - d0).astype(np.int32)
    nd = int(dl.max()) + 1
    lam = np.zeros((C, nd))
    S = V.copy()
    ar = np.arange(n)
    for _ in range(rounds):
        for c in range(C):
            Sc = S[:, c].copy()
            S[:, c] = -np.inf
            M = np.maximum(S.max(axis=1), 0.0)  # best competing score, or no ad
            S[:, c] = Sc
            gain = V[:, c] - M
            for d in range(nd):
                sel = dl[:, c] == d
                lam[c, d] = _dual_lambda(gain[sel], K[sel, c], 1.0, B[c], scale[c])
            S[:, c] = V[:, c] - (lam[c, dl[:, c]] * scale[c]) * K[:, c]
    S = V - np.stack([lam[c, dl[:, c]] * scale[c] for c in range(C)], axis=1) * K
    a = S.argmax(axis=1)
    best = S[ar, a]
    win = best > 0
    o_w, c_w = np.flatnonzero(win), a[win]
    d_w = dl[o_w, c_w]
    val = np.zeros((C, nd))
    spend = np.zeros((C, nd))
    np.add.at(val, (c_w, d_w), V[o_w, c_w])
    np.add.at(spend, (c_w, d_w), K[o_w, c_w])
    dual = float(np.maximum(S.max(axis=1), 0.0).astype(np.float64).sum()
                 + (lam * scale[:, None] * B[:, None]).sum())
    over = np.maximum(spend / B[:, None], 1.0)  # coordinate descent stops slightly short of feasibility
    val, spend = val / over, spend / over
    tab = pd.DataFrame({"campaign": np.repeat(np.arange(C), nd), "local_day": np.tile(np.arange(nd) + d0, C),
                        "ub_value": val.ravel(), "ub_spend": spend.ravel()})
    diag = {"primal_value": float(val.sum()), "dual_bound": dual, "gap": dual / max(float(val.sum()), 1e-300) - 1,
            "max_utilisation": float((spend / B[:, None]).max())}
    return tab, diag


def score_run(res, obs, ub_tab):
    """Per complete campaign-day: value, UB value and their ratio (study protocol Section 8, primary)."""
    from .metrics import campaign_days
    cd = campaign_days(res, obs)
    cd = cd.merge(ub_tab, on=["campaign", "local_day"], how="left")
    cd["ratio"] = cd["value"] / cd["ub_value"].where(cd["ub_value"] > 0)
    return cd
