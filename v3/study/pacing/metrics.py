"""Metrics of a harness run (study protocol Section 8). EVALUATION CODE."""
import numpy as np
import pandas as pd

from .observed import DAY


def planned_curve(obs):
    """Share of a campaign's local day traffic by the end of each local hour (training days, all
    opportunities): the even-pacing plan that smoothness is measured against. Shape (C, 24)."""
    tr = obs.ts < obs.train_days * DAY
    out = np.zeros((obs.C, 24))
    for c in range(obs.C):
        h = np.floor(np.mod(obs.ts[tr] / 3600 + obs.camp_tz[c], 24)).astype(int)
        cnt = np.bincount(h, minlength=24).astype(float)
        out[c] = np.cumsum(cnt) / cnt.sum()
    return out


def campaign_days(res, obs, plan=None):
    """One row per complete campaign-local day inside the run's days (straddling days are dropped)."""
    imps, B = res["impressions"], res["budget"]
    t0, t1 = res["days"][0] * DAY, res["days"][1] * DAY
    rows = []
    for c in range(obs.C):
        tz = obs.camp_tz[c]
        for d in range(-1, obs.days + 1):
            s, e = d * DAY - 3600 * tz, (d + 1) * DAY - 3600 * tz
            if s >= t0 and e <= t1:
                rows.append((c, d, s))
    cd = pd.DataFrame(rows, columns=["campaign", "local_day", "start"])
    if len(imps):
        g = imps.assign(absent_cost=imps["cost"] * imps["absent"]).groupby(["campaign", "local_day"])
        agg = g.agg(value=("value", "sum"), exp_clicks=("pctr", "sum"), clicks=("clicked", "sum"),
                    spend=("cost", "sum"), absent_spend=("absent_cost", "sum"), impressions=("opp", "size"))
        cd = cd.merge(agg.reset_index(), on=["campaign", "local_day"], how="left").fillna(0.0)
    cd["budget"] = B[cd["campaign"]]
    cd["utilisation"] = cd["spend"] / cd["budget"]
    # depletion hour and smoothness from the cumulative spend by local hour
    plan = planned_curve(obs) if plan is None else plan
    dep, mad = np.full(len(cd), np.nan), np.zeros(len(cd))
    if len(imps):
        lh = np.floor((imps["ts"].to_numpy() + 3600 * obs.camp_tz[imps["campaign"].to_numpy()]) / 3600
                      ).astype(int) % 24
        byh = (imps.assign(h=lh).groupby(["campaign", "local_day", "h"])["cost"].sum())
        for i, (c, d, b) in enumerate(zip(cd["campaign"], cd["local_day"], cd["budget"])):
            try:
                hv = byh.loc[(c, d)]
            except KeyError:
                continue
            cum = np.cumsum(np.bincount(hv.index.to_numpy(), weights=hv.to_numpy(), minlength=24)) / b
            hit = np.flatnonzero(cum >= 0.999)
            dep[i] = hit[0] + 1 if hit.size else np.nan
            mad[i] = np.mean(np.abs(cum - plan[c] * min(cum[-1], 1.0)))
    cd["depletion_hour"] = dep
    cd["early_depletion"] = dep < 23
    cd["smoothness_mad"] = mad
    return cd


def summary(cd):
    return {
        "campaign_days": int(len(cd)),
        "value": float(cd["value"].sum()),
        "expected_clicks": float(cd["exp_clicks"].sum()),
        "spend": float(cd["spend"].sum()),
        "utilisation_mean": float(cd["utilisation"].mean()),
        "overspend_max": float((cd["utilisation"] - 1).clip(lower=0).max()),
        "early_depletion_share": float(cd["early_depletion"].mean()),
        "smoothness_mad_mean": float(cd["smoothness_mad"].mean()),
        "wasted_spend_share": float(cd["absent_spend"].sum() / max(cd["spend"].sum(), 1e-300)),
    }
