"""Gate G3 (study protocol Sections 6 and 8.1): the harness reproduces the logged data.

    python validate_harness.py WORLD_DIR

A. Replaying the logged choices (no hard stop) must give exactly the logged true pCTR, fatigue and C1
   spend, and a click count within sampling error of the logged one (different random numbers).
B. Re-running the logging rule (P0) on the training days with the logging budgets must reproduce the
   logged total spend within 0.5%, and its per-campaign-day spend must be no further from the logs than
   two runs of the rule with different seeds are from each other (protocol deviation 1).
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from pacing.harness import World, run
from pacing.metrics import campaign_days, summary
from pacing.observed import DAY, Observed, campaign_local_day
from pacing.policies import EvenThrottle, LoggedActions


def main(world_dir):
    t = time.time()
    obs, world = Observed(world_dir), World(world_dir)
    print(f"loaded in {time.time() - t:.0f} s")
    rep = {}

    # A. logged-action replay over all days
    t = time.time()
    r = run(world, obs, LoggedActions(), cost="C1", rho=0.5, days=(0, obs.days), hard_stop=False)
    imps = r["impressions"].sort_values("opp")
    lg = world.logged
    order = np.argsort(lg["imp_opp"])
    same_opp = np.array_equal(imps["opp"].to_numpy(), lg["imp_opp"][order])
    pctr_err = float(np.abs(imps["pctr"].to_numpy() - lg["pctr"][order]).max()) if same_opp else None
    nuc_eq = bool(np.array_equal(imps["n_uc"].to_numpy(), lg["n_uc"][order])) if same_opp else None
    spend_err = float(np.abs(imps["cost"].to_numpy() - obs.log_cost["C1"][imps["opp"].to_numpy()]).max())
    n, k_log, k_h = len(imps), int(obs.log_clicked.sum()), int(imps["clicked"].sum())
    sd = float(np.sqrt((imps["pctr"] * (1 - imps["pctr"])).sum() * 2))  # sd of the difference
    rep["A"] = {"impressions": n, "same_opportunities": same_opp, "max_abs_pctr_diff": pctr_err,
                "n_uc_identical": nuc_eq, "max_abs_C1_cost_diff": spend_err, "clicks_logged": k_log,
                "clicks_harness": k_h, "click_diff_in_sd": (k_h - k_log) / sd, "seconds": round(time.time() - t)}
    rep["A"]["pass"] = bool(same_opp and pctr_err < 1e-12 and nuc_eq and spend_err == 0
                            and abs(rep["A"]["click_diff_in_sd"]) < 3)

    # B. the logging rule re-run on the training days
    t = time.time()
    r = run(world, obs, EvenThrottle(seed=12345), cost="C1", rho=0.5, days=(0, obs.train_days), hard_stop=False)
    cd = campaign_days(r, obs)
    r2 = run(world, obs, EvenThrottle(seed=54321), cost="C1", rho=0.5, days=(0, obs.train_days), hard_stop=False)
    cd2 = campaign_days(r2, obs)
    sh = obs.log_choice >= 0
    c = obs.log_choice[sh]
    lg_cd = pd.DataFrame({"campaign": c, "local_day": campaign_local_day(obs.ts[sh], obs.camp_tz[c]),
                          "logged": obs.log_cost["C1"][sh]}).groupby(["campaign", "local_day"]).sum()
    m = cd.merge(lg_cd.reset_index(), on=["campaign", "local_day"], how="left").fillna(0.0)
    rel = (m["spend"] - m["logged"]) / m["logged"].where(m["logged"] > 0)
    m2 = m.merge(cd2[["campaign", "local_day", "spend"]].rename(columns={"spend": "spend2"}),
                 on=["campaign", "local_day"], how="left").fillna(0.0)
    noise = ((m2["spend2"] - m2["spend"]) / m2["logged"].where(m2["logged"] > 0)).abs()
    ratio = {"median": float(rel.abs().median() / noise.median()), "p90": float(rel.abs().quantile(0.9) / noise.quantile(0.9))}
    rep["B"] = {"campaign_days": int(len(m)), "total_rel_diff": float(m["spend"].sum() / m["logged"].sum() - 1),
                "per_campaign_day_abs_rel_diff": {"median": float(rel.abs().median()),
                                                  "p90": float(rel.abs().quantile(0.9)),
                                                  "max": float(rel.abs().max())},
                "seed_to_seed_noise": {"median": float(noise.median()), "p90": float(noise.quantile(0.9))},
                "ratio_to_noise": ratio,
                "total_rel_diff_second_seed": float(m2["spend2"].sum() / m2["logged"].sum() - 1),
                "utilisation_harness": float(m["utilisation"].mean()),
                "utilisation_logged": float((m["logged"] / m["budget"]).mean()),
                "summary": summary(cd), "seconds": round(time.time() - t)}
    rep["B"]["pass"] = bool(abs(rep["B"]["total_rel_diff"]) <= 0.005 and max(ratio.values()) <= 1.25)
    print(json.dumps(rep, indent=1, default=float))
    return rep["A"]["pass"] and rep["B"]["pass"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
