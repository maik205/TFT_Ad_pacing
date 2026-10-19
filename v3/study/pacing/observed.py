"""Observed data of a world: the only door through which policies and models see it (generator spec R2)."""
import json
import os

import numpy as np
import pandas as pd

DAY = 86400
FORMATS = ("display", "preroll", "infeed")
COST_MODELS = ("C1", "C2", "C3", "C4", "C5", "C6")


def campaign_local_day(ts, tz):
    return np.floor((ts + 3600.0 * tz) / DAY).astype(np.int64)


class Observed:
    def __init__(self, world_dir):
        obs = os.path.join(world_dir, "observed")
        o = pd.read_parquet(os.path.join(obs, "synth_opportunities.parquet"))
        assert (o["opp_id"].to_numpy() == np.arange(len(o))).all()
        self.n = len(o)
        self.ts = o["ts_utc"].to_numpy()
        self.user = o["user_id"].to_numpy()
        self.visit = o["visit_id"].to_numpy()
        self.region = o["region_obs"].fillna(-1).to_numpy().astype(np.int64)  # -1 = not observed
        self.cats = list(o["context_category"].cat.categories)
        self.context = o["context_category"].cat.codes.to_numpy().astype(np.int64)
        self.slot = o["slot_index"].to_numpy()
        assert tuple(o["format"].cat.categories) == FORMATS
        self.format = o["format"].cat.codes.to_numpy().astype(np.int64)  # 0 display, 1 preroll, 2 infeed
        self.duration = o["video_duration_s"].to_numpy()
        self.n_users = int(self.user.max()) + 1

        r = pd.read_parquet(os.path.join(obs, "synth_regions.parquet"))
        self.region_tz = r.sort_values("region_code")["tz_offset"].to_numpy()

        c = pd.read_parquet(os.path.join(obs, "synth_campaigns.parquet")).sort_values("campaign_id")
        self.C = len(c)
        self.v = c["value_per_click"].to_numpy()
        self.m_cpm = c["cpm_unit_cost"].to_numpy()
        self.camp_tz = c["advertiser_tz"].to_numpy()
        # campaign category weights over context categories: 1 for the main category, 0.5 for a second
        self.camp_cats = np.zeros((self.C, len(self.cats)))
        self.camp_cats[np.arange(self.C), [self.cats.index(x) for x in c["category"]]] = 1.0
        for i, x in enumerate(c["category2"]):
            if isinstance(x, str):
                self.camp_cats[i, self.cats.index(x)] = 0.5
        self.budgets = {col[len("budget_"):]: c[col].to_numpy() for col in c.columns if col.startswith("budget_")}

        # the logged (training) data
        imp = pd.read_parquet(os.path.join(obs, "synth_impressions.parquet"))
        self.log_choice = imp["campaign_id"].fillna(-1).to_numpy().astype(np.int64)
        self.log_propensity = imp["propensity"].to_numpy()
        self.log_cost = {k: imp[f"cost_{k}"].to_numpy() for k in COST_MODELS}
        shown = pd.read_parquet(os.path.join(obs, "synth_clicks.parquet"))
        self.log_clicked = np.zeros(self.n, bool)
        self.log_clicked[shown["opp_id"].to_numpy()] = shown["clicked"].to_numpy()
        self.engagement = pd.read_parquet(os.path.join(obs, "synth_engagement.parquet"))

        meta = json.load(open(os.path.join(obs, "synth_meta.json")))
        self.days = meta["config"]["days"]
        self.train_days = meta["config"]["train_days"]
        self.steps_per_hour = meta["config"]["pacing_steps_per_hour"]
        self.config_hash = meta["config_hash"]
        self.world_seed = meta["world_seed"]

    def budget(self, cost, rho):
        return self.budgets[f"{cost}_rho{rho}"]
