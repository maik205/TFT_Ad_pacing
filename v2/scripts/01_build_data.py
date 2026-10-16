"""Build the hourly panel, campaign list, static features and impression store."""

import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import build_panel, load_impressions, p, select_campaigns, static_features
from pacing.provenance import write_meta

os.makedirs(p("work"), exist_ok=True)
os.makedirs(p("results"), exist_ok=True)

imp = load_impressions()
panel = build_panel(imp)
campaigns = select_campaigns(panel)
stat = static_features(panel)

panel.to_parquet(f"{p('work')}/panel.parquet", index=False)
pd.DataFrame({"campaign": campaigns}).to_parquet(f"{p('work')}/campaigns.parquet", index=False)
stat.to_parquet(f"{p('work')}/static.parquet")

# Impressions of selected campaigns in val+test hours, for the replay environment
keep = imp.campaign.isin(campaigns) & (imp.hour >= C.TRAIN_END) & (imp.hour < C.TEST_END)
imp.loc[keep, ["campaign", "hour", "cost", "click", "attribution"]].to_parquet(
    f"{p('work')}/impressions_eval.parquet", index=False)

sel = panel[panel.campaign.isin(campaigns)]
summary = {
    "n_impressions": int(len(imp)),
    "n_campaigns_total": int(imp.campaign.nunique()),
    "n_campaigns_selected": len(campaigns),
    "hours": [int(imp.hour.min()), int(imp.hour.max())],
    "selected_share_of_total_spend": float(sel.spend.sum() / panel.spend.sum()),
    "selected_share_of_impressions": float(sel.imps.sum() / panel.imps.sum()),
    "n_eval_impressions": int(keep.sum()),
    "test_episodes": len(campaigns) * (C.TEST_END - C.VAL_END) // 24,
}
with open(f"{p('results')}/data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
write_meta("data_summary")
print(json.dumps(summary, indent=2))
