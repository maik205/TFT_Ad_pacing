"""Forecast accuracy on the test origins for every saved forecast file (PROTOCOL 7)."""

import glob
import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing import metrics as M
from pacing.data import load_work, p, spend_matrix
from pacing.forecasters import STEPS
from pacing.provenance import write_meta

panel, campaigns, stat = load_work()
Y = spend_matrix(panel, campaigns)
k = {q: list(C.QUANTILES).index(q) for q in C.QUANTILES}

rows = []
for path in sorted(glob.glob(f"{p('results')}/forecasts/*.npz")):
    name = os.path.basename(path)[:-4]
    z = np.load(path)
    F, origins = z["F"].astype(np.float64), z["origins"]
    assert np.array_equal(z["campaigns"], np.array(campaigns)), f"{name}: campaign order differs"
    tgt = origins[:, None] + STEPS[None, :]
    ok = tgt < C.N_HOURS                               # targets that exist in the data
    for c in range(len(campaigns)):
        y = Y[c, np.minimum(tgt, C.N_HOURS - 1)][ok]
        f = F[c][ok]                                   # (n_points, n_quantiles)
        scale = M.seasonal_naive_scale(Y[c, :C.TRAIN_END])
        level = Y[c, :C.TRAIN_END].mean()
        rows.append({
            "forecast": name, "campaign": campaigns[c],
            "mase": M.mase(y, f[:, k[0.5]], scale),
            **{f"spl_{q}": M.pinball(y, f[:, k[q]], q) / level for q in (0.1, 0.5, 0.9)},
            "coverage80": M.coverage(y, f[:, k[0.1]], f[:, k[0.9]]),
            "crossing": M.crossing_rate(f),
        })

df = pd.DataFrame(rows)
df["model"] = df.forecast.str.replace(r"_seed\d+$", "", regex=True)
df.to_parquet(f"{p('results')}/forecast_metrics_per_campaign.parquet", index=False)

summary = {}
for model, g in df.groupby("model"):
    per_seed = g.groupby("forecast")[["mase", "spl_0.1", "spl_0.5", "spl_0.9", "coverage80", "crossing"]].mean()
    summary[model] = {
        "n_seeds": int(len(per_seed)),
        "mean": per_seed.mean().to_dict(),
        "std_over_seeds": per_seed.std(ddof=1).fillna(0).to_dict(),
        "median_mase_over_campaigns": float(g.groupby("campaign").mase.mean().median()),
    }
with open(f"{p('results')}/forecast_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
write_meta("forecast_metrics")
print(json.dumps(summary, indent=2))
