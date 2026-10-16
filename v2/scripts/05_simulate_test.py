"""Run every policy on the test days (evaluated once, after all choices are frozen)."""

import glob
import json
import os

import numpy as np
import pandas as pd

from pacing import config as C
from pacing.data import load_work, p
from pacing.env import ImpressionStore
from pacing.policies import EvenPacing, ForecastThrottle, NoPacing, TunedPID
from pacing.provenance import write_meta
from pacing.simulate import simulate

TEST_DAYS = list(range(C.VAL_END // 24, C.TEST_END // 24))
SEEDS = list(range(C.N_REPLAY_SEEDS))

panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)
with open(f"{p('results')}/pid_tuned.json") as f:
    pid_gains = {float(r): v["gains"] for r, v in json.load(f).items()}

forecasts = {}
for path in sorted(glob.glob(f"{p('results')}/forecasts/*.npz")):
    z = np.load(path)
    assert np.array_equal(z["campaigns"], np.array(campaigns))
    forecasts[os.path.basename(path)[:-4]] = (z["F"], z["origins"])


def factory():
    pols = [NoPacing(), EvenPacing(daily), TunedPID(daily, pid_gains)]
    for name, (F, origins) in forecasts.items():
        for q in C.HEDGE_QUANTILES:
            pols.append(ForecastThrottle(F, origins, q, name))
    return pols


rows = pd.DataFrame(simulate(store, daily, factory, range(C.MAX_CAMPAIGNS or len(campaigns)), TEST_DAYS, SEEDS))
rows.to_parquet(f"{p('results')}/pacing_test_episodes.parquet", index=False)
write_meta("pacing_test_episodes", test_days=TEST_DAYS, seeds=SEEDS, forecasts=sorted(forecasts),
           n_rows=len(rows))
print(rows.groupby(["rho", "policy"]).size().unstack(0).head())
