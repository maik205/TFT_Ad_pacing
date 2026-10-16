"""Grid-search PID gains on validation days, separately per budget level (PROTOCOL 5.3)."""

import itertools
import json

import pandas as pd

from pacing import config as C
from pacing.data import load_work, p
from pacing.env import ImpressionStore
from pacing.policies import PID
from pacing.provenance import write_meta
from pacing.simulate import simulate

KP = (0.03125, 0.125, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)
KI = (0, 0.05, 0.2, 0.5, 1)
KD = (0, 0.5, 2, 8, 32)
VAL_DAYS = list(range(C.TRAIN_END // 24, C.VAL_END // 24))
SEEDS = [0, 1, 2]

panel, campaigns, stat = load_work()
daily = stat.mean_daily_spend.to_numpy()
imp = pd.read_parquet(f"{p('work')}/impressions_eval.parquet")
store = ImpressionStore(imp, campaigns)


def factory():
    return [PID(daily, kp, ki, kd) for kp, ki, kd in itertools.product(KP, KI, KD)]


rows = pd.DataFrame(simulate(store, daily, factory, range(C.MAX_CAMPAIGNS or len(campaigns)), VAL_DAYS, SEEDS))
rows = rows[rows.policy.str.startswith("PID")]
score = rows.groupby(["rho", "policy"]).apply(lambda g: g.clicks.sum() / g.ref_clicks.sum(), include_groups=False)
best = {}
for rho, s in score.groupby(level=0):
    name = s.droplevel(0).idxmax()
    gains = [float(x.split("=")[1]) for x in name[4:-1].split(",")]
    best[str(rho)] = {"gains": gains, "val_click_ratio": float(s.max())}
score.rename("val_click_ratio").reset_index().to_csv(f"{p('results')}/pid_grid.csv", index=False)
with open(f"{p('results')}/pid_tuned.json", "w") as f:
    json.dump(best, f, indent=2)
write_meta("pid_tuned", val_days=VAL_DAYS, seeds=SEEDS, grid={"kp": KP, "ki": KI, "kd": KD})
print(json.dumps(best, indent=2))
