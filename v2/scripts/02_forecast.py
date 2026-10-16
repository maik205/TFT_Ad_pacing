"""Produce rolling-origin test forecasts for one forecaster/seed.

usage: 02_forecast.py {naive,lightgbm,TFT,NHITS} [--seed S] [--accelerator cpu|mps] [--max-steps N]
"""

import argparse
import os
import time

import numpy as np

from pacing.data import load_work, p, spend_matrix
from pacing.forecasters import lightgbm_forecast, neural_forecast, seasonal_naive, test_origins
from pacing.provenance import write_meta

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["naive", "lightgbm", "TFT", "NHITS"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--accelerator", default="cpu")
ap.add_argument("--max-steps", type=int, default=3000)
args = ap.parse_args()

panel, campaigns, stat = load_work()
Y = spend_matrix(panel, campaigns)
origins = test_origins()

t0 = time.time()
if args.model == "naive":
    F, info = seasonal_naive(Y, origins), {}
elif args.model == "lightgbm":
    F, info = lightgbm_forecast(Y, stat, seed=args.seed)
else:
    F, info = neural_forecast(args.model, panel, campaigns, stat, seed=args.seed,
                              accelerator=args.accelerator, max_steps=args.max_steps)
info["total_seconds"] = time.time() - t0

name = f"{args.model}_seed{args.seed}"
os.makedirs(f"{p('results')}/forecasts", exist_ok=True)
np.savez_compressed(f"{p('results')}/forecasts/{name}.npz", F=F.astype(np.float32),
                    origins=origins, campaigns=np.array(campaigns))
write_meta(f"forecasts/{name}", **info)
print(name, info)
