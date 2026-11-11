"""Spec Section 15 test 9 on a full-size generated world: fits hit their targets. The in-feed fit of
the frozen spec does not apply to the fork, which has no in-feed slot (FORK_SPEC.md, deviation F4).

    python tests/check_world.py OUT_DIR [--gamma-real]
"""
import json
import os
import sys

import numpy as np
import pandas as pd


def main(world_dir, gamma_real=False):
    meta = json.load(open(os.path.join(world_dir, "observed", "synth_meta.json")))
    ch, fails = meta["checks"], []
    imp = pd.read_parquet(os.path.join(world_dir, "observed", "synth_impressions.parquet"),
                          columns=["campaign_id", "cost_C5"]).dropna()
    lc = np.log(imp["cost_C5"].to_numpy())
    sd = np.std(lc - imp.assign(l=lc).groupby("campaign_id")["l"].transform("mean").to_numpy())
    if abs(sd / 0.865 - 1) > 0.02:
        fails.append(f"C5 within-campaign log-cost sd {sd:.4f} vs 0.865")
    if gamma_real:
        ex = ch["mi_excess_bits"]["excess"]
        if abs(ex / 0.0255 - 1) > 0.10:
            fails.append(f"MI excess {ex:.4f} vs 0.0255")
    print(json.dumps({"world": world_dir, "config_hash": meta["config_hash"], "fails": fails}, indent=1))
    return not fails


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], "--gamma-real" in sys.argv) else 1)
