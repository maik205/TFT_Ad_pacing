"""Ground-truth evaluation of CTR predictions (EVALUATION CODE ONLY: reads out/truth, rule R2).

    python -m evaluation.ctr_truth WORLDS_DIR RUN_DIR [RUN_DIR ...]

For each run and each world it was evaluated on, compares the test-split predictions with the true click
probability of the planted click model, which no model sees: mean absolute gap, the gap of the mean,
expected log loss under the true probability, and the same for an oracle that predicts the truth. Writes
RUN_DIR/truth_eval.json. Every number is from synthetic worlds (rule R1).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

FORMATS = ["banner", "interstitial", "rewarded"]


def xent(p_true, p):
    """Expected log loss of predictions p when clicks are Bernoulli(p_true)."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(p_true * np.log(p) + (1 - p_true) * np.log(1 - p)))


def compare(p_true, p):
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, 20)
    return {
        "n": int(p.size),
        "mean_abs_gap": float(np.mean(np.abs(p - p_true))),
        "mean_pred": float(p.mean()), "mean_true": float(p_true.mean()),
        "expected_logloss": xent(p_true, p),
        "oracle_expected_logloss": xent(p_true, p_true),
        "base_rate_expected_logloss": xent(p_true, np.full_like(p, p_true.mean())),
        # calibration against the truth: per prediction bin, mean prediction vs mean true probability
        "ece_vs_truth": float(sum(abs(p[i].mean() - p_true[i].mean()) * i.size for i in bins if i.size) / p.size),
        "corr": float(np.corrcoef(p, p_true)[0, 1]),
    }


def main(worlds, runs):
    for run in runs:
        out = {"synthetic": True}
        for f in sorted(os.listdir(run)):
            if not (f.startswith("pred_") and f.endswith(".npz")):
                continue
            world = f[len("pred_"):-len(".npz")]
            pr = dict(np.load(os.path.join(run, f)))
            ti = pd.read_parquet(os.path.join(worlds, world, "truth", "truth_impressions.parquet"),
                                 columns=["opp_id", "pctr"]).set_index("opp_id")
            p_true = ti.loc[pr["opp_id"], "pctr"].to_numpy()
            res = {"all": compare(p_true, pr["pred"].astype(np.float64))}
            for i, fm in enumerate(FORMATS):
                sel = pr["format"] == i + 1
                if sel.any():
                    res[fm] = compare(p_true[sel], pr["pred"][sel].astype(np.float64))
            out[world] = res
        with open(os.path.join(run, "truth_eval.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(run, json.dumps({w: r["all"] for w, r in out.items() if w != "synthetic"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
