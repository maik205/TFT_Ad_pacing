"""Export a trained model to ONNX with its manifest and a golden file for the ad server.

    python -m ctr.export --run runs/cadet_s0 --data data/default_s0.npz --out export/

Writes:
- `<name>.onnx`: the model, inputs and outputs as in contract.INPUTS / OUTPUTS, N candidates dynamic;
- `manifest.json`: version, sha256 of the ONNX file, the input contract (contract.manifest_schema()), the
  training world and the held-out metrics; marked synthetic (rule R1);
- `golden.json`: server-shaped requests (country, app category, served-ad history as IAB ids, formats and
  ages; candidates as IAB ids and recent views) with the tensors the contract turns them into and the
  probabilities the model returns. The ad server's tests rebuild the tensors from the raw request and
  compare both.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from . import contract as K
from .models import MODELS

TOKEN_IAB = {K.CAT_BASE + K.CATEGORIES.index(c): ids[0] for c, ids in K.CATEGORY_IAB.items()}
SERVER_FMT = {1: "BANNER", 2: "VIDEO", 3: "REWARDED"}


def load(run):
    ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
    m = MODELS[ck["model"]]()
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def request_from_row(data, i, n_cand, rng):
    """One request: row i's context and history, with n_cand candidates (row i's campaign first)."""
    L = K.HISTORY_LEN
    cand = [data["cand_cat"][i]]
    fat = [data["cand_fatigue"][i]]
    for _ in range(n_cand - 1):
        j = rng.integers(0, data["cand_cat"].shape[0])
        cand.append(data["cand_cat"][j])
        fat.append(float(rng.integers(0, 4)))
    t = {
        "ctx": data["ctx"][i:i + 1].astype(np.int64),
        "hist_cat": data["hist_cat"][i:i + 1].astype(np.int64),
        "hist_fmt": data["hist_fmt"][i:i + 1].astype(np.int64),
        "hist_click": data["hist_click"][i:i + 1].astype(np.int64),
        "hist_age_h": data["hist_age_h"][i:i + 1].astype(np.float32),
        "cand_cat": np.stack(cand).astype(np.int64),
        "cand_fatigue": np.array(fat, np.float32),
    }
    assert t["hist_cat"].shape[1] == L
    return t


def raw_request(t, ts_utc):
    """The server-side view of a tensor request: what the ad server knows before building inputs."""
    ctx = t["ctx"][0]
    region = int(ctx[2])
    hist = []
    for k in range(K.HISTORY_LEN):
        if t["hist_fmt"][0, k] == 0:
            break
        click = int(t["hist_click"][0, k])
        hist.append({
            "age_s": float(t["hist_age_h"][0, k]) * 3600.0,
            "categories": [TOKEN_IAB[int(c)] for c in t["hist_cat"][0, k] if c >= K.CAT_BASE],
            "format": SERVER_FMT[int(t["hist_fmt"][0, k])],
            # a recorded click on an ad younger than the outcome delay is masked by the builder
            "clicked": click == K.CLICK_YES or (click == K.CLICK_UNKNOWN and k % 2 == 0),
        })
    return {
        "utc_unix_s": ts_utc,
        "format": SERVER_FMT[int(ctx[0])],
        "country": K.REGION_COUNTRIES[region - 1] if region > 0 else "",
        "app_category": TOKEN_IAB[int(ctx[1])],
        "history": hist,
        "candidates": [{"categories": [TOKEN_IAB[int(c)] for c in cc if c >= K.CAT_BASE],
                        "views_24h": int(f)} for cc, f in zip(t["cand_cat"], t["cand_fatigue"])],
    }


def golden(sess, data, n=24, seed=0):
    """Requests whose raw form maps back to exactly these tensors. Rows are chosen so the UTC timestamp
    reproduces the context: synthetic time starts on a Monday, so it is shifted to a real Monday."""
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(data["split"] == 2)
    base = 1792972800  # 2026-10-26 00:00 UTC, a Monday
    assert time.gmtime(base).tm_wday == 0 and time.gmtime(base).tm_hour == 0
    cases = []
    picked = rng.choice(test, n, replace=False)
    for c, i in enumerate(picked):
        t = request_from_row(data, i, n_cand=1 + c % 6, rng=rng)
        ts = base + float(data["ts"][i]) - 7 * 86400 * np.floor(data["ts"][i] / (7 * 86400))
        # whole seconds so both sides derive the same hour; ages are measured from this instant
        ts = float(np.floor(ts))
        p = sess.run(["pctr"], t)[0]
        cases.append({"raw": raw_request(t, ts),
                      "tensors": {k: v.tolist() for k, v in t.items()},
                      "pctr": p.tolist()})
    return cases


def export(run, data_path, out):
    model, ck = load(run)
    os.makedirs(out, exist_ok=True)
    data = dict(np.load(data_path))
    rng = np.random.default_rng(1)
    sample = request_from_row(data, int(np.flatnonzero(data["split"] == 2)[0]), 3, rng)
    args = tuple(torch.from_numpy(sample[k]) for k in K.INPUTS)
    name = f"ctr-{ck['model']}"
    path = os.path.join(out, f"{name}.onnx")
    torch.onnx.export(model, args, path, input_names=list(K.INPUTS), output_names=list(K.OUTPUTS),
                      dynamic_axes={"cand_cat": {0: "N"}, "cand_fatigue": {0: "N"}, "pctr": {0: "N"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # parity with torch on random requests of several sizes
    worst = 0.0
    for n_cand in (1, 7, 40):
        for _ in range(20):
            i = int(rng.integers(0, data["split"].size))
            t = request_from_row(data, i, n_cand, rng)
            with torch.no_grad():
                pt = model(*(torch.from_numpy(t[k]) for k in K.INPUTS)).numpy()
            po = sess.run(["pctr"], t)[0]
            worst = max(worst, float(np.abs(pt - po).max()))
    if worst > 1e-5:
        raise RuntimeError(f"ONNX output differs from torch by {worst}")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    results = json.load(open(os.path.join(run, "results.json")))
    meta_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), "worlds")
    manifest = {
        "name": name,
        "version": f"{name}-c{K.CONTRACT_VERSION}-{sha[:12]}",
        "file": os.path.basename(path),
        "sha256": sha,
        "synthetic": True,
        "note": ("Trained and evaluated on the synthetic ad-format world only (TFT_Ad_pacing v4_ctr, rule R1). "
                 "Its metrics describe the pipeline on generated data, not real traffic."),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": ck["model"],
        "params": results["params"],
        "train_world": os.path.basename(results["train_world"]),
        "platt": results["platt"],
        "heldout": {w: {k: r["test"][k] for k in ("all", *K.FORMATS) if k in r["test"]}
                    for w, r in results["eval"].items()},
        "torch_onnx_max_abs_diff": worst,
        "contract": K.manifest_schema(),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out, "golden.json"), "w") as f:
        json.dump({"model_version": manifest["version"], "cases": golden(sess, data)}, f)
    print(json.dumps({"version": manifest["version"], "max_abs_diff": worst, "bytes": os.path.getsize(path)}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    export(a.run, a.data, a.out)


if __name__ == "__main__":
    main()
