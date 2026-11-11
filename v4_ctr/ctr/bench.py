"""CPU latency of one scoring call through ONNX Runtime (NFR-33: p99 <= 50 ms for all candidates of one
request in one call). The ad server measures the same thing in production; this is the offline check.

    python -m ctr.bench export/ctr-cadet.onnx [--threads 1]
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort

from . import contract as K


def request(rng, n):
    L = K.HISTORY_LEN
    return {
        "ctx": np.array([[2, 5, 7, 13, 3, 6]], np.int64),
        "hist_cat": rng.integers(0, K.N_CAT_TOKENS, (1, L, 2)).astype(np.int64),
        "hist_fmt": rng.integers(1, 4, (1, L)).astype(np.int64),
        "hist_click": rng.integers(1, 4, (1, L)).astype(np.int64),
        "hist_age_h": rng.uniform(0, 200, (1, L)).astype(np.float32),
        "cand_cat": rng.integers(2, K.N_CAT_TOKENS, (n, 2)).astype(np.int64),
        "cand_fatigue": rng.integers(0, 4, n).astype(np.float32),
    }


def bench(path, threads, sizes=(1, 10, 40, 100), runs=2000):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    out = {}
    for n in sizes:
        req = request(rng, n)
        for _ in range(50):
            sess.run(["pctr"], req)
        t = np.empty(runs)
        for i in range(runs):
            s = time.perf_counter()
            sess.run(["pctr"], req)
            t[i] = time.perf_counter() - s
        out[n] = {"p50_ms": float(np.percentile(t, 50) * 1e3), "p99_ms": float(np.percentile(t, 99) * 1e3),
                  "max_ms": float(t.max() * 1e3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    print(json.dumps({"model": a.model, "threads": a.threads, "candidates": bench(a.model, a.threads)}, indent=1))


if __name__ == "__main__":
    main()
