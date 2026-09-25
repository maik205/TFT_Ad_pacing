"""Determinism smoke test: same seed -> same result.

Point this at your real pipeline: implement run_once(seed) -> dict[str, float]
(a tiny train+eval that returns metrics) in src/train.py, and this asserts two
runs match exactly. Until that hook exists, it verifies the seeding utility
itself is deterministic.

    pytest tests/test_determinism.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.seeding import set_seed  # noqa: E402

try:
    # TODO: implement run_once(seed) in your training entry point.
    from src.train import run_once  # type: ignore
    HAVE_PIPELINE = True
except Exception:
    HAVE_PIPELINE = False


def _seeded_compute(seed: int) -> dict:
    """Fallback proxy: a seeded computation that must be reproducible."""
    set_seed(seed)
    import random
    out = {"py_sum": round(sum(random.random() for _ in range(1000)), 10)}
    try:
        import numpy as np
        out["np_mean"] = round(float(np.random.rand(1000).mean()), 10)
    except ImportError:
        pass
    try:
        import torch
        set_seed(seed)
        out["torch_sum"] = round(float(torch.rand(1000).sum()), 6)
    except ImportError:
        pass
    return out


def test_same_seed_same_result():
    run = run_once if HAVE_PIPELINE else _seeded_compute
    a = run(1234)
    b = run(1234)
    assert a == b, f"Nondeterministic under a fixed seed:\n{a}\n!=\n{b}"


def test_different_seed_differs():
    run = run_once if HAVE_PIPELINE else _seeded_compute
    assert run(1) != run(2), "Results identical across seeds; seeding may be a no-op."
