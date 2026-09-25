"""Checkpoint provenance + best-model tracking guardrails.

    pytest tests/test_checkpointing.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.checkpointing import (  # noqa: E402
    BestTracker,
    load_checkpoint,
    save_checkpoint,
    sha256_of,
)
from src.utils.results import log_metrics  # noqa: E402


def test_sha256_is_stable(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    assert sha256_of(f) == sha256_of(f)
    assert len(sha256_of(f)) == 64


def test_best_tracker_max():
    t = BestTracker(mode="max")
    assert t.update(0.5, step=0) is True
    assert t.update(0.4, step=1) is False  # not an improvement
    assert t.update(0.7, step=2) is True
    assert t.best == 0.7 and t.best_step == 2


def test_best_tracker_min():
    t = BestTracker(mode="min")
    assert t.update(1.0, step=0) is True
    assert t.update(0.8, step=1) is True
    assert t.update(0.9, step=2) is False


def test_metrics_record_checkpoint_provenance(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake-weights")
    out = log_metrics(tmp_path, "FACR", "Aff-Wild2",
                      {"macro_f1": [0.71, 0.69, 0.72]},
                      checkpoint=str(ckpt), seed=1234)
    d = json.loads(out.read_text())
    assert d["checkpoint"]["sha256"] == sha256_of(ckpt)
    assert d["metrics"]["macro_f1"] == [0.71, 0.69, 0.72]
    assert d["seed"] == 1234


def test_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 2)
    state = {"model": model.state_dict(), "epoch": 3}
    path = save_checkpoint(state, tmp_path, tag="best")
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 3
    for k, v in model.state_dict().items():
        assert torch.equal(loaded["model"][k], v)
