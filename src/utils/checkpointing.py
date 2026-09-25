"""Checkpoint saving/loading and best-model tracking, with file provenance.

The checkpoint that produced a reported number must be saved and referenced, so a
table entry traces to specific weights, not just a config. Pair save_checkpoint()
with results.log_metrics(..., checkpoint=path).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def sha256_of(path) -> str:
    """Return the SHA-256 of a file, for checkpoint provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(state: dict, run_dir, tag: str = "best") -> Path:
    """Save a checkpoint dict to run_dir/checkpoints/<tag>.pt and return its path.

    `state` should bundle everything needed to resume or evaluate, e.g.:
        {"model": model.state_dict(), "optimizer": opt.state_dict(),
         "epoch": e, "seed": seed, "metric": best_val}
    Weights live under runs/ and must not be committed to git.
    """
    import torch
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{tag}.pt"
    torch.save(state, path)
    return path


def load_checkpoint(path, map_location: Optional[str] = "cpu") -> dict:
    """Load a checkpoint dict saved by save_checkpoint()."""
    import torch
    return torch.load(path, map_location=map_location)


class BestTracker:
    """Track the best value of a monitored metric to decide when to checkpoint.

    Use it to save the best-on-validation checkpoint and then evaluate the test
    set with exactly that checkpoint, so you never tune on test:

        tracker = BestTracker(mode="max")            # e.g. macro-F1
        ...
        if tracker.update(val_metric, step=epoch):    # True when improved
            save_checkpoint(state, run_dir, tag="best")
    """

    def __init__(self, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self.best: Optional[float] = None
        self.best_step: Optional[int] = None

    def update(self, value: float, step: Optional[int] = None) -> bool:
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if improved:
            self.best = value
            self.best_step = step
        return improved
