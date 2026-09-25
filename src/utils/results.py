"""Write per-run results in the schema scripts/make_tables.py consumes, with a
provenance link to the checkpoint that produced them.

A reported number must trace to (commit + config + the evaluated checkpoint).
seeding.log_run_config() covers commit + config; this covers metrics + checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .checkpointing import sha256_of


def log_metrics(run_dir, method: str, dataset: str, metrics: dict,
                checkpoint: Optional[str] = None, seed: Optional[int] = None) -> Path:
    """Write run_dir/metrics.json.

    `metrics` maps a metric name to a list of per-fold (or per-seed) values, e.g.
        {"macro_f1": [0.71, 0.69, 0.72]}
    `checkpoint` is the path to the weights that were evaluated; its path and
    SHA-256 are recorded so the table entry traces to specific weights.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"method": method, "dataset": dataset, "metrics": metrics, "seed": seed}
    if checkpoint is not None:
        ck = Path(checkpoint)
        payload["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_of(ck) if ck.exists() else "missing",
        }
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    return out
