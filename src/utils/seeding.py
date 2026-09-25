"""Seed control and run-config logging for reproducible experiments.

Call set_seed() at the start of every entry point, and log_run_config() once the
config is resolved, so each run is reconstructable from (commit + config).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    With deterministic=True, also request deterministic algorithms where the
    framework supports them. Full determinism is not always achievable on GPU;
    document the residual nondeterminism rather than pretending it away.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' outside a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    versions = {"python": sys.version.split()[0]}
    for dist in ("numpy", "torch", "torchvision", "transformers", "accelerate",
                 "peft", "datasets", "scikit-learn", "scipy", "pandas"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            pass
    return versions


def log_run_config(config: Any, run_dir, seed: int | None = None) -> Path:
    """Dump resolved config + git hash + versions + seed to run_dir/config.json.

    `config` may be a dict or a dataclass. Returns the path written. Each run
    gets its own directory; nothing is overwritten.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_dataclass(config):
        config = asdict(config)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_hash(),
        "seed": seed,
        "versions": _package_versions(),
        "config": config,
    }
    out = run_dir / "config.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out
