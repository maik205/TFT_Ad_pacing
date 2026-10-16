"""Provenance record written next to every results file (PROTOCOL 8)."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from . import config as C
from .data import p

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def code_version():
    try:
        return subprocess.check_output(["git", "-C", CODE_DIR, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        stamp = os.path.join(CODE_DIR, "CODE_VERSION")
        return open(stamp).read().strip() if os.path.exists(stamp) else "unknown"


def write_meta(name, **extra):
    meta = {
        "name": name,
        "utc": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "host": platform.node(),
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
        **extra,
    }
    with open(f"{p('results')}/{name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
