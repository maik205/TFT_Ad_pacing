"""Ingest raw logs into processed hourly series + provenance manifest.

    python -m src.data.ingest --source criteo --raw ../../raw_data/criteo_attribution_dataset.tsv.gz
    python -m src.data.ingest --source criteo --raw ... --dry-run   # stats only, writes nothing

Outputs (not committed, see .gitignore): data/processed/<source>_hourly.parquet
Tracked: data/MANIFEST.json (raw file path, size, SHA-256, licence) and
         data/processed_stats/<source>.json (row counts, ranges, per-series stats).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import criteo
from .timesplit import CRITEO_SPLIT

ROOT = Path(__file__).resolve().parents[2]
LICENCES = {
    "criteo": "CC BY-NC-SA 4.0 (https://huggingface.co/datasets/criteo/criteo-attribution-dataset)",
    "ipinyou": "CC0 (figshare 5732328)",
    "auctionnet": "Apache-2.0 (https://github.com/alimama-tech/AuctionNet)",
}


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def update_manifest(source: str, raw: Path) -> dict:
    mpath = ROOT / "data" / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    entry = {
        "file": raw.name,
        "bytes": raw.stat().st_size,
        "sha256": sha256(raw),
        "licence": LICENCES[source],
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest.setdefault(source, {})[raw.name] = entry
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return entry


def criteo_stats(hourly, selected) -> dict:
    sp = CRITEO_SPLIT
    sel = hourly[hourly["campaign"].isin(selected)]
    per = sel.groupby("campaign")["spend"]
    return {
        "hours": [int(hourly["hour"].min()), int(hourly["hour"].max())],
        "n_campaigns_total": int(hourly["campaign"].nunique()),
        "split": {"train": [0, sp.train_end], "val": [sp.train_end, sp.val_end], "test": [sp.val_end, sp.end]},
        "selection_rule": "spend>0 in >=90% of TRAIN hours",
        "n_campaigns_selected": len(selected),
        "selected_spend_share_all_hours": float(sel["spend"].sum() / hourly["spend"].sum()),
        "total_impressions": int(hourly["impressions"].sum()),
        "total_spend": float(hourly["spend"].sum()),
        "per_series_mean_hourly_spend_quantiles": {
            str(q): float(v) for q, v in per.mean().quantile([0.1, 0.5, 0.9]).items()},
        "per_series_zero_hours_frac_median": float(
            sel.assign(z=sel["spend"] == 0).groupby("campaign")["z"].mean().median()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["criteo"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    raw = Path(args.raw).resolve()

    if args.source == "criteo":
        hourly = criteo.load_hourly(str(raw))
        selected = criteo.select_campaigns(hourly, CRITEO_SPLIT, min_active=0.9)
        stats = criteo_stats(hourly, selected)
    print(json.dumps(stats, indent=2))
    if args.dry_run:
        return stats

    entry = update_manifest(args.source, raw)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out / f"{args.source}_hourly.parquet", index=False)
    sdir = ROOT / "data" / "processed_stats"
    sdir.mkdir(parents=True, exist_ok=True)
    stats["raw_sha256"] = entry["sha256"]
    stats["selected_campaigns"] = [int(c) if isinstance(c, (np.integer, int)) else c for c in selected]
    (sdir / f"{args.source}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out / (args.source + '_hourly.parquet')} and {sdir / (args.source + '.json')}")
    return stats


if __name__ == "__main__":
    main()
