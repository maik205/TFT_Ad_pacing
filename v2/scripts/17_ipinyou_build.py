"""Build the iPinYou season-2 impression table for an impression replay (PROTOCOL 10.20).

    python 17_ipinyou_build.py [OUT_DIR]

Reads the 7 training days (2013-06-06 to 2013-06-12) of imp.*.txt.bz2 and clk.*.txt.bz2 from the season-2 zip, streamed.
Column layout (checked with v3/study/ipy_peek.py): 0 bid id, 1 timestamp YYYYMMDDHHmmssSSS (China local time),
20 paying price (per 1000 impressions), 22 advertiser. An impression is a click when its bid id appears in any click log.
Writes impressions.parquet (campaign = advertiser id, hour = hours since 2013-06-06 00:00, cost = paying price / 1000,
click, attribution = 0) and summary.json. Only the won (displayed) impressions are available, as in the Criteo logs."""

import bz2
import json
import os
import sys
import zipfile

import pandas as pd

Z = os.path.expanduser("~/datasets/ipinyou/season2.zip")
P = "ipinyou.contest.dataset-season2/training2nd/"
DAYS = [f"201306{d:02d}" for d in range(6, 13)]
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/ipinyou_work")
os.makedirs(OUT, exist_ok=True)
z = zipfile.ZipFile(Z)


def read(member, cols, names):
    parts = []
    with z.open(member) as f:
        for ch in pd.read_csv(bz2.open(f), sep="\t", header=None, usecols=cols, quoting=3, dtype=str,
                              on_bad_lines="skip", chunksize=1_000_000):
            ch = ch[cols]
            ch.columns = names
            parts.append(ch)
    return pd.concat(parts, ignore_index=True)


clicks = set()
for d in DAYS:
    c = read(f"{P}clk.{d}.txt.bz2", [0], ["bidid"])
    clicks.update(c.bidid)
    print("clicks", d, len(c), flush=True)

frames = []
for d in DAYS:
    imp = read(f"{P}imp.{d}.txt.bz2", [0, 1, 20, 22], ["bidid", "ts", "pay", "adv"])
    day = (pd.to_datetime(imp.ts.str[:8], format="%Y%m%d", errors="coerce") - pd.Timestamp("2013-06-06")).dt.days
    hour = pd.to_numeric(imp.ts.str[8:10], errors="coerce")
    out = pd.DataFrame({
        "campaign": pd.to_numeric(imp.adv, errors="coerce"),
        "hour": day * 24 + hour,
        "cost": pd.to_numeric(imp.pay, errors="coerce") / 1000.0,
        "click": imp.bidid.isin(clicks).astype("int64"),
    }).dropna()
    out["campaign"] = out.campaign.astype("int64")
    out["hour"] = out.hour.astype("int32")
    out["attribution"] = 0
    frames.append(out)
    print("imp", d, len(out), "clicks", int(out.click.sum()), flush=True)
imp = pd.concat(frames, ignore_index=True)
imp = imp[(imp.hour >= 0) & (imp.hour < 168)]
imp.to_parquet(os.path.join(OUT, "impressions.parquet"), index=False)
g = imp.groupby("campaign").agg(impressions=("cost", "size"), spend=("cost", "sum"), clicks=("click", "sum"))
json.dump({"rows": int(len(imp)), "hours": 168, "advertisers": {str(k): {c: float(v) for c, v in r.items()} for k, r in g.iterrows()}},
          open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(g)
