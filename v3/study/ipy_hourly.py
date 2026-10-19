"""Hourly counts per advertiser from the iPinYou season-2 training logs (streamed from the zip).

    python ipy_hourly.py OUT_DIR kind:day [kind:day ...]      kind in bid, imp, clk; day like 20130606

Writes OUT_DIR/{kind}.{day}.csv with columns day, hour, adv, n, bid_sum, pay_sum (pay only for imp).
Column layout (checked with ipy_peek.py): bid ts=1 bidprice=18 adv=19; imp ts=1 bidprice=19 pay=20 adv=22;
clk ts=1 adv=22.
"""
import bz2
import os
import sys
import zipfile

import pandas as pd

Z = r"C:\Users\maik\datasets\ipinyou\season2.zip"
P = "ipinyou.contest.dataset-season2/training2nd/"
COLS = {"bid": {"ts": 1, "bidp": 18, "adv": 19}, "imp": {"ts": 1, "bidp": 19, "pay": 20, "adv": 22},
        "clk": {"ts": 1, "adv": 22}}


def one(z, kind, day, out):
    cols = COLS[kind]
    names = list(cols)
    acc = []
    with z.open(f"{P}{kind}.{day}.txt.bz2") as f:
        rd = pd.read_csv(bz2.open(f), sep="\t", header=None, usecols=list(cols.values()), quoting=3, dtype=str,
                         on_bad_lines="skip", chunksize=1_000_000)
        for ch in rd:
            ch = ch[list(cols.values())]
            ch.columns = [n for n, _ in sorted(cols.items(), key=lambda kv: kv[1])]
            ch["day"] = ch["ts"].str[:8]
            ch["hour"] = pd.to_numeric(ch["ts"].str[8:10], errors="coerce")
            for c in ("bidp", "pay"):
                ch[c] = pd.to_numeric(ch[c], errors="coerce") if c in ch else 0.0
            g = ch.groupby(["day", "hour", "adv"]).agg(n=("ts", "size"), bid_sum=("bidp", "sum"), pay_sum=("pay", "sum"))
            acc.append(g)
    res = pd.concat(acc).groupby(level=[0, 1, 2]).sum().reset_index()
    res.to_csv(os.path.join(out, f"{kind}.{day}.csv"), index=False)
    print(kind, day, len(res), "rows, total", int(res.n.sum()), flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    z = zipfile.ZipFile(Z)
    for item in sys.argv[2:]:
        kind, day = item.split(":")
        one(z, kind, day, out)


if __name__ == "__main__":
    main()
