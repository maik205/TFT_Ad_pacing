"""Peek at the first rows of the iPinYou season-2 bid, impression and click logs (column layout)."""
import bz2
import zipfile

Z = r"C:\Users\maik\datasets\ipinyou\season2.zip"
P = "ipinyou.contest.dataset-season2/training2nd/"
z = zipfile.ZipFile(Z)
for kind in ("bid", "imp", "clk"):
    name = f"{P}{kind}.20130606.txt.bz2"
    with z.open(name) as f:
        b = bz2.open(f)
        rows = [b.readline().decode("utf-8", "replace").rstrip("\n").split("\t") for _ in range(3)]
    print(kind, "ncol", len(rows[0]))
    for i, v in enumerate(rows[0]):
        print(f"  {i:2d}  {v[:60]}")
