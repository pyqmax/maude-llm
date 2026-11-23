"""Median narrative length (words) over each task's full voluntary set and by writer, to verify the paper's figures."""
import glob
import os
import sys

import pandas as pd

d = sys.argv[1]
for task in ["device", "patient"]:
    files = glob.glob(os.path.join(d, f"*{task}*.parquet"))
    if not files:
        print(task, "no parquet found in", d, os.listdir(d))
        continue
    df = pd.concat(pd.read_parquet(f) for f in files)
    n = df["words_clean"]
    print(task, "rows", len(df), "median words_clean", n.median())
    print(n.groupby(df["set"]).median())
    if "writer" in df.columns:
        print(n.groupby(df["writer"]).median())
