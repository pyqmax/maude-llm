"""Rebuild runs/full/scores_frontier.tsv from the per-stage frontier score files (stages 2, 3, 4)."""
import sys
from pathlib import Path

import pandas as pd

runs = Path(sys.argv[1])
parts = [pd.read_csv(runs / f"scores_stage{s}.tsv", sep="\t", dtype={"key": str}) for s in (2, 3, 4)]
df = pd.concat(parts, ignore_index=True)
before = len(df)
df = df.drop_duplicates(["model", "task", "key", "arm"], keep="last")
df.to_csv(runs / "scores_frontier.tsv", sep="\t", index=False)
print(f"{before} rows read, {len(df)} written")
print(df.groupby(["model", "task", "arm"]).size().unstack("arm").to_string())
