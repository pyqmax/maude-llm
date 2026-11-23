"""Per report, how often does Arm B beat, tie or lose to Arm A for the frontier models?"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "runs/full/scores_frontier.tsv", sep="\t", dtype={"key": str})
d["arm"] = d.arm.astype(str)
for m in sorted(d.model.unique()):
    for t in ("device", "patient"):
        x = d[(d.model == m) & (d.task == t)].drop_duplicates(["arm", "key"])
        p = x.pivot(index="key", columns="arm", values="recall").dropna(subset=["A", "B"])
        print(m, t, "n", len(p), "B>A", int((p.B > p.A).sum()), "B=A", int((p.B == p.A).sum()),
              "B<A", int((p.B < p.A).sum()))
