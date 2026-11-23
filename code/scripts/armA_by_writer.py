"""Arm A soft recall by writer group, device task, open panel, and how much of the
comparison-minus-unseen gap the writer mix explains.

Usage: python -I scripts/armA_by_writer.py RUNS_DIR

For each model: unseen soft recall by writer, reweighted to the comparison set's
writer mix, gives the score the unseen reports would have with the older mix.
mix effect = reweighted - unseen; residual = comparison - reweighted.
Supports the memorisation paragraph (Section 5.4).
"""
import sys
from pathlib import Path

import pandas as pd

runs = Path(sys.argv[1])
# same sources and rules as paper_tables.py: stage1 + medgemma, unparsed answers score 0, first row kept
frames = [pd.read_csv(runs / f"scores_{s}.tsv", sep="\t", dtype={"key": str}) for s in ("stage1", "medgemma")]
df = pd.concat(frames, ignore_index=True)
df["soft_recall"] = df.soft_recall.fillna(0.0)
df = df[(df.task == "device") & (df.arm.astype(str) == "A") & df.set.isin(["comparison", "unseen"])]
df = df.drop_duplicates(["model", "set", "key"])
df = df[~df.model.isin(["claude-opus-4.8", "gpt-5.5"])]
by = df.groupby(["model", "set", "writer"]).soft_recall.mean().unstack("writer")
print(by.round(2).to_string())
print()
mix = df[df.set == "comparison"].drop_duplicates("key").writer.value_counts(normalize=True)
print("comparison mix:", mix.round(2).to_dict())
print("unseen mix:", df[df.set == "unseen"].drop_duplicates("key").writer.value_counts(normalize=True).round(2).to_dict())
print()
print("model\tcomparison\tunseen\tgap\tunseen_at_comp_mix\tmix_effect\tresidual")
for m in sorted(df.model.unique()):
    comp = df[(df.model == m) & (df.set == "comparison")].soft_recall.mean()
    uns = df[(df.model == m) & (df.set == "unseen")].soft_recall.mean()
    rew = sum(mix[w] * by.loc[(m, "unseen"), w] for w in mix.index)
    print(f"{m}\t{comp:.2f}\t{uns:.2f}\t{comp - uns:+.2f}\t{rew:.2f}\t{rew - uns:+.2f}\t{comp - rew:+.2f}")
