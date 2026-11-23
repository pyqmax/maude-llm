"""Paired comparisons on the frontier subsample (stages 2 and 3).

Usage: python -I analyze_frontier.py FRONTIER_SCORES.tsv STAGE1_SCORES.tsv CONTEXT_DIR

All contrasts are paired by report over the 500-per-task subsample, 2,000 bootstrap
resamples, 95% CI:
  1. frontier Arm C minus copy-the-nearest baseline
  2. frontier Arm C minus frontier Arm A (what retrieval adds for a frontier model)
  3. Opus minus GPT-5.5, Arm A and Arm C
  4. frontier Arm C minus each open model's Arm C on the same reports
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402

pd.set_option("display.width", 250)
rng = np.random.default_rng(20261114)
METS = ("recall", "soft_recall")


def ci(diff):
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    return round(diff.mean(), 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)


fr = pd.read_csv(sys.argv[1], sep="\t", dtype={"key": str})
s1 = pd.read_csv(sys.argv[2], sep="\t", dtype={"key": str})
for d in (fr, s1):
    d["model"] = d.model.str.replace("-instruct", "")
ctx = {t: pd.read_parquet(Path(sys.argv[3]) / f"context_{t}.parquet").set_index("mdr_report_key")
       for t in ("device", "patient")}


def table(d, task, arm, model, keys):
    g = d[(d.task == task) & (d.arm == arm) & (d.model == model)].set_index("key")
    g = g[~g.index.duplicated()].reindex(keys)
    return g[list(METS)].fillna(0)


rows = []
for task in ("device", "patient"):
    sub = fr[(fr.task == task) & (fr.arm == "C")].key.unique()
    c = ctx[task].loc[sub]
    base = pd.DataFrame([score(list(p), list(f)) for p, f in zip(c.nn_codes, c.filed)], index=c.index)
    frontier = sorted(fr.model.unique())
    for m in frontier:
        A, C = table(fr, task, "A", m, sub), table(fr, task, "C", m, sub)
        for met in METS:
            mean, lo, hi = ci((C[met] - base[met]).to_numpy())
            rows.append(dict(task=task, contrast=f"{m} C - nn_codes", metric=met, a=round(C[met].mean(), 3),
                             b=round(base[met].mean(), 3), diff=mean, ci_lo=lo, ci_hi=hi))
            mean, lo, hi = ci((C[met] - A[met]).to_numpy())
            rows.append(dict(task=task, contrast=f"{m} C - A", metric=met, a=round(C[met].mean(), 3),
                             b=round(A[met].mean(), 3), diff=mean, ci_lo=lo, ci_hi=hi))
    if len(frontier) == 2:
        m1, m2 = frontier
        for arm in ("A", "C"):
            X, Y = table(fr, task, arm, m1, sub), table(fr, task, arm, m2, sub)
            for met in METS:
                mean, lo, hi = ci((X[met] - Y[met]).to_numpy())
                rows.append(dict(task=task, contrast=f"{m1} - {m2}, arm {arm}", metric=met, a=round(X[met].mean(), 3),
                                 b=round(Y[met].mean(), 3), diff=mean, ci_lo=lo, ci_hi=hi))
    for m in frontier:
        C = table(fr, task, "C", m, sub)
        for om in sorted(s1.model.unique()):
            O = table(s1, task, "C", om, sub)
            for met in METS:
                mean, lo, hi = ci((C[met] - O[met]).to_numpy())
                rows.append(dict(task=task, contrast=f"{m} C - {om} C", metric=met, a=round(C[met].mean(), 3),
                                 b=round(O[met].mean(), 3), diff=mean, ci_lo=lo, ci_hi=hi))
out = pd.DataFrame(rows)
print(f"subsample: {fr[fr.arm == 'C'].groupby('task').key.nunique().to_dict()} reports per task")
print(out.to_string(index=False))
