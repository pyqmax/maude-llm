"""Tables from the stage 1 scores beyond what score_runs.py prints.

Usage: python -I analyze_stage1.py SCORES.tsv CONTEXT_DIR
  - unseen vs comparison (memorisation contrast), arms 0 and A
  - Arm C minus copy-the-nearest baseline per model, with a paired bootstrap 95% CI
  - per-writer table for Arm C, exact and soft recall
  - codes per answer by arm (overload signature)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402

pd.set_option("display.width", 250)
d = pd.read_csv(sys.argv[1], sep="\t", dtype={"key": str})
ctx = {t: pd.read_parquet(Path(sys.argv[2]) / f"context_{t}.parquet").set_index("mdr_report_key")
       for t in ("device", "patient")}
d["model"] = d.model.str.replace("-instruct", "")

print("=== memorisation contrast: soft recall, unseen (Feb-Jun 2026) vs comparison (2021-2024), arm A ===")
t = d[d.arm == "A"].groupby(["task", "model", "set"]).soft_recall.mean().unstack("set")
t["gap"] = t["comparison"] - t["unseen"]
print(t.round(3).to_string())

print("\n=== codes per answer, by arm (overload) ===")
print(d.groupby(["task", "model", "arm"]).n_pred.mean().unstack("arm").round(1).to_string())

print("\n=== Arm C vs copy-the-nearest baseline, unseen, exact and soft recall, paired bootstrap 95% CI ===")
rng = np.random.default_rng(20261114)
rows = []
for task in ("device", "patient"):
    c = ctx[task]
    c = c[c.set == "unseen"]
    base = pd.DataFrame([score(list(p), list(f)) for p, f in zip(c.nn_codes, c.filed)], index=c.index)
    for m, g in d[(d.task == task) & (d.arm == "C") & (d.set == "unseen")].groupby("model"):
        g = g.set_index("key").reindex(base.index)
        for met in ("recall", "soft_recall"):
            diff = (g[met].fillna(0) - base[met]).to_numpy()
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            rows.append({"task": task, "model": m, "metric": met, "model_mean": round(g[met].fillna(0).mean(), 3),
                         "baseline": round(base[met].mean(), 3), "diff": round(diff.mean(), 3),
                         "ci_lo": round(np.percentile(bs, 2.5), 3), "ci_hi": round(np.percentile(bs, 97.5), 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== Arm C by writer, unseen: exact recall / soft recall ===")
u = d[(d.arm == "C") & (d.set == "unseen")]
ex = u.groupby(["task", "writer", "model"]).recall.mean().unstack("model").round(2)
so = u.groupby(["task", "writer", "model"]).soft_recall.mean().unstack("model").round(2)
print("exact:\n" + ex.to_string())
print("\nsoft:\n" + so.to_string())
print("\nreports per writer:", u.groupby(["task", "writer"]).key.nunique().to_dict())
