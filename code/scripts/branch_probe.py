"""Two numbers for the paper: branch recall (ancestor or descendant match) of each
model in Arm C, and the copy-the-nearest baseline on the 100 reports of the frontier Arm 0 probe."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ctx_dir = Path(sys.argv[1])
s1 = pd.read_csv(ROOT / "runs/full/scores_stage1.tsv", sep="\t", dtype={"key": str})
mg = pd.read_csv(ROOT / "runs/full/scores_medgemma.tsv", sep="\t", dtype={"key": str})
fr = pd.read_csv(ROOT / "runs/full/scores_frontier.tsv", sep="\t", dtype={"key": str})
d = pd.concat([s1, mg, fr], ignore_index=True)
d["model"] = d.model.str.replace("-instruct", "")
d["arm"] = d.arm.astype(str)
d["branch_recall"] = d.branch_recall.fillna(0.0)
for t in ("device", "patient"):
    ctx = pd.read_parquet(ctx_dir / f"context_{t}.parquet").set_index("mdr_report_key")
    unseen = ctx.index[ctx.set == "unseen"]
    g = d[(d.task == t) & (d.arm == "C") & d.key.isin(unseen)].drop_duplicates(["model", "key"])
    print(t, "Arm C branch recall:", g.groupby("model").branch_recall.mean().round(2).to_dict())
    # share of reports where at least one FDA code matched by branch (report-level hit)
    hit = g.assign(hit=g.branch_recall > 0).groupby("model").hit.mean().round(2)
    print(t, "share of reports with any branch hit:", hit.to_dict())
    probe = d[(d.task == t) & (d.arm == "0") & (d.model == "claude-opus-4.8")].key.unique()
    c = ctx.loc[probe]
    b = pd.DataFrame([score(list(p), list(f)) for p, f in zip(c.nn_codes, c.filed)])
    print(t, f"copy-nearest baseline on the {len(probe)} probe reports: recall {b.recall.mean():.2f}, soft {b.soft_recall.mean():.2f}")
