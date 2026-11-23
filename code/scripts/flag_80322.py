"""Count 803.22 notifications (manufacturer-forwarded reports about another firm's device)
by writer group in the voluntary study set, and write a flag table.

Usage: python -I flag_80322.py VOLUNTARY_DIR OUT.tsv
"""
import re
import sys
from pathlib import Path

import pandas as pd

RX = re.compile(r"803\.22|803\s*\.\s*22|21\s*CFR\s*803", re.I)
vol = Path(sys.argv[1])
rows = []
for task in ("device", "patient"):
    d = pd.read_parquet(vol / f"{task}.parquet", columns=["mdr_report_key", "set", "writer", "text_event"])
    d["is_80322"] = d.text_event.fillna("").str.contains(RX)
    for (s, w), g in d.groupby(["set", "writer"]):
        rows.append(dict(task=task, set=s, writer=w, n=len(g), n_80322=int(g.is_80322.sum()),
                         share=round(g.is_80322.mean(), 3)))
    d[d.is_80322][["mdr_report_key", "set", "writer"]].assign(task=task).to_csv(
        sys.argv[2].replace(".tsv", f"_{task}_keys.tsv"), sep="\t", index=False)
out = pd.DataFrame(rows)
out.to_csv(sys.argv[2], sep="\t", index=False)
print(out.to_string(index=False))
