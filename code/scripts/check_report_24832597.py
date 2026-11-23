"""Who wrote report 24832597, what did FDA file, and do the legacy codes in its text match?"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1])
for task in ("device", "patient"):
    d = pd.read_parquet(root / f"{task}.parquet")
    r = d[d.mdr_report_key.astype(str) == "24832597"]
    print(task, "rows:", len(r), "unseen set size:", int((d.set == "unseen").sum()))
    for c in r.columns:
        if c != "text_event":
            print("   ", c, "=", r.iloc[0][c] if len(r) else None)
    if len(r):
        tx = r.iloc[0].text_event
        i = tx.find("Annex E")
        print("    text around list:", repr(tx[max(0, i - 200):i + 160]))
