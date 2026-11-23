"""Share of sample narratives with FDA-appended codes, by report source.

Usage: python -I embedded_by_source.py SAMPLE_DIR
"""
import re
import sys
from pathlib import Path

import pandas as pd

PAT = r"\b(?:DEVICE CODES?|PT CODES?|PATIENT CODES?|PT|DPC|HECC|ANNEX [A-G])\s*:\s*\d{4}"
for task in ("device", "patient"):
    d = pd.read_parquet(Path(sys.argv[1]) / f"{task}.parquet")
    if "report_source_code" not in d.columns:
        print(task, "sample lacks report_source_code; columns:", list(d.columns))
        continue
    d["embedded"] = d.text_event.fillna("").str.contains(PAT, regex=True, flags=re.I)
    print(f"\n{task}:")
    print(d.groupby(["report_source_code", "writer"]).embedded.agg(["mean", "size"]).round(3).to_string())
