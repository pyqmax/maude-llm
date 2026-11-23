"""How many sample narratives carry the reporter's or intake staff's own codes
in the text, e.g. "PT: 4582. DEVICE CODE: 2964." or "ANNEX E: 1776, 1770"?

Usage: python -I embedded_codes.py SAMPLE_DIR
"""
import re
import sys
from pathlib import Path

import pandas as pd

PAT = re.compile(r"\b(DEVICE CODES?|PT CODES?|PATIENT CODES?|PT|DPC|HECC|ANNEX [A-G])\s*:\s*\d{4}", re.I)

for task in ("device", "patient"):
    d = pd.read_parquet(Path(sys.argv[1]) / f"{task}.parquet")
    d["embedded"] = d.text_event.fillna("").str.contains(PAT)
    print(f"\n{task}: {d.embedded.mean():.1%} of all sample reports")
    print(d.groupby(d.stratum.where(d.stratum.isin(["lay", "professional", "industry matched"]), "industry"))
          .embedded.agg(["mean", "size"]).round(3).to_string())
    ex = d[d.embedded].text_event.str.extract(r"((?:DEVICE CODES?|PT|PATIENT CODES?|ANNEX [A-G])\s*:\s*[\d, ]+)", flags=re.I)[0]
    print("examples:", ex.dropna().head(5).tolist())
