"""Confirm the known personal-data slips are gone from the release copy and re-run the scan on it.

Usage: python -I release_verify.py RELEASE_DIR
"""
import re
import sys
from pathlib import Path

import pandas as pd

KNOWN = ["256 945 1953", "940-367-0886", "(940) 735 3119", "Height Street", "PHILIP.ZEUSCHNER", "GLASSMAN", "GOTTIMUKKALA"]
EMAIL = re.compile(r"(?<![\w.])[A-Z][A-Z0-9._%+-]*@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE = re.compile(r"(?<![\d.])(?:\+?1[\s.-]{0,2})?\(?\d{3}\)?[\s.-]{1,2}\d{3}[\s.-]{1,2}\d{4}(?!\d)")
MASK_RX = re.compile(r"\(B\)\(6\)", re.I)
rel = Path(sys.argv[1])
for task in ("device", "patient"):
    d = pd.read_parquet(rel / f"{task}_release.parquet", columns=["mdr_report_key", "text_event"])
    text = d.text_event.fillna("")
    n_masks = int(text.str.count(MASK_RX).sum())
    for k in KNOWN:
        n = int(text.str.contains(re.escape(k), case=False).sum())
        if n:
            print(f"{task}: '{k}' still present in {n} narratives")
    print(f"{task}: {len(d)} rows; residual emails {int(text.str.contains(EMAIL).sum())}, "
          f"residual phones {int(text.str.contains(PHONE).sum())}, (B)(6) markers {n_masks}")
print("verify done")
