"""Count narratives in the voluntary set that still carry an intake-note variant.

Usage: python -I check_residual_notes.py VOLUNTARY_DIR
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PAT = re.compile(r"\b(?:PT|PATIENT|DEVICE|DPC|HECC|HEALTH|PRODUCT)(?:\s+(?:PROBLEM|EFFECT|DEVICE))?(?:\s+CODES?)?\s*[-=]\s*\d{4}(?:\s*[,;.]\s*\d{4})*", re.I)
for task in ("device", "patient"):
    d = pd.read_parquet(Path(sys.argv[1]) / f"{task}.parquet", columns=["mdr_report_key", "set", "text_event"])
    m = d.text_event.str.contains(PAT)
    print(f"{task}: {int(m.sum())} of {len(d)} narratives match a residual code-like pattern "
          f"(unseen: {int((m & (d.set == 'unseen')).sum())})")
    for tx in d[m].text_event.head(12):
        s = PAT.search(tx)
        print("   ", repr(tx[max(0, s.start() - 40):s.end() + 40]))
