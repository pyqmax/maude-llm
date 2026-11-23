"""Summarise pii_scan output: top matches per pattern, plus the full email/phone/ssn lists."""
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1])
for t in ("device", "patient"):
    h = pd.read_csv(d / f"pii_{t}.tsv", sep="\t")
    for p in ("honorific", "url", "street"):
        s = h[h.pattern == p]
        print(f"== {t} {p} {len(s)}")
        print(s.match.str.upper().value_counts().head(15).to_string())
    for p in ("email", "phone", "ssn"):
        s = h[h.pattern == p]
        print(f"== {t} {p} {len(s)}")
        for c in s.context:
            print("  ", c[:200])
