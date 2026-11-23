"""Show what release_mask.py changed, rule by rule, for review.

Usage: python -I release_mask_check.py VOLUNTARY_DIR [N_PER_RULE]
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import release_mask as rm  # noqa: E402

vol = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
random.seed(1)
for task in ("device",):
    d = pd.read_parquet(vol / f"{task}.parquet", columns=["mdr_report_key", "text_event"])
    spans = {k: [] for k in list(rm.RULES) + ["honorific"]}
    for t in d.text_event:
        if not isinstance(t, str):
            continue
        for name, rx in rm.RULES.items():
            for m in rx.finditer(t):
                spans[name].append((m.group(0), t[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")))
        for m in rm.HONORIFIC.finditer(t):
            if m.group(2).lower() not in rm.COMMON:
                spans["honorific"].append((m.group(0), t[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")))
    for name, s in spans.items():
        print(f"\n== {task} {name}: {len(s)} spans")
        for span, ctx in random.sample(s, min(n, len(s))):
            print(f"  [{span}]  ...{ctx}...")
