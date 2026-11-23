"""Show the contexts of the word 'ANNEX' left in the stripped voluntary narratives."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
P = re.compile(r"\bANNEX\b", re.I)
for task in ("device", "patient"):
    d = pd.read_parquet(Path(sys.argv[1]) / f"{task}.parquet", columns=["mdr_report_key", "set", "text_event"])
    for k, s, tx in d[d.text_event.str.contains(P)][["mdr_report_key", "set", "text_event"]].itertuples(index=False):
        m = P.search(tx)
        print(task, k, s, repr(tx[max(0, m.start() - 80):m.end() + 80]))
