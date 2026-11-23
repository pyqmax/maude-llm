"""Show unknown-writer unseen device narratives that do not mention 803.22."""
import re
import sys
from pathlib import Path

import pandas as pd

RX = re.compile(r"803\.22|803\s*\.\s*22|21\s*CFR\s*803", re.I)
d = pd.read_parquet(Path(sys.argv[1]) / "device.parquet", columns=["mdr_report_key", "set", "writer", "text_event"])
d = d[(d.set == "unseen") & (d.writer == "unknown")]
d = d[~d.text_event.fillna("").str.contains(RX)]
print(len(d))
for t in d.text_event.head(12):
    print("-", t[:260].replace("\n", " "))
