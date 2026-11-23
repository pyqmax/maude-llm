"""Share of reports with an event narrative, by quarter received and report source.

Usage:
    python -I text_coverage.py EXTRACT_DIR

Streams record batches so the narrative text never sits in memory all at once.
"""
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.stdout.reconfigure(encoding="utf-8")
cols = ["date_received", "report_source_code", "text_event", "text_mfr"]
dset = ds.dataset(Path(sys.argv[1]), format="parquet")

n = defaultdict(int)
ev = defaultdict(int)
anyt = defaultdict(int)
for b in dset.to_batches(columns=cols, batch_size=200_000):
    d = b.column("date_received").to_pylist()
    src = b.column("report_source_code").to_pylist()
    e = pc.is_valid(b.column("text_event")).to_pylist()
    m = pc.is_valid(b.column("text_mfr")).to_pylist()
    for di, si, ei, mi in zip(d, src, e, m):
        q = f"{di[:4]}Q{(int(di[4:6]) - 1) // 3 + 1}"
        s = (si or "?").replace(" report", "")
        for key in ((q, "ALL"), (q, s)):
            n[key] += 1
            ev[key] += ei
            anyt[key] += ei or mi

quarters = sorted({k[0] for k in n})
sources = ["Manufacturer", "Voluntary", "User Facility", "Distributor"]
print(f"{'quarter':8} {'reports':>10} {'event%':>7} {'any%':>6}   " +
      "  ".join(f"{s[:12]:>12}" for s in sources) + "   (event% by source)")
for q in quarters:
    a = (q, "ALL")
    row = f"{q:8} {n[a]:>10,} {ev[a] / n[a]:>7.1%} {anyt[a] / n[a]:>6.1%}   "
    row += "  ".join(f"{(ev[(q, s)] / n[(q, s)]) if n[(q, s)] else float('nan'):>12.1%}" for s in sources)
    print(row)
