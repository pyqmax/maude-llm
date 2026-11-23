"""Narrative coverage of manufacturer reports received in 2026, by month and by week from July.

Usage:
    python -I text_coverage_2026.py EXTRACT_DIR
"""
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.stdout.reconfigure(encoding="utf-8")
dset = ds.dataset(Path(sys.argv[1]), format="parquet")
flt = (pc.field("date_received") >= "20260101") & (pc.field("report_source_code") == "Manufacturer report")
n, ev = defaultdict(int), defaultdict(int)
for b in dset.to_batches(columns=["date_received", "text_event"], filter=flt, batch_size=200_000):
    d = b.column("date_received").to_pylist()
    e = pc.is_valid(b.column("text_event")).to_pylist()
    for di, ei in zip(d, e):
        dt = date(int(di[:4]), int(di[4:6]), int(di[6:8]))
        key = di[:6] if dt < date(2026, 7, 1) else f"{di[:6]} wk{dt.isocalendar()[1]:02d}"
        n[key] += 1
        ev[key] += ei
for k in sorted(n):
    print(f"{k:14} {n[k]:>9,} reports  {ev[k] / n[k]:6.1%} with event narrative")
