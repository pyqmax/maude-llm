"""Voluntary reports with no device-problem term, no patient-problem term, or neither.

Usage: python -I scripts/uncoded_voluntary_any.py EXTRACT_DIR

Same window and source rule as uncoded_by_writer.py (received 1 January 2021 to
30 June 2026, report_source_code "Voluntary report", no deduplication). A field
counts as empty when it holds no term at all.
"""
import sys
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

START, END = "20210101", "20260630"
flt = ((pc.field("date_received") >= START) & (pc.field("date_received") <= END)
       & (pc.field("report_source_code") == "Voluntary report"))
tally = Counter()
for f in sorted(Path(sys.argv[1]).rglob("*.parquet")):
    t = ds.dataset(str(f), format="parquet").to_table(
        columns=["product_problems", "patient_problems"], filter=flt)
    for p, q in zip(t.column("product_problems").to_pylist(), t.column("patient_problems").to_pylist()):
        dev = any(x for x in (p or []))
        pat = any(x for x in (q or []))
        tally["reports"] += 1
        tally["no_device"] += not dev
        tally["no_patient"] += not pat
        tally["neither"] += not dev and not pat
n = tally["reports"]
for k in ("no_device", "no_patient", "neither"):
    print(f"{k}\t{tally[k]}\t{100 * tally[k] / n:.1f}%")
print(f"reports\t{n}")
