"""Define the voluntary (FDA-coded) study set and inspect how codes are appended.

Usage: python -I voluntary_set.py CLEAN.parquet SAMPLE_DIR
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

pd.set_option("display.width", 200)
sys.stdout.reconfigure(encoding="utf-8")
PAT = r"\b(?:DEVICE CODES?|PT CODES?|PATIENT CODES?|PT|DPC|HECC|ANNEX [A-G])\s*:\s*\d{4}"

t = pq.read_table(sys.argv[1], columns=["report_source_code", "writer", "period", "event_type",
                                        "in_device_task", "in_patient_task", "words_event"])
t = t.filter(pc.equal(t.column("report_source_code"), "Voluntary report"))
df = t.to_pandas()
print(f"voluntary reports in clean table: {len(df):,}")
for task in ("device", "patient"):
    s = df[df[f"in_{task}_task"]]
    print(f"\n{task} task, voluntary, writer x period:")
    print(pd.crosstab(s.writer, s.period, margins=True).to_string())
    print(f"median words: {s.words_event.median():.0f}")

# how FDA appends codes: tails of voluntary patient-task narratives with and without a match
d = pd.read_parquet(Path(sys.argv[2]) / "patient.parquet")
d = d[d.report_source_code == "Voluntary report"]
m = d.text_event.fillna("").str.contains(PAT, regex=True, flags=re.I)
print(f"\nvoluntary patient-task sample: {len(d)} reports, {m.mean():.1%} match the code pattern")
print("\n--- tails of 12 non-matching narratives ---")
for tx in d[~m].text_event.sample(12, random_state=1):
    print(repr(tx[-220:]))
print("\n--- tails of 6 matching narratives ---")
for tx in d[m].text_event.sample(6, random_state=1):
    print(repr(tx[-220:]))
