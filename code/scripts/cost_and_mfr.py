"""Measured cost per call from the pilot, and whether voluntary reports carry a
manufacturer narrative at all.

Usage: python -I cost_and_mfr.py runs/pilot/scores.tsv SAMPLE_DIR CLEAN.parquet
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

pd.set_option("display.width", 200)
df = pd.read_csv(sys.argv[1], sep="\t")
print("mean cost per call (USD) and prompt tokens, pilot:")
print(df.groupby(["model", "task", "arm"])[["cost", "in_tok", "out_tok"]].mean().round(5).to_string())

print("\nvoluntary reports with a manufacturer narrative (sample):")
for task in ("device", "patient"):
    d = pd.read_parquet(Path(sys.argv[2]) / f"{task}.parquet")
    v = d[d.report_source_code == "Voluntary report"]
    has = v.text_mfr.fillna("").str.split().str.len() >= 15
    print(f"  {task}: {has.mean():.1%} of {len(v)}")

print("\nfiled term quoted verbatim in the event narrative, by source (clean table):")
t = pq.read_table(sys.argv[3], columns=["report_source_code", "in_device_task", "in_patient_task",
                                        "quotes_term_event", "period"])
c = t.to_pandas()
for task in ("device", "patient"):
    s = c[c[f"in_{task}_task"]]
    print(f"  {task}:")
    print(s.groupby("report_source_code").quotes_term_event.agg(["mean", "size"]).round(3).to_string())
