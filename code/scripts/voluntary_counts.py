"""Count voluntary (FDA-coded) reports in the clean table by writer, task and period.

Usage: python -I voluntary_counts.py CLEAN.parquet
"""
import sys

import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 200)
t = pq.read_table(sys.argv[1], columns=["report_source_code", "reporter_occupation_code", "writer",
                                        "period", "in_device_task", "in_patient_task", "event_type"])
df = t.to_pandas()
print("report_source_code x writer (all clean reports):")
print(pd.crosstab(df.report_source_code.fillna("?"), df.writer.fillna("?")).to_string())
vol = df[df.writer.isin(["lay", "professional", "unknown"])]
print("\nnon-industry reports, occupation:")
print(vol.reporter_occupation_code.fillna("?").value_counts().head(15).to_string())
for task in ("device", "patient"):
    s = vol[vol[f"in_{task}_task"]]
    print(f"\n{task} task, non-industry reports by writer x period:")
    print(pd.crosstab(s.writer, s.period, margins=True).to_string())
