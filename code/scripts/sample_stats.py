"""Describe the clean device-task pool to size the sample.

Usage: python -I sample_stats.py CLEAN.parquet
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

cols = ["period", "writer", "product_code", "device_codes", "device_levels",
        "in_device_task", "in_patient_task", "words_event", "words_mfr",
        "quotes_term_event", "event_type"]
df = pq.read_table(sys.argv[1], columns=cols).to_pandas()
dv = df[df.in_device_task].copy()
print("device task:", len(dv))
print("\nperiod x writer")
print(pd.crosstab(dv.period, dv.writer, margins=True).to_string())

dv["ncodes"] = dv.device_codes.map(len)
print("\ncodes per report:", dv.ncodes.value_counts().sort_index().head(6).to_dict())
dv["maxlev"] = dv.device_levels.map(max)
print("deepest level per report:", dv.maxlev.value_counts().sort_index().to_dict())

# Rarity from the 2021-2024 pool
ref = Counter(c for cs in dv[dv.period == "2021-2024"].device_codes for c in cs)
print("\ndistinct codes 2021-2024:", len(ref))
vals = np.array(sorted(ref.values(), reverse=True))
cum = vals.cumsum() / vals.sum()
print("codes covering 50/80/95% of mentions:",
      [int((cum < q).sum()) + 1 for q in (0.5, 0.8, 0.95)])
for lo, hi in [(1, 99), (100, 999), (1000, 9999), (10000, 10**9)]:
    print(f"  codes with {lo}-{hi} reports: {int(((vals >= lo) & (vals <= hi)).sum())}")


def band(c):
    n = ref.get(c, 0)
    return "rare(<1k)" if n < 1000 else ("mid(1k-10k)" if n < 10000 else "common(10k+)")


order = {"rare(<1k)": 0, "mid(1k-10k)": 1, "common(10k+)": 2}
dv["rarity"] = dv.device_codes.map(lambda cs: min((band(c) for c in cs), key=order.get))

for p in ("2026H1 unseen", "2021-2024"):
    s = dv[dv.period == p]
    print(f"\n=== {p}: {len(s)} reports, {s.product_code.nunique()} product codes")
    pc_ = s.product_code.value_counts()
    print("top 5 product codes share:", f"{pc_.head(5).sum() / len(s):.1%}", pc_.head(5).to_dict())
    for cap in (5, 10, 20, 50):
        print(f"  pool after cap {cap}/product code: {int(np.minimum(pc_, cap).sum())}")
    print("writer x rarity")
    print(pd.crosstab(s.writer, s.rarity).to_string())
    print("writer x deepest level")
    print(pd.crosstab(s.writer, s.maxlev).to_string())
    # Cell sizes after a cap of 10 per product code within each cell
    g = s.groupby(["writer", "rarity", "maxlev", "product_code"]).size()
    capped = np.minimum(g, 10).groupby(level=[0, 1, 2]).sum().unstack(level=[2])
    print("cell pool after cap 10 per product code per cell (writer, rarity x level)")
    print(capped.fillna(0).astype(int).to_string())
    print("median words event / mfr:", s.words_event.median(), s.words_mfr.median())
    print("quotes term in event narrative:", f"{s.quotes_term_event.mean():.1%}")

pt = df[df.in_patient_task]
print("\npatient task period x writer")
print(pd.crosstab(pt.period, pt.writer, margins=True).to_string())
