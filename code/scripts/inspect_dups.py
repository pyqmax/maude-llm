"""Inspect the duplicate-narrative groups and the lay-writer losses.

Usage: python -I inspect_dups.py CLEAN.parquet EXTRACT_DIR
"""
import sys
from collections import Counter

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

t = pq.read_table(sys.argv[1], columns=["mdr_report_key", "text_key", "dup_text",
                                        "manufacturer_name", "product_code",
                                        "device_codes", "writer", "excl_device",
                                        "words_event"])
df = t.to_pandas()
g = df.groupby("text_key")
size = g.size().sort_values(ascending=False)
print("groups with >1 report:", int((size > 1).sum()), " reports in them:", int(size[size > 1].sum()))
print("size distribution of groups >1:")
for lo, hi in [(2, 2), (3, 10), (11, 100), (101, 1000), (1001, 10**9)]:
    s = size[(size >= lo) & (size <= hi)]
    print(f"  {lo}-{hi}: {len(s)} groups, {int(s.sum())} reports")

top = size.head(12)
keys = {}
for k in top.index:
    sub = df[df.text_key == k]
    keys[sub.mdr_report_key.iloc[0]] = k
    codes = Counter(tuple(c) for c in sub.device_codes).most_common(3)
    print(f"\nGROUP {k}: {len(sub)} reports, mfrs {sub.manufacturer_name.value_counts().head(2).to_dict()},"
          f" product {sub.product_code.value_counts().head(2).to_dict()}, codes {codes}")

dset = ds.dataset(sys.argv[2], format="parquet")
tx = dset.to_table(columns=["mdr_report_key", "text_event"],
                   filter=pc.field("mdr_report_key").isin(list(keys)))
for r in tx.to_pylist():
    print(f"\n--- {r['mdr_report_key']} (group {keys[r['mdr_report_key']]}):\n{r['text_event'][:400]}")

# Do duplicate groups agree on codes?
multi = df[df.text_key.isin(size[size > 1].index)].copy()
multi["cs"] = multi.device_codes.map(lambda c: tuple(sorted(c)))
agree = multi.groupby("text_key").cs.nunique()
print("\ngroups >1 whose reports carry identical device-code sets:",
      f"{(agree == 1).mean():.1%}")

print("\nlay writer, exclusion reasons:")
print(df[df.writer == "lay"].excl_device.value_counts().to_string())
