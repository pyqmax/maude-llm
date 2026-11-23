"""Share of MAUDE reports with no device-problem code, by who wrote them.

Usage: python -I scripts/uncoded_by_writer.py EXTRACT_DIR OUT_TSV

Reads every report received 1 January 2021 to 30 June 2026 from the parquet
extract, groups by writer with the same rule as build_clean.py (industry for
manufacturer and distributor reports; lay, professional and unknown by reporter
occupation for the rest), and counts reports whose product_problems field is
empty and reports whose only term is "Adverse Event Without Identified Device
or Use Problem". Narrative presence is also tallied so the share can be given
among reports that have a narrative. No deduplication.
"""
import sys
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

START, END = "20210101", "20260630"
LAY = {"PATIENT", "PATIENT FAMILY MEMBER OR FRIEND"}
AEWIDUP = "adverse event without identified device or use problem"
COLS = ["date_received", "report_source_code", "reporter_occupation_code",
        "product_problems", "text_event"]


def writer(source, occ):
    if source in ("Manufacturer report", "Distributor report"):
        return "industry"
    if occ in LAY:
        return "lay"
    if occ in (None, "", "UNKNOWN", "OTHER", "NOT APPLICABLE", "INVALID DATA"):
        return "unknown"
    return "professional"


extract = Path(sys.argv[1])
out = Path(sys.argv[2])
tally = Counter()
files = sorted(extract.rglob("*.parquet"))
flt = (pc.field("date_received") >= START) & (pc.field("date_received") <= END)
for i, f in enumerate(files, 1):
    t = ds.dataset(str(f), format="parquet").to_table(columns=COLS, filter=flt)
    src = t.column("report_source_code").to_pylist()
    occ = t.column("reporter_occupation_code").to_pylist()
    probs = t.column("product_problems").to_pylist()
    text = t.column("text_event").to_pylist()
    for s, o, p, x in zip(src, occ, probs, text):
        w = writer(s, o)
        vol = "voluntary" if s == "Voluntary report" else "other"
        key = (w, vol)
        terms = [str(q).strip().lower() for q in (p or []) if q]
        has_text = bool(x) and len(str(x).split()) >= 15
        tally[key + ("reports",)] += 1
        tally[key + ("with_text",)] += has_text
        if not terms:
            tally[key + ("no_code",)] += 1
            tally[key + ("no_code_with_text",)] += has_text
        elif all(q == AEWIDUP for q in terms):
            tally[key + ("only_aewidup",)] += 1
            tally[key + ("only_aewidup_with_text",)] += has_text
    if i % 20 == 0:
        print(f"{i}/{len(files)} files", flush=True)

groups = sorted({k[:2] for k in tally})
rows = ["writer\tsource\treports\tno_code\tonly_aewidup\tshare_no_code\tshare_no_or_aewidup\twith_text\tno_code_with_text\tshare_no_code_with_text"]
for g in groups:
    n = tally[g + ("reports",)]
    nc = tally[g + ("no_code",)]
    ae = tally[g + ("only_aewidup",)]
    wt = tally[g + ("with_text",)]
    nct = tally[g + ("no_code_with_text",)]
    rows.append(f"{g[0]}\t{g[1]}\t{n}\t{nc}\t{ae}\t{nc / n:.3f}\t{(nc + ae) / n:.3f}\t{wt}\t{nct}\t{(nct / wt if wt else 0):.3f}")
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
