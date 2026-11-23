"""Scan the cleaned voluntary narratives for personal data FDA's redaction may have missed.

Usage: python -I pii_scan.py VOLUNTARY_DIR OUT_DIR

Looks for emails, phone numbers, street addresses, SSN-like numbers, URLs and
honorific+surname patterns. Writes one TSV per task with key, pattern and the
matched span in context (60 chars each side), and prints counts. The text is
FDA's published text, so this is about not amplifying a slip, not about
discovering secrets; findings are reviewed by hand before any release.
"""
import re
import sys
from pathlib import Path

import pandas as pd

PATTERNS = {
    "email": r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    "phone": r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)",
    "ssn": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "url": r"\b(?:https?://|www\.)\S+",
    "street": r"\b\d{1,5}\s+(?:[A-Z]+\s){1,3}(?:STREET|ST|AVENUE|AVE|ROAD|RD|BLVD|BOULEVARD|DRIVE|DR|LANE|LN|COURT|CT|WAY)\b\.?",
    "honorific": r"\b(?:DR|MR|MRS|MS|MISS)\.?\s+[A-Z][A-Z'-]{2,}\b(?!\s*\(B\))",
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
# lot numbers, model numbers and FDA's own report numbers look like phones; drop spans next to these words
NOISE = re.compile(r"\b(?:LOT|REF|MODEL|SERIAL|SN|S/N|UDI|CATALOG|CAT|PART|P/N|REPORT|MDR|MW|NDC|ORDER|PO)\b\s*#?\s*:?\s*$", re.I)

vol, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
for task in ("device", "patient"):
    d = pd.read_parquet(vol / f"{task}.parquet", columns=["mdr_report_key", "set", "text_event"])
    rows = []
    for key, s, text in d.itertuples(index=False):
        if not isinstance(text, str):
            continue
        for name, rx in RX.items():
            for m in rx.finditer(text):
                before = text[max(0, m.start() - 60):m.start()]
                if name == "phone" and NOISE.search(before):
                    continue
                rows.append(dict(mdr_report_key=key, set=s, pattern=name, match=m.group(0),
                                 context=(before + "[[" + m.group(0) + "]]" + text[m.end():m.end() + 60]).replace("\n", " ")))
    hits = pd.DataFrame(rows)
    hits.to_csv(out_dir / f"pii_{task}.tsv", sep="\t", index=False)
    print(task, len(d), "narratives;", len(hits), "hits in", hits.mdr_report_key.nunique() if len(hits) else 0, "narratives")
    if len(hits):
        print(hits.pattern.value_counts().to_string())
