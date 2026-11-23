"""Residual intake-note check, widened for the paper's methods sentence (2026-11-18).

Two patterns over the stripped narratives of the voluntary set:
  1. a label variant followed by a four-digit code list, with colon, dash or equals separators
     (the original check_residual_notes.py accepted only dash and equals);
  2. the word-level check from textclean.RESIDUAL (CODE, CODES, ANNEX, MW######), with the
     context of every hit so the hits can be read.

Usage: python -I check_residual_notes2.py VOLUNTARY_DIR
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from textclean import RESIDUAL  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
LABEL = re.compile(
    r"\b(?:PT|PATIENT|DEVICE|DPC|HECC|HEALTH|PRODUCT|ANNEX\s*[A-G])"
    r"(?:\s+(?:PROBLEM|EFFECT|DEVICE))?(?:\s+CODES?)?\s*[-=:]\s*\d{4}(?:\s*[,;.]\s*\d{4})*",
    re.I,
)
for task in ("device", "patient"):
    d = pd.read_parquet(Path(sys.argv[1]) / f"{task}.parquet", columns=["mdr_report_key", "set", "text_event"])
    m1 = d.text_event.str.contains(LABEL)
    m2 = d.text_event.str.contains(RESIDUAL)
    print(f"{task}: {len(d)} narratives; label+codes pattern {int(m1.sum())}; word-level {int(m2.sum())} "
          f"(unseen {int((m2 & (d.set == 'unseen')).sum())})")
    for tx in d[m1].text_event.head(5):
        s = LABEL.search(tx)
        print("   LABEL:", repr(tx[max(0, s.start() - 40):s.end() + 40]))
    words = Counter()
    for tx in d[m2].text_event:
        for h in RESIDUAL.finditer(tx):
            words[h.group(0).upper()[:2] if h.group(0).upper().startswith("MW") else h.group(0).upper()] += 1
    print("   word hits:", dict(words))
    for tx in d[m2].text_event.head(40):
        s = RESIDUAL.search(tx)
        print("   WORD:", repr(tx[max(0, s.start() - 50):s.end() + 50]))
