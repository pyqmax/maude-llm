"""Print the table header and a few rows of annex sheets A and E."""
import sys

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
wb = openpyxl.load_workbook(sys.argv[1], read_only=True)
for name in ("A", "E"):
    rows = list(wb[name].iter_rows(values_only=True))
    for i, r in enumerate(rows[:14]):
        vals = [c for c in r if c is not None]
        if vals:
            print(name, i, [str(v)[:40] for v in vals][:12])
    print()
