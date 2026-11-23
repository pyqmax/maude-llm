"""Was FDA's code in the Arm C shortlist for a given report? Usage: shortlist_members.py CONTEXT_DIR TASK KEY [KEY ...]"""
import sys
from pathlib import Path

import pandas as pd

ctx_dir, task = sys.argv[1], sys.argv[2]
c = pd.read_parquet(Path(ctx_dir) / f"context_{task}.parquet").set_index("mdr_report_key")
for k in sys.argv[3:]:
    r = c.loc[k]
    sc = set(r.shortlist_C)
    print(k, "filed", list(r.filed), "in shortlist C:", {f: (f in sc) for f in r.filed}, "n_C", r.n_C)
