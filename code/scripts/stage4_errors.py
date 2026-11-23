"""Summarise the failed stage 4 calls: which model, what error, and when they occurred."""
import json
import sys
from collections import Counter

jobs = {json.loads(l)["id"] for l in open(sys.argv[1], encoding="utf-8")}
rows = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
rows = [r for r in rows if r["id"] in jobs]
ok = [r for r in rows if r["ok"]]
bad = [r for r in rows if not r["ok"]]
print("records for stage 4 ids:", len(rows), "ok", len(ok), "failed", len(bad))
print("ok by model:", Counter(r["model"] for r in ok))
print("failed by model:", Counter(r["model"] for r in bad))
errs = Counter()
for r in bad:
    e = r.get("error") or r.get("content") or ""
    errs[str(e)[:160]] += 1
for e, n in errs.most_common(8):
    print(n, repr(e))
print("sample failed record keys:", list(bad[0].keys()) if bad else None)
if bad:
    print(json.dumps(bad[0])[:800])
# order of outcomes in the file for stage 4 records
seq = "".join("." if r["ok"] else "x" for r in rows)
print("sequence (first 400):", seq[:400])
print("sequence (last 200):", seq[-200:])
