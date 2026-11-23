"""Which reports did the pilot runs use? Compare their report keys and dates with the unseen set.

Usage: python -I scripts/check_pilot.py
Reads runs/pilot/jobs.jsonl and runs/pilot2/jobs.jsonl, prints the fields seen,
the distribution of any set or date field, and the overlap with the unseen keys
in the frontier subsample files under data/.
"""
import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
unseen = set()
for f in root.glob("data/frontier_subsample_*.tsv"):
    for line in f.read_text(encoding="utf-8").splitlines()[1:]:
        unseen.add(line.split("\t")[0])
print("frontier subsample keys:", len(unseen))

for name in ["pilot", "pilot2"]:
    p = root / "runs" / name / "jobs.jsonl"
    if not p.exists():
        print(name, "missing")
        continue
    keys = collections.Counter()
    sets = collections.Counter()
    dates = collections.Counter()
    ids = set()
    n = 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            n += 1
            j = json.loads(line)
            keys.update(j.keys())
            for k in ("set", "split", "subset"):
                if k in j:
                    sets[(k, j[k])] += 1
            for k in ("date_received", "received", "date"):
                if k in j:
                    dates[str(j[k])[:7]] += 1
            for k in ("report_key", "key", "mdr_report_key", "report_id", "id"):
                if k in j:
                    ids.add(str(j[k]))
                    break
    print(f"\n{name}: {n} jobs; fields: {sorted(keys)}")
    print("  set fields:", dict(sets) if sets else "none")
    print("  months:", dict(sorted(dates.items())) if dates else "none")
    print("  distinct ids:", len(ids), "; overlap with frontier subsample:", len(ids & unseen))
