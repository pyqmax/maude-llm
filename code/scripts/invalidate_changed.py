"""Drop results whose prompt changed between two versions of a job file.

Usage: python -I invalidate_changed.py OLD_JOBS.jsonl NEW_JOBS.jsonl RESULTS.jsonl

Jobs are matched by id. For every id whose messages differ (or that is new),
the existing result is moved out of RESULTS.jsonl into RESULTS.invalidated.jsonl
so that run_openrouter.py re-asks it. Prints the counts.
"""
import hashlib
import json
import sys
from pathlib import Path


def digest(msgs):
    return hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


old = {}
for l in open(sys.argv[1], encoding="utf-8"):
    j = json.loads(l)
    old[j["id"]] = digest(j["messages"])
changed, new = set(), 0
for l in open(sys.argv[2], encoding="utf-8"):
    j = json.loads(l)
    h = old.get(j["id"])
    if h is None:
        new += 1
    elif h != digest(j["messages"]):
        changed.add(j["id"])
print(f"{len(changed)} jobs with a changed prompt, {new} new jobs")
keys = {c.split("|")[2] for c in changed}
print(f"{len(keys)} distinct reports affected")

res = Path(sys.argv[3])
keep, drop = [], []
for l in open(res, encoding="utf-8"):
    (drop if json.loads(l)["id"] in changed else keep).append(l)
with open(res.with_suffix(".invalidated.jsonl"), "a", encoding="utf-8") as f:
    f.writelines(drop)
with open(res, "w", encoding="utf-8") as f:
    f.writelines(keep)
print(f"moved {len(drop)} results out of {res.name}; {len(keep)} kept")
