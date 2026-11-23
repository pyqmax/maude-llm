"""Write the subset of a job file for given report keys and models (for case reading).

Usage: python -I subset_jobs.py JOBS.jsonl KEYS_JOBS.jsonl MODELS_CSV OUT.jsonl
KEYS_JOBS.jsonl: another job file whose report keys define the subset (e.g. the pilot).
"""
import json
import sys

keys = {json.loads(l)["key"] for l in open(sys.argv[2], encoding="utf-8")}
models = set(sys.argv[3].split(","))
n = 0
with open(sys.argv[4], "w", encoding="utf-8") as out:
    for l in open(sys.argv[1], encoding="utf-8"):
        j = json.loads(l)
        if j["key"] in keys and j["model"] in models:
            out.write(l)
            n += 1
print(n, "jobs")
