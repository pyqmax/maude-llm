"""Print cases for reading: narrative, FDA's codes, each model's answer per arm.

Usage:
    python -I show_cases.py JOBS.jsonl RESULTS.jsonl VOLUNTARY_DIR CODEBOOK.tsv TASK [ARMS]

ARMS is a string of arm letters to show, default "0ABC".
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prompts import load_codebook  # noqa: E402
from score_runs import parse  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    jobs_p, res_p, vol_dir, codebook, task = sys.argv[1:6]
    arms = sys.argv[6] if len(sys.argv) > 6 else "0ABC"
    book = load_codebook(codebook)
    name = {r["imdrf_code"]: r["term"] for a in book.values() for r in a}
    jobs = {j["id"]: j for j in map(json.loads, open(jobs_p, encoding="utf-8"))}
    res = {}
    for r in map(json.loads, open(res_p, encoding="utf-8")):
        if r["ok"] or r["id"] not in res:
            res[r["id"]] = r
    keys = sorted({j["key"] for j in jobs.values() if j["task"] == task})
    d = pd.read_parquet(Path(vol_dir) / f"{task}.parquet").set_index("mdr_report_key").loc[keys]

    def fmt(codes):
        return "; ".join(f"{c} {name.get(c, '??')}" for c in codes) or "(none)"

    for k in keys:
        r = d.loc[k]
        print("=" * 100)
        print(f"{k}  writer={r.writer}  product={r.product_code}  {r.event_type}  {r.date_received}")
        print(f"\nEVENT: {r.text_event[:1200]}")
        print(f"\nFDA: {fmt(list(r[f'{task}_codes']))}")
        for arm in arms:
            print(f"  -- arm {arm}")
            for jid, j in jobs.items():
                if j["task"] == task and j["key"] == k and j["arm"] == arm:
                    rr = res.get(jid)
                    pred = parse(rr.get("content")) if rr and rr["ok"] else None
                    print(f"     {j['model'].split('/')[-1]:<24} {fmt(pred) if pred is not None else '(unparsed)'}")
        print()


if __name__ == "__main__":
    main()
