"""Build the pilot job file: 50 unseen reports (25 device, 25 patient), both
arms and both input conditions for three open models, plus Arm A on 10 of the
reports for the two frontier models.

Usage:
    python -I make_pilot.py SAMPLE_DIR SHORTLIST_DIR CODEBOOK.tsv OUT_JOBS.jsonl
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prompts import VERSION, build_messages, load_codebook  # noqa: E402

SEED = 20261114
OPEN = ["meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-oss-120b"]
FRONTIER = ["anthropic/claude-opus-4.8", "openai/gpt-5.5"]
PER_GROUP = {"industry": 10, "industry matched": 5, "lay": 5, "professional": 5}


def main():
    sample_dir, short_dir, codebook, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    book = load_codebook(codebook)
    jobs = []
    for task in ("device", "patient"):
        d = pd.read_parquet(sample_dir / f"{task}.parquet")
        d = d[d.set == "unseen"]
        sl = pd.read_parquet(short_dir / f"shortlist_{task}.parquet")
        sl = sl[sl.set == "unseen"].set_index("mdr_report_key")
        d["grp"] = d.stratum.where(d.stratum.isin(PER_GROUP), "industry")
        pick = pd.concat(g.sample(n=PER_GROUP[k], random_state=SEED) for k, g in d.groupby("grp"))
        front = set(pick.groupby("grp").head(2).mdr_report_key.head(5)) | set(
            pick[pick.grp.isin(["lay", "professional"])].mdr_report_key.head(5))
        for r in pick.to_dict("records"):
            key = r["mdr_report_key"]
            for cond in ("event", "event+mfr"):
                for arm in ("A", "B"):
                    msgs = build_messages(task, arm, cond, r, book,
                                          shortlist=list(sl.at[key, "shortlist"]))
                    models = OPEN + (FRONTIER if arm == "A" and key in front else [])
                    for m in models:
                        jobs.append({"id": f"{m}|{task}|{key}|{cond}|{arm}", "model": m,
                                     "task": task, "key": key, "cond": cond, "arm": arm,
                                     "group": r["grp"], "prompt_version": VERSION,
                                     "messages": msgs})
    with open(out, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs written")
    print(pd.Series([j["model"] for j in jobs]).value_counts().to_string())


if __name__ == "__main__":
    main()
