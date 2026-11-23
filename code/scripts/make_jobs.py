"""Build a job file for the voluntary-set runs (pilot or full).

Usage:
    python -I make_jobs.py VOLUNTARY_DIR CONTEXT_DIR CODEBOOK.tsv OUT_JOBS.jsonl MODE

MODE "pilot":  25 unseen reports per task (10 lay, 8 professional, 7 unknown),
               arms 0, A, B, C for PILOT_OPEN + FRONTIER.
MODE "smoke":  3 unseen reports per task, arms A and B, for the open models not in the pilot.
MODE "stage1": every unseen report, arms 0, A, B, C, all OPEN models;
               comparison set, arms 0 and A, all OPEN models.
MODE "stage2": FRONTIER, arm A, on the fixed frontier subsample (SUB per task, stratified
               by writer; saved to data/frontier_subsample_{task}.tsv), plus arm 0 on
               the first 100 per task of it.
MODE "stage3": FRONTIER, arm C, on the frontier subsample.
MODE "stage4": FRONTIER, arm B, on the frontier subsample (added 2026-11-18 to complete the grid).
MODE "medgemma": the stage1 plan for the self-hosted MedGemma 4B (run on Fir, scripts/run_medgemma.py).
Job ids are model|task|report|arm, so stages and reruns never repeat a finished call.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prompts import VERSION, build_messages, load_codebook  # noqa: E402

SEED = 20261114
SUB = 500
PILOT_OPEN = ["meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-oss-120b"]
OPEN = PILOT_OPEN + ["microsoft/phi-4", "qwen/qwen-2.5-72b-instruct", "meta-llama/llama-4-maverick"]
FRONTIER = ["anthropic/claude-opus-4.8", "openai/gpt-5.5"]
SELF_HOSTED = ["google/medgemma-4b-it"]
PILOT_PER_WRITER = {"lay": 10, "professional": 8, "unknown": 7}
ARMS = ["0", "A", "B", "C"]
REPO = Path(__file__).resolve().parent.parent


def frontier_subsample(unseen, task):
    """Fixed SUB-report subsample, as even across writers as the data allows."""
    path = REPO / "data" / f"frontier_subsample_{task}.tsv"
    if path.exists():
        keys = pd.read_csv(path, sep="\t", dtype=str).mdr_report_key
        return unseen.loc[keys]
    rng = np.random.RandomState(SEED)
    groups = {w: g.sample(frac=1, random_state=rng) for w, g in unseen.groupby("writer")}
    quota, left, pending = {}, SUB, sorted(groups, key=lambda w: len(groups[w]))
    for i, w in enumerate(pending):  # smallest group first; give the remainder evenly to the rest
        quota[w] = min(len(groups[w]), left // (len(pending) - i))
        left -= quota[w]
    sub = pd.concat(groups[w].head(quota[w]) for w in pending)
    sub[["set", "writer"]].to_csv(path, sep="\t")
    print(f"{task}: frontier subsample {len(sub)} saved to {path.name}: {sub.writer.value_counts().to_dict()}")
    return sub


def main():
    vol_dir, ctx_dir, codebook, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    book = load_codebook(codebook)
    jobs = []
    for task in ("device", "patient"):
        d = pd.read_parquet(vol_dir / f"{task}.parquet", columns=["mdr_report_key", "set", "writer", "text_event",
                                                                   f"{task}_codes"])
        text = d.set_index("mdr_report_key").text_event
        codes = d.set_index("mdr_report_key")[f"{task}_codes"]
        ctx = pd.read_parquet(ctx_dir / f"context_{task}.parquet").set_index("mdr_report_key")
        unseen = ctx[ctx.set == "unseen"]
        if mode == "pilot":
            pick = pd.concat(g.sample(n=PILOT_PER_WRITER[w], random_state=SEED) for w, g in unseen.groupby("writer"))
            plan = [(pick, ARMS, PILOT_OPEN + FRONTIER)]
        elif mode == "smoke":
            plan = [(unseen.sample(n=3, random_state=SEED), ["A", "B"], [m for m in OPEN if m not in PILOT_OPEN])]
        elif mode == "stage1":
            plan = [(unseen, ARMS, OPEN), (ctx[ctx.set == "comparison"], ["0", "A"], OPEN)]
        elif mode == "stage2":
            sub = frontier_subsample(unseen, task)
            plan = [(sub, ["A"], FRONTIER), (sub.head(100), ["0"], FRONTIER)]
        elif mode == "stage3":
            plan = [(frontier_subsample(unseen, task), ["C"], FRONTIER)]
        elif mode == "stage4":
            plan = [(frontier_subsample(unseen, task), ["B"], FRONTIER)]
        elif mode == "medgemma":
            plan = [(unseen, ARMS, SELF_HOSTED), (ctx[ctx.set == "comparison"], ["0", "A"], SELF_HOSTED)]
        else:
            raise ValueError(mode)
        for rows, arms, models in plan:
            for key, r in rows.iterrows():
                report = {"text_event": text[key]}
                examples = [{"text": text[k], "codes": list(codes[k])} for k in r.example_keys]
                for arm in arms:
                    msgs = build_messages(task, arm, report, book,
                                          shortlist=list(r.shortlist_B if arm == "B" else r.shortlist_C),
                                          examples=examples if arm == "C" else None)
                    for m in models:
                        jobs.append({"id": f"{m}|{task}|{key}|{arm}", "model": m, "task": task, "key": key,
                                     "arm": arm, "set": r.set, "writer": r.writer,
                                     "prompt_version": VERSION, "messages": msgs})
    with open(out, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs written")
    print(pd.DataFrame(jobs).groupby(["task", "arm"]).size().unstack("arm").to_string())


if __name__ == "__main__":
    main()
