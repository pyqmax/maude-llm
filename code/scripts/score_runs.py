"""Score voluntary-set runs against FDA's codes.

Usage:
    python -I score_runs.py JOBS.jsonl RESULTS.jsonl CONTEXT_DIR CODEBOOK.tsv OUT.tsv

Per call: parsed, invalid_codes (not in the codebook), empty, n_pred, over_limit,
then the metrics of metrics.py on the first MAX_CODES valid codes.
Also the no-LLM baselines on the same reports: product prior top-1, nearest
neighbour's codes, k-NN vote top-1. And cache statistics from the usage field.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import METRICS, score  # noqa: E402
from prompts import load_codebook  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
MAX_CODES = 5


def parse(content):
    if not content:
        return None
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None
    try:
        codes = json.loads(m.group(0)).get("codes")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(codes, list):
        return None
    return [str(c).strip().split()[0].upper().rstrip(".,;:") for c in codes if str(c).strip()]


def main():
    jobs_p, res_p, ctx_dir, codebook, out = sys.argv[1:6]
    book = load_codebook(codebook)
    valid = {"device": {r["imdrf_code"] for r in book["A"]},
             "patient": {r["imdrf_code"] for r in book["E"]}}
    jobs = {j["id"]: j for j in map(json.loads, open(jobs_p, encoding="utf-8"))}
    ctx = {t: pd.read_parquet(Path(ctx_dir) / f"context_{t}.parquet").set_index("mdr_report_key")
           for t in ("device", "patient")}
    latest = {}
    for r in map(json.loads, open(res_p, encoding="utf-8")):
        if r["id"] not in jobs:  # results file may hold answers from other job files
            continue
        if r["ok"] or not latest.get(r["id"], {}).get("ok"):
            latest[r["id"]] = r
    rows = []
    for r in latest.values():
        j = jobs[r["id"]]
        filed = list(ctx[j["task"]].at[j["key"], "filed"])
        u = r.get("usage") or {}
        pd_ = u.get("prompt_tokens_details") or {}
        base = {"model": j["model"].split("/")[-1], "task": j["task"], "arm": j["arm"], "set": j["set"],
                "writer": j["writer"], "key": j["key"], "ok": r["ok"],
                "in_tok": u.get("prompt_tokens"), "cached_tok": pd_.get("cached_tokens"),
                "out_tok": u.get("completion_tokens"),
                "reason_tok": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": u.get("cost")}
        pred = parse(r.get("content")) if r["ok"] else None
        if pred is None:
            rows.append({**base, "parsed": 0})
            continue
        bad = [c for c in pred if c not in valid[j["task"]]]
        good = list(dict.fromkeys(c for c in pred if c in valid[j["task"]]))
        rows.append({**base, "parsed": 1, "invalid_codes": int(bool(bad)), "n_invalid": len(bad),
                     "empty": int(not good), "n_pred": len(good), "over_limit": int(len(good) > MAX_CODES),
                     **score(good[:MAX_CODES], filed)})
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)

    pd.set_option("display.width", 250)
    met = ["parsed", "invalid_codes", "empty", "n_pred"] + METRICS
    print("=== by task x arm x model ===")
    print(df.groupby(["task", "arm", "model"])[met].mean().round(2).to_string())
    print("\n=== soft recall by task x writer x arm x model ===")
    print(df.groupby(["task", "writer", "arm", "model"]).soft_recall.mean().round(2)
          .unstack("model").to_string())

    print("\n=== no-LLM baselines on the same reports ===")
    for t in ("device", "patient"):
        keys = df[df.task == t].key.unique()
        s = ctx[t].loc[keys]
        for name in ("prior_top1", "nn_codes", "knn_top1"):
            sc = pd.DataFrame([score(list(p), list(f)) for p, f in zip(s[name], s.filed)])
            print(f"{t:8s} {name:10s} " + ", ".join(f"{m} {sc[m].mean():.2f}" for m in METRICS))
        print(f"{t:8s} shortlist coverage: B {s.covered_B.mean():.2f}, C {s.covered_C.mean():.2f}")

    print("\n=== tokens, cache and cost per call ===")
    print(df.groupby(["model", "task", "arm"])[["in_tok", "cached_tok", "out_tok", "reason_tok", "cost"]]
          .mean().round(4).to_string())
    print(f"\ntotal cost ${df.cost.sum():.3f}")


if __name__ == "__main__":
    main()
