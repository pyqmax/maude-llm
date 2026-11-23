"""Score pilot results against the filed codes.

Usage:
    python -I score_pilot.py JOBS.jsonl RESULTS.jsonl SHORTLIST_DIR CODEBOOK.tsv OUT.tsv

Per call: parsed (valid JSON with a code list), invalid_codes (codes not in the
codebook), empty (no code), recall (share of filed codes predicted exactly),
branch_recall (filed code matched by a predicted code on the same branch, i.e.
ancestor or descendant), precision, exact (predicted set == filed set).
Also the no-LLM prior baseline (product's most common code) on the same reports.
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
    return [str(c).strip().split()[0].upper() for c in codes if str(c).strip()]


MAX_CODES = 5  # the prompt allows at most five; extra codes are not scored


def main():
    jobs_p, res_p, short_dir, codebook, out = sys.argv[1:6]
    book = load_codebook(codebook)
    valid = {"device": {r["imdrf_code"] for r in book["A"]},
             "patient": {r["imdrf_code"] for r in book["E"]}}
    jobs = {j["id"]: j for j in map(json.loads, open(jobs_p, encoding="utf-8"))}
    sl = {t: pd.read_parquet(Path(short_dir) / f"shortlist_{t}.parquet").query("set == 'unseen'")
          .set_index("mdr_report_key") for t in ("device", "patient")}
    latest = {}  # one result per job: a success beats a failure, later beats earlier
    for r in map(json.loads, open(res_p, encoding="utf-8")):
        if r["ok"] or not latest.get(r["id"], {}).get("ok"):
            latest[r["id"]] = r
    rows = []
    for r in latest.values():
        j = jobs[r["id"]]
        filed = list(sl[j["task"]].at[j["key"], "filed"])
        u = r.get("usage") or {}
        base = {"model": j["model"].split("/")[-1], "task": j["task"], "arm": j["arm"],
                "cond": j["cond"], "group": j["group"], "key": j["key"], "ok": r["ok"],
                "in_tok": u.get("prompt_tokens"), "out_tok": u.get("completion_tokens"),
                "reason_tok": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": u.get("cost")}
        pred = parse(r.get("content")) if r["ok"] else None
        if pred is None:
            rows.append({**base, "parsed": 0})
            continue
        bad = [c for c in pred if c not in valid[j["task"]]]
        good = list(dict.fromkeys(c for c in pred if c in valid[j["task"]]))
        rows.append({**base, "parsed": 1, "invalid_codes": int(bool(bad)), "empty": int(not good),
                     "n_pred": len(good), "over_limit": int(len(good) > MAX_CODES),
                     **score(good[:MAX_CODES], filed)})
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)

    pd.set_option("display.width", 250)
    met = ["parsed", "invalid_codes", "empty", "n_pred", "over_limit"] + METRICS
    print("=== by model x task x arm (both input conditions) ===")
    print(df.groupby(["task", "arm", "model"])[met].mean().round(2).to_string())
    print("\n=== by model x task x cond (arm A) ===")
    print(df[df.arm == "A"].groupby(["task", "cond", "model"])[["recall", "branch_recall", "soft_recall", "exact"]]
          .mean().round(2).to_string())
    print("\n=== by group (open models, arm A, event) ===")
    print(df[(df.arm == "A") & (df.cond == "event")].groupby(["task", "group", "model"])[
        ["recall", "branch_recall"]].mean().round(2).unstack("model").to_string())

    # prior baseline on the same pilot reports
    print("\n=== no-LLM baseline: product's most common code ===")
    for t in ("device", "patient"):
        keys = df[df.task == t].key.unique()
        s = sl[t].loc[keys]
        sc = pd.DataFrame([score(list(p), list(f)) for p, f in zip(s.prior_top1, s.filed)])
        print(f"{t}: " + ", ".join(f"{m} {sc[m].mean():.2f}" for m in METRICS))

    print("\n=== tokens and cost per call ===")
    print(df.groupby(["model", "task", "arm"])[["in_tok", "out_tok", "reason_tok", "cost"]]
          .mean().round(4).to_string())
    print(f"\npilot total cost ${df.cost.sum():.3f}")


if __name__ == "__main__":
    main()
