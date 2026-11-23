"""Draw a random sample of disagreements for hand reading.

Usage: python -I draw_cases.py VOLUNTARY_DIR CONTEXT_DIR CODEBOOK.tsv RESULTS.jsonl OUT_DIR [N=30]

For each task, among the frontier-subsample reports where Claude Opus 4.8 in Arm C did
not return every FDA code (exact recall < 1), draw N at random (seed 20261114) and print:
the narrative, FDA's codes with terms, and the Arm C answers of Opus, GPT-5.5 and
Qwen 2.5 72B, plus Opus's Arm A answer. Also prints how many reports were eligible.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402
from prompts import load_codebook  # noqa: E402
from score_runs import parse  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SHOW = [("anthropic/claude-opus-4.8", "C"), ("openai/gpt-5.5", "C"),
        ("qwen/qwen-2.5-72b-instruct", "C"), ("anthropic/claude-opus-4.8", "A")]


def main():
    vol_dir, ctx_dir, codebook, res_p, out_dir = map(Path, sys.argv[1:6])
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    out_dir.mkdir(parents=True, exist_ok=True)
    book = load_codebook(str(codebook))
    name = {r["imdrf_code"]: r["term"] for a in book.values() for r in a}
    res = {}
    for r in map(json.loads, open(res_p, encoding="utf-8")):
        if r["ok"] or r["id"] not in res:
            res[r["id"]] = r
    rng = np.random.default_rng(20261114)
    for task in ("device", "patient"):
        vol = pd.read_parquet(vol_dir / f"{task}.parquet", columns=["mdr_report_key", "writer", "product_code",
                                                                    "event_type", "date_received", "text_event",
                                                                    f"{task}_codes"]).set_index("mdr_report_key")
        sub = pd.read_csv(ROOT / "data" / f"frontier_subsample_{task}.tsv", sep="\t", dtype=str).mdr_report_key
        eligible = []
        for k in sub:
            r = res.get(f"anthropic/claude-opus-4.8|{task}|{k}|C")
            pred = parse(r.get("content")) if r and r["ok"] else None
            filed = list(vol.at[k, f"{task}_codes"])
            rec = score([c for c in (pred or []) if c in name][:5], filed)["recall"]
            if rec < 1:
                eligible.append(k)
        pick = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
        lines = [f"{task}: {len(eligible)} of {len(sub)} subsample reports have an Opus Arm C disagreement; {len(pick)} drawn\n"]
        for i, k in enumerate(pick, 1):
            v = vol.loc[k]
            lines.append("=" * 100)
            lines.append(f"[{i}] {k}  writer={v.writer}  product={v.product_code}  {v.event_type}  {v.date_received}\n")
            lines.append("EVENT: " + re.sub(r"\s+", " ", v.text_event or "") + "\n")
            lines.append("FDA: " + "; ".join(f"{c} {name.get(c, '?')}" for c in v[f"{task}_codes"]))
            for model, arm in SHOW:
                r = res.get(f"{model}|{task}|{k}|{arm}")
                pred = parse(r.get("content")) if r and r["ok"] else None
                shown = "(unparsed)" if pred is None else ("(none)" if not pred else
                        "; ".join(f"{c} {name.get(c, '??')}" for c in pred))
                lines.append(f"  {model.split('/')[-1]:24s} arm {arm}: {shown}")
            lines.append("  VERDICT: ")
            lines.append("")
        (out_dir / f"cases_read_{task}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(lines[0])


if __name__ == "__main__":
    main()
