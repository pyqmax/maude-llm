"""Estimate token counts and OpenRouter cost for the planned runs.

Usage: python -I cost_estimate.py CODEBOOK.tsv SAMPLE_DIR

Token counts use tiktoken cl100k_base as a proxy (model tokenizers differ by
roughly +-20%).
"""
import csv
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.stdout.reconfigure(encoding="utf-8")
enc = tiktoken.get_encoding("cl100k_base")
EXCLUDE = {"A24", "A25", "A26", "A27", "E2401", "E2402"}
INSTR = 350        # instructions + output format
OUT = 120          # answer tokens (non-reasoning)
SHORT = 35         # codes in the shortlist
ESCAPE = 0.25      # share of shortlist calls that go to the escape step
ESC_IN = 1200      # escape steps: 27 groups, then one group's children

n = lambda s: len(enc.encode(s))
book = {"A": [], "E": []}
with open(sys.argv[1], encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["primary"] == "True" and r["imdrf_code"] not in EXCLUDE:
            book[r["annex"]].append(r)
size = {}
for a, rows in book.items():
    names = n("\n".join(f"{'  ' * (int(r['level']) - 1)}{r['imdrf_code']} {r['term']}" for r in rows))
    per_def = sum(n(f"{r['imdrf_code']} {r['term']}: {r['definition']}") for r in rows) / len(rows)
    size[a] = (names, per_def)
    print(f"Annex {a}: {len(rows)} codes, full name list {names} tokens, {per_def:.0f} tokens per code with definition")

samp = Path(sys.argv[2])
calls = {}
for task, a in (("device", "A"), ("patient", "E")):
    d = pd.read_parquet(samp / f"{task}.parquet", columns=["text_event", "text_mfr"])
    ev = d.text_event.fillna("").map(n).mean()
    mf = d.text_mfr.fillna("").map(n).mean()
    names, per_def = size[a]
    for cond, narr in (("event", ev), ("event+mfr", ev + mf)):
        full = INSTR + names + narr
        short = INSTR + SHORT * per_def + narr + ESCAPE * (ESC_IN + narr)
        calls[(task, cond)] = (len(d), full, short)
        print(f"{task:7} {cond:9} n={len(d):,} narrative {narr:.0f} tok | full-list prompt {full:,.0f} | shortlist prompt {short:,.0f}")

# $ per million tokens (input, output), pinned providers from OpenRouter 2026-11-12
PRICES = {
    "Llama 3.1 8B (DeepInfra)": (0.02, 0.04, 0),
    "Phi-4 (DeepInfra)": (0.07, 0.14, 0),
    "Llama 3.3 70B (DeepInfra)": (0.10, 0.32, 0),
    "Qwen 2.5 72B (DeepInfra)": (0.36, 0.40, 0),
    "Llama 4 Maverick (Parasail)": (0.35, 1.00, 0),
    "gpt-oss-120b (fp4 host)": (0.15, 0.60, 600),   # + hidden reasoning tokens
}
FRONTIER = {  # batch 50% off; reasoning effort low assumed ~300 extra out tokens
    "Claude Opus 4.8": (2.5, 12.5, 300),
    "GPT-5.5": (2.5, 15.0, 300),
}


def cost(pin, pout, extra, arms):
    tot = 0.0
    for (task, cond), (k, full, short) in calls.items():
        for arm in arms:
            tin = full if arm == "full" else short
            tot += k * (tin * pin + (OUT + extra) * pout) / 1e6
    return tot


print("\nOpen panel, both arms, both input conditions:")
open_total = 0
for m, (pi, po, ex) in PRICES.items():
    c = cost(pi, po, ex, ("full", "short"))
    open_total += c
    print(f"  {m:30} ${c:8.2f}")
print(f"  {'MedGemma 4B (Fir, self-hosted)':30} $    0.00")
print(f"  open total ${open_total:.0f}")
print("\nFrontier reference, shortlist arm only (batch prices):")
fr = 0
for m, (pi, po, ex) in FRONTIER.items():
    c = cost(pi, po, ex, ("short",))
    fr += c
    print(f"  {m:30} ${c:8.2f}   (both arms would be ${cost(pi, po, ex, ('full', 'short')):.0f})")
print(f"  frontier total ${fr:.0f}")
print(f"\nGRAND TOTAL about ${open_total + fr:.0f}")
