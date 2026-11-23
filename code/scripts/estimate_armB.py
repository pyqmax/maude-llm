"""Estimate the cost of running Arm B for the two frontier models on the frontier subsample.

Prices per token are fitted from each model's uncached calls (Arm 0 and Arm C; Arm A used
prompt caching). The Arm B prompt length is taken from Qwen's Arm B calls on the same reports,
scaled by the model's own tokenizer ratio measured on Arm C.
"""
import json
import sys
from collections import defaultdict

results = sys.argv[1]
FRONTIER = ["anthropic/claude-opus-4.8", "openai/gpt-5.5"]
ref_model = "qwen/qwen-2.5-72b-instruct"

rows = defaultdict(list)          # (model, task, arm) -> [(key, prompt, compl, cost)]
with open(results, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        model, task, key, arm = r["id"].split("|")
        u = r.get("usage") or {}
        rows[(model, task, arm)].append((key, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost", 0) or 0))

def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def fit_prices(m):
    # least squares on uncached calls: cost = p*prompt + c*compl
    data = [x for task in ["device", "patient"] for arm in ["0", "C"] for x in rows[(m, task, arm)]]
    spp = sum(x[1] * x[1] for x in data); spc = sum(x[1] * x[2] for x in data); scc = sum(x[2] * x[2] for x in data)
    sp = sum(x[1] * x[3] for x in data); sc = sum(x[2] * x[3] for x in data)
    det = spp * scc - spc * spc
    p = (sp * scc - sc * spc) / det
    c = (spp * sc - spc * sp) / det
    resid = mean([abs(x[3] - p * x[1] - c * x[2]) / x[3] for x in data if x[3] > 0])
    return p, c, resid

total = 0
for m in FRONTIER:
    p, c, resid = fit_prices(m)
    print(f"{m}: prompt ${p*1e6:.2f}/M, completion ${c*1e6:.2f}/M (mean abs rel error {resid:.1%})")
    for task in ["device", "patient"]:
        keys = {x[0] for x in rows[(m, task, "C")]}
        qc = {x[0]: x[1] for x in rows[(ref_model, task, "C")] if x[0] in keys}
        mc = {x[0]: x[1] for x in rows[(m, task, "C")]}
        ratio = mean([mc[k] / qc[k] for k in keys if k in qc and qc[k]])
        qb = [x[1] for x in rows[(ref_model, task, "B")] if x[0] in keys]
        pb = mean(qb) * ratio
        cb = mean([x[2] for x in rows[(m, task, "A")] + rows[(m, task, "C")]])
        est = len(keys) * (pb * p + cb * c)
        total += est
        print(f"  {task:8s}: n={len(keys)} tokenizer ratio vs Qwen {ratio:.2f}, Arm B prompt ~{pb:.0f} tokens, completion ~{cb:.0f}  ->  ${est:.2f}")
print(f"total ~ ${total:.2f}")
