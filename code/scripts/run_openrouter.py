"""Send prompt jobs to OpenRouter and append results to a JSONL file.

Usage:
    python -I run_openrouter.py ENV_FILE JOBS.jsonl RESULTS.jsonl [WORKERS]

Each job line: {"id", "model", "messages", "params"}; the model's provider pin
and settings come from MODELS below. Jobs whose id is already in RESULTS are
skipped, so an interrupted run can simply be restarted. The API key is read
from ENV_FILE and never printed or written.
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import dotenv_values

# One pinned provider per model, at paper 3's precision. No fallbacks, so a
# result is never silently served by another host.
MODELS = {
    "meta-llama/llama-3.1-8b-instruct": {"order": ["DeepInfra"], "quantizations": ["fp8"]},
    "microsoft/phi-4": {"order": ["DeepInfra"], "quantizations": ["bf16"]},
    "meta-llama/llama-3.3-70b-instruct": {"order": ["DeepInfra"], "quantizations": ["fp8"]},
    "qwen/qwen-2.5-72b-instruct": {"order": ["DeepInfra"], "quantizations": ["fp8"]},
    "meta-llama/llama-4-maverick": {"order": ["Parasail"], "quantizations": ["fp8"]},
    "openai/gpt-oss-120b": {"order": ["Nebius"], "quantizations": ["fp4"]},
    "anthropic/claude-opus-4.8": {"order": ["Anthropic"]},
    "openai/gpt-5.5": {"order": ["OpenAI"]},
}
REASONING = {  # models that think; effort fixed and recorded
    "openai/gpt-oss-120b": {"effort": "low"},
    "anthropic/claude-opus-4.8": {"effort": "low"},
    "openai/gpt-5.5": {"effort": "low"},
}

lock = threading.Lock()
stop = threading.Event()  # set when the spend cap is reached; pending jobs then return unsent


def call(env, job, retries=5):
    model = job["model"]
    if stop.is_set():
        return {"id": job["id"], "model": model, "ok": False, "error": "budget cap reached, not sent"}
    prov = dict(MODELS[model])
    prov["allow_fallbacks"] = False
    messages = job["messages"]
    if model.startswith("anthropic/") and messages and messages[0]["role"] == "system":
        # Anthropic caches only on request: mark the system prompt (instructions and,
        # in Arm A, the whole codebook) as a cacheable prefix. OpenAI caches automatically.
        sys_msg = dict(messages[0])
        sys_msg["content"] = [{"type": "text", "text": messages[0]["content"],
                               "cache_control": {"type": "ephemeral"}}]
        messages = [sys_msg] + messages[1:]
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": 4000 if model in REASONING else 1000,
            "provider": prov, "usage": {"include": True}}
    if model in REASONING:
        body["reasoning"] = REASONING[model]
    body.update(job.get("params", {}))
    headers = {"Authorization": f"Bearer {env['OPENROUTER_API_KEY']}",
               "HTTP-Referer": env.get("OPENROUTER_REFERER", ""),
               "X-Title": env.get("OPENROUTER_TITLE", "")}
    url = env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=300)
            if r.status_code == 200:
                j = r.json()
                if "choices" in j:
                    ch = j["choices"][0]
                    return {"id": job["id"], "model": model, "ok": True,
                            "content": ch["message"].get("content"),
                            "finish_reason": ch.get("finish_reason"),
                            "provider": j.get("provider"), "usage": j.get("usage"),
                            "latency_s": round(time.time() - t0, 2), "attempts": attempt + 1}
                err = str(j)[:300]
            else:
                err = f"{r.status_code} {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break
        except requests.RequestException as e:
            err = repr(e)[:300]
        time.sleep(min(60, 2 ** attempt * 3))
    return {"id": job["id"], "model": model, "ok": False, "error": err}


def main():
    env = dotenv_values(sys.argv[1])
    jobs_path, out_path = Path(sys.argv[2]), Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else float("inf")  # dollars; stop sending past this
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip() and json.loads(l).get("ok")}
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run", flush=True)
    n_ok = n_err = 0
    cost = 0.0
    with ThreadPoolExecutor(workers) as ex, open(out_path, "a", encoding="utf-8") as out:
        futs = [ex.submit(call, env, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            res = fu.result()
            with lock:
                out.write(json.dumps(res) + "\n")
                out.flush()
            if res["ok"]:
                n_ok += 1
                cost += (res.get("usage") or {}).get("cost", 0) or 0
                if cost >= budget and not stop.is_set():
                    stop.set()
                    print(f"  budget cap ${budget:.2f} reached at ${cost:.4f}; remaining jobs not sent", flush=True)
            else:
                n_err += 1
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {n_err} errors, ${cost:.4f}", flush=True)
    print(f"finished: {n_ok} ok, {n_err} errors, cost ${cost:.4f}")


if __name__ == "__main__":
    main()
