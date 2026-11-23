"""Answer prompt jobs with self-hosted MedGemma 4B IT on a Fir GPU, batched, greedy.

Usage: python -P run_medgemma.py MODEL_DIR JOBS.jsonl RESULTS.jsonl [TOKEN_BUDGET] [MAX_NEW]

Same job and result schema as run_openrouter.py, so score_runs.py reads the output
unchanged (provider "fir_self_hosted", cost 0). Jobs already answered in RESULTS are
skipped, so a job that hits its time limit is simply resubmitted. Prompts are sorted
by length and packed into batches of at most TOKEN_BUDGET prompt tokens (default
48k) so the full-codebook arm (about 8k tokens a prompt) and the short arms both use
the GPU well. Decoding is greedy (temperature 0 equivalent), as for the API models.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_NAME = "google/medgemma-4b-it"
MAX_BATCH = 32


def main():
    model_dir, jobs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 48_000
    max_new = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip() and json.loads(l).get("ok")}
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"loaded {MODEL_NAME} on {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.mem_get_info()[1] / 1e9:.0f} GB", flush=True)

    # render every chat with the model's own template; the Gemma 3 template folds the
    # system turn into the first user turn itself
    texts = []
    for j in todo:
        msgs = [{"role": m["role"], "content": m["content"]} for m in j["messages"]]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        del j["messages"]  # keep host memory down: only the id is needed from here on
    del jobs
    lengths = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(todo)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:  # ascending length, so the newest prompt sets the padded width
        if cur and ((len(cur) + 1) * lengths[i] > budget or len(cur) >= MAX_BATCH):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    print(f"{len(batches)} batches; longest prompt {max(lengths)} tokens", flush=True)

    n_ok = n_err = 0
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for b, idx in enumerate(batches, 1):
            t0 = time.time()
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            try:
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                new = gen[:, enc.input_ids.shape[1]:]
                for k, i in enumerate(idx):
                    ids = new[k]
                    n_comp = int((ids != tok.pad_token_id).sum())
                    text = tok.decode(ids, skip_special_tokens=True)
                    finish = "length" if n_comp >= max_new else "stop"
                    out.write(json.dumps({
                        "id": todo[i]["id"], "model": MODEL_NAME, "ok": True, "content": text,
                        "finish_reason": finish, "provider": "fir_self_hosted",
                        "usage": {"prompt_tokens": lengths[i], "completion_tokens": n_comp, "cost": 0.0},
                        "latency_s": round((time.time() - t0) / len(idx), 3), "attempts": 1}) + "\n")
                    n_ok += 1
            except Exception as e:  # noqa: BLE001
                for i in idx:
                    out.write(json.dumps({"id": todo[i]["id"], "model": MODEL_NAME, "ok": False,
                                          "error": repr(e)[:300]}) + "\n")
                    n_err += 1
                torch.cuda.empty_cache()
            out.flush()
            if b % 10 == 0 or b == len(batches):
                el = time.time() - t_start
                print(f"  batch {b}/{len(batches)}, {n_ok} ok, {n_err} errors, {el / 60:.1f} min", flush=True)
    print(f"finished: {n_ok} ok, {n_err} errors")


if __name__ == "__main__":
    main()
