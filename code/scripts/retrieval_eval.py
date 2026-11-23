"""Offline check of training-free candidate retrieval: how often is the filed
code among the top-k retrieved codebook entries?

Usage:
    python -I retrieval_eval.py SAMPLE_DIR CODEBOOK.tsv OUT.tsv [SET] [MODEL_DIR]

Run on the cluster (GPU if present), not the laptop.

SET is the sample set to evaluate on (default "comparison", so that choices
about the retriever and k are not made on the unseen set).

Retrievers: BM25 over term + definition + parent terms; dense (bge-base-en-v1.5)
over the same text; hybrid = reciprocal-rank fusion of the two.
Queries: the event narrative alone, and event plus manufacturer narrative.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")
EXCLUDE = {"A24", "A25", "A26", "A27", "E2401", "E2402"}
KS = (5, 10, 20, 30, 50)
DENSE = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def load_codebook(path, annex):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["annex"] == annex and r["primary"] == "True" and r["imdrf_code"] not in EXCLUDE:
                rows.append(r)
    docs = [f"{r['term']}. {r['definition']} ({r['path_terms']})" for r in rows]
    return [r["imdrf_code"] for r in rows], docs


def related(a, b):
    """Same branch: one code is an ancestor of (or equal to) the other."""
    return a.startswith(b) or b.startswith(a)


def rrf(*rankings, c=60):
    score = {}
    for rk in rankings:
        for i, j in enumerate(rk):
            score[j] = score.get(j, 0.0) + 1.0 / (c + i + 1)
    return sorted(score, key=score.get, reverse=True)


def main():
    sample_dir, codebook, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    which = sys.argv[4] if len(sys.argv) > 4 else "comparison"
    model_dir = sys.argv[5] if len(sys.argv) > 5 else DENSE
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    model = SentenceTransformer(model_dir, device=device)
    rows = []
    for task, annex, col in (("device", "A", "device_codes"), ("patient", "E", "patient_codes")):
        codes, docs = load_codebook(codebook, annex)
        bm25 = BM25Okapi([tok(d) for d in docs])
        demb = model.encode(docs, batch_size=64, normalize_embeddings=True)
        df = pd.read_parquet(sample_dir / f"{task}.parquet")
        df = df[df.set == which].reset_index(drop=True)
        print(f"{task}: {len(df)} reports ({which}), {len(codes)} candidate codes", flush=True)
        for cond in ("event", "event+mfr"):
            q = df.text_event.fillna("")
            if cond == "event+mfr":
                q = q + " " + df.text_mfr.fillna("")
            qemb = model.encode([QUERY_PREFIX + s for s in q], batch_size=32,
                                normalize_embeddings=True, show_progress_bar=False)
            dense_rank = np.argsort(-(qemb @ demb.T), axis=1)
            bm25_rank = [list(np.argsort(-bm25.get_scores(tok(t)))) for t in q]
            print(f"  {cond}: embedded and scored", flush=True)
            for name in ("bm25", "dense", "hybrid"):
                hits = {k: [] for k in KS}
                fam = {k: [] for k in KS}
                expd = {k: [] for k in KS}
                for i in range(len(q)):
                    if name == "dense":
                        rk = list(dense_rank[i])
                    elif name == "bm25":
                        rk = bm25_rank[i]
                    else:
                        rk = rrf(bm25_rank[i], list(dense_rank[i]))
                    ranked = [codes[j] for j in rk]
                    filed = set(df.at[i, col])
                    for k in KS:
                        top = set(ranked[:k])
                        hits[k].append(len(filed & top) / len(filed))
                        # filed code on the same branch as some retrieved code
                        fam[k].append(sum(any(related(f, t) for t in top) for f in filed) / len(filed))
                        # retrieved codes plus all their ancestors
                        anc = {t[:n] for t in top for n in range(3, len(t) + 1, 2)}
                        expd[k].append(len(filed & anc) / len(filed))
                rec = {"task": task, "set": which, "query": cond, "retriever": name}
                rec.update({f"recall@{k}": round(float(np.mean(hits[k])), 3) for k in KS})
                rec.update({f"branch@{k}": round(float(np.mean(fam[k])), 3) for k in KS})
                rec.update({f"withparents@{k}": round(float(np.mean(expd[k])), 3) for k in KS})
                rows.append(rec)
                print(rec, flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
