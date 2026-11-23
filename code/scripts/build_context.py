"""Build the retrieval context for every test report in the voluntary set:
the Arm B shortlist, the Arm C examples, and the no-LLM baselines.

Usage:
    python -P build_context.py VOLUNTARY_DIR CODEBOOK.tsv MODEL_DIR OUT_DIR

For each task, the pool (set == "pool", voluntary 2021 to 2025) supplies:
  prior     codes most often filed for the report's product code (global ranking
            if the product has under MIN_PC pool reports)
  retrieval top N_RETR codes by similarity between the narrative and the code
            definitions (dense for device, BM25+dense for patient, as chosen on
            the earlier comparison)
  examples  the K pool narratives most similar to the report, with their FDA
            codes; near-duplicates (cosine >= DUP_SIM, or same text_key) are
            skipped so the examples cannot simply be copied
Shortlist B = prior top N_PRIOR + retrieval top N_RETR + ancestors.
Shortlist C = shortlist B + the examples' codes + ancestors.
Baselines: prior top-1; nearest neighbour's codes; k-NN vote top-1.

Test reports are set == "unseen" and set == "comparison". Writes
OUT_DIR/context_{task}.parquet and OUT_DIR/context_coverage.tsv.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EXCLUDE = {"A24", "A25", "A26", "A27", "E2401", "E2402"}
N_PRIOR, N_RETR, K, MIN_PC, DUP_SIM = 20, 10, 8, 20, 0.95
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def anc(c):
    return [c[:n] for n in range(3, len(c) + 1, 2)]


def rrf(*rankings, c=60):
    score = {}
    for rk in rankings:
        for i, j in enumerate(rk):
            score[j] = score.get(j, 0.0) + 1.0 / (c + i + 1)
    return sorted(score, key=score.get, reverse=True)


def cover(filed, codes):
    return len(filed & set(codes)) / len(filed)


def main():
    vol_dir, codebook, model_dir, out_dir = map(Path, sys.argv[1:5])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    model = SentenceTransformer(str(model_dir), device=dev)

    book = {"A": [], "E": []}
    with open(codebook, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["primary"] == "True" and r["imdrf_code"] not in EXCLUDE:
                book[r["annex"]].append(r)

    cov_rows = []
    for task, annex, col, method in (("device", "A", "device_codes", "dense"),
                                     ("patient", "E", "patient_codes", "hybrid")):
        rows = book[annex]
        codes = [r["imdrf_code"] for r in rows]
        order = {c: i for i, c in enumerate(codes)}
        valid = set(codes)
        docs = [f"{r['term']}. {r['definition']} ({r['path_terms']})" for r in rows]
        demb = model.encode(docs, batch_size=64, normalize_embeddings=True)
        bm25 = BM25Okapi([tok(d) for d in docs])

        d = pd.read_parquet(vol_dir / f"{task}.parquet",
                            columns=["mdr_report_key", "set", "writer", "product_code", "text_key",
                                     "text_event", col])
        pool = d[d.set == "pool"].reset_index(drop=True)
        test = d[d.set != "pool"].reset_index(drop=True)
        print(f"{task}: pool {len(pool):,}, test {len(test):,}", flush=True)

        by_pc, glob = defaultdict(Counter), Counter()
        for p, cs in zip(pool.product_code, pool[col]):
            by_pc[p].update(cs)
            glob.update(cs)
        glob_rank = [c for c, _ in glob.most_common() if c in valid]

        pemb = model.encode([t or "" for t in pool.text_event], batch_size=128,
                            normalize_embeddings=True, show_progress_bar=False)
        qemb = model.encode([QUERY_PREFIX + (t or "") for t in test.text_event], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=False)
        temb = model.encode([t or "" for t in test.text_event], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=False)
        code_rank = np.argsort(-(qemb @ demb.T), axis=1)
        pool_sim = temb @ pemb.T  # symmetric narrative-to-narrative similarity
        pool_codes = list(pool[col])
        pool_keys = list(pool.mdr_report_key)
        pool_tk = list(pool.text_key)

        out = []
        for i, r in enumerate(test.itertuples(index=False)):
            pc_hist = by_pc.get(r.product_code)
            if pc_hist and sum(pc_hist.values()) >= MIN_PC:
                prior = [c for c, _ in pc_hist.most_common() if c in valid]
                prior += [c for c in glob_rank if c not in pc_hist]
                fallback = False
            else:
                prior, fallback = glob_rank, True
            if method == "dense":
                rk = list(code_rank[i])
            else:
                b = list(np.argsort(-bm25.get_scores(tok(r.text_event or ""))))
                rk = rrf(b, list(code_rank[i]))
            retr = [codes[j] for j in rk[:N_RETR]]

            sims = pool_sim[i]
            ex_idx, ex_sim, skipped = [], [], 0
            for j in np.argsort(-sims)[:K + 50]:
                if sims[j] >= DUP_SIM or pool_tk[j] == r.text_key or pool_keys[j] == r.mdr_report_key:
                    skipped += 1
                    continue
                ex_idx.append(int(j))
                ex_sim.append(float(sims[j]))
                if len(ex_idx) == K:
                    break
            ex_codes = [list(pool_codes[j]) for j in ex_idx]

            base_b = set(prior[:N_PRIOR]) | set(retr)
            full_b = {a for c in base_b for a in anc(c) if a in valid}
            full_c = full_b | {a for cs in ex_codes for c in cs for a in anc(c) if a in valid}
            filed = set(getattr(r, col))
            vote = Counter(c for cs in ex_codes for c in cs)
            knn1 = [vote.most_common(1)[0][0]] if vote else []
            out.append({
                "mdr_report_key": r.mdr_report_key, "set": r.set, "writer": r.writer,
                "product_code": r.product_code, "filed": sorted(filed),
                "shortlist_B": sorted(full_b, key=order.get), "n_B": len(full_b),
                "shortlist_C": sorted(full_c, key=order.get), "n_C": len(full_c),
                "prior_fallback": fallback, "prior_top1": prior[:1], "prior_top3": prior[:3],
                "example_keys": [pool_keys[j] for j in ex_idx], "example_sims": ex_sim,
                "n_dup_skipped": skipped,
                "nn_codes": ex_codes[0] if ex_codes else [],
                "knn_top1": knn1,
                "covered_B": cover(filed, full_b), "all_covered_B": filed <= full_b,
                "covered_C": cover(filed, full_c), "all_covered_C": filed <= full_c,
                "covered_prior": cover(filed, {a for c in prior[:N_PRIOR] for a in anc(c)}),
                "covered_retr": cover(filed, {a for c in retr for a in anc(c)}),
                "covered_examples": cover(filed, {c for cs in ex_codes for c in cs}),
                "nn_recall": cover(filed, ex_codes[0]) if ex_codes else 0.0,
                "knn_top1_recall": cover(filed, knn1),
            })
        o = pd.DataFrame(out)
        o.to_parquet(out_dir / f"context_{task}.parquet", index=False)
        for (s, w), p in o.groupby(["set", "writer"]):
            rec = {"task": task, "set": s, "writer": w, "reports": len(p),
                   "n_B": round(p.n_B.mean(), 1), "n_C": round(p.n_C.mean(), 1),
                   "covered_B": round(p.covered_B.mean(), 3), "covered_C": round(p.covered_C.mean(), 3),
                   "prior": round(p.covered_prior.mean(), 3), "retrieval": round(p.covered_retr.mean(), 3),
                   "examples": round(p.covered_examples.mean(), 3),
                   "nn_recall": round(p.nn_recall.mean(), 3), "knn_top1_recall": round(p.knn_top1_recall.mean(), 3),
                   "mean_top_sim": round(p.example_sims.map(lambda s: s[0] if len(s) else 0).mean(), 3)}
            cov_rows.append(rec)
            print(rec, flush=True)
    pd.DataFrame(cov_rows).to_csv(out_dir / "context_coverage.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
