"""Build the Arm B shortlist for every sampled report and measure its coverage.

Usage:
    python -P build_shortlists.py CLEAN.parquet SAMPLE_DIR CODEBOOK.tsv MODEL_DIR OUT_DIR

Shortlist = product-code prior top 20 (2021 to 2024 history, sampled reports
excluded; global ranking if the product has under 20 past reports)
  + retrieved top 10 (event narrative only, so both input conditions share one
    shortlist; dense for device, hybrid for patient, the best per task on the
    comparison set)
  + every ancestor of those codes.
Uninformative codes are never candidates. Codes are stored in codebook order
(not rank order) so list position carries no hint.

Writes OUT_DIR/shortlist_{task}.parquet (one row per sampled report, with the
prior baseline predictions) and OUT_DIR/shortlist_coverage.tsv.

Coverage per report: share of filed codes on the list ("covered"), and whether
all filed codes are on it ("all_covered").
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EXCLUDE = {"A24", "A25", "A26", "A27", "E2401", "E2402"}
N_PRIOR, N_RETR = 20, 10
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


def main():
    clean, sample_dir, codebook, model_dir, out_dir = sys.argv[1:6]
    sample_dir, out_dir = Path(sample_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    model = SentenceTransformer(model_dir, device=device)

    book = {"A": [], "E": []}
    with open(codebook, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["primary"] == "True" and r["imdrf_code"] not in EXCLUDE:
                book[r["annex"]].append(r)

    hist_cols = ["mdr_report_key", "period", "product_code", "device_codes",
                 "patient_codes", "in_device_task", "in_patient_task"]
    df = pq.read_table(clean, columns=hist_cols).to_pandas()
    cov_rows = []
    for task, annex, col, flag, method in (("device", "A", "device_codes", "in_device_task", "dense"),
                                           ("patient", "E", "patient_codes", "in_patient_task", "hybrid")):
        rows = book[annex]
        codes = [r["imdrf_code"] for r in rows]
        order = {c: i for i, c in enumerate(codes)}
        valid = set(codes)
        docs = [f"{r['term']}. {r['definition']} ({r['path_terms']})" for r in rows]
        demb = model.encode(docs, batch_size=64, normalize_embeddings=True)
        bm25 = BM25Okapi([tok(d) for d in docs])

        smp = pd.read_parquet(sample_dir / f"{task}.parquet")
        hist = df[df[flag] & (df.period == "2021-2024") & ~df.mdr_report_key.isin(smp.mdr_report_key)]
        by_pc, glob = defaultdict(Counter), Counter()
        for p, cs in zip(hist.product_code, hist[col]):
            by_pc[p].update(cs)
            glob.update(cs)
        glob_rank = [c for c, _ in glob.most_common() if c in valid]

        q = [QUERY_PREFIX + (t or "") for t in smp.text_event]
        qemb = model.encode(q, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        dense_rank = np.argsort(-(qemb @ demb.T), axis=1)

        out = []
        for i, r in enumerate(smp.itertuples(index=False)):
            pc_hist = by_pc.get(r.product_code)
            if pc_hist and sum(pc_hist.values()) >= 20:
                prior = [c for c, _ in pc_hist.most_common() if c in valid]
                prior += [c for c in glob_rank if c not in pc_hist]
                fallback = False
            else:
                prior, fallback = glob_rank, True
            if method == "dense":
                rk = list(dense_rank[i])
            else:
                b = list(np.argsort(-bm25.get_scores(tok(r.text_event or ""))))
                rk = rrf(b, list(dense_rank[i]))
            retr = [codes[j] for j in rk[:N_RETR]]
            base = set(prior[:N_PRIOR]) | set(retr)
            full = {a for c in base for a in anc(c) if a in valid}
            shortlist = sorted(full, key=order.get)
            filed = set(getattr(r, col))
            out.append({
                "mdr_report_key": r.mdr_report_key, "set": r.set, "stratum": r.stratum,
                "product_code": r.product_code, "filed": sorted(filed),
                "shortlist": shortlist, "n_shortlist": len(shortlist),
                "prior_fallback": fallback,
                "prior_top1": prior[:1], "prior_top3": prior[:3],
                "covered": len(filed & full) / len(filed),
                "all_covered": filed <= full,
                "covered_prior_only": len(filed & {a for c in prior[:N_PRIOR] for a in anc(c)}) / len(filed),
                "covered_retr_only": len(filed & {a for c in retr for a in anc(c)}) / len(filed),
            })
        o = pd.DataFrame(out)
        o.to_parquet(out_dir / f"shortlist_{task}.parquet", index=False)
        o["group"] = np.where(o.stratum.isin(["lay", "professional"]), o.stratum,
                              np.where(o.stratum == "industry matched", "matched", "industry"))
        for (s, g), p in o.groupby(["set", "group"]):
            rec = {"task": task, "set": s, "group": g, "reports": len(p),
                   "mean_size": round(p.n_shortlist.mean(), 1),
                   "covered": round(p.covered.mean(), 3), "all_covered": round(p.all_covered.mean(), 3),
                   "prior_only": round(p.covered_prior_only.mean(), 3),
                   "retrieval_only": round(p.covered_retr_only.mean(), 3)}
            cov_rows.append(rec)
            print(rec, flush=True)
    pd.DataFrame(cov_rows).to_csv(out_dir / "shortlist_coverage.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
