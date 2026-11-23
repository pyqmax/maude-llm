"""How often is the filed code among the codes historically used for the
report's product code? (The "what a specialised human coder keeps in mind" list.)

Usage:
    python -P prior_eval.py CLEAN.parquet SAMPLE_DIR OUT.tsv

History = 2021 to 2024 reports in the task's clean pool, minus every sampled
report. Ranked by frequency within product code. Reports whose product code
has no history fall back to the global ranking (counted separately).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NS = (5, 10, 20, 30, 50)


def anc(c):
    return {c[:n] for n in range(3, len(c) + 1, 2)}


def main():
    clean, sample_dir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    cols = ["mdr_report_key", "period", "product_code", "device_codes", "patient_codes",
            "in_device_task", "in_patient_task"]
    df = pq.read_table(clean, columns=cols).to_pandas()
    rows = []
    for task, col, flag in (("device", "device_codes", "in_device_task"),
                            ("patient", "patient_codes", "in_patient_task")):
        smp = pd.read_parquet(sample_dir / f"{task}.parquet",
                              columns=["mdr_report_key", "set", "stratum", "product_code", col])
        hist = df[df[flag] & (df.period == "2021-2024") & ~df.mdr_report_key.isin(smp.mdr_report_key)]
        by_pc = defaultdict(Counter)
        glob = Counter()
        for pc_, cs in zip(hist.product_code, hist[col]):
            by_pc[pc_].update(cs)
            glob.update(cs)
        n_hist = {k: sum(v.values()) for k, v in by_pc.items()}
        glob_rank = [c for c, _ in glob.most_common()]
        print(f"{task}: history {len(hist):,} reports, {len(by_pc):,} product codes", flush=True)

        smp["group"] = np.where(smp.stratum.isin(["lay", "professional"]), smp.stratum,
                                np.where(smp.stratum == "industry matched", "matched", "industry"))
        for (s, g), part in smp.groupby(["set", "group"]):
            res = {n: [] for n in NS}
            resp = {n: [] for n in NS}
            nohist = 0
            for pc_, filed in zip(part.product_code, part[col]):
                if n_hist.get(pc_, 0) >= 20:
                    rank = [c for c, _ in by_pc[pc_].most_common()]
                    rank += [c for c in glob_rank if c not in by_pc[pc_]]
                else:
                    rank = glob_rank
                    nohist += 1
                filed = set(filed)
                for n in NS:
                    top = set(rank[:n])
                    res[n].append(len(filed & top) / len(filed))
                    withp = set().union(*(anc(c) for c in top))
                    resp[n].append(len(filed & withp) / len(filed))
            rec = {"task": task, "set": s, "group": g, "reports": len(part),
                   "no_history_share": round(nohist / len(part), 3)}
            rec.update({f"recall@{n}": round(float(np.mean(res[n])), 3) for n in NS})
            rec.update({f"withparents@{n}": round(float(np.mean(resp[n])), 3) for n in NS})
            rows.append(rec)
            print(rec, flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
