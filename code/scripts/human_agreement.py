"""Human ceiling: how well does one filed code set predict another filed code
set for the same narrative?

Reports whose normalised narrative is identical (same text_key) were coded
separately. Within each such group, every ordered pair of reports (A, B) is
scored as if A's codes were a prediction and B's codes the filed answer, with
the same metrics used for the models (metrics.py). Both reports must carry at
least one informative code for the task; the share of pairs where only one
side has codes is reported separately.

Big template groups (thousands of copies) are capped at CAP reports, and
results are given per group (each narrative counts once) and per pair.

Usage:
    python -P human_agreement.py CLEAN.parquet OUT_PREFIX
writes OUT_PREFIX_{device,patient}.tsv (one row per group) and prints summaries.
"""
import random
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from metrics import METRICS, score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SEED, CAP, MIN_WORDS = 20261114, 30, 15
BINS = [(2, 2, "2"), (3, 10, "3-10"), (11, 100, "11-100"), (101, 1000, "101-1000"), (1001, 10**9, ">1000")]


def size_bin(n):
    return next(lab for lo, hi, lab in BINS if lo <= n <= hi)


def run(df, task, out):
    col = f"{task}_codes"
    if task == "patient":
        df = df[df.event_type.isin(["Injury", "Death"])]
    df = df[df.words_event >= MIN_WORDS]
    n = df.groupby("text_key").mdr_report_key.transform("size")
    df = df[n > 1]
    print(f"\n######## {task}: {df.text_key.nunique():,} narratives filed more than once, {len(df):,} reports")
    rng = random.Random(SEED)
    groups, pairs = [], []
    for tk, g in df.groupby("text_key", sort=False):
        recs = g[["manufacturer_name", "writer", "period", col]].to_dict("records")
        if len(recs) > CAP:
            recs = rng.sample(recs, CAP)
        coded = [r for r in recs if len(r[col])]
        one_sided = sum(1 for a, b in permutations(recs, 2) if bool(len(a[col])) != bool(len(b[col])))
        if len(coded) < 2:
            groups.append({"text_key": tk, "size": len(g), "n_coded": len(coded), "one_sided_pairs": one_sided})
            continue
        sc = []
        for a, b in permutations(coded, 2):
            s = score(a[col], b[col])
            s.update(same_mfr=a["manufacturer_name"] == b["manufacturer_name"],
                     writer_pair="/".join(sorted([a["writer"] or "?", b["writer"] or "?"])),
                     period=b["period"], size_bin=size_bin(len(g)))
            sc.append(s)
        pairs.extend(sc)
        p = pd.DataFrame(sc)
        groups.append({"text_key": tk, "size": len(g), "n_coded": len(coded), "one_sided_pairs": one_sided,
                       "size_bin": size_bin(len(g)), "any_diff_mfr": not p.same_mfr.all(),
                       **p[METRICS].mean().to_dict()})
    G = pd.DataFrame(groups)
    P = pd.DataFrame(pairs)
    G.to_csv(out, sep="\t", index=False)
    S = G.dropna(subset=["recall"])
    print(f"groups with >=2 coded reports: {len(S):,} of {len(G):,}; "
          f"pairs scored {len(P):,}; one-sided pairs (codes on one side only) {int(G.one_sided_pairs.sum()):,}")
    pd.set_option("display.width", 250)
    print("\n-- per group (each narrative once) --")
    print(S[METRICS].mean().round(3).to_string())
    print("\n-- per group, by group size --")
    print(S.groupby("size_bin")[METRICS].agg(["mean"]).droplevel(1, axis=1)
          .assign(groups=S.groupby("size_bin").size()).round(3).to_string())
    print("\n-- per pair, same vs different manufacturer --")
    print(P.groupby("same_mfr")[METRICS].mean().assign(pairs=P.groupby("same_mfr").size()).round(3).to_string())
    print("\n-- per pair, writer pair --")
    print(P.groupby("writer_pair")[METRICS].mean().assign(pairs=P.groupby("writer_pair").size())
          .round(3).to_string())
    print("\n-- per pair, period of the reference report --")
    print(P.groupby("period")[METRICS].mean().assign(pairs=P.groupby("period").size()).round(3).to_string())


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    cols = ["mdr_report_key", "text_key", "manufacturer_name", "writer", "event_type", "period",
            "words_event", "device_codes", "patient_codes"]
    t = pq.read_table(src, columns=cols)
    vc = pc.value_counts(t.column("text_key"))
    multi = pc.filter(vc.field("values"), pc.greater(vc.field("counts"), 1))
    t = t.filter(pc.is_in(t.column("text_key"), value_set=multi))
    df = t.to_pandas()
    print("writer values:", df.writer.value_counts(dropna=False).to_dict())
    print("period values:", df.period.value_counts(dropna=False).to_dict())
    for task in ("device", "patient"):
        run(df, task, f"{prefix}_{task}.tsv")


if __name__ == "__main__":
    main()
