"""Draw the study samples (design in the paper, Section 3.2) and attach the narratives.

Usage:
    python -I draw_sample.py CLEAN.parquet EXTRACT_DIR OUT_DIR COUNTS.tsv

Writes OUT_DIR/device.parquet and OUT_DIR/patient.parquet: the clean-table
columns plus set (unseen / comparison), stratum labels, text_event, text_mfr.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
SEED = 20261112
CAP = 10
CELL = {"unseen": 200, "comparison": 100}
PATIENT_INDUSTRY = 500
UNSEEN_FROM = "20260201"
SETS = {"2026H1 unseen": "unseen", "2021-2024": "comparison"}
RANK = {"rare": 0, "mid": 1, "common": 2}


def capped(df, rng, cap=CAP):
    """Random order, then at most `cap` rows per product code."""
    df = df.sample(frac=1, random_state=rng)
    return df[df.groupby("product_code").cumcount() < cap]


def take(df, n, rng):
    return df if len(df) <= n else df.sample(n=n, random_state=rng)


def main():
    clean, extract, out_dir, counts = sys.argv[1:5]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    df = pq.read_table(clean).to_pandas()
    df = df[df.period.isin(SETS)].copy()
    df["set"] = df.period.map(SETS)
    # Claude Opus 4.8 (reference) has a January 2026 training cutoff, so the
    # unseen set starts in February.
    df = df[(df.set == "comparison") | (df.date_received >= UNSEEN_FROM)]

    ref = Counter(c for cs in df[df.in_device_task & (df.period == "2021-2024")].device_codes
                  for c in cs)

    def band(c):
        n = ref.get(c, 0)
        return "rare" if n < 1000 else ("mid" if n < 10000 else "common")

    dv = df[df.in_device_task].copy()
    dv["rarity"] = dv.device_codes.map(lambda cs: min((band(c) for c in cs), key=RANK.get))
    dv["level"] = dv.device_levels.map(max)

    parts = []
    for s in ("unseen", "comparison"):
        x = dv[dv.set == s]
        ind = x[x.writer == "industry"]
        for (r, lv), cell in ind.groupby(["rarity", "level"]):
            parts.append(take(capped(cell, rng), CELL[s], rng).assign(stratum=f"industry {r} L{lv}"))
        if s == "unseen":
            for w in ("lay", "professional"):
                parts.append(x[x.writer == w].assign(stratum=w))
    dev = pd.concat(parts)

    pt = df[df.in_patient_task]
    parts = []
    for s in ("unseen", "comparison"):
        x = pt[pt.set == s]
        parts.append(take(capped(x[x.writer == "industry"], rng), PATIENT_INDUSTRY, rng).assign(stratum="industry"))
        if s == "unseen":
            for w in ("lay", "professional"):
                parts.append(x[x.writer == w].assign(stratum=w))
    pat = pd.concat(parts)

    # Matched manufacturer controls (drawn last so the draws above are unchanged):
    # one per unseen lay or professional report, same product code, not already sampled.
    def matched(sample, pool):
        x = pool[(pool.set == "unseen") & (pool.writer == "industry")
                 & ~pool.mdr_report_key.isin(sample.mdr_report_key)]
        need = sample[(sample.set == "unseen") & sample.stratum.isin(["lay", "professional"])]
        want = need.groupby("product_code").size()
        x = x[x.product_code.isin(want.index)].sample(frac=1, random_state=rng)
        x = x[x.groupby("product_code").cumcount() < x.product_code.map(want)]
        short = int(want.sum() - len(x))
        print(f"matched controls: {len(x)} of {int(want.sum())} ({short} product-code slots unfilled)")
        return pd.concat([sample, x.assign(stratum="industry matched")])

    dev = matched(dev, dv)
    pat = matched(pat, pt)

    keys = sorted(set(dev.mdr_report_key) | set(pat.mdr_report_key))
    tx = ds.dataset(extract, format="parquet").to_table(
        columns=["mdr_report_key", "text_event", "text_mfr"],
        filter=pc.field("mdr_report_key").isin(keys)).to_pandas()
    tx = tx.drop_duplicates("mdr_report_key")

    rows = []
    for name, d in (("device", dev), ("patient", pat)):
        d = d.merge(tx, on="mdr_report_key", how="left")
        assert d.text_event.notna().all(), f"{name}: missing narratives"
        assert not d.duplicated(["set", "mdr_report_key"]).any()
        d = d.sort_values(["set", "stratum", "mdr_report_key"]).reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False),
                       out_dir / f"{name}.parquet", compression="zstd")
        for (s, st), g in d.groupby(["set", "stratum"]):
            rows.append((name, s, st, len(g), g.product_code.nunique(),
                         int(g.words_event.median()), f"{g.quotes_term_event.mean():.3f}"))
        print(f"{name}: {len(d)} reports")

    with open(counts, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "set", "stratum", "reports", "product_codes",
                    "median_words_event", "share_quoting_term"])
        w.writerows(rows)
    for r in rows:
        print("\t".join(str(v) for v in r))


if __name__ == "__main__":
    main()
