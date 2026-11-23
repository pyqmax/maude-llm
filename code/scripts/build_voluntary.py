"""Build the voluntary (FDA-coded) study set with narratives, intake note stripped.

Usage:
    python -I build_voluntary.py CLEAN.parquet EXTRACT_DIR OUT_DIR COUNTS.tsv

Voluntary MedWatch reports only (report_source_code == "Voluntary report"),
one file per task:
  OUT_DIR/device.parquet   reports in the device task
  OUT_DIR/patient.parquet  reports in the patient task (injury and death)
Columns: the clean-table columns, text_event (note stripped), had_note,
words_clean, set:
  "unseen"      received 1 Feb to 30 Jun 2026 (after the frontier cutoffs), all of them
  "comparison"  COMPARISON reports drawn from 2021 to 2024 for the memorisation check
  "pool"        every other 2021 to 2025 report: few-shot example pool and product prior
Reports whose narrative falls under MIN_WORDS after stripping are dropped.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from textclean import strip_note  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SEED = 20261114
COMPARISON = 500
UNSEEN_FROM = "20260201"
MIN_WORDS = 15


def main():
    clean, extract, out_dir, counts = sys.argv[1:5]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    t = pq.read_table(clean)
    t = t.filter(pc.equal(t.column("report_source_code"), "Voluntary report"))
    df = t.to_pandas()
    df = df[df.in_device_task | df.in_patient_task].copy()
    print(f"voluntary reports in a task: {len(df):,}")

    tx = ds.dataset(extract, format="parquet").to_table(
        columns=["mdr_report_key", "text_event"],
        filter=pc.field("mdr_report_key").isin(sorted(df.mdr_report_key))).to_pandas()
    tx = tx.drop_duplicates("mdr_report_key")
    df = df.merge(tx, on="mdr_report_key", how="left")
    assert df.text_event.notna().all(), "missing narratives"

    res = df.text_event.map(strip_note)
    df["text_event"] = [r[0] for r in res]
    df["had_note"] = [r[1] for r in res]
    df["words_clean"] = df.text_event.str.split().str.len()
    short = df.words_clean < MIN_WORDS
    print(f"note stripped in {df.had_note.mean():.1%}; dropped {int(short.sum())} reports under {MIN_WORDS} words after stripping")
    df = df[~short]

    df["set"] = np.where(df.period == "2026H1 unseen", "unseen", "pool")
    df = df[(df.set == "pool") | (df.date_received >= UNSEEN_FROM)]

    rows = []
    for task in ("device", "patient"):
        d = df[df[f"in_{task}_task"]].copy()
        cand = d[(d.period == "2021-2024")].sample(n=COMPARISON, random_state=rng).mdr_report_key
        d.loc[d.mdr_report_key.isin(cand), "set"] = "comparison"
        d = d.sort_values(["set", "mdr_report_key"]).reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False),
                       out_dir / f"{task}.parquet", compression="zstd")
        for (s, w), g in d.groupby(["set", "writer"]):
            rows.append((task, s, w, len(g), g.product_code.nunique(), int(g.words_clean.median()),
                         f"{g.had_note.mean():.3f}"))
        print(f"{task}: {len(d):,} reports; " + ", ".join(f"{s} {n:,}" for s, n in d.set.value_counts().items()))

    with open(counts, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "set", "writer", "reports", "product_codes", "median_words", "share_had_note"])
        w.writerows(rows)
    for r in rows:
        print("\t".join(str(v) for v in r))


if __name__ == "__main__":
    main()
