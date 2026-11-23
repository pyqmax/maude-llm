"""Build the cleaned study table (one row per report, no narrative text).

Usage:
    python -I build_clean.py EXTRACT_DIR CODEBOOK.tsv ALIASES.tsv OUT.parquet COUNTS.tsv [WORKERS]

Rules (fixed before any model is run; see the paper, Section 3.1):
  window      date_received 2021-01-01 to 2026-06-30
  device      terms mapped to IMDRF Annex A by exact name, normalised name, or
              the alias table. Uninformative codes are removed as answers
              (A24 No Apparent Device Problem, A25 No Apparent Adverse Event,
              A26 Insufficient Information, A27 Term not Available), as are
              alias rows marked exclude or drop and any unmatched term.
  patient     same for Annex E. Legacy no-harm names become E2403. E2401
              (Insufficient Information), E2402 (Term not Available) and alias
              exclusions are removed. Patient task only for Injury and Death.
  text        event narrative (text_event) must have at least MIN_WORDS words
  duplicates  same mdr_report_key: keep one. Same normalised narrative (lower
              case, letters only, digits removed so templated reports that
              differ only in lot numbers collapse): keep the earliest received.
  flag        quotes_term: the narrative contains a filed term verbatim.

The text stays in the extract; join on mdr_report_key when sampling.
"""
import csv
import hashlib
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

START, END = "20210101", "20260630"
MIN_WORDS = 15
UNINFORMATIVE = {"A24", "A25", "A26", "A27", "E2401", "E2402"}
NO_HARM = "E2403"
LAY = {"PATIENT", "PATIENT FAMILY MEMBER OR FRIEND"}
COLS = ["mdr_report_key", "report_number", "date_received", "event_type",
        "report_source_code", "reporter_occupation_code", "health_professional",
        "product_code", "device_class", "manufacturer_name",
        "product_problems", "patient_problems", "text_event", "text_mfr"]

MAPS = None  # set in each worker by init()


def norm(s):
    s = s.replace("Â\xa0", " ").replace("\xa0", " ")
    return " ".join(s.lower().split())


def squash(s):
    return re.sub(r"[^a-z0-9]", "", norm(s).replace("labelling", "labeling"))


def load_maps(codebook, aliases):
    book = {"A": {}, "E": {}}
    info = {}
    with open(codebook, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["primary"] != "True":
                continue
            info[r["imdrf_code"]] = (int(r["level"]), r["term"])
            book[r["annex"]][squash(r["term"])] = r["imdrf_code"]
    alias = {"A": {}, "E": {}}
    with open(aliases, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["report_term"].startswith("("):
                continue
            act = r["action"]
            if act == "map":
                alias[r["annex"]][squash(r["report_term"])] = r["imdrf_code"]
            elif act == "no_harm":
                alias[r["annex"]][squash(r["report_term"])] = NO_HARM
            else:
                alias[r["annex"]][squash(r["report_term"])] = "#" + act
    return {"book": book, "alias": alias, "info": info}


def init(codebook, aliases):
    global MAPS
    MAPS = load_maps(codebook, aliases)


def map_terms(terms, annex, tally):
    """Return (codes, terms_kept) after mapping and removals."""
    codes, kept = [], []
    for t in terms or []:
        k = squash(t)
        c = MAPS["alias"][annex].get(k) or MAPS["book"][annex].get(k)
        if c is None:
            tally[f"{annex} term unmatched (dropped)"] += 1
            continue
        if c.startswith("#"):
            tally[f"{annex} term alias {c[1:]}"] += 1
            continue
        if c in UNINFORMATIVE:
            tally[f"{annex} term uninformative {c}"] += 1
            continue
        tally[f"{annex} term kept"] += 1
        if c not in codes:
            codes.append(c)
            kept.append(t)
    return codes, kept


def text_key(s):
    s = re.sub(r"[^a-z]", "", (s or "").lower())
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") >> 1


def writer(source, occ):
    if source in ("Manufacturer report", "Distributor report"):
        return "industry"
    if occ in LAY:
        return "lay"
    if occ in (None, "", "UNKNOWN", "OTHER", "NOT APPLICABLE", "INVALID DATA"):
        return "unknown"
    return "professional"


def process(path):
    flt = (pc.field("date_received") >= START) & (pc.field("date_received") <= END)
    t = ds.dataset(path, format="parquet").to_table(columns=COLS, filter=flt)
    tally = Counter()
    out = {k: [] for k in ("mdr_report_key", "report_number", "date_received",
                           "event_type", "report_source_code",
                           "reporter_occupation_code", "health_professional",
                           "product_code", "device_class", "manufacturer_name",
                           "writer", "device_codes", "device_levels",
                           "patient_codes", "patient_levels",
                           "n_device_terms_filed", "words_event", "words_mfr",
                           "text_key", "quotes_term_event", "quotes_term_mfr")}
    info = MAPS["info"]
    for r in t.to_pylist():
        dc, dterms = map_terms(r["product_problems"], "A", tally)
        pcodes, _ = map_terms(r["patient_problems"], "E", tally)
        ev, mf = r["text_event"] or "", r["text_mfr"] or ""
        evl, mfl = norm(ev), norm(mf)
        terms_l = [norm(x) for x in dterms]
        for k in ("mdr_report_key", "report_number", "date_received", "event_type",
                  "report_source_code", "reporter_occupation_code",
                  "health_professional", "product_code", "device_class",
                  "manufacturer_name"):
            out[k].append(r[k])
        out["writer"].append(writer(r["report_source_code"], r["reporter_occupation_code"]))
        out["device_codes"].append(dc)
        out["device_levels"].append([info[c][0] for c in dc])
        out["patient_codes"].append(pcodes)
        out["patient_levels"].append([info[c][0] for c in pcodes])
        out["n_device_terms_filed"].append(len(r["product_problems"] or []))
        out["words_event"].append(len(ev.split()))
        out["words_mfr"].append(len(mf.split()))
        out["text_key"].append(text_key(ev))
        out["quotes_term_event"].append(any(x in evl for x in terms_l))
        out["quotes_term_mfr"].append(any(x in mfl for x in terms_l))
    return pa.table(out), tally


def main():
    extract, codebook, aliases = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    out_path, counts_path = Path(sys.argv[4]), Path(sys.argv[5])
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    files = sorted(str(p) for p in extract.rglob("*.parquet"))
    tables, tally = [], Counter()
    with Pool(workers, initializer=init, initargs=(codebook, aliases)) as pool:
        for i, (tb, tl) in enumerate(pool.imap_unordered(process, files), 1):
            if tb.num_rows:
                tables.append(tb)
            tally.update(tl)
            if i % 20 == 0:
                print(f"  {i}/{len(files)} files", flush=True)
    t = pa.concat_tables(tables)
    del tables
    n = t.num_rows

    # Duplicates: order by date_received then key, keep first per key and per text.
    order = pc.sort_indices(t, [("date_received", "ascending"),
                                ("mdr_report_key", "ascending")])
    t = t.take(order)
    keys = t.column("mdr_report_key").to_numpy(zero_copy_only=False)
    _, first = np.unique(keys, return_index=True)
    dup_key = np.ones(n, bool)
    dup_key[first] = False
    tk = t.column("text_key").to_numpy()
    tk_live = np.where(dup_key, -1, tk)  # key-duplicates do not claim a text
    _, first = np.unique(tk_live, return_index=True)
    dup_text = np.ones(n, bool)
    dup_text[first] = False
    dup_text[dup_key] = False

    words = t.column("words_event").to_numpy()
    no_text = words == 0
    short = (words > 0) & (words < MIN_WORDS)
    n_dev = pc.list_value_length(t.column("device_codes")).to_numpy(zero_copy_only=False)
    n_pat = pc.list_value_length(t.column("patient_codes")).to_numpy(zero_copy_only=False)
    ev = t.column("event_type").to_numpy(zero_copy_only=False)
    harm = np.isin(ev, ["Injury", "Death"])

    def reason(code_ok, eligible=None):
        r = np.full(n, "", dtype=object)
        steps = [("duplicate report key", dup_key)]
        if eligible is not None:
            steps.append(("not injury or death", ~eligible))
        steps += [("no narrative", no_text),
                  (f"narrative under {MIN_WORDS} words", short),
                  ("duplicate narrative", dup_text),
                  ("no codable code", ~code_ok)]
        for label, m in steps:
            r[(r == "") & m] = label
        return r

    rd = reason(n_dev > 0)
    rp = reason(n_pat > 0, harm)
    d = t.column("date_received").to_numpy(zero_copy_only=False)
    period = np.where(d >= "20260101", "2026H1 unseen",
                      np.where(d >= "20250101", "2025", "2021-2024"))
    t = t.append_column("period", pa.array(period))
    t = t.append_column("dup_text", pa.array(dup_text))
    t = t.append_column("excl_device", pa.array(rd, pa.string()))
    t = t.append_column("excl_patient", pa.array(rp, pa.string()))
    t = t.append_column("in_device_task", pa.array(rd == ""))
    t = t.append_column("in_patient_task", pa.array(rp == ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out_path, compression="zstd")

    rows = [("all", "reports in window", "", n)]
    for task, r in (("device", rd), ("patient", rp)):
        left = n
        for label in ["duplicate report key", "not injury or death", "no narrative",
                      f"narrative under {MIN_WORDS} words", "duplicate narrative",
                      "no codable code"]:
            c = int((r == label).sum())
            if task == "device" and label == "not injury or death":
                continue
            left -= c
            rows.append((task, "removed: " + label, c, left))
        for p in ("2021-2024", "2025", "2026H1 unseen"):
            rows.append((task, f"kept, {p}", "", int(((r == "") & (period == p)).sum())))
        wr = t.column("writer").to_numpy(zero_copy_only=False)
        for w in ("industry", "professional", "lay", "unknown"):
            rows.append((task, f"kept, writer {w}", "", int(((r == "") & (wr == w)).sum())))
    for k, v in sorted(tally.items()):
        rows.append(("terms", k, v, ""))
    with open(counts_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "step", "removed", "remaining"])
        w.writerows(rows)
    for row in rows:
        print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()
