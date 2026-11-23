"""Count every distinct device- and patient-problem term in the study window and
match each against the IMDRF codebook.

Usage:
    python -I term_inventory.py EXTRACT_DIR CODEBOOK.tsv OUT_DIR [START] [END]

START/END are yyyymmdd bounds on date_received (inclusive); defaults
20210101 to 20260630. Writes OUT_DIR/terms_A.tsv and OUT_DIR/terms_E.tsv with
columns term, reports, match (exact / squashed / none), imdrf_code, level,
codebook_term, status. Prints the unmatched terms by frequency.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.stdout.reconfigure(encoding="utf-8")


def norm(s):
    # FDA's own files carry mojibake ("Â\xa0") and non-breaking spaces.
    s = s.replace("Â\xa0", " ").replace("\xa0", " ")
    return " ".join(s.lower().split())


def squash(s):
    s = norm(s).replace("labelling", "labeling")
    return re.sub(r"[^a-z0-9]", "", s)


def main():
    extract, codebook, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    start = sys.argv[4] if len(sys.argv) > 4 else "20210101"
    end = sys.argv[5] if len(sys.argv) > 5 else "20260630"
    out_dir.mkdir(parents=True, exist_ok=True)

    book = {"A": {}, "E": {}}
    with open(codebook, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["primary"] == "True":
                book[r["annex"]][r["term"]] = r

    counts = {"A": Counter(), "E": Counter()}
    flt = (pc.field("date_received") >= start) & (pc.field("date_received") <= end)
    dset = ds.dataset(extract, format="parquet")
    n = 0
    for b in dset.to_batches(columns=["product_problems", "patient_problems"],
                             filter=flt, batch_size=200_000):
        n += b.num_rows
        for terms in b.column("product_problems").to_pylist():
            counts["A"].update(terms or [])
        for terms in b.column("patient_problems").to_pylist():
            counts["E"].update(terms or [])
    print(f"{n:,} reports received {start} to {end}\n")

    for annex in ("A", "E"):
        exact = {norm(t): r for t, r in book[annex].items()}
        squashed = {squash(t): r for t, r in book[annex].items()}
        rows = []
        for term, c in counts[annex].most_common():
            r, how = exact.get(norm(term)), "exact"
            if r is None:
                r, how = squashed.get(squash(term)), "squashed"
            if r is None:
                how = "none"
            rows.append({"term": term, "reports": c, "match": how,
                         "imdrf_code": r["imdrf_code"] if r else "",
                         "level": r["level"] if r else "",
                         "codebook_term": r["term"] if r else "",
                         "status": r["status"] if r else ""})
        with open(out_dir / f"terms_{annex}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        tot = sum(r["reports"] for r in rows)
        by = Counter()
        for r in rows:
            by[r["match"]] += r["reports"]
        print(f"Annex {annex}: {len(rows)} distinct terms, {tot:,} term mentions")
        for k in ("exact", "squashed", "none"):
            print(f"   {k:9} {sum(r['match'] == k for r in rows):>5} terms  {by[k] / tot:7.2%} of mentions")
        for r in rows:
            if r["match"] == "squashed":
                print(f"   SQUASHED {r['reports']:>10,}  {r['term']!r} -> {r['codebook_term']!r}")
        for r in rows:
            if r["match"] == "none":
                print(f"   UNMATCHED {r['reports']:>9,}  {r['term']}")
        print()


if __name__ == "__main__":
    main()
