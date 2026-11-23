"""Build a flat codebook (TSV) for IMDRF Annex A and E from FDA's MDR codes workbook.

Usage:
    python -I build_codebook.py WORKBOOK.xlsx OUT.tsv

One row per term: annex, imdrf_code, level (1 to 3), fda_code, term,
definition, status, path (IMDRF codes root to leaf), path_terms.
"""
import csv
import sys

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
HEADER_ROW = 7  # zero-based index of the column-header row in sheets A and E


def clean(v):
    return "" if v is None else str(v).strip()


def main():
    wb = openpyxl.load_workbook(sys.argv[1], read_only=True)
    out = []
    for annex in ("A", "E"):
        rows = list(wb[annex].iter_rows(values_only=True))
        head = [clean(h) for h in rows[HEADER_ROW]]
        col = {h: i for i, h in enumerate(head) if h}
        hier_col = col.get("CodeHierarchy")
        by_code = {}
        for r in rows[HEADER_ROW + 1:]:
            r = list(r) + [None] * (len(head) - len(r))
            terms = [clean(r[col[f"Level {k} Term"]]) for k in (1, 2, 3)]
            code = clean(r[col["IMDRF Code"]])
            if not code:
                continue
            level = max(k for k in (1, 2, 3) if terms[k - 1]) if any(terms) else 0
            path = clean(r[hier_col]) if hier_col is not None else ""
            if not path:  # sheet E has no hierarchy column: derive from code
                path = "|".join(code[:n] for n in range(3, len(code) + 1, 2))
            rec = {
                "annex": annex,
                "imdrf_code": code,
                "level": level,
                "fda_code": clean(r[col["FDA Code"]]),
                "term": terms[level - 1] if level else "",
                "definition": clean(r[col["Definition"]]),
                "status": clean(r[col["Status"]]),
                "path": path,
            }
            # A code can appear under several parents (e.g. Annex E repeats
            # perforations under Procedural Complications). The first
            # occurrence is its primary location.
            rec["primary"] = code not in by_code
            if rec["primary"]:
                by_code[code] = rec
            out.append(rec)
        for rec in out:
            if rec["annex"] == annex:
                rec["path_terms"] = " > ".join(
                    by_code[c]["term"] for c in rec["path"].split("|") if c in by_code)

    fields = ["annex", "imdrf_code", "level", "fda_code", "term", "definition",
              "status", "primary", "path", "path_terms"]
    with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(out)

    for annex in ("A", "E"):
        recs = [r for r in out if r["annex"] == annex]
        lv = {k: sum(r["level"] == k for r in recs) for k in (1, 2, 3)}
        statuses = {}
        for r in recs:
            statuses[r["status"] or "(blank)"] = statuses.get(r["status"] or "(blank)", 0) + 1
        print(f"Annex {annex}: {len(recs)} terms, by level {lv}, status {statuses}")


if __name__ == "__main__":
    main()
