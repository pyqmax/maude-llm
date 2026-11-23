"""Check how many openFDA problem terms (from count queries) match the codebook.

Usage:
    python -I check_term_match.py CODEBOOK.tsv COUNTS.json ANNEX
"""
import csv
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")


def norm(s):
    return " ".join(s.lower().split())


def main():
    book = {}
    with open(sys.argv[1], encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["annex"] == sys.argv[3]:
                book[norm(r["term"])] = r
    res = json.load(open(sys.argv[2], encoding="utf-8"))["results"]
    total = sum(r["count"] for r in res)
    hit = [r for r in res if norm(r["term"]) in book]
    miss = [r for r in res if norm(r["term"]) not in book]
    print(f"{sys.argv[2]}: {len(hit)}/{len(res)} terms matched, "
          f"{sum(r['count'] for r in hit) / total:.1%} of mentions")
    lv = {}
    for r in hit:
        k = book[norm(r["term"])]["level"]
        lv[k] = lv.get(k, 0) + r["count"]
    print("   mentions by level:", {k: f"{v / total:.1%}" for k, v in sorted(lv.items())})
    for r in miss:
        print(f"   UNMATCHED {r['count']:>9,}  {r['term']}")


if __name__ == "__main__":
    main()
