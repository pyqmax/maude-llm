"""Download openFDA device-event bulk files for a range of years, politely.

Usage:
    python -I download_maude.py INDEX_JSON OUT_DIR YEAR [YEAR ...]

One file at a time, a pause between files, retries with backoff, resumable
(files that already exist and pass a zip integrity test are skipped).
Writes MANIFEST.tsv (file, bytes, sha256, url) to OUT_DIR at the end.
"""
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PAUSE_S = 3
RETRIES = 5
UA = "research-download-script/1.0 (academic use)"


def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main():
    index_path, out_dir, years = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3:])
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    parts = [p for p in idx["results"]["device"]["event"]["partitions"]
             if p["display_name"][:4] in years]
    parts.sort(key=lambda p: p["file"])
    print(f"{len(parts)} files to fetch into {out_dir}", flush=True)

    rows = []
    for i, p in enumerate(parts, 1):
        url = p["file"]
        # e.g. .../device/event/2021q1/device-event-0001-of-0004.json.zip
        quarter = url.rstrip("/").split("/")[-2]
        dest = out_dir / quarter / url.rsplit("/", 1)[-1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and zip_ok(dest):
            print(f"[{i}/{len(parts)}] skip (have) {quarter}/{dest.name}", flush=True)
        else:
            for attempt in range(1, RETRIES + 1):
                try:
                    fetch(url, dest)
                    if zip_ok(dest):
                        break
                    raise IOError("zip integrity test failed")
                except Exception as e:  # noqa: BLE001
                    wait = PAUSE_S * 2 ** attempt
                    print(f"   attempt {attempt} failed ({e}); waiting {wait}s", flush=True)
                    time.sleep(wait)
            else:
                print(f"[{i}/{len(parts)}] FAILED {url}", flush=True)
                continue
            print(f"[{i}/{len(parts)}] ok {quarter}/{dest.name} "
                  f"{dest.stat().st_size / 1e6:.1f} MB", flush=True)
            time.sleep(PAUSE_S)
        rows.append((f"{quarter}/{dest.name}", dest.stat().st_size, sha256(dest), url))

    with open(out_dir / "MANIFEST.tsv", "w", encoding="utf-8") as f:
        f.write("file\tbytes\tsha256\turl\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"done: {len(rows)}/{len(parts)} files, manifest written", flush=True)


if __name__ == "__main__":
    main()
