"""Stream openFDA device-event bulk zips into compact Parquet files.

Usage:
    python -I extract_reports.py IN_DIR OUT_DIR [WORKERS] [GLOB]

Reads every IN_DIR/<quarter>/*.json.zip without unzipping to disk and writes
one Parquet file per zip to OUT_DIR/<quarter>/<name>.parquet (zstd). Files
whose Parquet output already exists are skipped, so the run is resumable.

Kept per report: identifiers, dates, event type, report source, reporter
occupation, first device's product code and names, device-problem terms,
patient-problem terms, and narrative text split by type.
"""
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

EVENT_TEXT = "Description of Event or Problem"
MFR_TEXT = "Additional Manufacturer Narrative"

SCHEMA = pa.schema([
    ("mdr_report_key", pa.string()),
    ("report_number", pa.string()),
    ("date_received", pa.string()),
    ("date_of_event", pa.string()),
    ("event_type", pa.string()),
    ("report_source_code", pa.string()),
    ("reporter_occupation_code", pa.string()),
    ("health_professional", pa.string()),
    ("type_of_report", pa.list_(pa.string())),
    ("n_devices", pa.int16()),
    ("product_code", pa.string()),
    ("generic_name", pa.string()),
    ("brand_name", pa.string()),
    ("device_name", pa.string()),
    ("device_class", pa.string()),
    ("manufacturer_name", pa.string()),
    ("n_patients", pa.int16()),
    ("product_problems", pa.list_(pa.string())),
    ("patient_problems", pa.list_(pa.string())),
    ("text_event", pa.string()),
    ("text_mfr", pa.string()),
    ("text_other", pa.string()),
])


def s(v):
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def uniq(items):
    seen, out = set(), []
    for x in items:
        x = s(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def join_texts(texts):
    return "\n\n".join(uniq(texts)) or None


def flatten(rec):
    devices = rec.get("device") or []
    d0 = devices[0] if devices else {}
    ofda = d0.get("openfda") or {}
    patients = rec.get("patient") or []
    texts = {"event": [], "mfr": [], "other": []}
    for t in rec.get("mdr_text") or []:
        kind = t.get("text_type_code")
        key = "event" if kind == EVENT_TEXT else "mfr" if kind == MFR_TEXT else "other"
        texts[key].append(t.get("text"))
    tor = rec.get("type_of_report")
    return {
        "mdr_report_key": s(rec.get("mdr_report_key")),
        "report_number": s(rec.get("report_number")),
        "date_received": s(rec.get("date_received")),
        "date_of_event": s(rec.get("date_of_event")),
        "event_type": s(rec.get("event_type")),
        "report_source_code": s(rec.get("report_source_code")),
        "reporter_occupation_code": s(rec.get("reporter_occupation_code")),
        "health_professional": s(rec.get("health_professional")),
        "type_of_report": uniq(tor if isinstance(tor, list) else [tor]),
        "n_devices": len(devices),
        "product_code": s(d0.get("device_report_product_code")),
        "generic_name": s(d0.get("generic_name")),
        "brand_name": s(d0.get("brand_name")),
        "device_name": s(ofda.get("device_name")),
        "device_class": s(ofda.get("device_class")),
        "manufacturer_name": s(d0.get("manufacturer_d_name")),
        "n_patients": len(patients),
        "product_problems": uniq(rec.get("product_problems") or []),
        "patient_problems": uniq(pp for p in patients for pp in (p.get("patient_problems") or [])),
        "text_event": join_texts(texts["event"]),
        "text_mfr": join_texts(texts["mfr"]),
        "text_other": join_texts(texts["other"]),
    }


def process(args):
    src, dst = args
    if dst.exists():
        return src.name, -1, 0.0
    t0 = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.part")
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    n, batch = 0, []
    with zipfile.ZipFile(src) as z:
        member = z.infolist()[0]
        with z.open(member) as f:
            for rec in ijson.items(f, "results.item", use_float=True):
                batch.append(flatten(rec))
                if len(batch) >= 20000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    n += len(batch)
                    batch = []
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        n += len(batch)
    writer.close()
    tmp.replace(dst)
    return src.name, n, time.time() - t0


def main():
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    pattern = sys.argv[4] if len(sys.argv) > 4 else "*/*.json.zip"
    jobs = [(z, out_dir / z.parent.name / z.name.replace(".json.zip", ".parquet"))
            for z in sorted(in_dir.glob(pattern))]
    print(f"{len(jobs)} zips, {workers} workers", flush=True)
    total, t0 = 0, time.time()
    with Pool(workers) as pool:
        for i, (name, n, secs) in enumerate(pool.imap_unordered(process, jobs), 1):
            if n < 0:
                print(f"[{i}/{len(jobs)}] skip {name}", flush=True)
                continue
            total += n
            print(f"[{i}/{len(jobs)}] {name}: {n:,} reports in {secs:.0f}s "
                  f"(total {total:,}, elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"done: {total:,} reports", flush=True)


if __name__ == "__main__":
    main()
