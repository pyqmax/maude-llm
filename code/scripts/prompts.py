"""Prompt construction for the MAUDE coding runs (v2, voluntary reports).

build_messages(task, arm, report, codebook, shortlist=None, examples=None) -> chat messages

task   "device" (IMDRF Annex A) or "patient" (Annex E)
arm    "0" no code list (what the model knows from training)
       "A" full list of code names in hierarchy
       "B" shortlist with definitions
       "C" shortlist with definitions plus retrieved coded examples

Everything identical across reports (instructions, and in Arm A the codebook)
goes in the system message so providers can cache it; the report comes last.
The narrative must already have FDA's intake note stripped (textclean.py).
"""
import csv

EXCLUDE = {"A24", "A25", "A26", "A27", "E2401", "E2402"}
VERSION = "prompts-v2-2026-11-14"

TASK_TEXT = {
    "device": (
        "You assign IMDRF Annex A medical device problem codes to FDA MedWatch adverse event reports. "
        "A device problem code describes what went wrong with the device itself (for example a leak, "
        "a break, an incorrect reading, a software failure), not what happened to the patient."
    ),
    "patient": (
        "You assign IMDRF Annex E health effect codes (clinical signs, symptoms and conditions) to FDA "
        "MedWatch adverse event reports. A health effect code describes what happened to the patient, not "
        "what went wrong with the device. Use E2403 (No Clinical Signs, Symptoms or Conditions) only when "
        "the report describes no effect on the patient at all. A death, an injury, a retained object or "
        "fragment, a hospital visit or any symptom is a clinical consequence and must be coded."
    ),
}

RULES = (
    "Rules:\n"
    "1. Use only codes from the list{where}. Never invent a code.\n"
    "2. Assign every distinct problem the report supports, usually one and at most five codes.\n"
    "3. Choose the most specific code the text supports. If the text supports only a general "
    "category, use the broader code instead of guessing a specific one.\n"
    "4. Base your answer only on the report text.\n"
)

RULES_NOLIST = (
    "Rules:\n"
    "1. Use official IMDRF {annex_name} codes (for example {example}) exactly as published. "
    "Never invent a code.\n"
    "2. Assign every distinct problem the report supports, usually one and at most five codes.\n"
    "3. Choose the most specific code the text supports. If the text supports only a general "
    "category, use the broader code instead of guessing a specific one.\n"
    "4. Base your answer only on the report text.\n"
)

NONE_RULE = "5. If none of the listed codes fits, answer with an empty list.\n"

OUTPUT = 'Answer with JSON only, no other text: {"codes": ["<code>", ...]}'

ANNEX = {"device": "A", "patient": "E"}
ANNEX_NAME = {"device": "Annex A device problem", "patient": "Annex E health effect"}
ANNEX_EXAMPLE = {"device": "A0504", "patient": "E1714"}


def load_codebook(path):
    """{annex: [row dicts in codebook order]} for primary, informative codes."""
    book = {"A": [], "E": []}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["primary"] == "True" and r["imdrf_code"] not in EXCLUDE:
                book[r["annex"]].append(r)
    return book


def full_list(rows):
    return "\n".join(f"{'  ' * (int(r['level']) - 1)}{r['imdrf_code']} {r['term']}" for r in rows)


def short_list(rows, keep):
    keep = set(keep)
    return "\n".join(
        f"{'  ' * (int(r['level']) - 1)}{r['imdrf_code']} {r['term']}: {r['definition'].strip()}"
        for r in rows if r["imdrf_code"] in keep)


def _s(v):
    return v.strip() if isinstance(v, str) else ""


def report_text(report):
    return f"Report:\n\n{_s(report['text_event'])}"


def examples_text(examples, names):
    """examples: list of {"text": narrative, "codes": [codes]} (FDA-coded past reports)."""
    out = ["Worked examples. Similar past reports with the codes FDA assigned to them:"]
    for i, ex in enumerate(examples, 1):
        codes = "; ".join(f"{c} {names.get(c, '')}".strip() for c in ex["codes"])
        out.append(f"\nExample {i}:\n{_s(ex['text'])}\nCodes: {codes}")
    return "\n".join(out)


def build_messages(task, arm, report, codebook, shortlist=None, examples=None):
    rows = codebook[ANNEX[task]]
    names = {r["imdrf_code"]: r["term"] for r in rows}
    if arm == "0":
        rules = RULES_NOLIST.format(annex_name=ANNEX_NAME[task], example=ANNEX_EXAMPLE[task])
        system = f"{TASK_TEXT[task]}\n\n{rules}\n{OUTPUT}"
        user = report_text(report)
    elif arm == "A":
        system = (f"{TASK_TEXT[task]}\n\n{RULES.format(where=' below')}\n{OUTPUT}\n\n"
                  f"Code list (indentation shows the hierarchy):\n{full_list(rows)}")
        user = report_text(report)
    elif arm in ("B", "C"):
        system = f"{TASK_TEXT[task]}\n\n{RULES.format(where=' of candidate codes')}{NONE_RULE}\n{OUTPUT}"
        parts = []
        if arm == "C":
            parts.append(examples_text(examples, names))
        parts.append("Candidate codes with definitions (indentation shows the hierarchy):\n"
                     + short_list(rows, shortlist))
        parts.append(report_text(report))
        user = "\n\n".join(parts)
    else:
        raise ValueError(arm)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
