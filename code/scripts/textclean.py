"""Remove FDA's intake note from voluntary MedWatch narratives.

FDA staff append the codes they assigned, and the sibling report numbers, to
the narrative, e.g. "PATIENT CODES: 1807, 1880. DEVICE CODES: 1250, 3023.
REFERENCE REPORT#: MW5184570, MW5184571." or "PT: 4582. DEVICE CODE: 2964.
REF: MW5184076." The numbers are FDA legacy codes that map one-to-one onto the
filed IMDRF codes, so they are the answer key and must never reach a model.

strip_note(text) -> (clean_text, had_note)
"""
import re

_CODES = r"\d{4}(?:\s*[,;.]?\s*(?:AND\s+)?\d{4})*"
_LABEL = r"(?:PATIENT|PT|DEVICE|PRODUCT|HEALTH|DPC|HECC|DP|PP)(?:\s+(?:PROBLEM|EFFECT|DEVICE))?"
_PATTERNS = [
    # "PATIENT CODES: 1807, 1880." / "DEVICE CODE: 2964" / "HEALTH EFFECT CODE: 4582" / "PRODUCT PROBLEM CODE:"
    rf"\b{_LABEL}\s*CODES?\s*#?\s*[:=-]?\s*{_CODES}\.?",
    # "PT: 4582." / "DEVICE: 2423, 2338." / "PATIENT DEVICE: 1943, 2033." / "DPC: 3016." / "PT-4577, 1754. DEVICE-2682."
    rf"\b{_LABEL}\s*#?\s*[:=-]\s*{_CODES}\.?",
    # "ANNEX E: 1776, 1770."
    rf"\bANNEX\s+[A-G]\s*:\s*{_CODES}\.?",
    # "REF: MW5188804, MW5188805." / "REFERENCE REPORT#: MW..." / "REFERENCING REPORTS MW... AND MW..."
    r"\b(?:REF(?:ERENCE)?|REFERENCING)\s*(?:REPORTS?)?\s*#?\s*:?\s*(?:MW\s?\d{6,8}\s*(?:,|;|AND|&)?\s*)+\.?",
    # a bare list of MedWatch numbers left over
    r"\b(?:MW\s?\d{6,8}\s*(?:,|;|AND|&)?\s*){1,}\.?",
    # dangling intake fragments: "REFERENCED REPORTS-", "REFERENCE REPORTS:", "PROCODE-JJX."
    r"\bREFERENCED?\s+REPORTS?\s*[-:]\s*(?=\s|$|\n)",
    r"\bPROCODE\s*[-:]\s*[A-Z]{3}\.?",
]
_RX = [re.compile(p, re.I) for p in _PATTERNS]
_WS = re.compile(r"[ \t]+")
_PUNCT = re.compile(r"(\s*\.){2,}")


def strip_note(text):
    if not isinstance(text, str):
        return text, False
    out = text
    for rx in _RX:
        out = rx.sub(" ", out)
    had = out != text
    if had:
        out = _WS.sub(" ", out)
        out = _PUNCT.sub(".", out)
        out = re.sub(r"\s+([.,;])", r"\1", out)
        out = "\n".join(line.strip() for line in out.split("\n")).strip()
    return out, had


# residual check: anything that still looks like a code list
RESIDUAL = re.compile(r"\b(?:CODE|CODES|ANNEX|MW\d{6})\b", re.I)
