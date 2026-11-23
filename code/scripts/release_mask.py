"""Make the release copy of the cleaned voluntary set with residual personal data masked.

Usage: python -I release_mask.py VOLUNTARY_DIR OUT_DIR

FDA already redacts names and dates as "(B)(6)". This pass masks, by rule, what the
scan (pii_scan.py) showed slipping through: phone numbers, email addresses, street
addresses, SSN-shaped numbers, and an honorific followed by a word that is not an
ordinary English word (a surname). Each is replaced by "(B)(6)", FDA's own marker,
so the released text reads like the rest of MAUDE. Over-masking a company's
toll-free number costs nothing. The models never saw this file; it is only for
redistribution. Writes {task}_release.parquet and a masking count table.
"""
import re
import sys
from pathlib import Path

import pandas as pd

MASK = "(B)(6)"
RULES = {
    # local part must start with a letter: "1V@0.8MS" is a pacing threshold, not an address
    "email": re.compile(r"(?<![\w.])[A-Z][A-Z0-9._%+-]*@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    "phone": re.compile(r"(?<![\d.])(?:\+?1[\s.-]{0,2})?\(?\d{3}\)?[\s.-]{1,2}\d{3}[\s.-]{1,2}\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    # house number, one to four plain words (no punctuation, so a sentence boundary breaks the
    # match), then either a long street word, or a short one followed by a city/state/ZIP tail
    # house number, one to four plain words, a street word, then a city, state and ZIP tail;
    # every real address in the scan had the tail, and common words such as "place" or
    # "drive" never do. House numbers can look like years (2054 Height Street), so no year
    # filter. The tail is masked too.
    "street": re.compile(
        r"(?<!\d)\d{1,5}\s+(?:[A-Z][A-Z'-]*\s+){1,4}"
        r"(?:STREET|AVENUE|ROAD|BOULEVARD|DRIVE|LANE|COURT|PLACE|CIRCLE|TERRACE|HIGHWAY|PARKWAY|"
        r"ST|AVE|RD|BLVD|DR|LN|CT|PL|CIR|HWY|PKWY)\b\.?"
        r"\s*,?\s*(?:(?:STE|SUITE|APT|UNIT|#)\s*\w+\s*,?\s*)?"
        r"[A-Z][A-Z\s.]{1,30}?,\s*[A-Z]{2},?\s*\d{5}\b", re.I),
}
# not after a digit or slash: "0.4 MS. OUTPUT" is milliseconds, "LUX DR" a device name
HONORIFIC = re.compile(r"(?<![\d/.]\s)(?<![\d/.])\b(DR|MR|MRS|MS|PROF)\.?\s+([A-Z][A-Za-z'-]{2,})(?:\s+([A-Z][A-Za-z'-]{2,}))?",
                       re.I)  # "Miss" is left out: in MAUDE it is almost always "near miss" or "miss fire"
# a tiny English word list is enough: medical narrative words that follow "Dr" and are not names
COMMON = set("""and the was said who that did for told this visits office had any also but not has have one two about
after again all appointment appt are asked at because before being called came can checked could didn does even explained
felt finally found from gave got her him his how immediately informed instructed is it just keep later looked made may
mentioned met my never new next noted now offices only or ordered our out performed prescribed put recommended referred
removed replaced reported saw says say seen sent she showed since so stated states still suggested surgery take then there
they thought took tried until visit wanted wants went were when where which while will with would wrote you appointments
advised agreed confirmed decided determined diagnosed examined inserted notes nurse orders placed recommends remove saying
scheduled see sees spoke state suspects thinks whom years believes confirms indicated admitted call calls check come
contacted emailed feels going having held insisted know left let listed look make noticed offered opinion order please
prescribes pulled requested sends should switched test tested think upon used wait wasn whats writes yet anyone everything
something nothing someone anything fit fits becomes sensing major minor occurrence prior unable diagnosis advise informs
notice report recommendation infusion unipolar bipolar dose doses attempted conditional work asking events noise aware fda
school tests during due notified system their atrial pacemaker environment multiple continued unit three patient became
several takes results broke appts icd cartridge treatment symptoms nor reports explant deployed assessed led first insert
table preferred ran wills office visit visits scholl pepper appointment told said who was had has have this that then
there these those what when where which while with would could should might must shall will can may also very much more
most some such than too into onto over under again further once here both each few other same own just don doesn didn
isn aren wasn weren hasn haven hadn won wouldn shouldn couldn mightn mustn needn immediately eventually finally
hospital clinic emergency room surgeon surgery procedure replacement revision implant device pump sensor monitor
pulled removed placed inserted explanted implanted continues stating concluded retired reviewing findings technical
output high last safety loop recorder didn't doesn't wasn't isn't couldn't wouldn't won't can't hasn't haven't aren't
weren't attribute attributed appointment appointments prescribed increased decreased changed suggested recommended
replaced suspected suspected believes believed wanted wants needs needed office offices hospital clinic regarding
described ended like cardiologist mri provider providers request requests why decided wondering explained
specialist specialists surgeon surgeons oncologist neurologist dermatologist urologist gynecologist orthopedic
pediatrician optometrist ophthalmologist dentist radiologist anesthesiologist psychiatrist therapist ever vice
environmental rather discretion started stopped finished mentioned mention mentions ignoring ignored agreed""".split())


def mask_honorific(m):
    title, w1, w2 = m.group(1), m.group(2), m.group(3)
    if w1.lower() in COMMON:
        return m.group(0)
    if w2 and w2.lower() in COMMON:  # "Dr Kurita decided": keep the verb
        return f"{title} {MASK} {w2}"
    return f"{title} {MASK}"


def mask(text):
    if not isinstance(text, str):
        return text, {}
    counts = {}
    for name, rx in RULES.items():
        text, n = rx.subn(MASK, text)
        if n:
            counts[name] = n
    text, n = HONORIFIC.subn(mask_honorific, text)
    if n:
        counts["honorific_checked"] = n
    return text, counts


vol, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
for task in ("device", "patient"):
    d = pd.read_parquet(vol / f"{task}.parquet")
    masked, tally = [], {}
    for t in d.text_event:
        mt, c = mask(t)
        masked.append(mt)
        for k, v in c.items():
            tally[k] = tally.get(k, 0) + v
    changed = int(sum(a != b for a, b in zip(masked, d.text_event)))
    d["text_event"] = masked
    d.to_parquet(out_dir / f"{task}_release.parquet", index=False)
    rows.append(dict(task=task, narratives=len(d), changed=changed, **tally))
    print(task, len(d), "narratives;", changed, "changed;", tally)
pd.DataFrame(rows).to_csv(out_dir / "mask_counts.tsv", sep="\t", index=False)
