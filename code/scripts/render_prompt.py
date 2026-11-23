"""Print the prompts of all four arms for one report, for the paper's appendix.

Usage: python -I render_prompt.py VOLUNTARY_DIR CONTEXT_DIR CODEBOOK.tsv TASK KEY [OUT.txt]

The Arm A codebook is shown with its first and last lines only.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_messages, load_codebook  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    vol_dir, ctx_dir, codebook, task, key = sys.argv[1:6]
    out = sys.argv[6] if len(sys.argv) > 6 else None
    book = load_codebook(codebook)
    vol = pd.read_parquet(Path(vol_dir) / f"{task}.parquet").set_index("mdr_report_key")
    ctx = pd.read_parquet(Path(ctx_dir) / f"context_{task}.parquet").set_index("mdr_report_key")
    r, c = vol.loc[key], ctx.loc[key]
    examples = [{"text": vol.at[k, "text_event"], "codes": list(vol.at[k, f"{task}_codes"])} for k in c.example_keys]
    lines = []
    for arm in "0ABC":
        msgs = build_messages(task, arm, {"text_event": r.text_event}, book,
                              shortlist=list(c.shortlist_B if arm == "B" else c.shortlist_C), examples=examples)
        lines.append(f"########## ARM {arm}")
        for m in msgs:
            body = m["content"]
            if arm == "A" and m["role"] == "system":
                head, _, book_txt = body.partition("Code list (indentation shows the hierarchy):\n")
                bl = book_txt.split("\n")
                body = head + "Code list (indentation shows the hierarchy):\n" + "\n".join(bl[:6]) + \
                    f"\n  [... {len(bl) - 9} more lines ...]\n" + "\n".join(bl[-3:])
            lines.append(f"--- {m['role']}\n{body}\n")
    lines.append(f"FDA codes: {list(r[f'{task}_codes'])}")
    text = "\n".join(lines)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
