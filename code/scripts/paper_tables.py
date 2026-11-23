"""Every number in the paper comes from here.

Usage: python -I paper_tables.py CONTEXT_DIR PAPER_DIR

Reads runs/full/scores_{stage1,frontier,medgemma}.tsv, the retrieval context
(baselines, shortlist coverage) and data/voluntary_counts.tsv, and writes

  PAPER_DIR/main.tex   regions between "%% BEGIN GEN <name>" and "%% END GEN <name>"
                       are replaced; everything else in the file is left alone
  PAPER_DIR/figures/ladder.pdf     exact recall by arm, open panel, unseen set
  PAPER_DIR/figures/contrasts.pdf  paired contrasts on the 500-report subsample
  PAPER_DIR/numbers.txt            every generated number, for checking prose

Generated regions: numbers (macros), tab:data, tab:main, tab:contrasts,
tab:writer, tab:behaviour, tab:memorisation.

Bootstraps use seed 20261114 and 2,000 resamples, as analyze_stage1.py and
analyze_frontier.py do; point estimates are identical to those files, intervals
can differ in the third decimal because the draw order differs.
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "full"
TASKS = ("device", "patient")
ARMS = ("0", "A", "B", "C")
SEED, NBOOT = 20261114, 2000

# display names and order; the order is the order in every table and figure
OPEN = ["medgemma-4b-it", "llama-3.1-8b", "phi-4", "llama-3.3-70b", "qwen-2.5-72b",
        "llama-4-maverick", "gpt-oss-120b"]
FRONTIER = ["gpt-5.5", "claude-opus-4.8"]
NAME = {"medgemma-4b-it": "MedGemma 4B", "llama-3.1-8b": "Llama 3.1 8B", "phi-4": "Phi-4 14B",
        "llama-3.3-70b": "Llama 3.3 70B", "qwen-2.5-72b": "Qwen 2.5 72B",
        "llama-4-maverick": "Llama 4 Maverick", "gpt-oss-120b": "gpt-oss-120b",
        "gpt-5.5": "GPT-5.5", "claude-opus-4.8": "Claude Opus 4.8"}
SHORT = {"medgemma-4b-it": "MedGemma", "llama-3.1-8b": "Llama 8B", "phi-4": "Phi-4",
         "llama-3.3-70b": "Llama 70B", "qwen-2.5-72b": "Qwen 72B",
         "llama-4-maverick": "Maverick", "gpt-oss-120b": "gpt-oss",
         "gpt-5.5": "GPT-5.5", "claude-opus-4.8": "Opus 4.8"}
TASKNAME = {"device": "Device problem", "patient": "Patient effect"}
WRITER = {"lay": "Patient or family", "professional": "Health professional", "unknown": "Manufacturer notification"}
# validated categorical palette (dataviz skill, light mode, adjacent pairs pass)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

rng = np.random.default_rng(SEED)
numbers = {}


def num(key, value, fmt="{:.2f}"):
    numbers[key] = fmt.format(value) if not isinstance(value, str) else value
    return numbers[key]


def ci(diff):
    """Percentile bootstrap. A fresh generator with the fixed seed per call, so every contrast
    uses the same resampling indices and adding a contrast never changes another one's interval
    (the global-stream version shifted a borderline interval when the Arm B contrasts were added)."""
    diff = np.asarray(diff)
    r = np.random.default_rng(SEED)
    bs = [diff[r.integers(0, len(diff), len(diff))].mean() for _ in range(NBOOT)]
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def pm(x):
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def r2(x):
    """Round to two decimals with negative zero folded into zero, as printed."""
    v = round(float(x), 2)
    return 0.0 if v == 0 else v


def fmt_ci(mean, lo, hi):
    return f"{pm(mean)} [{r2(lo):.2f}, {r2(hi):.2f}]"


def sig(lo, hi):
    """Significance as the printed interval shows it: both bounds rounded to two decimals, so an
    interval printed as [0.00, 0.06] is never starred."""
    return r2(lo) > 0 or r2(hi) < 0


def load():
    s1 = pd.read_csv(RUNS / "scores_stage1.tsv", sep="\t", dtype={"key": str})
    fr = pd.read_csv(RUNS / "scores_frontier.tsv", sep="\t", dtype={"key": str})
    mg = pd.read_csv(RUNS / "scores_medgemma.tsv", sep="\t", dtype={"key": str})
    d = pd.concat([s1, mg, fr], ignore_index=True)
    d["model"] = d.model.str.replace("-instruct", "")
    d["arm"] = d.arm.astype(str)
    for c in ("recall", "soft_recall", "branch_recall", "precision", "soft_precision", "exact", "n_pred"):
        d[c] = d[c].fillna(0.0)
    d["parsed"] = d.parsed.fillna(0).astype(int)
    d["invalid_codes"] = d.invalid_codes.fillna(0)
    return d


def per_report(d, task, arm, model, keys, mets=("recall", "soft_recall")):
    g = d[(d.task == task) & (d.arm == arm) & (d.model == model)].set_index("key")
    g = g[~g.index.duplicated()].reindex(keys)
    return g[list(mets)].fillna(0.0)


def baseline_frame(ctx, keys, col):
    c = ctx.loc[keys]
    return pd.DataFrame([score(list(p), list(f)) for p, f in zip(c[col], c.filed)], index=c.index)


def replace_region(tex, name, body):
    pat = re.compile(rf"(%% BEGIN GEN {re.escape(name)}\n)(.*?)(%% END GEN {re.escape(name)})", re.S)
    if not pat.search(tex):
        raise SystemExit(f"region {name} not found in main.tex")
    return pat.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(3), tex)


def main():
    ctx_dir, paper = Path(sys.argv[1]), Path(sys.argv[2])
    ctx = {t: pd.read_parquet(ctx_dir / f"context_{t}.parquet").set_index("mdr_report_key") for t in TASKS}
    d = load()
    unseen = {t: ctx[t].index[ctx[t].set == "unseen"] for t in TASKS}
    sub = {t: pd.read_csv(ROOT / "data" / f"frontier_subsample_{t}.tsv", sep="\t", dtype=str).mdr_report_key for t in TASKS}
    sub = {t: pd.Index(s.values) for t, s in sub.items()}
    regions = {}

    # ---------------------------------------------------------------- data table
    vc = pd.read_csv(ROOT / "data" / "voluntary_counts.tsv", sep="\t")
    rows = []
    for t in TASKS:
        for s, label in (("pool", "Pool"), ("comparison", "Comparison"), ("unseen", "Unseen")):
            v = vc[(vc.task == t) & (vc.set == s)].set_index("writer")
            tot = int(v.reports.sum())
            cells = [f"{int(v.at[w, 'reports']):,}" for w in ("lay", "professional", "unknown")]
            rows.append(f"{TASKNAME[t] if s == 'pool' else ''} & {label} & {tot:,} & " + " & ".join(cells) + "\\\\")
            num(f"n_{t}_{s}", tot, "{:,}")
            for w in ("lay", "professional", "unknown"):
                num(f"n_{t}_{s}_{w}", int(v.at[w, "reports"]), "{:,}")
        num(f"n_{t}_total", int(vc[vc.task == t].reports.sum()), "{:,}")
        rows.append("\\midrule")
    rows.pop()
    regions["tab:data"] = "\n".join(rows)

    # ---------------------------------------------------------------- baselines
    base = {}
    for t in TASKS:
        for pop, keys in (("unseen", unseen[t]), ("sub", sub[t])):
            for col in ("prior_top1", "nn_codes", "knn_top1"):
                b = baseline_frame(ctx[t], keys, col)
                base[(t, pop, col)] = b
                num(f"base_{col}_{t}_{pop}_recall", b.recall.mean())
                num(f"base_{col}_{t}_{pop}_soft", b.soft_recall.mean())
            c = ctx[t].loc[keys]
            num(f"cover_B_{t}_{pop}", c.covered_B.mean())
            num(f"cover_C_{t}_{pop}", c.covered_C.mean())
            num(f"cover_gain_{t}_{pop}", c.covered_C.mean() - c.covered_B.mean())
            num(f"nB_{t}_{pop}", c.n_B.mean(), "{:.0f}")
            num(f"nC_{t}_{pop}", c.n_C.mean(), "{:.0f}")

    # ---------------------------------------------------------------- main table
    def cell(t, arm, m, keys):
        """Mean over the reports this model/arm actually answered (unparsed answers score 0;
        reports never sent, such as the 400 outside the frontier Arm 0 probe, are not counted)."""
        answered = d[(d.task == t) & (d.arm == arm) & (d.model == m) & d.key.isin(keys)].key.unique()
        if len(answered) == 0:
            return None, None, 0
        g = per_report(d, t, arm, m, pd.Index(answered))
        return g.recall.mean(), g.soft_recall.mean(), len(answered)

    lines = []
    for t in TASKS:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{TASKNAME[t]}: open panel on all {len(unseen[t]):,} unseen reports}}}}\\\\")
        for m in OPEN:
            cells = []
            for arm in ARMS:
                r, s, n = cell(t, arm, m, unseen[t])
                cells.append(f"{r:.2f} ({s:.2f})")
                num(f"r_{m}_{t}_{arm}", r)
                num(f"s_{m}_{t}_{arm}", s)
            lines.append(f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
        nn = base[(t, "unseen", "nn_codes")]
        lines.append(f"Copy nearest coded report & \\multicolumn{{4}}{{c}}{{{nn.recall.mean():.2f} ({nn.soft_recall.mean():.2f})}}\\\\")
        lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{TASKNAME[t]}: frontier models on the fixed subsample of {len(sub[t]):,} reports}}}}\\\\")
        for m in FRONTIER:
            cells = []
            for arm in ARMS:
                r, s, n = cell(t, arm, m, sub[t])
                if n == 0:
                    cells.append("not run")
                elif arm == "0":
                    cells.append(f"{r:.2f} ({s:.2f})$^{{\\dagger}}$")
                    num(f"r_{m}_{t}_{arm}", r); num(f"s_{m}_{t}_{arm}", s); num(f"n0_{m}_{t}", n, "{:d}")
                else:
                    cells.append(f"{r:.2f} ({s:.2f})")
                    num(f"r_{m}_{t}_{arm}", r); num(f"s_{m}_{t}_{arm}", s)
            lines.append(f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
        nn = base[(t, "sub", "nn_codes")]
        lines.append(f"Copy nearest coded report & \\multicolumn{{4}}{{c}}{{{nn.recall.mean():.2f} ({nn.soft_recall.mean():.2f})}}\\\\")
        if t == "device":
            lines.append("\\midrule")
    regions["tab:main"] = "\n".join(lines)

    # open-panel ranges per arm, for the prose
    for t in TASKS:
        for arm in ARMS:
            rs = [float(numbers[f"r_{m}_{t}_{arm}"]) for m in OPEN]
            ss = [float(numbers[f"s_{m}_{t}_{arm}"]) for m in OPEN]
            num(f"rmin_{t}_{arm}", min(rs)); num(f"rmax_{t}_{arm}", max(rs))
            num(f"smin_{t}_{arm}", min(ss)); num(f"smax_{t}_{arm}", max(ss))
            best = OPEN[int(np.argmax(rs))]
            num(f"rbest_{t}_{arm}", NAME[best])
    # ladder check: does every open model climb 0 < A < B < C on exact recall?
    monotone = all(float(numbers[f"r_{m}_{t}_{a}"]) < float(numbers[f"r_{m}_{t}_{b}"])
                   for m in OPEN for t in TASKS for a, b in zip(ARMS, ARMS[1:]))
    num("ladder_monotone_all", "yes" if monotone else "NO")

    # ---------------------------------------------------------------- contrasts
    con = []  # rows: task, label, metric, mean, lo, hi, group
    for t in TASKS:
        keys = unseen[t]
        nnb = base[(t, "unseen", "nn_codes")]
        for m in OPEN:
            C = per_report(d, t, "C", m, keys)
            for met in ("recall", "soft_recall"):
                mean, lo, hi = ci((C[met] - nnb[met]).to_numpy())
                con.append(dict(task=t, group="open_vs_nn", label=NAME[m], metric=met, mean=mean, lo=lo, hi=hi))
                num(f"c_{m}_C_nn_{t}_{met}", fmt_ci(mean, lo, hi))
        keys = sub[t]
        nnb = base[(t, "sub", "nn_codes")]
        for m in FRONTIER:
            A, B, C = (per_report(d, t, a, m, keys) for a in ("A", "B", "C"))
            for met in ("recall", "soft_recall"):
                mean, lo, hi = ci((C[met] - nnb[met]).to_numpy())
                con.append(dict(task=t, group="frontier_vs_nn", label=NAME[m], metric=met, mean=mean, lo=lo, hi=hi))
                num(f"c_{m}_C_nn_{t}_{met}", fmt_ci(mean, lo, hi))
                for g, X, Y in (("C_A", C, A), ("B_A", B, A), ("C_B", C, B)):
                    mean, lo, hi = ci((X[met] - Y[met]).to_numpy())
                    con.append(dict(task=t, group=f"frontier_{g}", label=NAME[m], metric=met, mean=mean, lo=lo, hi=hi))
                    num(f"c_{m}_{g}_{t}_{met}", fmt_ci(mean, lo, hi))
        for arm in ("A", "B", "C"):
            X, Y = per_report(d, t, arm, "claude-opus-4.8", keys), per_report(d, t, arm, "gpt-5.5", keys)
            for met in ("recall", "soft_recall"):
                mean, lo, hi = ci((X[met] - Y[met]).to_numpy())
                con.append(dict(task=t, group=f"opus_gpt_{arm}", label=f"Arm {arm}", metric=met, mean=mean, lo=lo, hi=hi))
                num(f"c_opus_gpt_{arm}_{t}_{met}", fmt_ci(mean, lo, hi))
        for fm in FRONTIER:
            F = per_report(d, t, "C", fm, keys)
            for om in OPEN:
                O = per_report(d, t, "C", om, keys)
                for met in ("recall", "soft_recall"):
                    mean, lo, hi = ci((F[met] - O[met]).to_numpy())
                    con.append(dict(task=t, group=f"{fm}_vs_open", label=NAME[om], metric=met, mean=mean, lo=lo, hi=hi))
                    num(f"c_{fm}_{om}_C_{t}_{met}", fmt_ci(mean, lo, hi))
    con = pd.DataFrame(con)

    # contrast table: exact recall, both tasks side by side
    def crow(label, g, lab):
        cells = []
        for t in TASKS:
            r = con[(con.task == t) & (con.group == g) & (con.label == lab) & (con.metric == "recall")].iloc[0]
            s = con[(con.task == t) & (con.group == g) & (con.label == lab) & (con.metric == "soft_recall")].iloc[0]
            cells.append(fmt_ci(r["mean"], r.lo, r.hi) + ("*" if sig(r.lo, r.hi) else ""))
            cells.append(fmt_ci(s["mean"], s.lo, s.hi) + ("*" if sig(s.lo, s.hi) else ""))
        return f"{label} & " + " & ".join(cells) + "\\\\"

    STEP = {"B_A": "B minus A", "C_B": "C minus B", "C_A": "C minus A"}
    lines = ["\\multicolumn{5}{@{}l}{\\emph{Does retrieval help a frontier model? Same model, one arm minus another}}\\\\"]
    for m in FRONTIER:
        for g in ("B_A", "C_B", "C_A"):
            lines.append(crow(f"{SHORT[m]}, {STEP[g]}", f"frontier_{g}", NAME[m]))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Claude Opus 4.8 minus GPT-5.5, same arm}}\\\\")
    for arm in ("A", "B", "C"):
        lines.append(crow(f"Arm {arm}", f"opus_gpt_{arm}", f"Arm {arm}"))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Claude Opus 4.8, Arm C, minus each open model, Arm C}}\\\\")
    for om in OPEN:
        lines.append(crow(NAME[om], "claude-opus-4.8_vs_open", NAME[om]))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{GPT-5.5, Arm C, minus each open model, Arm C}}\\\\")
    for om in OPEN:
        lines.append(crow(NAME[om], "gpt-5.5_vs_open", NAME[om]))
    regions["tab:contrasts"] = "\n".join(lines)

    # ---------------------------------------------------------------- writer table (Arm C)
    lines = []
    for t in TASKS:
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{TASKNAME[t]}}}}}\\\\")
        for m in OPEN + FRONTIER:
            keys = sub[t] if m in FRONTIER else unseen[t]
            g = d[(d.task == t) & (d.arm == "C") & (d.model == m) & d.key.isin(keys)]
            g = g.drop_duplicates("key")
            cells = []
            for w in ("lay", "professional", "unknown"):
                gw = g[g.writer == w]
                cells.append(f"{gw.recall.mean():.2f} ({gw.soft_recall.mean():.2f})")
                num(f"w_{m}_{t}_{w}_r", gw.recall.mean()); num(f"w_{m}_{t}_{w}_s", gw.soft_recall.mean())
            mark = "$^{\\ddagger}$" if m in FRONTIER else ""
            lines.append(f"{NAME[m]}{mark} & " + " & ".join(cells) + "\\\\")
        if t == "device":
            lines.append("\\midrule")
    regions["tab:writer"] = "\n".join(lines)
    for t in TASKS:
        for w in ("lay", "professional", "unknown"):
            n = d[(d.task == t) & (d.arm == "C") & (d.model == OPEN[0]) & d.key.isin(unseen[t]) & (d.writer == w)].key.nunique()
            num(f"nw_{t}_{w}", n, "{:,}")
            n = d[(d.task == t) & (d.arm == "C") & (d.model == FRONTIER[0]) & d.key.isin(sub[t]) & (d.writer == w)].key.nunique()
            num(f"nws_{t}_{w}", n, "{:,}")
    # Arm C, report-level: share of reports on which at least one of FDA's codes was matched exactly
    # or by an ancestor or descendant (the "right code or its branch is on the menu" figure), in percent
    for t in TASKS:
        for m in OPEN + FRONTIER:
            keys = sub[t] if m in FRONTIER else unseen[t]
            g = d[(d.task == t) & (d.arm == "C") & (d.model == m) & d.key.isin(keys)].drop_duplicates("key")
            num(f"hit_{m}_{t}_C", 100 * (g.branch_recall > 0).mean(), "{:.0f}")
        # copy-the-nearest baseline on the 100 reports of the frontier Arm 0 probe
        probe = d[(d.task == t) & (d.arm == "0") & (d.model == FRONTIER[0])].key.unique()
        b = baseline_frame(ctx[t], pd.Index(probe), "nn_codes")
        num(f"base_nn_codes_{t}_probe_recall", b.recall.mean())
        num(f"base_nn_codes_{t}_probe_soft", b.soft_recall.mean())
    # lay minus professional, patient task, exact recall, per model
    for m in OPEN + FRONTIER:
        num(f"gap_layprof_{m}", pm(float(numbers[f"w_{m}_patient_lay_r"]) - float(numbers[f"w_{m}_patient_professional_r"])))

    # ---------------------------------------------------------------- behaviour table (parsed, invalid, codes per answer)
    lines = []
    for m in OPEN + FRONTIER:
        cells = []
        for t in TASKS:
            keys = sub[t] if m in FRONTIER else unseen[t]
            for arm in ("A", "C"):
                g = d[(d.task == t) & (d.arm == arm) & (d.model == m) & d.key.isin(keys)].drop_duplicates("key")
                if len(g) == 0:
                    cells.append("--")
                    continue
                cells.append(f"{100 * (1 - g.parsed.mean()):.0f} & {100 * g.invalid_codes.mean():.0f} & {g.n_pred.mean():.1f}")
                num(f"unparsed_{m}_{t}_{arm}", 100 * (1 - g.parsed.mean()), "{:.0f}")
                num(f"invalid_{m}_{t}_{arm}", 100 * g.invalid_codes.mean(), "{:.0f}")
                num(f"npred_{m}_{t}_{arm}", g.n_pred.mean(), "{:.1f}")
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
    regions["tab:behaviour"] = "\n".join(lines)
    # Arm 0 invented codes, for the prose
    for m in OPEN + FRONTIER:
        for t in TASKS:
            g = d[(d.task == t) & (d.arm == "0") & (d.model == m)].drop_duplicates("key")
            if len(g):
                num(f"invalid0_{m}_{t}", 100 * g.invalid_codes.mean(), "{:.0f}")

    # ---------------------------------------------------------------- memorisation table
    lines = []
    for m in OPEN:
        cells = []
        for t in TASKS:
            g = d[(d.task == t) & (d.arm == "A") & (d.model == m)].drop_duplicates(["key", "set"])
            comp = g[g.set == "comparison"].soft_recall.mean()
            uns = g[g.set == "unseen"].soft_recall.mean()
            cells.append(f"{comp:.2f} & {uns:.2f} & {pm(comp - uns)}")
            num(f"mem_{m}_{t}", pm(comp - uns))
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
    regions["tab:memorisation"] = "\n".join(lines)
    gaps = [float(numbers[f"mem_{m}_{t}"]) for m in OPEN for t in TASKS]
    num("mem_min", pm(min(gaps))); num("mem_max", pm(max(gaps)))

    # ------------------------------------------------ uncoded reports by writer
    # From scripts/uncoded_by_writer.py over the whole 2021 to mid-2026 extract:
    # share of reports with no device-problem term at all, by writer group.
    root = Path(__file__).resolve().parent.parent
    tsv = (root / "data" / "uncoded_by_writer.tsv").read_text(encoding="utf-8").splitlines()[1:]
    tot = {"industry": [0, 0], "nonindustry": [0, 0]}
    for line in tsv:
        w, src, n, nc = line.split("\t")[:4]
        key = "industry" if w == "industry" else "nonindustry"
        tot[key][0] += int(n)
        tot[key][1] += int(nc)
        if src == "voluntary" and w in ("lay", "professional"):
            num(f"uncoded_{w}_pct", 100 * int(nc) / int(n), "{:.0f}")
    num("uncoded_nonindustry_pct", 100 * tot["nonindustry"][1] / tot["nonindustry"][0], "{:.0f}")
    num("uncoded_industry_pct", 100 * tot["industry"][1] / tot["industry"][0], "{:.1f}")

    # ---------------------------------------------------------------- cost
    cost = d[d.cost.notna()]
    num("cost_total", cost.cost.sum(), "{:.0f}")
    for m in OPEN + FRONTIER:
        for t in TASKS:
            for arm in ("A", "B", "C"):
                g = cost[(cost.model == m) & (cost.task == t) & (cost.arm == arm)]
                if len(g):
                    num(f"cost_{m}_{t}_{arm}", 100 * g.cost.mean(), "{:.2f}")  # cents per call

    # ---------------------------------------------------------------- figure 1: ladder
    def spread(ys, gap):
        """Push label positions apart by at least gap, keeping their order; returns new positions."""
        order = np.argsort(ys)
        pos = np.array(ys, dtype=float)
        for a, b in zip(order, order[1:]):
            if pos[b] - pos[a] < gap:
                pos[b] = pos[a] + gap
        return pos

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, t in zip(axes, TASKS):
        x = np.arange(len(ARMS))
        ends = []
        for i, m in enumerate(OPEN):
            y = [float(numbers[f"r_{m}_{t}_{a}"]) for a in ARMS]
            ax.plot(x, y, color=PALETTE[i], lw=1.6, marker="o", ms=4, mec="white", mew=0.6, zorder=3)
            ends.append(y[-1])
        lab = spread(ends, 0.036)
        for i, m in enumerate(OPEN):
            ax.plot([3.03, 3.14], [ends[i], lab[i]], color="#bbbbbb", lw=0.6, zorder=2)
            ax.text(3.17, lab[i], SHORT[m], va="center", fontsize=6.5, color="#333333")
        nnv = float(numbers[f"base_nn_codes_{t}_unseen_recall"])
        ax.axhline(nnv, color="#777777", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate("copy nearest coded report", (-0.1, nnv), xytext=(0, -3), textcoords="offset points",
                    fontsize=6.5, color="#555555", va="top", ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels(["0\nno list", "A\nfull\ncodebook", "B\nshortlist", "C\nshortlist +\nexamples"], fontsize=7.5)
        ax.set_title(TASKNAME[t], fontsize=9.5, loc="left")
        ax.set_ylim(0, 0.72)
        ax.set_xlim(-0.2, 3.9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#999999"); ax.spines["bottom"].set_color("#999999")
        ax.tick_params(colors="#333333", labelsize=7.5)
    axes[0].set_ylabel("Exact recall of FDA's codes", fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(paper / "figures" / "ladder.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2: contrasts forest plot (exact recall)
    blocks = [("Retrieval for a frontier model",
               [(f"frontier_{g}", NAME[m], f"{SHORT[m]}, {STEP[g]}") for m in FRONTIER for g in ("B_A", "C_B", "C_A")]),
              ("Opus 4.8 minus GPT-5.5", [(f"opus_gpt_{a}", f"Arm {a}", f"Arm {a}") for a in ("A", "B", "C")]),
              ("Opus 4.8 minus open model\n(both Arm C)", [("claude-opus-4.8_vs_open", NAME[m], NAME[m]) for m in reversed(OPEN)]),
              ("GPT-5.5 minus open model\n(both Arm C)", [("gpt-5.5_vs_open", NAME[m], NAME[m]) for m in reversed(OPEN)])]
    labels, groups, ypos, spans = [], [], [], []
    y = 0
    for title, items in blocks:
        start = y
        for g, lab, disp in items:
            labels.append(disp); groups.append((g, lab)); ypos.append(y); y += 1
        spans.append((title, start, y - 1))
        y += 1.6
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 6.2), sharey=True)
    for ax, t in zip(axes, TASKS):
        for (g, lab), yy in zip(groups, ypos):
            r = con[(con.task == t) & (con.group == g) & (con.label == lab) & (con.metric == "recall")].iloc[0]
            filled = sig(r.lo, r.hi)
            ax.plot([r.lo, r.hi], [yy, yy], color="#2a78d6", lw=1.4, zorder=2)
            ax.plot(r["mean"], yy, marker="o", ms=5, color="#2a78d6" if filled else "white",
                    mec="#2a78d6", mew=1.2, zorder=3)
        ax.axvline(0, color="#777777", lw=0.9, zorder=1)
        for i, (title, s, e) in enumerate(spans):
            if i % 2:
                ax.axhspan(s - 0.5, e + 0.5, color="#f3f3f1", zorder=0)
            ax.text(0.0, s - 0.6, title, fontsize=7, color="#555555", va="bottom", ha="left",
                    style="italic", transform=ax.get_yaxis_transform())
        ax.set_title(TASKNAME[t], fontsize=9.5, loc="left")
        ax.set_xlabel("Difference in exact recall, 95% CI", fontsize=8)
        ax.set_xlim(-0.12, 0.27)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=7.5, colors="#333333")
        ax.tick_params(axis="x", labelsize=7.5, colors="#333333")
    axes[0].set_ylim(ypos[-1] + 0.8, -1.6)  # reversed limits: first block at the top
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(paper / "figures" / "contrasts.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- macros and write-out
    def macro_name(k):
        # LaTeX macro names: letters only
        tr = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
              "8": "Eight", "9": "Nine", ".": "", "-": "", "_": ""}
        parts = re.split(r"[_.-]", k)
        out = "".join(p[:1].upper() + p[1:] for p in parts if p)
        return "".join(tr.get(ch, ch) for ch in out)
    macros = "\n".join(f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}" for k, v in sorted(numbers.items()))
    regions["numbers"] = macros

    tex_path = paper / "main.tex"
    tex = tex_path.read_text(encoding="utf-8")
    for name, body in regions.items():
        tex = replace_region(tex, name, body)
    tex_path.write_text(tex, encoding="utf-8", newline="\n")
    with open(paper / "numbers.txt", "w", encoding="utf-8") as f:
        for k, v in sorted(numbers.items()):
            f.write(f"{k}\t{macro_name(k)}\t{v}\n")
    print(f"wrote {len(regions)} regions, {len(numbers)} numbers, 2 figures")
    print("ladder monotone for every open model:", numbers["ladder_monotone_all"])


if __name__ == "__main__":
    main()
