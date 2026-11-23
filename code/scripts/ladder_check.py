"""List every model/task/arm step where exact or soft recall does not rise. Usage: ladder_check.py PAPER_DIR/numbers.txt"""
import sys

nums = {}
for line in open(sys.argv[1], encoding="utf-8"):
    k, _, v = line.rstrip("\n").split("\t")
    nums[k] = v
models = sorted({k.split("_")[1] for k in nums if k.startswith("r_") and k.count("_") == 3})
arms = ["0", "A", "B", "C"]
for met in ("r", "s"):
    for m in models:
        for t in ("device", "patient"):
            vals = [nums.get(f"{met}_{m}_{t}_{a}") for a in arms]
            for a, b, va, vb in zip(arms, arms[1:], vals, vals[1:]):
                if va is None or vb is None:
                    continue
                if float(vb) <= float(va):
                    print(f"{'exact' if met == 'r' else 'soft'} {m:18s} {t:8s} {a}->{b}: {va} -> {vb}")
