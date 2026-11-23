"""Set-level scores for one predicted code set against one reference set.

Used for the models (prediction vs filed codes) and for the human ceiling
(one coder's codes vs another coder's codes on the same narrative), so both
are measured with exactly the same yardstick.

exact recall    share of reference codes predicted exactly
branch recall   reference code matched by a predicted ancestor or descendant
soft recall     each reference code earns its best similarity to any predicted
                code, where similarity follows the IMDRF tree (Wu-Palmer):
                sim(a, b) = 2 * depth(deepest shared ancestor) / (depth(a) + depth(b)).
                Same code 1.0; parent and child (levels 1 and 2) 0.67; siblings at
                level 2 0.5, at level 3 0.67; different top-level branch 0.
soft precision  the same from the predicted side
exact set       predicted set equals reference set
"""


def level(c):
    return (len(c) - 1) // 2  # A01 -> 1, A0101 -> 2, A010101 -> 3


def related(a, b):
    return a.startswith(b) or b.startswith(a)


def sim(a, b):
    la, lb = level(a), level(b)
    shared = 0
    for k in range(1, min(la, lb) + 1):
        if a[:1 + 2 * k] != b[:1 + 2 * k]:
            break
        shared = k
    return 2 * shared / (la + lb)


def _soft(ref, pred):
    if not pred:
        return 0.0
    return sum(max(sim(r, p) for p in pred) for r in ref) / len(ref)


def score(pred, ref):
    """pred, ref: iterables of codes; ref must be non-empty."""
    pred, ref = set(pred), set(ref)
    return {
        "recall": len(pred & ref) / len(ref),
        "branch_recall": sum(any(related(r, p) for p in pred) for r in ref) / len(ref),
        "soft_recall": _soft(ref, pred),
        "precision": len(pred & ref) / len(pred) if pred else 0.0,
        "soft_precision": _soft(pred, ref) if pred else 0.0,
        "exact": float(pred == ref),
    }


METRICS = ["recall", "branch_recall", "soft_recall", "precision", "soft_precision", "exact"]
