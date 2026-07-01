"""H3 — is the cheap probe signal a valid/leading indicator of true (full-eval) recall?
(HYPOTHESIS.md H3.) Reads a trajectory CSV; per (f, policy) computes Spearman between the probe
signal and (a) concurrent full recall [proxy validity] and (b) next-window full recall [lead].
Uses the P0-none trajectory by default (pure drift, no repair confound). Pure numpy (no scipy).
Usage: python analyze_h3.py results/trajectory_sift_SIFTv4finegrain.csv [policy-substr]
"""
from __future__ import annotations
import sys, csv
from collections import defaultdict
import numpy as np


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "P0-none"
    series = defaultdict(list)  # (f,policy) -> [(op, probe, recall)]
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if want not in r["policy"]:
                continue
            series[(r["f"], r["policy"])].append((int(r["op"]), float(r["probe"]), float(r["recall"])))
    print(f"H3 signal validity/lead ({want}); Spearman probe↔recall (concurrent) and probe[t]↔recall[t+1] (lead):")
    for (f, pol), rows in sorted(series.items()):
        rows.sort()
        probe = [x[1] for x in rows]; rec = [x[2] for x in rows]
        s_now = spearman(probe, rec)
        s_lead = spearman(probe[:-1], rec[1:])
        print(f"  [f={f}] {pol:14s} n={len(rows):3d}  concurrent ρ={s_now:+.3f}   lead ρ={s_lead:+.3f}")


if __name__ == "__main__":
    main()
