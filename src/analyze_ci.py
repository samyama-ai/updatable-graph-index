"""Multi-seed CI aggregation for the matched-budget H1 tail-win. Takes several summary CSVs (one per
stream seed, same f), computes for each seed the P2-vs-P1 matched-budget delta (interpolating that
seed's OWN P1 curve, so budget-matching is within-seed), then reports mean ± 95% t-CI across seeds
per P2 threshold. Establishes the tail win is not a single-seed artifact.
Usage: python analyze_ci.py summary_a.csv summary_b.csv ...
"""
from __future__ import annotations
import sys, csv
from collections import defaultdict
import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_consol"] = int(r["n_consol"])
            for c in ("avg_recall", "min_recall"):
                r[c] = float(r[c])
            rows.append(r)
    return rows


def seed_deltas(rows):
    """For one seed's rows (single f), return {p2_policy: (d_avg, d_min, consol)}."""
    p0 = [r for r in rows if r["policy"].startswith("P0")]
    p1 = sorted([r for r in rows if r["policy"].startswith("P1")], key=lambda r: r["n_consol"])
    p2 = [r for r in rows if r["policy"].startswith("P2")]
    xs = ([0] if p0 else []) + [r["n_consol"] for r in p1]
    av = ([p0[0]["avg_recall"]] if p0 else []) + [r["avg_recall"] for r in p1]
    mn = ([p0[0]["min_recall"]] if p0 else []) + [r["min_recall"] for r in p1]
    xs, av, mn = map(np.array, (xs, av, mn)); o = np.argsort(xs); xs, av, mn = xs[o], av[o], mn[o]
    out = {}
    for r in p2:
        c = r["n_consol"]
        out[r["policy"]] = (r["avg_recall"] - float(np.interp(c, xs, av)),
                            r["min_recall"] - float(np.interp(c, xs, mn)), c)
    return out


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2:
        return m, float("nan")
    # 95% t (approx t for small n)
    tval = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57}.get(n, 2.45)
    return m, tval * x.std(ddof=1) / np.sqrt(n)


def main():
    per_policy_avg = defaultdict(list); per_policy_min = defaultdict(list); per_policy_c = defaultdict(list)
    for p in sys.argv[1:]:
        for pol, (da, dm, c) in seed_deltas(load(p)).items():
            per_policy_avg[pol].append(da); per_policy_min[pol].append(dm); per_policy_c[pol].append(c)
    print(f"Matched-budget P2−P1 deltas across {len(sys.argv)-1} seeds (mean ± 95% t-CI):")
    for pol in sorted(per_policy_avg, key=lambda k: -np.mean(per_policy_c[k])):
        ma, ha = tci(per_policy_avg[pol]); mm, hm = tci(per_policy_min[pol])
        cbar = np.mean(per_policy_c[pol])
        npos_a = sum(1 for x in per_policy_avg[pol] if x > 0); npos_m = sum(1 for x in per_policy_min[pol] if x > 0)
        n = len(per_policy_avg[pol])
        print(f"  {pol:16s} ~{cbar:4.1f} consol   "
              f"Δavg={ma:+.4f}±{ha:.4f} ({npos_a}/{n}>0)   Δmin={mm:+.4f}±{hm:.4f} ({npos_m}/{n}>0)")


if __name__ == "__main__":
    main()
