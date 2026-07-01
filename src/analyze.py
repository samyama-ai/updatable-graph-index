"""Matched-budget H1 analysis (HYPOTHESIS.md H1). Reads a summary CSV and, for each signal-triggered
(P2) operating point, interpolates the fixed-cadence (P1) recall at the SAME consolidation count and
reports the delta. P2 wins at matched budget where delta > 0. Reports avg- and min-recall deltas.
Apparatus-independent (pure stdlib+numpy); run natively.
Usage: python analyze.py results/summary_sift_SIFTv2.csv
"""
from __future__ import annotations
import sys, csv
import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_consol"] = int(r["n_consol"])
            for c in ("avg_recall", "min_recall", "final_recall", "total_wall_s"):
                r[c] = float(r[c])
            rows.append(r)
    return rows


def analyze_f(rows, f):
    rf = [r for r in rows if abs(float(r["f"]) - f) < 1e-9]
    p0 = [r for r in rf if r["policy"].startswith("P0")]
    p1 = sorted([r for r in rf if r["policy"].startswith("P1")], key=lambda r: r["n_consol"])
    p2 = sorted([r for r in rf if r["policy"].startswith("P2")], key=lambda r: r["n_consol"])
    if not p1 or not p2:
        print(f"[f={f}] need both P1 and P2 points"); return
    # P1 curve incl the P0 (0-consolidation) anchor
    base = p0[0] if p0 else None
    xs = ([0] if base else []) + [r["n_consol"] for r in p1]
    avg = ([base["avg_recall"]] if base else []) + [r["avg_recall"] for r in p1]
    mn = ([base["min_recall"]] if base else []) + [r["min_recall"] for r in p1]
    xs, avg, mn = map(np.array, (xs, avg, mn))
    o = np.argsort(xs); xs, avg, mn = xs[o], avg[o], mn[o]

    print(f"\n[f={f}] P1 fixed-cadence curve (consol -> avg/min recall):")
    for x, a, m in zip(xs, avg, mn):
        print(f"    {int(x):3d} consol   avg={a:.4f}  min={m:.4f}")
    print(f"[f={f}] P2 signal-triggered vs P1 at MATCHED budget:")
    wins_avg = wins_min = 0
    for r in p2:
        c = r["n_consol"]
        p1a = float(np.interp(c, xs, avg)); p1m = float(np.interp(c, xs, mn))
        da, dm = r["avg_recall"] - p1a, r["min_recall"] - p1m
        wins_avg += da > 0; wins_min += dm > 0
        print(f"    {r['policy']:16s} consol={c:3d}  avg={r['avg_recall']:.4f} (P1@{c}={p1a:.4f}, Δ={da:+.4f})"
              f"   min={r['min_recall']:.4f} (Δ={dm:+.4f})")
    n = len(p2)
    print(f"[f={f}] H1 (P2>P1 at matched budget): avg-recall {wins_avg}/{n} points, min-recall {wins_min}/{n} points")


def main():
    rows = load(sys.argv[1])
    for f in sorted(set(float(r["f"]) for r in rows)):
        analyze_f(rows, f)


if __name__ == "__main__":
    main()
