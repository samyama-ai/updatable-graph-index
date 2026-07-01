"""Generate paper figures from the results CSVs into results/figures/. Local (matplotlib).
Robust to missing inputs (skips a figure with a note). Run: python src/make_figures.py
"""
from __future__ import annotations
import os, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)


def read_traj(path):
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["policy"]].append((int(r["op"]), float(r["recall"]), int(r["consolidated"])))
    for k in d:
        d[k].sort()
    return d


def read_summary(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_consol"] = int(r["n_consol"]); r["avg_recall"] = float(r["avg_recall"]); r["min_recall"] = float(r["min_recall"])
            rows.append(r)
    return rows


def fig_trajectory(tag="SIFTv4finegrain"):
    p = os.path.join(RES, f"trajectory_sift_{tag}.csv")
    if not os.path.exists(p):
        print("skip trajectory:", p); return
    d = read_traj(p)
    plt.figure(figsize=(7, 4))
    show = {"P0-none": ("no repair", "0.6"), "P1-fixed@50000": ("fixed-cadence (~1 consol)", "C1"),
            "P2-signal@0.030": ("signal-triggered (~1 consol)", "C2")}
    for pol, (lbl, col) in show.items():
        if pol not in d:
            continue
        ops = [x[0] for x in d[pol]]; rec = [x[1] for x in d[pol]]
        plt.plot(ops, rec, label=lbl, color=col, lw=1.6)
        for o, r, c in d[pol]:
            if c:
                plt.scatter([o], [r], color=col, s=18, zorder=5)
    plt.xlabel("operations"); plt.ylabel("recall@10 (live set)")
    plt.title("Recall under bursty churn at matched budget (~1 repair)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_trajectory.png"), dpi=140); plt.close()
    print("wrote fig1_trajectory.png")


def fig_pareto(tag="SIFTv4finegrain"):
    p = os.path.join(RES, f"summary_sift_{tag}.csv")
    if not os.path.exists(p):
        print("skip pareto:", p); return
    rows = read_summary(p)
    p0 = [r for r in rows if r["policy"].startswith("P0")]
    p1 = sorted([r for r in rows if r["policy"].startswith("P1")], key=lambda r: r["n_consol"])
    p2 = sorted([r for r in rows if r["policy"].startswith("P2")], key=lambda r: r["n_consol"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ttl in ((axs[0], "avg_recall", "mean recall"), (axs[1], "min_recall", "min (tail) recall")):
        xs = ([0] if p0 else []) + [r["n_consol"] for r in p1]
        ys = ([p0[0][key]] if p0 else []) + [r[key] for r in p1]
        ax.plot(xs, ys, "-o", color="C1", label="P1 fixed-cadence")
        ax.plot([r["n_consol"] for r in p2], [r[key] for r in p2], "s", color="C2", ms=8, label="P2 signal-triggered")
        ax.set_xlabel("# consolidations (repair budget)"); ax.set_ylabel(key); ax.set_title(ttl); ax.legend(fontsize=8)
    fig.suptitle("Recall vs repair budget — P2 above P1 at matched budget (esp. tail, scarce budget)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_pareto.png"), dpi=140); plt.close()
    print("wrote fig2_pareto.png")


def fig_regime():
    # tail-delta (Δmin at scarcest P2 point) vs DRIFT SEVERITY (graph degree R): lower R = sparser
    # graph = more fragile under churn = bigger expected win. (f-sweep is invalid under burst mode.)
    pts = []
    for R, tag in [(16, "SIFTv4finegrain"), (24, "bR24s7"), (32, "bR32s7")]:
        p = os.path.join(RES, f"summary_sift_{tag}.csv")
        if not os.path.exists(p):
            continue
        rows = read_summary(p)
        p0 = [r for r in rows if r["policy"].startswith("P0")]
        p1 = sorted([r for r in rows if r["policy"].startswith("P1")], key=lambda r: r["n_consol"])
        p2 = sorted([r for r in rows if r["policy"].startswith("P2")], key=lambda r: r["n_consol"])
        if not (p0 and p1 and p2):
            continue
        xs = np.array([0] + [r["n_consol"] for r in p1]); mn = np.array([p0[0]["min_recall"]] + [r["min_recall"] for r in p1])
        fired = [r for r in p2 if r["n_consol"] >= 1]  # scarcest point where P2 actually fired
        if not fired:
            continue
        r = min(fired, key=lambda r: r["n_consol"])
        pts.append((R, r["min_recall"] - float(np.interp(r["n_consol"], xs, mn))))
    if not pts:
        print("skip regime (no R-sweep yet)"); return
    rs, ds = zip(*sorted(pts))
    plt.figure(figsize=(6, 4)); plt.plot(rs, ds, "-o", color="C2")
    plt.axhline(0, color="0.6", lw=0.8); plt.xlabel("graph degree R (higher = more robust index)")
    plt.ylabel("tail-recall win Δmin (scarce budget)")
    plt.title("Where signal-triggering helps: tail win vs drift severity"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_regime.png"), dpi=140); plt.close()
    print("wrote fig3_regime.png")


if __name__ == "__main__":
    fig_trajectory(); fig_pareto(); fig_regime()
    print("figures in", FIG)
