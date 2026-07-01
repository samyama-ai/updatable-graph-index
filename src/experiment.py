"""Driver: replay one churn stream through each repair policy on a FRESH index (fair, same stream
+ seed), record the recall trajectory and repair-budget accounting. Emits per-window trajectory CSV
+ a per-(f,policy) summary CSV into results/. Runs inside the amd64 container on mini.

Pre-registered comparison (HYPOTHESIS.md H1): compare P1 (fixed-cadence) vs P2 (signal-triggered)
at MATCHED consolidation-pass count. NC controls: P0 floor; static/insert-only via f knob.
Usage: python experiment.py --dataset synth|sift [--sift-path ...] [--tag PRELIM]
"""
from __future__ import annotations
import os, sys, csv, argparse, json
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
from index_wrap import VamanaIndex
from nav_signal import probe_recall as recall_on      # generic: recall@k on any query set
from policies import NoRepair, FixedCadence, SignalTriggered
from make_runbook import make_stream


def make_synth(n_pool, d, seed=0, n_clusters=50):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, (n_clusters, d)).astype(np.float32)
    lbl = rng.integers(0, n_clusters, n_pool)
    base = (centers[lbl] + rng.normal(0, 1.0, (n_pool, d))).astype(np.float32)
    # held-out queries near clusters (disjoint from base)
    def q(nq):
        l = rng.integers(0, n_clusters, nq)
        return (centers[l] + rng.normal(0, 1.0, (nq, d))).astype(np.float32)
    return base, q(1000), q(200)


def load_sift(path):
    """ann-benchmarks SIFT-128-euclidean HDF5: 'train' (1M x 128), 'test' (10k x 128)."""
    import h5py
    with h5py.File(path, "r") as f:
        base = np.asarray(f["train"][:], dtype=np.float32)
        test = np.asarray(f["test"][:], dtype=np.float32)
    return base, test[:1000], test[1000:1200]


def run_policy(policy, base, eval_q, probe_q, stream, k, dim, R, L, alpha, max_vectors):
    idx = VamanaIndex(dim, max_vectors, R=R, L=L, alpha=alpha)
    ops_since = 0
    baseline = None
    traj = []
    op_index = 0
    for op in stream:
        if op.kind == "insert":
            idx.insert(base[op.id], op.id); ops_since += 1; op_index += 1
        elif op.kind == "delete":
            idx.delete(op.id); ops_since += 1; op_index += 1
        else:  # eval window
            probe = recall_on(idx, probe_q, base, k)
            rec = recall_on(idx, eval_q, base, k)
            if baseline is None:
                baseline = probe
            fire = policy.decide(op_index=op_index, ops_since_consol=ops_since,
                                 probe_signal=probe, baseline_signal=baseline)
            if fire:
                idx.consolidate(op_index)
                ops_since = 0
                baseline = recall_on(idx, probe_q, base, k)  # post-repair healthy level
            traj.append(dict(op=op_index, live=len(idx.live), probe=round(probe, 4),
                             recall=round(rec, 4), consolidated=int(fire)))
    return traj, idx.repair_summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synth", choices=["synth", "sift"])
    ap.add_argument("--sift-path", default="/data/sift")
    ap.add_argument("--tag", default="PRELIM")
    ap.add_argument("--n-pool", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--steady", type=int, default=120000)
    ap.add_argument("--eval-every", type=int, default=6000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--f-list", default="0.3,0.5")
    ap.add_argument("--burst-block", type=int, default=0,
                    help="0=uniform churn; >0 alternates delete-bursts/insert-calm of this block size")
    ap.add_argument("--seed", type=int, default=7, help="churn-stream seed (bootstrap over streams)")
    args = ap.parse_args()

    if args.dataset == "synth":
        base, eval_q, probe_q = make_synth(args.n_pool, d=32)
    else:
        base, eval_q, probe_q = load_sift(args.sift_path)
        if args.n_pool and args.n_pool < base.shape[0]:
            base = base[: args.n_pool]          # cap pool to control turnover ratio
        else:
            args.n_pool = base.shape[0]
    dim = base.shape[1]
    max_vectors = args.n_pool + 100
    f_list = [float(x) for x in args.f_list.split(",")]

    outdir = os.path.join(HERE, "..", "results")
    os.makedirs(outdir, exist_ok=True)
    traj_path = os.path.join(outdir, f"trajectory_{args.dataset}_{args.tag}.csv")
    summ_path = os.path.join(outdir, f"summary_{args.dataset}_{args.tag}.csv")
    tf = open(traj_path, "w", newline=""); tw = csv.writer(tf)
    tw.writerow(["dataset", "f", "policy", "op", "live", "probe", "recall", "consolidated"])
    sf = open(summ_path, "w", newline=""); sw = csv.writer(sf)
    sw.writerow(["dataset", "f", "policy", "n_consol", "total_wall_s", "total_cleared",
                 "avg_recall", "min_recall", "final_recall"])

    for f in f_list:
        stream = make_stream(n_pool=args.n_pool, n_warmup=args.warmup, n_steady=args.steady,
                             delete_fraction=f, eval_every=args.eval_every, seed=args.seed,
                             burst_block=args.burst_block)
        # budget grid: fixed-cadence pass-counts B (via cadence) and signal drop-thresholds,
        # chosen so P1 and P2 fire across an OVERLAPPING #consolidations range (matched-budget H1).
        policies = [NoRepair(),
                    FixedCadence(args.steady // 4), FixedCadence(args.steady // 10),
                    FixedCadence(args.steady // 20), FixedCadence(args.steady // 40),
                    SignalTriggered(0.004), SignalTriggered(0.008),
                    SignalTriggered(0.015), SignalTriggered(0.03)]
        for pol in policies:
            traj, rep = run_policy(pol, base, eval_q, probe_q, stream, args.k, dim,
                                   args.R, args.L, args.alpha, max_vectors)
            for row in traj:
                tw.writerow([args.dataset, f, pol.name, row["op"], row["live"],
                             row["probe"], row["recall"], row["consolidated"]])
            recs = [r["recall"] for r in traj]
            sw.writerow([args.dataset, f, pol.name, rep["n_consolidations"],
                         round(rep["total_wall"], 4), rep["total_cleared"],
                         round(float(np.mean(recs)), 4), round(float(np.min(recs)), 4),
                         round(recs[-1], 4)])
            sf.flush(); tf.flush()
            print(f"[f={f}] {pol.name:16s} consol={rep['n_consolidations']:2d} "
                  f"avg_recall={np.mean(recs):.4f} min={np.min(recs):.4f} "
                  f"wall={rep['total_wall']:.2f}s", flush=True)
    tf.close(); sf.close()
    print("WROTE", traj_path, summ_path)


if __name__ == "__main__":
    main()
