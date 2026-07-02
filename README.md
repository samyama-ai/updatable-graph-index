# updatable-graph-index — when to repair a graph ANN index under churn

**Honest one-line claim:** triggering *local* graph-edge repair on a measured **navigability-degradation
signal** (instead of a fixed clock) protects **worst-case (tail) recall** of a graph ANN index under
bursty insert/delete churn — **at matched repair budget** — with the gain largest when budget is scarce and
the index is fragile. The *mean*-recall gain is modest; this is **not** a new index and **not** a mean-recall
claim.

Daily-research problem #13 · catalog id [`28-vector-similarity-search/updatable-graph-index`](https://github.com/samyama-ai/dbms_research).
This repo is a **reproducible baseline + a mechanism**, not a production system.

**Preprint:** *When to Repair a Graph ANN Index: Navigability-Signal-Triggered Local Repair Protects
Tail Recall Under Bursty Churn* — [arXiv:2607.00728](https://arxiv.org/abs/2607.00728) (cs.IR; cross-list cs.DB).

## Result (real SIFT-128, bursty churn, matched budget, 4 stream seeds)

| repair budget (# consolidations) | mean-recall win (P2−P1) | **tail (min)-recall win (P2−P1)** |
|---|---|---|
| ~1 | +0.0056 ± 0.0006 | **+0.0136 ± 0.0093** (4/4 seeds) |
| ~2 | +0.0045 ± 0.0010 | **+0.0112 ± 0.0088** (4/4 seeds) |

P2 = signal-triggered local repair; P1 = fixed-cadence (FreshDiskANN-style), compared **at equal
consolidation count**. Tail win tracks a drift-severity gradient (bigger for sparser graphs) and fades to
parity when the index is robust or budget is ample. The cheap probe signal leads true recall at ρ≈0.95.
Figures in `results/figures/`.

## Repro (one command; real data, no mocks)
Compute runs in an amd64 container (diskannpy manylinux wheel) — on Apple Silicon use Docker emulation.
```
./run.sh            # sync → build image → run SIFT sweep on the configured host → figures
# or directly:
python src/experiment.py --dataset sift --sift-path <sift-128-euclidean.hdf5> \
   --n-pool 200000 --warmup 20000 --steady 200000 --eval-every 2000 --R 16 --burst-block 20000
python src/analyze.py results/summary_sift_*.csv     # matched-budget H1
python src/analyze_ci.py results/summary_sift_*seed*.csv   # multi-seed CIs
python src/analyze_h3.py results/trajectory_sift_*.csv     # signal-lead indicator
python src/make_figures.py
```

## Layout
`src/` harness (index wrapper + repair-budget instrumentation, navigability signal, policies, driver,
analyzers, figures) · `bench/make_runbook.py` churn-stream generator · `tests/` recall-oracle + smoke ·
`results/` committed CSVs + figures · `paper/` preprint content.

## Honesty
Mean-recall win is small and **below our pre-registered ≥2-point bar** (set before we saw this high-recall
regime); the >2-pt effect is **tail-only**. Advantage is regime-bound. A delete-fraction sweep was found
invalid under the burst generator and **discarded, not reported** (see `paper/paper-draft.md` §Limitations).
Theory + data-drift coupling are future work. License: Apache-2.0 (code); catalog content CC BY 4.0.
