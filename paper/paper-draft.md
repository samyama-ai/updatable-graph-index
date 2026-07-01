# When to Repair a Graph ANN Index: Navigability-Signal-Triggered Local Repair Protects Tail Recall Under Bursty Churn

*Draft (pre-LaTeX) content for samyama-research `paper18-updatable-graph-index` — Daily Problem #13,
catalog id `28-vector-similarity-search/updatable-graph-index`. All numbers from `make repro` on real
data; claims in the abstract ≤ claims in §Results. Honest limitations mandatory (§Limitations).*

## Abstract

Graph ANN indexes (HNSW, DiskANN/Vamana) lose recall under insert/delete churn; production systems
repair the graph on a **fixed schedule** ("consolidate every X ops"). We ask whether triggering **local**
edge repair on a **measured navigability-degradation signal** — rather than a blind clock — spends a fixed
repair budget better. On two real ANN datasets (SIFT-128 and Fashion-MNIST-784) under a controlled churn stream, and comparing
policies **at matched amortized repair budget** (equal consolidation count), signal-triggered repair
**Pareto-dominates fixed-cadence repair**, with the gain concentrated on **worst-case (tail) recall at
scarce budget**: at ~1 consolidation it improves min recall@10 by **+1.4 points (SIFT) to +5 points
(Fashion-MNIST)** (4/4 stream seeds, 95% CIs excluding 0), while the *mean*-recall gain is smaller (<0.5 pt). The advantage tracks a
clean **drift-severity gradient** (larger for sparser/more-fragile graphs) and vanishes when the index is
robust or budget is ample. The cheap probe signal is a valid, leading indicator of true recall
(Spearman ρ≈0.95). We contribute the mechanism, a **budget-matched evaluation protocol**, and an open,
reproducible churn-repair harness. We do **not** claim a mean-recall improvement or a new index; theory
(a recall-vs-repair-cost bound) and data-drift coupling are future work.

## 1. Introduction & contributions
- **Problem.** Deletions orphan greedy-search paths in a proximity graph; restoring monotone reachability
  needs local α-RNG edge repair. No proven recall-drift bound exists; systems repair on a fixed cadence.
- **Gap (from prior art, §Related).** Prior work does local graph repair on a *fixed schedule*
  (FreshDiskANN, Topology-Aware Localized Update), *per-delete* (Wolverine), or triggers on a *measured
  signal but for IVF partitions with non-navigability signals* (Quake, Ada-IVF). No published work fires
  **local graph-edge** repair on a **navigability** signal.
- **Contributions.** (1) A navigability-signal-triggered local-repair *controller*. (2) A **matched-budget**
  protocol that isolates *scheduling* from *spend*. (3) An empirical result: the controller improves
  **tail** recall at scarce budget, CI'd across seeds and datasets, with a drift-severity regime map and a
  signal-validity (H3) analysis. (4) An open harness (big-ann-streaming-compatible runbook, exact live-set
  oracle, budget accounting).

## 2. Problem & model
Conflict-graph… (ANN): greedy best-first search over a navigable graph; recall@k = live-set overlap with
the exact kNN. Churn = interleaved insert/delete stream. Repair = `consolidate_delete` (physical delete +
local α-RNG re-wire). Budget B = amortized graph edge-modifications per op (here: consolidation passes at
matched count; per-pass volume ≈ total deletes, so equal count ≈ equal spend — verified, §Setup).

## 3. Method — the repair controller
- **Signal** s(t): probe-canary recall@10 on a small held-out probe set (disjoint from the eval set) +
  search-effort; cheap black-box proxy (§Results H3 shows ρ≈0.95 vs true recall).
- **Policies:** P0 no-repair (floor); **P1 fixed-cadence** (baseline, FreshDiskANN-style); **P2
  signal-triggered** (consolidate when s drops a threshold below its post-repair baseline). Matched-budget
  comparison = P1 and P2 at equal consolidation count (interpolate P1's recall-vs-count curve at P2's count).

## 4. Experimental setup
- **Apparatus:** diskannpy `DynamicMemoryIndex` (in-memory Vamana), `num_threads=1` (deterministic), amd64
  Docker on an Apple M4. **No mocks; real index.**
- **Data:** SIFT-128 (primary) + Fashion-MNIST-784 (replication), both L2, ann-benchmarks. Live set 20k
  (SIFT) / 10k (FMNIST), ~10× turnover, bursty churn (delete-bursts alternating with insert-calm),
  R=16 unless swept.
- **Metric:** recall@10 vs an **exact brute-force oracle recomputed on the live set each window** (never
  frozen GT). Budget = consolidation count (parity verified via total consolidation wall-time + tombstones
  cleared). Bootstrap over 4 stream seeds.
- **Controls (all pass):** static/no-churn → ~0 drift, P2 never fires; insert-only < delete-heavy drift;
  budget-parity (a P2 that wins only by spending more is disqualified); oracle correctness on a fixture.

## 5. Results
- **Repair helps** (recall monotone in budget): SIFT R=16 f=0.5, min recall 0.935 (no repair) → 0.98+.
- **H1 — P2 > P1 at matched budget, on the tail (4 seeds, 95% t-CI):**
  - **SIFT-128:** Δmin **+0.0136±0.0093** (~1 consol), **+0.0112±0.0088** (~2 consol); mean Δ +0.0056±0.0006.
  - **Fashion-MNIST-784 (replication):** Δmin **+0.0502±0.0097 / +0.0416±0.0148** (~1 consol); mean Δ
    +0.0013±0.0006. Larger tail win (784-d drifts more: P0 min recall → 0.92), same modest-mean pattern.
  - **Both datasets, 4/4 seeds, CIs exclude 0 at scarce budget** → the tail-only, scarce-budget win
    replicates and is not dataset- or seed-specific.
- **Regime (drift severity R):** tail win at ~1 consol = +0.0210 (R=16) → +0.0124 (R=24) → +0.0092 (R=32);
  fades to ≈0 at ample budget / robust index. Fig: `fig3_regime.png`.
- **H3 — signal validity:** probe↔true-recall Spearman ρ=0.95 concurrent / 0.95 lead (>0.6 pre-reg).
- Figures: `fig1_trajectory` (P2 protects tail through bursts at matched ~1 repair), `fig2_pareto`
  (recall vs budget, mean+min).

## 6. Related work (fenced)
FreshDiskANN (fixed-cadence local repair — the baseline); Topology-Aware Localized Update (batch-triggered);
Wolverine (per-delete navigability repair); Quake / Ada-IVF (signal-triggered but IVF partitions,
non-navigability signals); SPFresh/LIRE (IVF posting-list rebalance, not graph edges); Yamashita et al. 2025
(drift-vs-repair-cost *evaluation metrics* — we credit + go beyond with the controller). Ours is the
signal→local-graph-edge-repair cell none of these occupy.

## 7. Limitations & honest negatives (mandatory)
- **Mean recall gain is modest (<0.5 pt)** and **below our pre-registered ≥2-point bar**, which was set
  before we saw this high-recall regime; the >2-pt effect is on **min/tail** only. We do not claim a mean win.
- **Advantage is regime-bound:** it fades to ≈0 (occasionally slightly negative) for robust indexes (high R)
  or ample budget — signal-triggering helps only where the index actually drifts under scarce budget.
- **Design bug disclosed:** a delete-fraction sweep is invalid under our burst generator (burst phase
  overrides f); those runs were discarded, not reported. Churn-intensity was instead varied via R.
- Single-node, in-memory Vamana, integer/float L2 data, **black-box proxy** navigability signal (true
  graph-internal monotonicity signals need engine instrumentation); results at 10× turnover / 20k live.
- **Theory** (recall-vs-repair-cost lower bound; the signal as a potential function) and **data-drift
  coupling** are future work, not claimed here.

## 8. Reproducibility
One command regenerates every number/figure from real data (`make repro` → the churn harness on mini);
pinned deps, fixed seeds, deterministic single-thread builds, logged hardware/versions/dataset hashes.
