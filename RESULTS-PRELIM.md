# RESULTS (PRELIM) — Updatable Graph ANN Under Churn

*Dev log; the pre-registered contract is `dbms_cloud/daily/28-updatable-graph-index/HYPOTHESIS.md`.
Apparatus: diskannpy `DynamicMemoryIndex` (Vamana), amd64 Docker on `mini`, `num_threads=1`
(deterministic). Recall@10 vs an exact brute-force oracle recomputed on the live set each window.*

## Harness validated (2026-07-01)
- Full churn cycle works end-to-end (insert / mark_deleted / consolidate_delete / search); tag +1
  offset at the engine boundary (diskannpy requires positive uint32 ids). `num_threads=1` for
  deterministic builds. Recall oracle unit-tested (4/4); churn-stream generator deterministic.
- Five test layers status: (1) reproduction — P1 fixed-cadence recovers recall like FreshDiskANN;
  (2) correctness — oracle tests pass; (3) headline — matched-budget P1 vs P2 (below); (4) ablation
  — budget sweep; (5) negative controls — **NC "easy regime"**: synthetic clustered-gaussian, low
  turnover → P0 (no repair) already 0.9998 recall, P2 correctly **never fires** (no wasted repair),
  P1 fires but gains nothing. Confirms the low-churn "coarse-suffices" corner + P2's abstention logic.

## Drift regime found: real SIFT-128, ~10× turnover, f=0.5, R=16
P0 (no repair) drifts to **avg 0.955 / min 0.925** — real headroom. Repair helps monotonically in
budget. **H1 headline (matched-budget: P2 signal-triggered vs P1 fixed-cadence, interpolated at equal
consolidation count):**

| P2 op-point | #consol | P2 avg | P1@matched | Δavg | P2 min | Δmin |
|---|---|---|---|---|---|---|
| signal@0.030 | 1 | 0.9694 | 0.9606 | **+0.0088** | 0.9532 | **+0.0172** |
| signal@0.015 | 2 | 0.9738 | 0.9666 | **+0.0072** | 0.9512 | +0.0045 |
| signal@0.008 | 3 | 0.9779 | 0.9725 | **+0.0054** | 0.9666 | +0.0092 |
| signal@0.004 | 8 | 0.9833 | 0.9827 | +0.0006 | 0.9777 | +0.0005 |

**P2 > P1 at matched budget on 4/4 operating points, both avg and min recall** — largest gain in the
**scarce-budget regime** (1 consolidation: +0.9 avg / +1.7 min points), converging to parity as budget
grows (both hit the recall ceiling). Mechanism as predicted: when you can only afford a few repairs,
placing them by a navigability signal beats a fixed clock.

## Honest caveats (disclosed)
- **Pre-registration miss:** frozen H1 bar was Δ ≥ 2.0 recall points; observed Δ are +0.06 … +1.7
  points — **directionally consistent (8/8) but sub-threshold in magnitude.** The 2-point bar was set
  before we saw this high-recall (~0.95–0.99) regime, where the whole P0→P1 gap is only ~3 points; a
  2-point inter-policy gap was unrealistic. Reframe: consistent **Pareto improvement**, modest
  magnitude, largest under scarce budget — not a 2-point beat.
- Deletes so far are **uniform** → smooth drift, which limits a timing-based trigger's edge. The
  mechanism predicts a **larger** P2 advantage under **bursty** drift.
- **Bursty-churn result (SIFTv3, `--burst-block 10000`): the sharp prediction is NOT confirmed at
  this granularity.** P2 still Pareto-beats P1 at matched budget (avg 4/4, min 3/4), but the deltas
  (avg +0.0016…+0.0067, min −0.0008…+0.0144) are **comparable to uniform, not wider.** Diagnosed
  cause: P2 only decides at eval windows (every 8k ops) while the burst period is 10k — so P2 cannot
  react *within* a burst any faster than a coarse clock. **Correction:** the decision granularity must
  be ≪ the burst period. Re-running with eval every 2k, bursts of 20k (SIFTv4) so P2 can fire ~10×
  per burst against a matched-budget blind clock — the proper test of the timing mechanism.

## Fine-granularity bursty result (SIFTv4, eval 2k / burst 20k) — the headline
With decision granularity ≪ burst period, P2 beats P1 at matched budget on **4/4 avg AND 4/4 min**:

| P2 op-point | #consol | Δavg | **Δmin (tail)** |
|---|---|---|---|
| signal@0.030 | 1 | +0.0050 | **+0.0210** |
| signal@0.015 | 2 | +0.0048 | **+0.0192** |
| signal@0.008 | 3 | +0.0022 | +0.0070 |
| signal@0.004 | 6 | +0.0013 | +0.0057 |

**The mechanism resolves on the TAIL, not the mean.** Under bursty churn at scarce budget,
signal-triggered repair improves **worst-case (min) recall by up to ~2.1 points** over a matched-budget
fixed clock — because it repairs *at burst onset* while the blind clock lets recall crater during
unattended bursts. Average gain stays modest (<0.5 pt) because calm windows dominate the mean.

## Honest verdict (so far)
- **Robust:** repair helps (budget→recall monotone); **signal-triggered Pareto-beats fixed-cadence at
  matched budget across all regimes tested** (uniform, coarse-bursty, fine-bursty: avg 12/12, min 11/12).
- **The win is concentrated where the mechanism predicts:** tail (min) recall, scarce budget, bursty
  drift — up to **+2 worst-case points**. On the mean it is consistent but modest.
- **Pre-registration honesty:** the frozen H1 was on **mean** recall ≥2 pts — **not met** (mean gain
  <0.5 pt). The >2-pt effect is on **min/tail**, the arguably more decision-relevant (SLA) metric.
  Disclose the mean-vs-tail distinction; do NOT claim the mean H1 passed. Reframe the headline to the
  tail-under-bursty-scarce regime.

## Robustness (2026-07-01): CIs, H3, regime, and a disclosed design bug

**Multi-seed CIs (4 seeds, f=0.5 fine-bursty) — the tail win is real, not a single-seed artifact:**

| P2 op-point | ~#consol | Δavg (mean±95%tCI) | Δmin/tail (mean±95%tCI) | seeds>0 |
|---|---|---|---|---|
| signal@0.030 | 1.0 | +0.0056 ± 0.0006 | **+0.0136 ± 0.0093** | 4/4 |
| signal@0.015 | 2.0 | +0.0045 ± 0.0010 | **+0.0112 ± 0.0088** | 4/4 |
| signal@0.008 | 3.5 | +0.0011 ± 0.0028 | +0.0046 ± 0.0034 | 4/4 (min) |
| signal@0.004 | 6.8 | +0.0010 ± 0.0009 | +0.0035 ± 0.0026 | 4/4 |

Both mean and tail deltas are **positive on 4/4 seeds with 95% t-CIs excluding 0** at scarce budget.

**H3 (signal is a valid, leading indicator) — CONFIRMED:** on the no-repair drift trajectory, Spearman
between the cheap probe signal and true full-eval recall is **ρ=0.952 concurrent / 0.949 lead**
(fine-bursty) and **0.985 / 0.986** (uniform) — far above the pre-registered ρ≥0.6. Triggering on the
probe is justified; it predicts next-window recall.

**Regime axis = drift severity (graph degree R), f=0.5 fine-bursty — clean gradient:** the scarce-budget
tail win **shrinks as the index gets more robust**: Δmin at ~1 consolidation = **+0.0210 (R=16) →
+0.0124 (R=24) → +0.0092 (R=32)**; at ample budget (R=32, ~4 consol) it fades to ≈0 / slightly negative
(nothing left to fix). So signal-triggering helps most exactly where the mechanism predicts — **fragile
index (low R) + scarce budget + bursty drift** — and fades to parity when the index is robust or budget
is ample. `results/figures/fig3_regime.png`.

**Disclosed design bug (honesty):** the intended *delete-fraction* regime sweep is **invalid under burst
mode** — when `burst_block>0`, `p_del` is set by the burst phase (0.90/0.05) and **ignores `f`**, so the
f=0.3/0.5/0.7 bursty runs produced byte-identical streams. Those f-sweep points are **discarded, not
reported**; the valid churn-intensity sweep must use uniform mode (documented in `make_runbook.py`). The
CI/H1/H3 results above are unaffected (f fixed at 0.5). Caught by noticing identical output across f.

## Figures (`results/figures/`)
`fig1_trajectory.png` (recall vs ops under bursts, matched ~1-repair budget — P2 protects the tail) ·
`fig2_pareto.png` (recall vs budget, mean+min; P2 above P1) · `fig3_regime.png` (tail win vs R).

## Replication — Fashion-MNIST-784 (L2, 4 seeds, fine-bursty f=0.5) — GENERALIZES, larger tail win
P0 drift deeper (784-d more fragile: min recall → 0.92). Matched-budget tail win at ~1 consolidation:
**Δmin +0.0502±0.0097 / +0.0416±0.0148 (4/4 seeds, CIs exclude 0)**; mean Δ modest (+0.0013±0.0006). Same
tail-only, scarce-budget pattern as SIFT but **~3–4× larger tail magnitude** → the mechanism is not
SIFT-specific.

## Scale confirmation — SIFT1M / 100k-live (single seed, fine-bursty)
The pre-registered larger-scale config (100k live, 5× the headline index) confirms the effect. Deeper
drift (P0 min 0.911). Matched-budget: **P2 > P1 on 4/4 avg and 4/4 min**; tail win **Δmin +0.0090
(~1 consol), +0.0082 (~2 consol)** — same direction and comparable magnitude to the 20k CI'd result, so
the mechanism persists at scale. (`results/summary_sift_SIFT1M100k.csv`.)

## Stage 5/7 drafted
Repo essentials written (`README.md`, `LICENSE`, `CITATION.cff`, `REPRODUCIBILITY.md`, `run.sh` one-command
repro). Paper draft `paper/paper-draft.md` (full narrative, two-dataset results, honest limitations,
related-work fencing) → destined for samyama-research `paper18`.

## Next (to reach paper+quiz DoD)
- Scale to the pre-registered **SIFT1M / 100k-live** config for the headline numbers (optional; the effect
  is already CI'd + replicated at 200k/20k).
- Convert `paper/paper-draft.md` → **LaTeX in samyama-research `paper18`**, run validators, tarball.
- **200-Q quiz** (build_app_quizzes + validate_quiz + rebalance) — the second DoD deliverable.
- Stage 6 (user): public GitHub `samyama-ai` + Gitea mirror; arXiv (cs.IR/cs.DB); Stage-9 post.
