# Stage-9 post draft (the human presses send)

> Guardrails honored: invitation not conquest; lead with others' contributions; every claim maps to a
> number in the repo; the modest/tail-only nature is stated plainly; ≤ a handful of tags.

---

**When should you repair a vector index — on a clock, or when it actually drifts?**

Graph ANN indexes (HNSW, DiskANN/Vamana) lose recall under insert/delete churn: a deletion orphans the
greedy-search paths that routed through it. The standard fix — pioneered by FreshDiskANN — repairs the
graph locally on a **fixed schedule**. We tried a small change: trigger that same local repair on a
**measured navigability signal** (a cheap probe-set recall) instead of a blind clock, and compare the two
**at matched repair budget** so neither wins just by repairing more.

The honest result, on real SIFT-128 and Fashion-MNIST-784 under bursty churn, 4 seeds:
- Signal-triggering **Pareto-beats fixed-cadence on worst-case (tail) recall** at scarce budget:
  **+0.014 (SIFT) to +0.050 (Fashion-MNIST) min-recall**, CIs excluding zero.
- The **mean-recall gain is small (<0.005)** — we do not claim a mean win. The effect is a *tail* effect,
  largest exactly where a degradation-triggered controller should help: fragile index, scarce budget,
  bursty drift. It fades to parity when the index is robust or budget is ample.
- The cheap signal leads true recall at ρ≈0.95.

It's a mechanism + a budget-matched protocol + an open one-command harness — not a new index, and not a
mean-recall claim. Code, figures, and reproduction: [github.com/samyama-ai/updatable-graph-index]. Preprint
inside.

We'd genuinely value pushback: is the matched-budget framing the right way to separate *when* from *how
much*? And where would a navigability-triggered controller matter most in a deployed engine?

---

## Tag list (rationale; human resolves handles, tags ≤ a handful)
- **Harsha Vardhan Simhadri / Ravishankar Krishnaswamy** (DiskANN, FreshDiskANN) — the local-repair
  baseline we build on and compare against; lead with their contribution.
- **Christian S. Jensen et al.** (Wolverine, PVLDB'25) — closest navigability-aware graph repair
  (per-delete); we position relative to it.
- **Daichi Amagata / Yusuke Matsui** (deletion-evaluation metrics, 2025) — whose measurement framing we
  adopt and extend with the controller.
- (Optional) **SPFresh / Quake teams** — signal/cost-triggered maintenance on the IVF side; we distinguish
  index family and signal type.

Framing note: invitation, not a "we beat X." The win is modest and tail-only; say so — that honesty is
why practitioners in this area reply.
