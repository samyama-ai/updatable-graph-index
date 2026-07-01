# Reproducibility

## Environment
- **Index:** `diskannpy` (DiskANN Python bindings) `DynamicMemoryIndex`, in-memory Vamana, L2.
- **Determinism:** built with `num_threads=1` (single-thread build → deterministic graph). Fixed churn-
  stream seeds. Search is read-only (thread count doesn't affect recall).
- **Runtime:** amd64 `python:3.11-slim` container (`Dockerfile` → image `uidx:latest`). On Apple Silicon
  (arm64) this runs under Docker's `linux/amd64` emulation (diskannpy ships x86-64 manylinux wheels). Native
  x86-64 Linux runs without emulation.
- **Deps:** `diskannpy numpy h5py` (+ `matplotlib` for figures). Python 3.11.
- **Hardware used:** Apple M4 (Mac mini), emulated amd64, single-thread per run.

## Data (real; downloaded, not vendored)
- SIFT-128-euclidean and Fashion-MNIST-784-euclidean HDF5 from ann-benchmarks (`train`=base, `test`=queries).
  `run.sh` downloads them on first use. No synthetic data in the reported results (a synthetic clustered-
  gaussian config exists only as the low-churn negative-control fixture).

## Definitions (frozen; see the pre-registered HYPOTHESIS)
- **recall@10** vs an exact brute-force kNN oracle **recomputed on the current live set after each eval
  window** — never against frozen initial ground truth (which would manufacture drift as points are deleted).
- **repair budget** = number of consolidation passes (matched across policies; per-pass volume ≈ total
  deletes so equal count ≈ equal spend — verified via logged consolidation wall-time and tombstones cleared).
- **matched-budget comparison** = interpolate P1's recall-vs-consolidation-count curve at P2's count.

## One command
```
./run.sh                         # end-to-end on the host in $UIDX_HOST (default: mini)
```
Regenerates `results/summary_*.csv`, `results/trajectory_*.csv`, and `results/figures/*.png`.
Every number in `paper/` and `README.md` comes from these.

## Negative controls (must pass)
static/no-churn → ~0 drift and P2 never fires; insert-only drifts less than delete-heavy; budget-parity
(a P2 that wins only by spending more repair is disqualified); recall oracle matches a freshly-built static
index and flags the single-delete orphaning fixture. See `tests/`.
