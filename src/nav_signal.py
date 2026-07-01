"""Navigability signal (black-box, day-1): probe-canary recall@k on a SMALL, held-out probe query
set, disjoint from the evaluation query set. Cheap monitor the triggered policy reacts to; the
reported metric uses the larger eval set. (BRIEF.md §3; HYPOTHESIS.md §0.)

Named nav_signal (not `signal`) to avoid shadowing Python's stdlib `signal` module, since src/ is
on sys.path for the whole process.
"""
from __future__ import annotations
import numpy as np
from recall import exact_knn, recall_at_k


def probe_recall(index, queries: np.ndarray, base: np.ndarray, k: int) -> float:
    """recall@k of the live index on `queries` vs the exact oracle over the current live set.
    Generic over any query set (used for both the small probe signal and the full eval metric)."""
    if len(index.live) < k:
        return 1.0
    live_ids = np.fromiter(index.live, dtype=np.int64)
    live_vecs = base[live_ids]
    truth = exact_knn(live_vecs, live_ids, queries, k)
    approx = index.search(queries, k)
    return recall_at_k(approx, truth)
