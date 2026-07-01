"""Exact brute-force kNN oracle + recall@k, computed against the CURRENT LIVE SET.

Pre-registration guardrail (HYPOTHESIS.md §0): recall drift is measured vs an exact oracle
recomputed on the live set after each evaluation window — NEVER against frozen initial ground
truth (that would manufacture drift as points are deleted). This module is apparatus-independent
(no diskannpy dependency) so it can be unit-tested anywhere, incl. native macOS.
"""
from __future__ import annotations
import numpy as np


def exact_knn(live_vectors: np.ndarray, live_ids: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Return, for each query, the ids of its true k nearest neighbours among the LIVE set (L2).

    live_vectors: (M, d) float32 of currently-present vectors
    live_ids:     (M,)  ids aligned with live_vectors
    queries:      (Q, d) float32
    -> (Q, k) array of ids (true neighbours), k clipped to M.
    """
    M = live_vectors.shape[0]
    kk = min(k, M)
    # squared L2 via (a-b)^2 = a^2 - 2ab + b^2; chunk queries to bound memory.
    out = np.empty((queries.shape[0], kk), dtype=live_ids.dtype)
    v_sq = np.einsum("ij,ij->i", live_vectors, live_vectors)  # (M,)
    CH = 1024
    for s in range(0, queries.shape[0], CH):
        q = queries[s : s + CH]
        d2 = v_sq[None, :] - 2.0 * (q @ live_vectors.T) + np.einsum("ij,ij->i", q, q)[:, None]
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        # order the kk by true distance so ties/order are deterministic
        row = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(row, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        out[s : s + CH] = live_ids[idx]
    return out


def recall_at_k(approx_ids: np.ndarray, true_ids: np.ndarray) -> float:
    """Mean over queries of |approx ∩ true| / k. Both (Q, k) arrays of ids."""
    assert approx_ids.shape[0] == true_ids.shape[0]
    hits = 0
    total = 0
    for a, t in zip(approx_ids, true_ids):
        tset = set(int(x) for x in t)
        k = len(tset)
        if k == 0:
            continue
        hits += sum(1 for x in a[:k] if int(x) in tset)
        total += k
    return hits / total if total else float("nan")
