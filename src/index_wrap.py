"""Thin wrapper over diskannpy.DynamicMemoryIndex (in-memory Vamana) with repair-budget
instrumentation. Runs inside the amd64 container on mini (needs diskannpy).

We track the live id-set ourselves and ALWAYS filter returned neighbours against it, so recall is
measured over live points regardless of diskannpy's internal lazy-delete result semantics.
Repair budget accounting: every consolidate() records (wall_seconds, tombstones_cleared) so the
pre-registered budget-parity check (NC3) can verify policies are matched on real repair cost.
"""
from __future__ import annotations
import time
import numpy as np
import diskannpy as dap


class VamanaIndex:
    def __init__(self, dim, max_vectors, R=32, L=64, alpha=1.2):
        self.dim = dim
        self.search_L = L
        self.idx = dap.DynamicMemoryIndex(
            distance_metric="l2", vector_dtype=np.float32, dimensions=dim,
            max_vectors=max_vectors, complexity=L, graph_degree=R, alpha=alpha,
            num_threads=1,   # deterministic single-thread build (reproducibility gate)
        )
        self.live: set[int] = set()
        self._pending_deletes = 0      # tombstoned since last consolidate
        self.consolidations: list[dict] = []  # [{op, wall, cleared}]

    # diskannpy tags must be POSITIVE uint32, so we store tag = id+1 at the engine boundary and
    # map back on search; base indexing, self.live, and the oracle all stay 0-based.
    def insert(self, vec: np.ndarray, i: int):
        self.idx.insert(np.asarray(vec, dtype=np.float32), np.uint32(int(i) + 1))
        self.live.add(int(i))

    def delete(self, i: int):
        self.idx.mark_deleted(np.uint32(int(i) + 1))
        self.live.discard(int(i))
        self._pending_deletes += 1

    def consolidate(self, op_index: int):
        t0 = time.perf_counter()
        self.idx.consolidate_delete()
        wall = time.perf_counter() - t0
        rec = {"op": op_index, "wall": wall, "cleared": self._pending_deletes}
        self.consolidations.append(rec)
        self._pending_deletes = 0
        return rec

    def search(self, queries: np.ndarray, k: int):
        """Return (Q, k) live neighbour ids (deleted tags filtered out). Over-fetch then filter."""
        q = np.asarray(queries, dtype=np.float32)
        over = min(len(self.live), k * 4 + 10)  # over-fetch to survive filtering
        try:
            ids, _ = self.idx.batch_search(q, k_neighbors=over, complexity=max(self.search_L, over + 10),
                                           num_threads=0)
        except Exception:
            ids = np.stack([self.idx.search(qq, k_neighbors=over,
                                            complexity=max(self.search_L, over + 10))[0] for qq in q])
        out = np.full((q.shape[0], k), -1, dtype=np.int64)
        for r in range(q.shape[0]):
            # engine tags are 1-based; map back to 0-based ids and keep only live ones
            live_hits = [int(t) - 1 for t in ids[r] if (int(t) - 1) in self.live][:k]
            out[r, : len(live_hits)] = live_hits
        return out

    # budget summary for parity checks
    def repair_summary(self):
        return {
            "n_consolidations": len(self.consolidations),
            "total_wall": float(sum(c["wall"] for c in self.consolidations)),
            "total_cleared": int(sum(c["cleared"] for c in self.consolidations)),
        }
