"""Smoke test: does diskannpy DynamicMemoryIndex support the insert/delete/consolidate/search
cycle our churn harness needs? Run inside a linux container on mini. Prints PASS/FAIL per step.
No mocks — exercises the real Vamana index."""
import sys
import numpy as np

try:
    import diskannpy as dap
except Exception as e:  # noqa
    print("IMPORT_FAIL:", repr(e)); sys.exit(2)

print("diskannpy:", getattr(dap, "__version__", "?"))
D, N = 16, 200
rng = np.random.default_rng(0)
X = rng.random((N, D), dtype=np.float32)

# Construct a dynamic in-memory Vamana index. API names vary slightly by version;
# probe the constructor signature defensively.
def build():
    kw = dict(distance_metric="l2", vector_dtype=np.float32, dimensions=D,
              max_vectors=N + 50, complexity=64, graph_degree=32)
    try:
        return dap.DynamicMemoryIndex(**kw)
    except TypeError as e:
        print("CTOR_TYPEERROR:", e)
        # older/newer arg names
        kw2 = dict(metric="l2", vector_dtype=np.float32, dim=D,
                   max_points=N + 50, complexity=64, graph_degree=32)
        return dap.DynamicMemoryIndex(**kw2)

try:
    idx = build()
    print("PASS ctor")
except Exception as e:
    print("FAIL ctor:", repr(e)); sys.exit(1)

ids = np.arange(1, N + 1, dtype=np.uint32)  # tags must be positive
try:
    idx.batch_insert(X, ids)
    print("PASS batch_insert")
except Exception as e:
    try:
        for i in range(N):
            idx.insert(X[i], int(ids[i]))
        print("PASS insert (per-vector)")
    except Exception as e2:
        print("FAIL insert:", repr(e), "|", repr(e2)); sys.exit(1)

try:
    nbr, dist = idx.search(X[0], k_neighbors=5, complexity=64)
    print("PASS search pre-delete:", nbr[:5].tolist())
except Exception as e:
    print("FAIL search:", repr(e)); sys.exit(1)

try:
    for t in range(1, 31):           # delete 30 tags
        idx.mark_deleted(np.uint32(t))
    print("PASS mark_deleted x30")
except Exception as e:
    print("FAIL mark_deleted:", repr(e)); sys.exit(1)

try:
    idx.consolidate_delete()         # THE repair pass our lever tunes
    print("PASS consolidate_delete")
except Exception as e:
    print("FAIL consolidate_delete:", repr(e)); sys.exit(1)

try:
    nbr2, _ = idx.search(X[100], k_neighbors=5, complexity=64)
    assert all(int(t) > 30 for t in nbr2), f"deleted tag returned: {nbr2[:5]}"
    print("PASS search post-consolidate (no deleted tags):", nbr2[:5].tolist())
except Exception as e:
    print("FAIL post-consolidate search:", repr(e)); sys.exit(1)

print("ALL_SMOKE_PASS")
