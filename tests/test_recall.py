"""Correctness tests for the recall oracle (apparatus-independent; runs natively).
Layer 2 of the pre-registered five test layers (HYPOTHESIS.md §4)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recall import exact_knn, recall_at_k  # noqa: E402


def test_exact_knn_matches_naive():
    rng = np.random.default_rng(0)
    V = rng.random((300, 8), dtype=np.float32)
    ids = np.arange(300, dtype=np.uint32)
    Q = rng.random((20, 8), dtype=np.float32)
    got = exact_knn(V, ids, Q, k=5)
    # naive reference
    for qi, q in enumerate(Q):
        d2 = ((V - q) ** 2).sum(1)
        ref = set(int(x) for x in np.argsort(d2)[:5])
        assert set(int(x) for x in got[qi]) == ref, f"mismatch at q{qi}"
    print("PASS exact_knn matches naive")


def test_recall_self_is_one():
    rng = np.random.default_rng(1)
    V = rng.random((200, 8), dtype=np.float32)
    ids = np.arange(200, dtype=np.uint32)
    Q = rng.random((15, 8), dtype=np.float32)
    truth = exact_knn(V, ids, Q, k=10)
    assert abs(recall_at_k(truth, truth) - 1.0) < 1e-9
    print("PASS recall(oracle,oracle)==1.0")


def test_recall_shuffled_is_low():
    rng = np.random.default_rng(2)
    V = rng.random((1000, 8), dtype=np.float32)
    ids = np.arange(1000, dtype=np.uint32)
    Q = rng.random((30, 8), dtype=np.float32)
    truth = exact_knn(V, ids, Q, k=10)
    bad = rng.integers(0, 1000, size=truth.shape).astype(np.uint32)  # random ids
    r = recall_at_k(bad, truth)
    assert r < 0.10, f"random recall too high: {r}"
    print(f"PASS recall(random,oracle)={r:.3f} < 0.10")


def test_live_set_shrinks_k():
    # k clipped to live-set size (deletion case)
    rng = np.random.default_rng(3)
    V = rng.random((4, 8), dtype=np.float32)
    ids = np.arange(4, dtype=np.uint32)
    Q = rng.random((2, 8), dtype=np.float32)
    got = exact_knn(V, ids, Q, k=10)
    assert got.shape[1] == 4
    print("PASS k clipped to live-set size")


if __name__ == "__main__":
    test_exact_knn_matches_naive()
    test_recall_self_is_one()
    test_recall_shuffled_is_low()
    test_live_set_shrinks_k()
    print("ALL_RECALL_TESTS_PASS")
