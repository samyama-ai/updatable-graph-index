"""Knobbed insert/delete/search churn-stream generator (apparatus-independent).

Produces a deterministic operation schedule over a pool of base-dataset ids. `delete_fraction`
f is the fraction of steady-state ops that are deletes (HYPOTHESIS.md froze f in {0,0.1,0.3,0.5});
the rest are inserts of fresh ids. `eval` markers are emitted every `eval_every` steady ops; at an
eval the harness measures recall@k of the live index vs the exact oracle on the current live set.

A live-size floor keeps deletes from draining the index; an insert pool cap keeps it from exhausting
the base set. f=0.5 is ~constant-size turnover; f<0.5 grows, f>0.5 shrinks (a delete-heavy regime).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Op:
    kind: str          # "insert" | "delete" | "eval"
    id: int = -1       # base-dataset id for insert/delete; -1 for eval


def make_stream(n_pool: int, n_warmup: int, n_steady: int, delete_fraction: float,
                eval_every: int, seed: int = 0, live_floor: int = 1000, burst_block: int = 0):
    """Yield-list of Ops. n_pool = #available base ids (e.g. 1_000_000 for SIFT1M).

    Phase 1 (warm-up): insert n_warmup fresh ids (no evals).
    Phase 2 (steady):  n_steady ops; each is delete w.p. f (if live>floor and live nonempty)
                       else insert (if pool left). An `eval` Op every eval_every steady ops.
    Ids are consumed from a shuffled pool so inserts never collide; deletes pick a uniformly
    random currently-live id. Fully determined by `seed`.
    """
    rng = np.random.default_rng(seed)
    pool = list(rng.permutation(n_pool))
    assert n_warmup + 1 <= n_pool, "pool too small for warm-up"
    live: list[int] = []
    live_set: set[int] = set()
    ops: list[Op] = []

    def do_insert():
        if not pool:
            return False
        i = pool.pop()
        live.append(i)
        live_set.add(i)
        ops.append(Op("insert", i))
        return True

    def do_delete():
        if not live:
            return False
        j = int(rng.integers(len(live)))
        i = live[j]
        live[j] = live[-1]      # O(1) swap-remove
        live.pop()
        live_set.discard(i)
        ops.append(Op("delete", i))
        return True

    for _ in range(n_warmup):
        do_insert()

    for t in range(1, n_steady + 1):
        if burst_block > 0:
            # alternate delete-heavy bursts with insert-heavy calm so drift is BURSTY not smooth.
            # NOTE: in burst mode p_del is fixed by phase (0.90/0.05) and IGNORES delete_fraction —
            # so a delete-fraction sweep MUST use uniform mode (burst_block=0); vary drift severity
            # under burst via R (graph degree) or burst_block instead.
            p_del = 0.90 if (((t - 1) // burst_block) % 2 == 1) else 0.05
        else:
            p_del = delete_fraction
        if rng.random() < p_del and len(live) > live_floor:
            if not do_delete():
                do_insert()
        else:
            if not do_insert():
                do_delete()
        if t % eval_every == 0:
            ops.append(Op("eval"))
    return ops


if __name__ == "__main__":  # quick self-check
    s = make_stream(n_pool=10_000, n_warmup=2_000, n_steady=4_000,
                    delete_fraction=0.5, eval_every=500, seed=1)
    from collections import Counter
    c = Counter(o.kind for o in s)
    print("ops:", dict(c), "total:", len(s))
