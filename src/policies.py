"""Repair-scheduling policies. Each decides, at an evaluation window, whether to fire a local
consolidation (the α-RNG repair pass). All policies do the SAME per-pass repair (consolidate_delete);
they differ only in WHEN — the pre-registered matched-budget comparison compares them at equal
consolidation-pass count (HYPOTHESIS.md §0/H1)."""
from __future__ import annotations


class NoRepair:
    """P0 — floor: never consolidate (pure lazy tombstoning)."""
    name = "P0-none"
    def decide(self, *, op_index, ops_since_consol, probe_signal, baseline_signal):
        return False


class FixedCadence:
    """P1 — consolidate every `cadence` ops (FreshDiskANN-style fixed schedule)."""
    def __init__(self, cadence: int):
        self.cadence = cadence
        self.name = f"P1-fixed@{cadence}"
    def decide(self, *, op_index, ops_since_consol, probe_signal, baseline_signal):
        return ops_since_consol >= self.cadence


class SignalTriggered:
    """P2 — consolidate when the navigability signal (probe recall) drops `drop` below its
    post-repair baseline. Triggering on measured degradation, not a fixed clock."""
    def __init__(self, drop: float):
        self.drop = drop
        self.name = f"P2-signal@{drop:.3f}"
    def decide(self, *, op_index, ops_since_consol, probe_signal, baseline_signal):
        if probe_signal is None or baseline_signal is None:
            return False
        return probe_signal <= baseline_signal - self.drop
