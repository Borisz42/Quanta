"""QUANTA Neuro-Symbolic Verification and MUC Repair Package (Phase 4).

Provides:
1. ClingoVerificationGate: Formal semantic verification gate wrapping Clingo ASP
   with translation of Minimal Unsatisfiable Cores (MUCs) into model-actionable
   [REPAIR REQUEST] diagnostics.
2. MUCRepairManager: Closed-loop repair orchestrator enforcing a strict repair
   attempt cap (<= 2 retries) to eliminate structural hallucinations without infinite loops.
3. MUCDiagnostic & MUCDiagnosticResult: Rich diagnostic data models for ontological,
   temporal, causal, and structural contradiction analysis.
4. RepairResult: Result container for closed-loop repair operations.
"""

from __future__ import annotations

from verification.clingo_gate import (
    ClingoVerificationGate,
    MUCDiagnostic,
    MUCDiagnosticResult,
    MUCRepairManager,
    RepairResult,
)
from verification.lattice_gate import (
    ContradictionDetail,
    LatticeInvarianceGate,
    LatticeMeetResult,
)

__all__ = [
    "ClingoVerificationGate",
    "MUCRepairManager",
    "MUCDiagnostic",
    "MUCDiagnosticResult",
    "RepairResult",
    "ContradictionDetail",
    "LatticeInvarianceGate",
    "LatticeMeetResult",
]
