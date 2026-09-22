"""Closed-Loop Round-Trip Lattice Meet Gate for QUANTA (Phase 4 / Section 3).

Computes slot-wise lattice meet invariance:
    v_meet = v_orig ⊓ v_reparsed

Verifies semantic cycle-consistency, detects epistemic contradictions (TRUE vs. FALSE),
and audits propositional slot preservation across forward parsing and reverse realization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_index, get_slot_by_name
from core.types import (
    BandContract,
    QuantaVector,
    QuaternaryValue,
    StructuralValue,
    RegisterValue,
)


@dataclass
class ContradictionDetail:
    """Detailed record of an epistemic or structural contradiction between two vectors."""
    slot_index: int
    slot_name: str
    band_id: int
    contract: str
    orig_value: int
    reparsed_value: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "slot_name": self.slot_name,
            "band_id": self.band_id,
            "contract": self.contract,
            "orig_value": self.orig_value,
            "reparsed_value": self.reparsed_value,
            "description": self.description,
        }


@dataclass
class LatticeMeetResult:
    """Comprehensive diagnostic result of a lattice meet invariance check."""
    is_sound: bool
    preservation_rate: float
    hamming_distance: int
    contradiction_count: int
    contradictions: List[ContradictionDetail] = field(default_factory=list)
    dropped_slots: List[Dict[str, Any]] = field(default_factory=list)
    meet_vector: Optional[QuantaVector] = None
    orig_vector: Optional[QuantaVector] = None
    reparsed_vector: Optional[QuantaVector] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sound": self.is_sound,
            "preservation_rate": round(self.preservation_rate, 4),
            "hamming_distance": self.hamming_distance,
            "contradiction_count": self.contradiction_count,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "dropped_slot_count": len(self.dropped_slots),
            "errors": self.errors,
        }


class LatticeInvarianceGate:
    """Enforces closed-loop lattice meet consistency across round-trip translations.

    In Belnap 4-valued logic:
        0 = IRRELEVANT (bottom)
        1 = TRUE
        2 = FALSE
        3 = UNKNOWN / CONFLICT (top)

    Knowledge Meet (⊓_k):
        1 ⊓ 1 = 1
        2 ⊓ 2 = 2
        1 ⊓ 2 = 0  (Epistemic clash wipes out common information)
        3 ⊓ x = x
        0 ⊓ x = 0

    Epistemic Contradiction:
        Occurs when v_orig asserts TRUE (1) and v_reparsed asserts FALSE (2),
        or vice versa. This indicates factual drift or polarity inversion during
        realization and re-parsing.
    """

    def __init__(
        self,
        default_min_preservation: float = 0.50,
        strict_contradiction_check: bool = True,
    ):
        self.default_min_preservation = default_min_preservation
        self.strict_contradiction_check = strict_contradiction_check

    def _ensure_vector(self, item: Union[QuantaGraph, QuantaVector, QuantaNode]) -> QuantaVector:
        """Converts graph, node, or vector into a proposition vector."""
        if isinstance(item, QuantaGraph):
            return item.to_proposition_vector()
        elif isinstance(item, QuantaNode):
            return item.vector
        elif isinstance(item, QuantaVector):
            return item
        raise TypeError(f"Expected QuantaGraph, QuantaNode, or QuantaVector, got {type(item)}")

    def compute_lattice_meet(
        self,
        orig: Union[QuantaGraph, QuantaVector, QuantaNode],
        reparsed: Union[QuantaGraph, QuantaVector, QuantaNode],
    ) -> QuantaVector:
        """Computes element-wise polymorphic lattice meet (v_orig ⊓ v_reparsed)."""
        v_orig = self._ensure_vector(orig)
        v_rep = self._ensure_vector(reparsed)
        return v_orig.meet(v_rep)

    def detect_epistemic_contradictions(
        self,
        v_orig: QuantaVector,
        v_reparsed: QuantaVector,
    ) -> List[ContradictionDetail]:
        """Scans for epistemic contradictions (1 vs. 2) between two vectors."""
        contradictions: List[ContradictionDetail] = []
        n_slots = min(len(v_orig), len(v_reparsed))

        for idx in range(n_slots):
            val_orig = int(v_orig[idx])
            val_rep = int(v_reparsed[idx])

            # In Epistemic bands (or general truth slots): 1 vs 2 is a direct contradiction
            if (val_orig == 1 and val_rep == 2) or (val_orig == 2 and val_rep == 1):
                slot_def = get_slot_by_index(idx)
                slot_name = slot_def.name if slot_def else f"SLOT_{idx}"
                band_id = int(slot_def.band) if slot_def else idx // 128
                contract_str = slot_def.contract.value if slot_def else "EPISTEMIC"

                orig_label = "TRUE" if val_orig == 1 else "FALSE"
                rep_label = "TRUE" if val_rep == 1 else "FALSE"

                contradictions.append(
                    ContradictionDetail(
                        slot_index=idx,
                        slot_name=slot_name,
                        band_id=band_id,
                        contract=contract_str,
                        orig_value=val_orig,
                        reparsed_value=val_rep,
                        description=(
                            f"Epistemic contradiction in {slot_name} (slot {idx}): "
                            f"orig={orig_label} ({val_orig}) vs reparsed={rep_label} ({val_rep})"
                        ),
                    )
                )

        return contradictions

    def compute_slot_preservation_rate(
        self,
        v_orig: QuantaVector,
        v_reparsed: QuantaVector,
        use_meet: bool = True,
    ) -> float:
        """Computes the slot preservation rate of active original slots.

        If use_meet=True:
            Checks whether v_meet[s] == v_orig[s] for all slots s active in v_orig.
        Else:
            Checks agreement over union of active slots.
        """
        if use_meet:
            v_meet = v_orig.meet(v_reparsed)
            active_orig = [s for s in range(len(v_orig)) if int(v_orig[s]) != 0]
            if not active_orig:
                return 1.0
            preserved = sum(1 for s in active_orig if int(v_meet[s]) == int(v_orig[s]))
            return preserved / float(len(active_orig))
        else:
            active_union = set(v_orig.active_slots().keys()) | set(v_reparsed.active_slots().keys())
            if not active_union:
                return 1.0
            matching = sum(1 for s in active_union if int(v_orig[s]) == int(v_reparsed[s]))
            return matching / float(len(active_union))

    def audit_round_trip(
        self,
        orig: Union[QuantaGraph, QuantaVector, QuantaNode],
        reparsed: Union[QuantaGraph, QuantaVector, QuantaNode],
        min_preservation_rate: Optional[float] = None,
    ) -> LatticeMeetResult:
        """Performs full diagnostic audit of round-trip lattice meet invariance."""
        v_orig = self._ensure_vector(orig)
        v_rep = self._ensure_vector(reparsed)
        threshold = min_preservation_rate if min_preservation_rate is not None else self.default_min_preservation

        v_meet = v_orig.meet(v_rep)
        contradictions = self.detect_epistemic_contradictions(v_orig, v_rep)
        preservation = self.compute_slot_preservation_rate(v_orig, v_rep, use_meet=True)
        hamming = v_orig.hamming_distance(v_rep)

        # Audit dropped slots
        dropped_slots: List[Dict[str, Any]] = []
        for s in range(len(v_orig)):
            val1 = int(v_orig[s])
            val2 = int(v_rep[s])
            if val1 in (1, 2) and val2 == 0:
                s_def = get_slot_by_index(s)
                dropped_slots.append({
                    "slot_index": s,
                    "slot_name": s_def.name if s_def else f"SLOT_{s}",
                    "band_id": int(s_def.band) if s_def else s // 128,
                    "orig_value": val1,
                })

        errors: List[str] = []
        for c in contradictions:
            errors.append(c.description)

        if preservation < threshold:
            errors.append(
                f"Lattice meet preservation rate {preservation:.4f} is below threshold {threshold:.4f}"
            )

        is_sound = (len(contradictions) == 0) and (preservation >= threshold)

        return LatticeMeetResult(
            is_sound=is_sound,
            preservation_rate=preservation,
            hamming_distance=hamming,
            contradiction_count=len(contradictions),
            contradictions=contradictions,
            dropped_slots=dropped_slots,
            meet_vector=v_meet,
            orig_vector=v_orig,
            reparsed_vector=v_rep,
            errors=errors,
        )

    def verify_round_trip_invariance(
        self,
        orig_graph: Union[QuantaGraph, QuantaVector],
        reparsed_graph: Union[QuantaGraph, QuantaVector],
        threshold: Optional[float] = None,
    ) -> Tuple[bool, int, List[str]]:
        """Verifies round-trip invariance and returns (is_sound, error_count, error_messages).

        Adheres strictly to Roadmap Task 3.1 specification:
            verify_round_trip_invariance(orig_graph, reparsed_graph) -> Tuple[bool, int, List[str]]
        """
        audit = self.audit_round_trip(orig_graph, reparsed_graph, min_preservation_rate=threshold)
        return audit.is_sound, len(audit.errors), audit.errors


__all__ = [
    "ContradictionDetail",
    "LatticeMeetResult",
    "LatticeInvarianceGate",
]
