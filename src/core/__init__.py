"""QUANTA Core: Quaternary logic, 1024-dimensional canonical slots, and Merkle-addressed ASG."""

from core.types import QuaternaryValue, QuantaVector
from core.slots import (
    SlotBand,
    CANONICAL_SLOTS,
    get_slot_by_name,
    get_slot_by_index,
    get_slot_names,
    export_canonical_slots_layout,
)
from core.asg import QuantaNode, QuantaGraph
from core.valency import TypeConstraintRegistry, ValencyConstraint, validate_valency

__all__ = [
    "QuaternaryValue",
    "QuantaVector",
    "SlotBand",
    "CANONICAL_SLOTS",
    "get_slot_by_name",
    "get_slot_by_index",
    "get_slot_names",
    "export_canonical_slots_layout",
    "QuantaNode",
    "QuantaGraph",
    "TypeConstraintRegistry",
    "ValencyConstraint",
    "validate_valency",
]
