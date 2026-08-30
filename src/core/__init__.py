"""QUANTA Core: Quaternary logic, polymorphic 2-bit typing, 1024-dimensional canonical slots, and Merkle-addressed ASG."""

from core.types import (
    BandContract,
    QuaternaryValue,
    EpistemicValue,
    StructuralValue,
    RoutingValue,
    RegisterValue,
    QuantaVector,
)
from core.slots import (
    SlotBand,
    BAND_CONTRACTS,
    get_band_contract,
    get_slot_contract,
    CANONICAL_SLOTS,
    get_slot_by_name,
    get_slot_by_index,
    get_slot_names,
    export_canonical_slots_layout,
)
from core.asg import QuantaNode, QuantaGraph
from core.valency import TypeConstraintRegistry, ValencyConstraint, validate_valency

__all__ = [
    "BandContract",
    "QuaternaryValue",
    "EpistemicValue",
    "StructuralValue",
    "RoutingValue",
    "RegisterValue",
    "QuantaVector",
    "SlotBand",
    "BAND_CONTRACTS",
    "get_band_contract",
    "get_slot_contract",
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
