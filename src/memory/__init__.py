"""QUANTA Virtual Page-Table & Long-Horizon Memory Architecture (Phase 6)."""

from memory.node_interner import (
    CanonicalNodeInterner,
    get_global_interner,
    set_global_interner,
    reset_global_interner,
)
from memory.page_table import (
    ActiveCanvas,
    PageTable,
    SemanticPageFaultHandler,
    SimdHammingIndex,
    StringInternTable,
    batch_quaternary_hamming,
    topk_hamming_search,
)
from memory.spreading_activation import SpreadingActivationRetriever
from memory.world_state import (
    EntityStateRecord,
    WorldStateManager,
    parse_timestamp,
)

__all__ = [
    "ActiveCanvas",
    "CanonicalNodeInterner",
    "EntityStateRecord",
    "PageTable",
    "SemanticPageFaultHandler",
    "SimdHammingIndex",
    "SpreadingActivationRetriever",
    "StringInternTable",
    "WorldStateManager",
    "batch_quaternary_hamming",
    "get_global_interner",
    "parse_timestamp",
    "reset_global_interner",
    "set_global_interner",
    "topk_hamming_search",
]

