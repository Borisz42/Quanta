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

__all__ = [
    "ActiveCanvas",
    "CanonicalNodeInterner",
    "PageTable",
    "SemanticPageFaultHandler",
    "SimdHammingIndex",
    "SpreadingActivationRetriever",
    "StringInternTable",
    "batch_quaternary_hamming",
    "get_global_interner",
    "reset_global_interner",
    "set_global_interner",
    "topk_hamming_search",
]
