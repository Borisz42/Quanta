"""QUANTA Virtual Page-Table, Long-Horizon Memory & Global Knowledge Base Architecture (Phases 6 & 10)."""

from memory.global_kb import GlobalKnowledgeBase
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
from memory.hipporag_ppr import HippoRAGRetriever
from memory.poprag_gating import PoPRAGGating
from memory.passage_store import PassageRecord, PassageStore
from memory.context_assembler import (
    BipartiteProjector,
    DualStreamContext,
    DualStreamContextAssembler,
    ProjectedPassage,
)
from memory.spreading_activation import SpreadingActivationRetriever
from memory.world_state import (
    EntityStateRecord,
    WorldStateManager,
    parse_timestamp,
)

__all__ = [
    "ActiveCanvas",
    "BipartiteProjector",
    "CanonicalNodeInterner",
    "DualStreamContext",
    "DualStreamContextAssembler",
    "EntityStateRecord",
    "GlobalKnowledgeBase",
    "HippoRAGRetriever",
    "PageTable",
    "PassageRecord",
    "PassageStore",
    "PoPRAGGating",
    "ProjectedPassage",
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
