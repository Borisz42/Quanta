"""QUANTA: Quaternary Universal Abstract Natural Topology Architecture."""

from core.types import (
    BandContract,
    QuantaVector,
    QuaternaryValue,
    EpistemicValue,
    StructuralValue,
    RoutingValue,
    RegisterValue,
)
from core.slots import CANONICAL_SLOTS, get_slot_by_index, get_slot_by_name, get_slot_names, get_band_contract, get_slot_contract
from core.asg import QuantaNode, QuantaGraph, fold_subgraph, unfold_subgraph
from parser.nlp_forward import NLPForwardParser
from parser.lexical_grounder import WordNetLexicalGrounder
from parser.fol_parser import FOLParser
from parser.ast_parser import ASTForwardParser
from solver.validator_gate import ValidationGate, ValidationResult
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector
from realizer.english_nlg import EnglishRealizer
from realizer.fol_emitter import FOLEmitter
from realizer.code_emitter import CodeEmitter
from pipeline.translator_pipeline import TwoWayTranslationPipeline, TranslationResult, RoundTripResult
from memory.page_table import PageTable, ActiveCanvas, SemanticPageFaultHandler, SimdHammingIndex

__version__ = "0.1.0"

__all__ = [
    "BandContract",
    "QuantaVector",
    "QuaternaryValue",
    "EpistemicValue",
    "StructuralValue",
    "RoutingValue",
    "RegisterValue",
    "CANONICAL_SLOTS",
    "get_slot_by_index",
    "get_slot_by_name",
    "get_slot_names",
    "get_band_contract",
    "get_slot_contract",
    "QuantaNode",
    "QuantaGraph",
    "fold_subgraph",
    "unfold_subgraph",
    "PageTable",
    "ActiveCanvas",
    "SemanticPageFaultHandler",
    "SimdHammingIndex",
    "NLPForwardParser",
    "WordNetLexicalGrounder",
    "FOLParser",
    "ASTForwardParser",
    "ValidationGate",
    "ValidationResult",
    "QuantaInformationProfiler",
    "MRMRSelector",
    "EnglishRealizer",
    "FOLEmitter",
    "CodeEmitter",
    "TwoWayTranslationPipeline",
    "TranslationResult",
    "RoundTripResult",
]

