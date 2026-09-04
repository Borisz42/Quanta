"""QUANTA Parser package for translating Natural Language, First-Order Logic, and Code into ASGs."""

from parser.lexical_grounder import WordNetLexicalGrounder, GroundedLexicalConcept
from parser.nlp_forward import NLPForwardParser
from parser.fol_parser import FOLParser
from parser.ast_parser import ASTForwardParser
from parser.chunker import DiscourseChunk, SentenceSpan, DiscourseChunker
from parser.entity_manifest import (
    EntityRecord,
    EntityStorage,
    ActiveEntityManifest,
    EntityMatcher,
    EntityPagingEngine,
)
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
    ExtractedProposition,
)
from parser.transducer import (
    BaseDiscourseTransducer,
    LMStudioTransducer,
    LocalGGUFTransducer,
    MockTransducer,
    create_transducer,
)

__all__ = [
    "WordNetLexicalGrounder",
    "GroundedLexicalConcept",
    "NLPForwardParser",
    "FOLParser",
    "ASTForwardParser",
    "DiscourseChunk",
    "SentenceSpan",
    "DiscourseChunker",
    "EntityRecord",
    "EntityStorage",
    "ActiveEntityManifest",
    "EntityMatcher",
    "EntityPagingEngine",
    "DiscourseExtractionResult",
    "ExtractedEntity",
    "ExtractedEvent",
    "ExtractedRelation",
    "ExtractedProposition",
    "BaseDiscourseTransducer",
    "LMStudioTransducer",
    "LocalGGUFTransducer",
    "MockTransducer",
    "create_transducer",
]

