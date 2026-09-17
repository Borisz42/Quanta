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
    ExtractedTimeInterval,
)
from parser.transducer import (
    BaseDiscourseTransducer,
    LMStudioTransducer,
    LocalGGUFTransducer,
    MockTransducer,
    create_transducer,
)
from parser.asg_compiler import ASGCompiler, ASGCompilationError
from parser.sexpr_parser import (
    SExprTokenType,
    SExprToken,
    SExprLexer,
    SExprAtom,
    SExprList,
    SExprParser,
    SExprSyntaxError,
    SExprASTConverter,
    parse_sexpr,
    to_sexpr,
    serialize_to_sexpr,
    parse_to_asg,
)
from parser.unsloth_transducer import (
    UnslothTransducer,
    MockUnslothTransducer,
    MockSExprTransducer,
)
from parser.graph_stitcher import (
    GraphStitcher,
    stitch,
    stitch_to_graph,
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
    "ExtractedTimeInterval",
    "ExtractedRelation",
    "ExtractedProposition",
    "BaseDiscourseTransducer",
    "LMStudioTransducer",
    "LocalGGUFTransducer",
    "MockTransducer",
    "create_transducer",
    "ASGCompiler",
    "ASGCompilationError",
    "SExprTokenType",
    "SExprToken",
    "SExprLexer",
    "SExprAtom",
    "SExprList",
    "SExprParser",
    "SExprSyntaxError",
    "SExprASTConverter",
    "parse_sexpr",
    "to_sexpr",
    "serialize_to_sexpr",
    "parse_to_asg",
    "UnslothTransducer",
    "MockUnslothTransducer",
    "MockSExprTransducer",
    "GraphStitcher",
    "stitch",
    "stitch_to_graph",
]



