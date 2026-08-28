"""QUANTA Parser & Grounder: Lexical anchors, WordNet hierarchy mapping, forward translation, and AST parsing."""

from quanta.parser.lexical_grounder import WordNetLexicalGrounder, GroundedLexicalConcept
from quanta.parser.nlp_forward import NLPForwardParser
from quanta.parser.fol_parser import FOLParser
from quanta.parser.ast_parser import ASTForwardParser

__all__ = [
    "WordNetLexicalGrounder",
    "GroundedLexicalConcept",
    "NLPForwardParser",
    "FOLParser",
    "ASTForwardParser",
]
