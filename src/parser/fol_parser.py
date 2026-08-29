"""First-Order Logic (FOL) formula parser for FOLIO dataset integration."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue
from parser.lexical_grounder import WordNetLexicalGrounder


class FOLTokenType(Enum):
    FORALL = auto()      # ∀, \forall, FORALL
    EXISTS = auto()      # ∃, \exists, EXISTS
    IMPLIES = auto()     # →, ->, \rightarrow, IMPLIES
    AND = auto()         # ∧, &, \land, AND
    OR = auto()          # ∨, |, \lor, OR
    XOR = auto()         # ⊕, \oplus, XOR
    NOT = auto()         # ¬, ~, \neg, NOT, !
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    COMMA = auto()       # ,
    IDENTIFIER = auto()  # Predicate name or variable
    EOF = auto()


@dataclass
class FOLToken:
    type: FOLTokenType
    value: str
    pos: int = 0


class FOLTokenizer:
    """Tokenizes First-Order Logic formula strings."""

    KEYWORDS = {
        "\\forall": FOLTokenType.FORALL,
        "forall": FOLTokenType.FORALL,
        "∀": FOLTokenType.FORALL,
        "\\exists": FOLTokenType.EXISTS,
        "exists": FOLTokenType.EXISTS,
        "∃": FOLTokenType.EXISTS,
        "\\rightarrow": FOLTokenType.IMPLIES,
        "->": FOLTokenType.IMPLIES,
        "→": FOLTokenType.IMPLIES,
        "implies": FOLTokenType.IMPLIES,
        "\\land": FOLTokenType.AND,
        "&": FOLTokenType.AND,
        "∧": FOLTokenType.AND,
        "and": FOLTokenType.AND,
        "\\lor": FOLTokenType.OR,
        "|": FOLTokenType.OR,
        "∨": FOLTokenType.OR,
        "or": FOLTokenType.OR,
        "\\oplus": FOLTokenType.XOR,
        "⊕": FOLTokenType.XOR,
        "xor": FOLTokenType.XOR,
        "\\neg": FOLTokenType.NOT,
        "¬": FOLTokenType.NOT,
        "~": FOLTokenType.NOT,
        "!": FOLTokenType.NOT,
        "not": FOLTokenType.NOT,
    }

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def tokenize(self) -> List[FOLToken]:
        tokens: List[FOLToken] = []
        while self.pos < self.length:
            if self.text[self.pos].isspace():
                self.pos += 1
                continue

            start_pos = self.pos
            ch = self.text[self.pos]

            if ch == "(":
                tokens.append(FOLToken(FOLTokenType.LPAREN, "(", start_pos))
                self.pos += 1
                continue
            elif ch == ")":
                tokens.append(FOLToken(FOLTokenType.RPAREN, ")", start_pos))
                self.pos += 1
                continue
            elif ch == ",":
                tokens.append(FOLToken(FOLTokenType.COMMA, ",", start_pos))
                self.pos += 1
                continue

            if self.text.startswith("->", self.pos):
                tokens.append(FOLToken(FOLTokenType.IMPLIES, "->", start_pos))
                self.pos += 2
                continue

            if ch in self.KEYWORDS:
                tokens.append(FOLToken(self.KEYWORDS[ch], ch, start_pos))
                self.pos += 1
                continue

            if ch == "\\" or ch.isalpha() or ch == "_":
                ident_match = re.match(r"\\[A-Za-z]+|[A-Za-z_][A-Za-z0-9_]*", self.text[self.pos:])
                if ident_match:
                    ident = ident_match.group(0)
                    self.pos += len(ident)
                    ident_lower = ident.lower()
                    if ident in self.KEYWORDS:
                        tokens.append(FOLToken(self.KEYWORDS[ident], ident, start_pos))
                    elif ident_lower in self.KEYWORDS:
                        tokens.append(FOLToken(self.KEYWORDS[ident_lower], ident, start_pos))
                    else:
                        tokens.append(FOLToken(FOLTokenType.IDENTIFIER, ident, start_pos))
                    continue

            self.pos += 1

        tokens.append(FOLToken(FOLTokenType.EOF, "", self.pos))
        return tokens


# AST Definitions
@dataclass
class FOLAST:
    pass


@dataclass
class TermAST(FOLAST):
    name: str
    is_variable: bool = True


@dataclass
class PredicateAST(FOLAST):
    name: str
    args: List[TermAST] = field(default_factory=list)
    is_negated: bool = False


@dataclass
class UnaryOpAST(FOLAST):
    op: str
    operand: FOLAST


@dataclass
class BinaryOpAST(FOLAST):
    op: str
    left: FOLAST
    right: FOLAST


@dataclass
class QuantifierAST(FOLAST):
    quantifier: str  # "FORALL" or "EXISTS"
    variable: str
    body: FOLAST


class FOLASTParser:
    """Recursive descent parser for FOL formulas producing typed AST."""

    def __init__(self, tokens: List[FOLToken]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> FOLToken:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def advance(self) -> FOLToken:
        tok = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def match(self, *expected_types: FOLTokenType) -> bool:
        if self.current().type in expected_types:
            self.advance()
            return True
        return False

    def parse(self) -> Optional[FOLAST]:
        if self.current().type == FOLTokenType.EOF:
            return None
        return self.parse_quantified()

    def parse_quantified(self) -> FOLAST:
        if self.current().type in (FOLTokenType.FORALL, FOLTokenType.EXISTS):
            q_tok = self.advance()
            q_type = "FORALL" if q_tok.type == FOLTokenType.FORALL else "EXISTS"
            var_name = "x"
            if self.current().type == FOLTokenType.IDENTIFIER:
                var_name = self.advance().value
            body = self.parse_quantified()
            return QuantifierAST(quantifier=q_type, variable=var_name, body=body)

        return self.parse_implication()

    def parse_implication(self) -> FOLAST:
        left = self.parse_or()
        while self.current().type == FOLTokenType.IMPLIES:
            self.advance()
            right = self.parse_or()
            left = BinaryOpAST(op="->", left=left, right=right)
        return left

    def parse_or(self) -> FOLAST:
        left = self.parse_and()
        while self.current().type in (FOLTokenType.OR, FOLTokenType.XOR):
            op_tok = self.advance()
            op_str = "^" if op_tok.type == FOLTokenType.XOR else "|"
            right = self.parse_and()
            left = BinaryOpAST(op=op_str, left=left, right=right)
        return left

    def parse_and(self) -> FOLAST:
        left = self.parse_unary()
        while self.current().type == FOLTokenType.AND:
            self.advance()
            right = self.parse_unary()
            left = BinaryOpAST(op="&", left=left, right=right)
        return left

    def parse_unary(self) -> FOLAST:
        if self.current().type == FOLTokenType.NOT:
            self.advance()
            operand = self.parse_unary()
            return UnaryOpAST(op="~", operand=operand)
        return self.parse_primary()

    def parse_primary(self) -> FOLAST:
        if self.current().type == FOLTokenType.LPAREN:
            self.advance()
            expr = self.parse_quantified()
            self.match(FOLTokenType.RPAREN)
            return expr

        if self.current().type == FOLTokenType.IDENTIFIER:
            pred_name = self.advance().value
            args: List[TermAST] = []
            if self.match(FOLTokenType.LPAREN):
                while self.current().type != FOLTokenType.RPAREN and self.current().type != FOLTokenType.EOF:
                    if self.current().type == FOLTokenType.IDENTIFIER:
                        arg_val = self.advance().value
                        is_var = arg_val.islower() and len(arg_val) <= 2
                        args.append(TermAST(name=arg_val, is_variable=is_var))
                    if not self.match(FOLTokenType.COMMA):
                        break
                self.match(FOLTokenType.RPAREN)
            else:
                args.append(TermAST(name="x", is_variable=True))

            return PredicateAST(name=pred_name, args=args)

        tok = self.advance()
        return PredicateAST(name=tok.value or "P", args=[TermAST(name="x", is_variable=True)])


class FOLParser:
    """Parses First-Order Logic formulas into canonical QUANTA ASG graphs with 4-valued polarity."""

    def __init__(self, offline_cache_path: Optional[str] = None):
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)

    def parse_formula(self, fol_str: str, is_query: bool = False) -> QuantaGraph:
        """Parses a First-Order Logic string into a QuantaGraph with full quaternary polarities."""
        fol = fol_str.strip()
        graph = QuantaGraph()

        # Tokenize and parse AST
        tokenizer = FOLTokenizer(fol)
        tokens = tokenizer.tokenize()
        ast_parser = FOLASTParser(tokens)
        ast = ast_parser.parse()

        # Construct Proposition Root Node
        root_node = QuantaNode(literal=fol)
        root_node.set_slot("MODALITY_LITERAL", 1)
        root_node.set_slot("GRAPH_ROOT_NODE", 1)
        root_node.set_slot("TYPE_PROPOSITION", 3 if is_query else 1)
        root_node.set_slot("GRAPH_ASSERTION_CLAIM", 1)
        root_node.set_slot("EPIST_DEDUCTIVE_INFERENCE", 1)
        root_node.set_slot("SOLVER_PROOF_VALIDATED", 1)

        if is_query:
            root_node.set_slot("GRAPH_QUERY_TARGET", 3)
            root_node.set_slot("EPIST_PROB_MARGINAL", 3)
            root_node.set_slot("MODALITY_HYPOTHETICAL", 3)

        # Walk AST to extract quantifiers, connectives, and predicates
        current = ast
        while isinstance(current, QuantifierAST):
            if current.quantifier == "FORALL":
                root_node.set_slot("NSM_ALL", 1)
                root_node.set_slot("LJB_RO_ALL_QUANT", 1)
            elif current.quantifier == "EXISTS":
                root_node.set_slot("NSM_SOME", 1)
                root_node.set_slot("LJB_SUO_AT_LEAST_ONE", 1)
            current = current.body

        # Check connectives & negation
        has_negation = any(t.type == FOLTokenType.NOT for t in tokens)
        has_implication = any(t.type == FOLTokenType.IMPLIES for t in tokens)
        has_and = any(t.type == FOLTokenType.AND for t in tokens)
        has_or = any(t.type == FOLTokenType.OR for t in tokens)
        has_xor = any(t.type == FOLTokenType.XOR for t in tokens)

        if has_implication:
            root_node.set_slot("LJB_GANAI_IF_THEN", 3 if is_query else 1)
            root_node.set_slot("GRAPH_ENTAILMENT_EDGE", 1)
            root_node.set_slot("GRAPH_BRANCH_COND", 1)
            root_node.set_slot("GRAPH_BRANCH_THEN", 1)

        if has_and:
            root_node.set_slot("LJB_JE_AND", 1)
        if has_or:
            root_node.set_slot("LJB_JA_OR", 1)
        if has_xor:
            root_node.set_slot("LJB_JON_XOR", 1)

        if has_negation:
            root_node.set_slot("LJB_NA_NEGATION", 2)
        else:
            root_node.set_slot("NSM_TRUE", 1)

        graph.add_node(root_node, set_as_root=True)

        # Collect predicates from AST
        predicates: List[Tuple[PredicateAST, Optional[str]]] = []  # (pred, branch_role: "COND"|"THEN"|None)

        def collect_predicates(node: Optional[FOLAST], branch_role: Optional[str] = None):
            if node is None:
                return
            if isinstance(node, QuantifierAST):
                collect_predicates(node.body, branch_role)
            elif isinstance(node, BinaryOpAST):
                if node.op == "->":
                    collect_predicates(node.left, "COND")
                    collect_predicates(node.right, "THEN")
                else:
                    collect_predicates(node.left, branch_role)
                    collect_predicates(node.right, branch_role)
            elif isinstance(node, UnaryOpAST):
                if isinstance(node.operand, PredicateAST):
                    pred = PredicateAST(name=node.operand.name, args=node.operand.args, is_negated=True)
                    predicates.append((pred, branch_role))
                else:
                    collect_predicates(node.operand, branch_role)
            elif isinstance(node, PredicateAST):
                predicates.append((node, branch_role))

        collect_predicates(ast)

        # Shared variable nodes across predicates
        variable_nodes: Dict[str, QuantaNode] = {}

        for pred_ast, branch_role in predicates:
            pred_name = pred_ast.name
            pred_lemma = pred_name.lower()
            pred_polarity = 2 if pred_ast.is_negated else (3 if is_query else 1)

            try:
                concept = self.grounder.ground_synset(pred_lemma)
                pred_node = QuantaNode(vector=concept.vector.copy(), anchor=concept.synset_name, literal=pred_name)
            except Exception:
                pred_node = QuantaNode(literal=pred_name)

            pred_node.set_slot("GRAPH_IS_SUB_EXP", 1)
            pred_node.set_slot("TYPE_RELATION_ROLE", pred_polarity)
            pred_node.set_slot("VAL_X1_AGENT", 1)

            root_cat = self.grounder.get_wordnet_root_category(pred_node.anchor or pred_lemma)
            if root_cat is not None:
                pred_node.vector[root_cat] = 1
            elif pred_lemma in ("animal", "fauna", "creature", "beast", "organism"):
                pred_node.set_slot("WN_ANIMAL_FAUNA", 1)

            if pred_ast.is_negated:
                for slot_idx, val in pred_node.vector.active_slots().items():
                    pred_node.vector[slot_idx] = QuaternaryValue.FALSE
                pred_node.set_slot("LJB_NA_NEGATION", 2)

            graph.add_node(pred_node)

            # Connect arguments
            for i, arg in enumerate(pred_ast.args):
                arg_name = arg.name
                if arg_name in variable_nodes:
                    arg_node = variable_nodes[arg_name]
                else:
                    arg_node = QuantaNode(literal=arg_name)
                    arg_node.set_slot("GRAPH_LEAF", 1)
                    arg_node.set_slot("VAL_X1_AGENT", 1)
                    if arg.is_variable:
                        arg_node.set_slot("GRAPH_VARIABLE_BIND", 1)
                        arg_node.anchor = f"var:{arg_name}"
                    else:
                        arg_node.set_slot("TYPE_HUMAN", 1)
                        arg_node.set_slot("ROLE_AGENT_CAPABLE", 1)
                        arg_node.anchor = f"const:{arg_name}"
                    variable_nodes[arg_name] = arg_node
                    graph.add_node(arg_node)

                rel_name = f"VAL_X{min(i+1, 5)}_AGENT" if i == 0 else f"VAL_X{min(i+1, 5)}_PATIENT"
                graph.add_edge(pred_node, rel_name, arg_node)

            # Connect root to predicate
            graph.add_edge(root_node, "GRAPH_IS_SUB_EXP", pred_node)
            if branch_role == "COND":
                graph.add_edge(root_node, "GRAPH_BRANCH_COND", pred_node)
            elif branch_role == "THEN":
                graph.add_edge(root_node, "GRAPH_BRANCH_THEN", pred_node)

        return graph

    def formula_to_vector(self, fol_str: str, is_query: bool = False) -> QuantaVector:
        """Helper returning the 1024-d vector of the parsed FOL proposition ASG."""
        graph = self.parse_formula(fol_str, is_query=is_query)
        return graph.to_proposition_vector()

