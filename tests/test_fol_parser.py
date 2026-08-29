"""Unit tests for FOLTokenizer, FOLASTParser, and FOLParser (Phase 6C)."""

import pytest
from parser.fol_parser import (
    FOLParser,
    FOLTokenizer,
    FOLTokenType,
    FOLASTParser,
    QuantifierAST,
    BinaryOpAST,
    PredicateAST,
    TermAST,
)
from core.types import QuaternaryValue


@pytest.fixture(scope="module")
def fol_parser():
    return FOLParser()


def test_fol_tokenizer():
    """Verify Phase 6 Item 6C.1: Tokenize FOL strings."""
    formula = r"\forall x (Dog(x) \rightarrow Animal(x)) \land \exists y (\neg Cat(y) \lor Bird(y))"
    tokenizer = FOLTokenizer(formula)
    tokens = tokenizer.tokenize()

    token_types = [t.type for t in tokens if t.type != FOLTokenType.EOF]
    assert token_types[0] == FOLTokenType.FORALL
    assert token_types[1] == FOLTokenType.IDENTIFIER  # x
    assert token_types[2] == FOLTokenType.LPAREN
    assert token_types[3] == FOLTokenType.IDENTIFIER  # Dog
    assert token_types[4] == FOLTokenType.LPAREN
    assert token_types[5] == FOLTokenType.IDENTIFIER  # x
    assert token_types[6] == FOLTokenType.RPAREN
    assert token_types[7] == FOLTokenType.IMPLIES     # \rightarrow
    assert token_types[8] == FOLTokenType.IDENTIFIER  # Animal


def test_fol_ast_parser():
    """Verify Phase 6 Item 6C.2: Recursive descent AST parser."""
    formula = r"\forall x (Dog(x) -> Animal(x))"
    tokens = FOLTokenizer(formula).tokenize()
    ast = FOLASTParser(tokens).parse()

    assert isinstance(ast, QuantifierAST)
    assert ast.quantifier == "FORALL"
    assert ast.variable == "x"
    assert isinstance(ast.body, BinaryOpAST)
    assert ast.body.op == "->"
    assert isinstance(ast.body.left, PredicateAST)
    assert ast.body.left.name == "Dog"
    assert isinstance(ast.body.right, PredicateAST)
    assert ast.body.right.name == "Animal"


def test_fol_example_e_quanta_graph(fol_parser):
    """Verify Phase 6 Item 6C.3, 6C.4: Canonical Example E ASG mapping."""
    formula = r"\forall x (Dog(x) \rightarrow Animal(x))"
    graph = fol_parser.parse_formula(formula)

    assert graph.root is not None
    root = graph.root

    # Root Implication Head Node
    assert root.get_slot("NSM_ALL") == 1
    assert root.get_slot("LJB_RO_ALL_QUANT") == 1
    assert root.get_slot("LJB_GANAI_IF_THEN") == 1
    assert root.get_slot("GRAPH_ROOT_NODE") == 1
    assert root.get_slot("GRAPH_BRANCH_COND") == 1
    assert root.get_slot("GRAPH_BRANCH_THEN") == 1
    assert root.get_slot("GRAPH_ENTAILMENT_EDGE") == 1
    assert root.get_slot("TYPE_PROPOSITION") == 1
    assert root.get_slot("MODALITY_LITERAL") == 1
    assert root.get_slot("EPIST_DEDUCTIVE_INFERENCE") == 1
    assert root.get_slot("SOLVER_PROOF_VALIDATED") == 1

    # Branch edges
    assert "GRAPH_BRANCH_COND" in root.edges
    assert "GRAPH_BRANCH_THEN" in root.edges
    assert len(root.edges["GRAPH_BRANCH_COND"]) == 1
    assert len(root.edges["GRAPH_BRANCH_THEN"]) == 1

    antecedent_cid = root.edges["GRAPH_BRANCH_COND"][0]
    consequent_cid = root.edges["GRAPH_BRANCH_THEN"][0]

    antecedent = graph.get_node(antecedent_cid)
    consequent = graph.get_node(consequent_cid)

    assert antecedent is not None
    assert consequent is not None

    # Antecedent Node (Dog)
    assert antecedent.literal == "Dog"
    assert antecedent.anchor == "wn:dog.n.01"
    assert antecedent.get_slot("GRAPH_IS_SUB_EXP") == 1
    assert antecedent.get_slot("VAL_X1_AGENT") == 1
    assert antecedent.get_slot("TYPE_RELATION_ROLE") == 1
    assert antecedent.get_slot("WN_ANIMAL_FAUNA") == 1

    # Consequent Node (Animal)
    assert consequent.literal == "Animal"
    assert consequent.anchor == "wn:animal.n.01"
    assert consequent.get_slot("GRAPH_IS_SUB_EXP") == 1
    assert consequent.get_slot("VAL_X1_AGENT") == 1
    assert consequent.get_slot("TYPE_RELATION_ROLE") == 1
    assert consequent.get_slot("WN_ANIMAL_FAUNA") == 1

    # Bound Variable (x)
    assert "VAL_X1_AGENT" in antecedent.edges
    assert "VAL_X1_AGENT" in consequent.edges
    var_cid_ante = antecedent.edges["VAL_X1_AGENT"][0]
    var_cid_cons = consequent.edges["VAL_X1_AGENT"][0]
    assert var_cid_ante == var_cid_cons  # Shared variable node

    var_node = graph.get_node(var_cid_ante)
    assert var_node is not None
    assert var_node.literal == "x"
    assert var_node.anchor == "var:x"
    assert var_node.get_slot("GRAPH_VARIABLE_BIND") == 1
    assert var_node.get_slot("GRAPH_LEAF") == 1
    assert var_node.get_slot("VAL_X1_AGENT") == 1


def test_fol_complex_conjunction_and_negation(fol_parser):
    """Verify FOL parser handles conjunction, disjunction, and negation."""
    formula = r"\exists y (\neg Cat(y) \land Dog(y))"
    graph = fol_parser.parse_formula(formula)

    assert graph.root is not None
    root = graph.root

    assert root.get_slot("NSM_SOME") == 1
    assert root.get_slot("LJB_SUO_AT_LEAST_ONE") == 1
    assert root.get_slot("LJB_JE_AND") == 1
    assert root.get_slot("LJB_NA_NEGATION") == 2
