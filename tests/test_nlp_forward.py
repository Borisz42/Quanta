"""Tests for NLPForwardParser and First-Order Logic formula parser."""

import pytest
from parser.nlp_forward import NLPForwardParser
from parser.fol_parser import FOLParser
from core.types import QuaternaryValue


@pytest.fixture(scope="module")
def nlp_parser():
    return NLPForwardParser()


@pytest.fixture(scope="module")
def fol_parser():
    return FOLParser()


def test_action_sentence_parsing(nlp_parser):
    # "A brown dog chased the mailman into the garden."
    graph = nlp_parser.parse_sentence("A brown dog chased the mailman into the garden.")
    
    assert graph.root is not None
    root = graph.root
    
    # Root event slots
    assert root.get_slot("TYPE_EVENT") == QuaternaryValue.TRUE
    assert root.get_slot("NSM_MOVE") == QuaternaryValue.TRUE
    assert root.get_slot("LJB_PU_PAST_TENSE") == QuaternaryValue.TRUE
    
    # Graph edges
    assert "VAL_X1_AGENT" in root.edges
    assert "VAL_X2_PATIENT" in root.edges
    assert "VAL_X3_DESTINATION" in root.edges
    
    # Agent node (dog)
    agent_cid = root.edges["VAL_X1_AGENT"][0]
    agent_node = graph.get_node(agent_cid)
    assert agent_node is not None
    assert agent_node.get_slot("TYPE_ANIMATE") == QuaternaryValue.TRUE
    assert agent_node.get_slot("ROLE_AGENT_CAPABLE") == QuaternaryValue.TRUE


def test_mental_predicate_parsing(nlp_parser):
    # "Alice believes that the earth is round."
    graph = nlp_parser.parse_sentence("Alice thinks about astronomy.")
    assert graph.root is not None
    assert graph.root.get_slot("NSM_THINK") == QuaternaryValue.TRUE


def test_fol_implication_parsing(fol_parser):
    # "\forall x (Human(x) \rightarrow Mortal(x))"
    graph = fol_parser.parse_formula(r"\forall x (Human(x) \rightarrow Mortal(x))")
    assert graph.root is not None
    root = graph.root

    assert root.get_slot("NSM_ALL") == QuaternaryValue.TRUE
    assert root.get_slot("LJB_RO_ALL_QUANT") == QuaternaryValue.TRUE
    assert root.get_slot("LJB_GANAI_IF_THEN") == QuaternaryValue.TRUE
    assert root.get_slot("TYPE_PROPOSITION") == QuaternaryValue.TRUE
    assert root.get_slot("EPIST_DEDUCTIVE_INFERENCE") == QuaternaryValue.TRUE


def test_negated_past_sentence_parsing(nlp_parser):
    # "A person did not see a cat."
    graph = nlp_parser.parse_sentence("A person did not see a cat.")
    assert graph.root is not None
    root = graph.root

    assert root.literal == "A person did not see a cat."
    assert root.get_slot("NSM_SEE") == QuaternaryValue.FALSE
    assert root.get_slot("LJB_NA_NEGATION") == QuaternaryValue.FALSE
    assert root.get_slot("LJB_PU_PAST_TENSE") == QuaternaryValue.TRUE
    assert root.get_slot("NSM_DIE") == QuaternaryValue.IRRELEVANT
    assert "VAL_X1_AGENT" in root.edges
    assert "VAL_X2_PATIENT" in root.edges

